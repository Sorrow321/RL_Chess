"""One generation of batched self-play (docs/05), driving the docs/02 Runner.

    ev = Evaluator(net)
    gen = play_generation(ev, total_games=256, sims=400, n_parallel=256, seed=1)
    chunk = examples_to_chunk(gen["examples"])     # -> replay buffer
    stats = result_stats(gen["results"])           # -> the CSV row

The whole fleet is one GPU forward per cycle: `pending()` returns every game's
leaf that needs an evaluation, `feed()` resumes all of them. Nothing here
touches a tree — the C++ side owns the search, this file owns the loop around
it and the bookkeeping the training loop needs.

**Positions are not in the dump.** docs/02 stores raw facts per move (the move
played, the root's legal indices and visits) and reconstructs planes by
replaying moves through the C++ encoder, which keeps dumps at ~1 KB/move and
the encoder the single source of truth. `examples_to_chunk` does that replay
breadth-first — one `encode_batch` per ply for the entire generation rather
than one per position — so it costs a fraction of a second against minutes of
search.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce


def play_generation(evaluator, total_games, sims, n_parallel=256, c_puct=1.5,
                    dirichlet_alpha=0.3, dirichlet_eps=0.25, temp_plies=15,
                    resign_threshold=-0.95, seed=0, progress=None, progress_every=30.0):
    """Play `total_games` with the current net. -> dict of dumps and throughput.

    `progress(games_done, total, elapsed, evals)` is called at most every
    `progress_every` seconds — a generation is minutes long and a silent
    process is indistinguishable from a hung one.

    The returned `avg_batch` is fleet utilization made visible: it is the mean
    number of leaves per GPU forward, and it decays over a generation because
    slots that finish their last game go idle (with `n_parallel == total_games`
    there is nothing left to refill them with). If it sits far below
    `n_parallel`, raise `total_games` before blaming the net.
    """
    runner = ce.Runner(total_games=total_games, n_parallel=min(n_parallel, total_games),
                       sims=sims, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha,
                       dirichlet_eps=dirichlet_eps, temp_plies=temp_plies,
                       resign_threshold=resign_threshold, seed=seed)

    t0 = last = time.time()
    cycles = evals = 0
    while True:
        batch = runner.pending()
        if not batch.size:
            break
        runner.feed(*evaluator(batch))
        cycles += 1
        evals += batch.shape[0]
        now = time.time()
        if progress and now - last >= progress_every:
            last = now
            progress(runner.games_completed, total_games, now - t0, evals)

    seconds = time.time() - t0
    return {"examples": runner.get_examples(), "results": runner.get_results(),
            "seconds": seconds, "cycles": cycles, "evals": evals,
            "avg_batch": evals / max(cycles, 1),
            "evals_per_s": evals / max(seconds, 1e-9)}


def result_stats(results):
    """Per-generation game statistics — docs/05's watch-list, computed once.

    `resign_fp_frac` is the docs/02 validation guard: among the 10% of games
    that play on with resignation disabled, how often the side that *would*
    have resigned went on to salvage a draw or a win. Above ~5% the threshold
    is too hot and is throwing away games that were not lost.
    """
    n = len(results["game_id"])
    if n == 0:
        return {"games": 0, "avg_plies": 0.0, "draw_frac": 0.0, "resign_frac": 0.0,
                "white_score": 0.0, "audit_games": 0, "resign_triggers": 0,
                "resign_fp_frac": float("nan"), "terminations": {}}

    result = results["result"].astype(np.int64)        # +1 white, -1 black, 0 draw
    plies = results["plies"].astype(np.int64)
    term = results["termination"].astype(np.int64)
    audit = results["resign_disabled"].astype(bool)
    trig_side = results["would_resign_side"].astype(np.int64)   # -1 none, 0 white, 1 black

    # In audit games the resignation was recorded but not acted on, so the game
    # played to its real end: that is what makes the false-positive measurable.
    triggered = audit & (trig_side >= 0)
    side_result = np.where(trig_side == 1, -result, result)     # from the trigger side's view
    salvaged = int((triggered & (side_result >= 0)).sum())
    n_trig = int(triggered.sum())

    counts = {}
    for t in np.unique(term):
        counts[ce.Termination(int(t)).name.lower()] = int((term == t).sum())

    return {
        "games": n,
        "avg_plies": float(plies.mean()),
        "draw_frac": float((result == 0).mean()),
        "resign_frac": float((term == int(ce.Termination.RESIGN)).mean()),
        # 0.5 = balanced. Persistent drift means the value head or the search
        # has a colour bias, which is a bug long before it is an opening book.
        "white_score": float((result.mean() + 1.0) / 2.0),
        "audit_games": int(audit.sum()),
        "resign_triggers": n_trig,
        "resign_fp_frac": (salvaged / n_trig) if n_trig else float("nan"),
        "terminations": counts,
    }


# --- example dump -> replay buffer chunk --------------------------------------

def empty_chunk():
    return {"planes": np.empty((0, ce.NUM_PLANES, 8, 8), np.uint8),
            "legal_off": np.empty(0, np.int64),
            "n_legal": np.empty(0, np.int32),
            "legal_idx": np.empty(0, np.uint16),
            "visits": np.empty(0, np.uint16),
            "z": np.empty(0, np.float32),
            "ply": np.empty(0, np.int32)}


def examples_to_chunk(examples):
    """Runner dump -> the arrays the replay buffer stores.

    Rows keep the dump's own order, so `legal_idx`/`visits` keep the offsets the
    dump already implies and nothing ragged has to be rebuilt. Only the *replay*
    needs games grouped and ply-ordered, and that grouping is a lexsort over the
    row indices, not a permutation of the data.

    A corrupt dump raises here rather than poisoning the buffer: `push_u16`
    resolves every move against the legal movelist of the position the replay
    actually reached.
    """
    n = len(examples["game_id"])
    if n == 0:
        return empty_chunk()

    n_legal = examples["n_legal"].astype(np.int64)
    off = np.concatenate([[0], np.cumsum(n_legal)])
    if off[-1] != len(examples["legal_idx"]) or off[-1] != len(examples["visits"]):
        raise ValueError("example dump is inconsistent: n_legal does not sum to the flat arrays")

    game_id, ply, move = examples["game_id"], examples["ply"], examples["move"]
    order = np.lexsort((ply, game_id))          # rows of each game, in ply order
    g = game_id[order]
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    counts = np.diff(np.concatenate([starts, [n]]))

    planes = np.empty((n, ce.NUM_PLANES, 8, 8), np.uint8)
    games = [ce.Game() for _ in starts]
    alive = np.arange(len(starts))
    k = 0
    while alive.size:
        rows = order[starts[alive] + k]
        # One encoder call for every game's k-th position at once. planes is
        # filled at the *dump's* row indices, so it stays aligned with the
        # flat legal_idx/visits arrays.
        planes[rows] = ce.encode_batch([games[i] for i in alive])
        for i, row in zip(alive, rows):
            games[i].push_u16(int(move[row]))
        k += 1
        alive = alive[counts[alive] > k]

    return {"planes": planes,
            "legal_off": off[:-1].copy(),
            "n_legal": n_legal.astype(np.int32),
            "legal_idx": np.ascontiguousarray(examples["legal_idx"]),
            "visits": np.ascontiguousarray(examples["visits"]),
            # docs/03: value target is the game result from the mover's
            # perspective, which is exactly what the dump's `result` already is.
            "z": examples["result"].astype(np.float32),
            "ply": ply.astype(np.int32)}


def chunk_stats(chunk):
    """Cheap summary of one generation's positions, for the run log."""
    n = len(chunk["z"])
    if n == 0:
        return {"positions": 0, "draw_frac": 0.0, "legal_mean": 0.0, "visit_entropy": 0.0}
    # Entropy of the search's visit distribution, in nats, averaged over
    # positions: how sharp the tree's opinion is. It falls as the net learns
    # and rises when sims are raised — the cheapest read on both.
    off, n_legal = chunk["legal_off"], chunk["n_legal"].astype(np.int64)
    take = np.arange(0, n, max(1, n // 4096))
    ent = 0.0
    for i in take:
        v = chunk["visits"][off[i]:off[i] + n_legal[i]].astype(np.float64)
        s = v.sum()
        if s > 0:
            p = v[v > 0] / s
            ent -= float((p * np.log(p)).sum())
    return {"positions": n,
            "draw_frac": float((chunk["z"] == 0).mean()),
            "legal_mean": float(n_legal.mean()),
            "visit_entropy": ent / max(len(take), 1)}
