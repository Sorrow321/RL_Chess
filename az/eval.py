"""Elo evaluation harness (docs/06-eval.md) — the number every phase exits on.

    python -m az.eval --ckpt runs/bootstrap/bootstrap.pt --anchor random,sf1320
    python -m az.eval --ckpt ... --sims 1600 --anchor sf1700,sf2000 --games 400
    python -m az.eval --calibrate --anchor random,skill0,skill3,sf1320,sf1700
    python -m az.eval --ckpt ... --backend cutechess --dry-run

Two backends, same protocol and the same anchors:

* **inproc** (default) — plays the match in this process. Our side searches
  through `chess_engine.Searcher`, so every game in flight contributes its
  pending leaves to one GPU forward; a 100-game match costs about what a
  handful of games would. This is docs/06's named fallback, and it is the
  backend that the tests cover end to end.
* **cutechess** — emits and runs a `cutechess-cli` command against the same
  UCI wrapper (`az.uci`) and the same EPD book (`az.book`). That is docs/06's
  preferred path because cutechess is battle-tested at exactly this job, and
  it is what unlocks SPRT and `ordo` over accumulated PGNs. It needs
  `cutechess-cli` on PATH; `--dry-run` prints the command without running it.

## What the numbers mean

Elo comes from the match score with a 95% band, so it is a *relative* number:
"+120 vs sf1320", never "1440 Elo" on its own. docs/06 asks for two adjacent
anchors at all times, because a single anchor saturates — pass `--anchor a,b`
and both get played and reported.

Anchor strength is frozen by node count (`--nodes`, default `ANCHOR_NODES`),
not by wall clock: Stockfish's own time management would otherwise make every
result a statement about how busy this box was. docs/06's first caveat is that
`UCI_Elo` calibration assumes that time management, so the ladder's real
spacing has to be *measured* — that is what `--calibrate` does, and the node
count must not move afterwards.

Draw adjudication is deliberately just self-play's rule: the 250-ply cap of
`chess_engine.MAX_PLIES`. docs/06 warns that eval and self-play drift apart if
their adjudication philosophies differ, and a cap both sides already agree on
is the cheapest way not to drift. cutechess's score-based `-draw`/`-resign`
adjudication is available through `--cutechess-arg` when a marathon needs
cutting short, but it is opt-in.

Reproducibility: our side is deterministic given `--seed` (no root noise in
match play), and the seed, simulation count and checkpoint land in the PGN
tags. Stockfish's `Skill Level`/`UCI_Elo` randomness is seeded from its own
clock and is not reproducible — that is the anchor's noise, and it is the
reason a pairing needs 100 games rather than 10.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

from az.book import lines as book_lines

WHITE, BLACK = 0, 1

# Frozen anchor strength (docs/06 caveat 1). Every Elo ever recorded by this
# harness is relative to anchors playing at this budget; changing it silently
# invalidates the whole history, so change it with a new column in eval.csv and
# a fresh calibration run, not in passing.
ANCHOR_NODES = 10_000

# The ladder of docs/06, in order. Names are parsed rather than looked up, so
# `sf1450` or `skill7` work too; this list is what --calibrate defaults to and
# what the help text advertises.
LADDER = ["random", "skill0", "skill1", "skill3", "skill5", "sf1320", "sf1700", "sf2000", "sf2400"]

# Stockfish's calibrated floor. Below it, `Skill Level` is the only honest way
# down, and its levels are not Elo — hence the calibration mode.
SF_ELO_FLOOR = 1320


# --- statistics --------------------------------------------------------------
# az/gates.py carries its own copy of the score->Elo conversion so the bootstrap
# gate can run before this module exists. This is the canonical one.

def score_to_elo(score):
    """Match score in (0, 1) -> Elo difference. A shutout has no finite Elo, so
    the score is clamped and 0% reads as -2400 rather than -inf."""
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


# A uniform result — every game a win, or a loss, or a draw — has zero sample
# variance, and the normal approximation then reports perfect precision from
# the single least informative outcome there is. Floor the variance so those
# matches get the rule-of-three bound instead: 0 wins in n games still allows a
# true score up to 3/n away. Chosen so 1.96*se == 3/n, and small enough that it
# never touches a match with a mixed result.
UNIFORM_RESULT_VAR = (3.0 / 1.96) ** 2


def elo_from_wdl(wins, draws, losses):
    """Elo difference and a 95% band, from the actual win/draw/loss split.

    The band uses the observed variance of the per-game score rather than a
    binomial on the score alone, because a draw is not half a coin flip. A
    binomial band assumes every game is decisive and is the worst case: ±69 Elo
    over 100 even games. docs/06 quotes ±35, which is that same 100 games at
    the draw rate a real match has — and this reproduces either, because it
    reads the draw rate off the results instead of assuming one.
    """
    n = wins + draws + losses
    if n == 0:
        return 0.0, 0.0, 0.0
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score ** 2) / n
    se = math.sqrt(max(var, UNIFORM_RESULT_VAR / n) / n)
    return score_to_elo(score), score_to_elo(score - 1.96 * se), score_to_elo(score + 1.96 * se)


def los(wins, losses):
    """Likelihood of superiority: P(we are stronger) given the decisive games."""
    if wins + losses == 0:
        return 0.5
    return 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2.0 * (wins + losses))))


def sprt_llr(wins, draws, losses, elo0, elo1):
    """Log-likelihood ratio for H1(elo1) over H0(elo0), draw ratio held fixed.

    The standard trinomial SPRT: the observed draw rate is taken as given, so
    only the win/loss split carries evidence. Compare against the bounds from
    `sprt_bounds` — cross the upper one and the improvement is real at the
    chosen error rates, cross the lower one and it is not, in between keep
    playing.
    """
    n = wins + draws + losses
    if n == 0 or wins == 0 or losses == 0:
        return 0.0
    draw_ratio = draws / n

    def split(elo):
        s = 1.0 / (1.0 + 10.0 ** (-elo / 400.0))
        w = s - draw_ratio / 2.0
        return (min(max(w, 1e-9), 1.0), min(max(1.0 - s - draw_ratio / 2.0, 1e-9), 1.0))

    w0, l0 = split(elo0)
    w1, l1 = split(elo1)
    return wins * math.log(w1 / w0) + losses * math.log(l1 / l0)


def sprt_bounds(alpha=0.05, beta=0.05):
    """(lower, upper) LLR bounds: below -> accept H0, above -> accept H1."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


# --- players -----------------------------------------------------------------
# Protocol: `moves(records) -> [uci]`, one move per record, records being the
# match loop's live games. Batched on purpose — the net side is one GPU forward
# per ply of the whole fleet, not per game.

class RandomPlayer:
    """The floor of the ladder. Beating it ~100% is a legality check, not Elo."""

    def __init__(self, seed=0):
        self.name = "random"
        self.rng = np.random.default_rng(seed)

    def moves(self, recs):
        return [self.rng.choice(r["game"].legal_moves()) for r in recs]

    def close(self):
        pass


class SearchPlayer:
    """The agent: net + MCTS at a fixed simulation count, fleet-batched.

    `n_slots` must cover the match's concurrency; larger matches are chunked
    rather than refused, which costs a smaller batch but never a wrong move.
    """

    def __init__(self, evaluator, sims, n_slots, c_puct=1.5, noise=0.0, seed=0, label=None):
        self.name = label or f"az-{sims}s"
        self.ev = evaluator
        self.sims = sims
        self.n_slots = max(1, n_slots)
        self.searcher = ce.Searcher(n_slots=self.n_slots, sims=sims, c_puct=c_puct,
                                    dirichlet_eps=noise, seed=seed)
        self.evals = 0  # positions sent to the net, for the throughput line

    def moves(self, recs):
        out = []
        for start in range(0, len(recs), self.n_slots):
            chunk = recs[start:start + self.n_slots]
            for i, rec in enumerate(chunk):
                self.searcher.set_position(i, rec["game"])
            while True:
                batch = self.searcher.pending()
                if not batch.size:
                    break
                self.searcher.feed(*self.ev(batch))
                self.evals += batch.shape[0]
            out.extend(self.searcher.best_move(i) for i in range(len(chunk)))
        return out

    def close(self):
        pass


class GreedyPolicyPlayer:
    """`--sims 0`: argmax over the policy, no search. docs/04's gate 1 player.

    Reuses az.gates rather than reimplementing it — this is the same measurement
    the bootstrap gate makes, just placed on the Elo ladder.
    """

    def __init__(self, evaluator):
        from az.gates import GreedyPlayer
        self.name = "az-greedy"
        self.inner = GreedyPlayer(evaluator)

    def moves(self, recs):
        return self.inner.moves([r["game"] for r in recs])

    def close(self):
        pass


class StockfishPlayer:
    """One node-limited Stockfish process, either `UCI_Elo` or `Skill Level`.

    Node-limited so the anchor's strength does not depend on how busy this box
    is (docs/06). It keeps a python-chess board per live game, advanced move by
    move, so Stockfish sees the real history and can claim repetitions — a
    Stockfish fed bare FENs repeats winning positions and hands us draws we did
    not earn.
    """

    def __init__(self, path, elo=None, skill=None, nodes=ANCHOR_NODES, threads=1, hash_mb=16, label=None):
        import chess.engine

        self.name = label or (f"sf{elo}" if elo is not None else f"skill{skill}")
        self.nodes = nodes
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        options = {"Threads": threads, "Hash": hash_mb}
        if skill is not None:
            options["Skill Level"] = skill
        else:
            options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.engine.configure(options)
        self.limit = chess.engine.Limit(nodes=nodes)

    @staticmethod
    def board_of(rec):
        """The record's python-chess board, advanced to the current position."""
        import chess

        board = rec.get("sf_board")
        if board is None:
            board = rec["sf_board"] = chess.Board(rec["start_fen"])
            rec["sf_plies"] = 0
        for uci in rec["moves"][rec["sf_plies"]:]:
            board.push(chess.Move.from_uci(uci))
        rec["sf_plies"] = len(rec["moves"])
        return board

    def moves(self, recs):
        return [self.engine.play(self.board_of(r), self.limit).move.uci() for r in recs]

    def close(self):
        self.engine.close()


def parse_anchor(name):
    """`random` | `skill<0-20>` | `sf<elo>` -> a spec dict. Raises on garbage."""
    name = name.strip().lower()
    if name == "random":
        return {"kind": "random", "name": "random"}
    if name.startswith("skill"):
        level = int(name[5:])
        if not 0 <= level <= 20:
            raise ValueError(f"skill level out of range: {name}")
        return {"kind": "stockfish", "skill": level, "name": name}
    if name.startswith("sf"):
        elo = int(name[2:])
        if elo < SF_ELO_FLOOR:
            raise ValueError(f"{name}: Stockfish's UCI_Elo floor is {SF_ELO_FLOOR}; "
                             f"use skill0-5 for anchors below it")
        return {"kind": "stockfish", "elo": elo, "name": name}
    if name == "az":
        return {"kind": "az", "name": "az"}
    raise ValueError(f"unknown anchor {name!r} (expected az, random, skill<N> or sf<elo>)")


def make_player(spec, args, n_slots, evaluator=None, seed=0):
    """Spec dict -> a player. The caller owns closing it."""
    if spec["kind"] == "random":
        return RandomPlayer(seed=seed)
    if spec["kind"] == "stockfish":
        if not args.stockfish:
            raise SystemExit(f"anchor {spec['name']} needs a Stockfish binary: "
                             "pass --stockfish /path/to/stockfish or set $STOCKFISH")
        return StockfishPlayer(args.stockfish, elo=spec.get("elo"), skill=spec.get("skill"),
                               nodes=args.nodes, threads=args.sf_threads, label=spec["name"])
    if args.sims <= 0:
        return GreedyPolicyPlayer(evaluator)
    return SearchPlayer(evaluator, args.sims, n_slots, c_puct=args.c_puct,
                        noise=args.noise, seed=seed, label=f"az-{args.sims}s")


# --- the match ---------------------------------------------------------------

def load_book(path=None, plies=4, limit=None):
    """-> [(name, moves|None, fen)]. `moves` is None for an external EPD book."""
    if not path:
        return book_lines(limit=limit, plies=plies)

    entries = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fen, _, rest = line.partition(";")
            name = rest.strip()
            if name.startswith('id "') and name.endswith('";'):
                name = name[4:-2]
            entries.append((name or f"epd{len(entries)}", None, fen.strip()))
    if not entries:
        raise SystemExit(f"{path}: no positions in the opening book")
    return entries[:limit]


def new_record(game_id, openings, max_plies):
    """One game: an opening, a colour, and the two views of the position."""
    name, moves, fen = openings[(game_id // 2) % len(openings)]
    start_fen = ce.STARTPOS if moves is not None else fen
    game = ce.Game(start_fen, max_plies)
    for uci in moves or []:
        game.push(uci)
    return {"id": game_id, "opening": name, "our_color": game_id % 2,
            "game": game, "moves": list(moves or []), "start_fen": start_fen,
            "book_plies": len(moves or [])}


def play_match(player, opponent, n_games, openings, max_plies=ce.MAX_PLIES, concurrency=32,
               progress=None):
    """`n_games` games as colour-swapped opening pairs, played from a rolling pool.

    A pool rather than one lockstep wave: games end at wildly different lengths,
    and a wave spends its tail searching two positions per GPU forward. Slot
    frees up, next game starts, batch stays full — the same shape the self-play
    runner uses, for the same reason.
    """
    pool = [None] * min(max(concurrency, 1), n_games)
    launched, done, t0 = 0, [], time.time()

    while True:
        for k in range(len(pool)):
            if pool[k] is None and launched < n_games:
                pool[k] = new_record(launched, openings, max_plies)
                launched += 1

        live = []
        for k in range(len(pool)):
            rec = pool[k]
            if rec is None:
                continue
            out = rec["game"].outcome()
            if out.over:
                # Outcome.value is from the side to move; store it for white.
                rec["result_white"] = out.value if rec["game"].white_to_move else -out.value
                rec["termination"] = int(out.reason)
                rec["plies"] = rec["game"].ply
                rec.pop("sf_board", None)  # python-chess boards do not need to outlive the game
                done.append(rec)
                pool[k] = None
                if progress:
                    progress(len(done), n_games, time.time() - t0)
            else:
                live.append(rec)

        if not live:
            if launched >= n_games:
                break
            continue  # every live game just ended; refill and carry on

        ours, theirs = [], []
        for rec in live:
            side = WHITE if rec["game"].white_to_move else BLACK
            (ours if side == rec["our_color"] else theirs).append(rec)
        for group, who in ((ours, player), (theirs, opponent)):
            if not group:
                continue
            played = who.moves(group)
            # A player that returns the wrong number of moves would leave games
            # unadvanced and spin this loop forever; say so instead.
            if len(played) != len(group):
                raise RuntimeError(f"{who.name} returned {len(played)} moves for {len(group)} positions")
            for rec, uci in zip(group, played):
                rec["game"].push(uci)
                rec["moves"].append(uci)

    done.sort(key=lambda r: r["id"])
    return summarize(done, player.name, opponent.name, time.time() - t0)


def summarize(records, player_name, opponent_name, seconds):
    n = len(records)
    wins = sum(1 for r in records if r["result_white"] == (1 if r["our_color"] == WHITE else -1))
    losses = sum(1 for r in records if r["result_white"] == (-1 if r["our_color"] == WHITE else 1))
    draws = n - wins - losses
    score = (wins + 0.5 * draws) / n if n else 0.0
    elo, lo, hi = elo_from_wdl(wins, draws, losses)
    counts = {}
    for r in records:
        key = ce.Termination(r["termination"]).name.lower()
        counts[key] = counts.get(key, 0) + 1
    return {"player": player_name, "opponent": opponent_name, "games": n,
            "wins": wins, "draws": draws, "losses": losses, "score": score,
            "elo": elo, "elo_lo": lo, "elo_hi": hi, "los": los(wins, losses),
            "mean_plies": float(np.mean([r["plies"] for r in records])) if n else 0.0,
            "terminations": counts, "seconds": seconds, "records": records}


def report(match, sprt=None):
    m = match
    print(f"{m['player']} vs {m['opponent']}: +{m['wins']} ={m['draws']} -{m['losses']} "
          f"({m['score']:.1%})  Elo {m['elo']:+.0f} [{m['elo_lo']:+.0f}, {m['elo_hi']:+.0f}] 95% CI, "
          f"LOS {m['los']:.1%}")
    print(f"  {m['games']} games, {m['mean_plies']:.0f} plies avg, {m['seconds']:.0f}s "
          f"({m['games'] / max(m['seconds'], 1e-6) * 3600:,.0f} games/h)"
          + (f", {m['net_evals'] / max(m['seconds'], 1e-6):,.0f} net evals/s" if m.get("net_evals") else ""))
    # How the games ended is the diagnosis: a weak side draws won positions by
    # fifty-move and repetition, which reads as a bad score but is a conversion
    # problem. ply_cap games are the ones that never got anywhere at all.
    print("  terminations: " + ", ".join(f"{k} {v}" for k, v in
                                         sorted(m["terminations"].items(), key=lambda kv: -kv[1])))
    if sprt:
        elo0, elo1 = sprt
        llr = sprt_llr(m["wins"], m["draws"], m["losses"], elo0, elo1)
        low, high = sprt_bounds()
        verdict = "H1 accepted" if llr >= high else "H0 accepted" if llr <= low else "inconclusive, keep playing"
        print(f"  SPRT [{elo0:+.0f}, {elo1:+.0f}]: LLR {llr:+.2f} "
              f"(bounds {low:+.2f}/{high:+.2f}) — {verdict}")


# --- outputs -----------------------------------------------------------------

def write_pgn(match, path, event, tags=None):
    """Every game, with the reproducibility tags docs/06 asks for.

    `ordo` consumes exactly this, which is how several anchors and pairings turn
    into one joint rating list once enough PGNs pile up.
    """
    import chess
    import chess.pgn

    with open(path, "w") as fh:
        for rec in match["records"]:
            board = chess.Board(rec["start_fen"])
            for uci in rec["moves"]:
                board.push(chess.Move.from_uci(uci))
            game = chess.pgn.Game.from_board(board)
            we_are_white = rec["our_color"] == WHITE
            game.headers["Event"] = event
            game.headers["Site"] = "az.eval"
            game.headers["Round"] = str(rec["id"] + 1)
            game.headers["White"] = match["player"] if we_are_white else match["opponent"]
            game.headers["Black"] = match["opponent"] if we_are_white else match["player"]
            # from_board() reads the result off the board, which says "*" for a
            # game the ply cap ended; the match knows better.
            game.headers["Result"] = {1: "1-0", 0: "1/2-1/2", -1: "0-1"}[rec["result_white"]]
            game.headers["Termination"] = ce.Termination(rec["termination"]).name
            game.headers["Opening"] = rec["opening"]
            game.headers["PlyCount"] = str(rec["plies"])
            if rec["start_fen"] != ce.STARTPOS:
                game.headers["SetUp"] = "1"
                game.headers["FEN"] = rec["start_fen"]
            for key, value in (tags or {}).items():
                game.headers[key] = str(value)
            print(game, file=fh, end="\n\n")
    return path


CSV_COLUMNS = ["timestamp", "tag", "player", "opponent", "games", "wins", "draws", "losses",
               "score", "elo", "elo_lo", "elo_hi", "los", "mean_plies", "sims", "anchor_nodes",
               "seed", "ckpt", "pgn", "seconds"]


def append_csv(path, match, row_extra):
    """One row per pairing, appended forever (lesson #7: the curve, not memory of it)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        row = {k: match.get(k) for k in CSV_COLUMNS}
        row.update(row_extra)
        for key in ("score", "elo", "elo_lo", "elo_hi", "los", "mean_plies", "seconds"):
            if row.get(key) is not None:
                row[key] = f"{row[key]:.4f}"
        writer.writerow(row)


def summary_json(match):
    """The match dict minus the move logs — the PGN already holds those."""
    return {k: v for k, v in match.items() if k != "records"}


# --- cutechess backend -------------------------------------------------------

def engine_spec(name, cmd, args=(), options=None, extra=()):
    """One `-engine` clause as cutechess-cli's key=value tokens."""
    tokens = [f"name={name}", f"cmd={cmd}", "proto=uci"]
    tokens += [f"arg={a}" for a in args]
    tokens += [f"option.{k}={v}" for k, v in (options or {}).items()]
    tokens += list(extra)
    return tokens


def cutechess_command(args, anchor, book_path, pgn_path, repo_root):
    """The full cutechess-cli argv for one pairing.

    Our engine is node-limited by its own `Sims` option and told `tc=inf`: a
    clock would make the result a statement about this machine. Stockfish gets
    `nodes=` at the frozen budget for the same reason.
    """
    seed = args.seed
    our_name = f"az-{args.sims}s-seed{seed}"
    ours = engine_spec(our_name, sys.executable,
                       args=["-m", "az.uci", "--ckpt", os.path.abspath(args.ckpt),
                             "--sims", str(args.sims), "--seed", str(seed),
                             "--c-puct", str(args.c_puct)],
                       extra=[f"dir={repo_root}", "tc=inf"])

    if anchor["kind"] == "random":
        theirs = engine_spec("random", sys.executable, args=["-m", "az.uci", "--random", "--seed", str(seed)],
                             extra=[f"dir={repo_root}", "tc=inf"])
    else:
        options = ({"Skill Level": anchor["skill"]} if "skill" in anchor
                   else {"UCI_LimitStrength": "true", "UCI_Elo": anchor["elo"]})
        options.update({"Threads": args.sf_threads, "Hash": 16})
        theirs = engine_spec(anchor["name"], args.stockfish or "stockfish", options=options,
                             extra=["tc=inf", f"nodes={args.nodes}"])

    cmd = [args.cutechess_bin, "-engine", *ours, "-engine", *theirs,
           "-each", "proto=uci",
           "-openings", f"file={book_path}", "format=epd", "order=sequential",
           "-repeat", "2", "-games", "2", "-rounds", str(max(1, args.games // 2)),
           "-maxmoves", str(args.max_plies // 2),
           "-pgnout", pgn_path,
           "-concurrency", str(args.concurrency),
           "-recover", "-wait", "10"]
    if args.sprt:
        elo0, elo1 = args.sprt
        cmd += ["-sprt", f"elo0={elo0:g}", f"elo1={elo1:g}", "alpha=0.05", "beta=0.05"]
    cmd += list(args.cutechess_arg or [])
    return cmd


CUTECHESS_SCORE = "Score of "


def run_cutechess(cmd):
    """Run it, echo it, and pull the final `Score of A vs B: W - L - D` line."""
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    last = None
    for line in proc.stdout.splitlines():
        if line.startswith(CUTECHESS_SCORE):
            last = line
    if last is None:
        return None
    head, _, tail = last.partition(":")
    names = head[len(CUTECHESS_SCORE):].split(" vs ")
    counts = tail.split("[")[0].replace("-", " ").split()
    wins, losses, draws = (int(x) for x in counts[:3])
    return summarize_counts(names[0].strip(), names[-1].strip(), wins, draws, losses)


def summarize_counts(player, opponent, wins, draws, losses):
    """A match summary from bare counts — what a cutechess run gives back."""
    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n if n else 0.0
    elo, lo, hi = elo_from_wdl(wins, draws, losses)
    return {"player": player, "opponent": opponent, "games": n, "wins": wins, "draws": draws,
            "losses": losses, "score": score, "elo": elo, "elo_lo": lo, "elo_hi": hi,
            "los": los(wins, losses), "mean_plies": 0.0, "terminations": {}, "seconds": 0.0,
            "records": []}


# --- entry point -------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="checkpoint to evaluate (az.net format)")
    ap.add_argument("--sims", type=int, default=400,
                    help="simulations per move for our side; 0 = greedy policy, no search. "
                         "docs/06: 400 for tracking, 1600 for milestone attempts")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Dirichlet eps at the root; 0 (default) keeps match play deterministic")
    ap.add_argument("--device", default=None, help="default: cuda if available")

    ap.add_argument("--anchor", default="random",
                    help=f"comma-separated opponents; docs/06 wants two adjacent ones. Ladder: {', '.join(LADDER)}")
    ap.add_argument("--player", default="az",
                    help="who plays the anchors (default az); an anchor name here pits anchors against each other")
    ap.add_argument("--games", type=int, default=100,
                    help="games per pairing (docs/06: 100 minimum, 400 for a milestone claim)")
    ap.add_argument("--concurrency", type=int, default=32, help="games in flight; also the search fleet size")
    ap.add_argument("--max-plies", type=int, default=ce.MAX_PLIES,
                    help="draw adjudication cap, deliberately self-play's own (docs/02)")

    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH"), help="path to a stockfish binary")
    ap.add_argument("--nodes", type=int, default=ANCHOR_NODES,
                    help=f"frozen node budget per anchor move (default {ANCHOR_NODES}; see docs/06 caveat 1)")
    ap.add_argument("--sf-threads", type=int, default=1)

    ap.add_argument("--book", default=None, help="EPD opening book (default: az.book's 40 lines)")
    ap.add_argument("--book-plies", type=int, default=4)
    ap.add_argument("--book-limit", type=int, default=None)

    ap.add_argument("--out", default="runs/eval", help="dir for PGNs, eval.csv and summaries")
    ap.add_argument("--tag", default=None, help="label for this run in eval.csv (default: the checkpoint's name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sprt", default=None, metavar="ELO0,ELO1",
                    help="report the sequential test for H0=ELO0 vs H1=ELO1")

    ap.add_argument("--backend", default="inproc", choices=["inproc", "cutechess"])
    ap.add_argument("--cutechess-bin", default="cutechess-cli")
    ap.add_argument("--cutechess-arg", action="append", default=[],
                    help="extra cutechess-cli argument, repeatable (e.g. --cutechess-arg=-draw "
                         "--cutechess-arg=movenumber=40)")
    ap.add_argument("--dry-run", action="store_true", help="cutechess backend: print the command, run nothing")
    ap.add_argument("--calibrate", action="store_true",
                    help="round-robin among --anchor instead of playing them: measures the ladder's real "
                         "spacing at the frozen node count (docs/06 caveat 1)")
    return ap


def pairings(args):
    """-> [(player_spec, anchor_spec)] for this run."""
    anchors = [parse_anchor(a) for a in args.anchor.split(",") if a.strip()]
    if not anchors:
        raise SystemExit("--anchor is empty")
    if args.calibrate:
        return [(anchors[i], anchors[j]) for i in range(len(anchors)) for j in range(i + 1, len(anchors))]
    return [(parse_anchor(args.player), anchor) for anchor in anchors]


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.sprt:
        args.sprt = tuple(float(x) for x in args.sprt.split(","))

    specs = pairings(args)
    plays_net = any(s["kind"] == "az" for pair in specs for s in pair)
    if plays_net and not args.ckpt:
        raise SystemExit("--ckpt is required to evaluate the agent "
                         "(use --calibrate or --player <anchor> for anchor-only matches)")

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or (os.path.splitext(os.path.basename(args.ckpt))[0] if args.ckpt else "anchors")
    openings = load_book(args.book, plies=args.book_plies, limit=args.book_limit)

    # The cutechess backend runs the net in a separate az.uci process, so this
    # one must not load torch: a --dry-run should print its command on a box
    # with no GPU and no valid checkpoint at all.
    evaluator = None
    if plays_net and args.backend == "inproc":
        import torch

        from az.net import Evaluator
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = Evaluator.from_checkpoint(args.ckpt, device=device)
        print(f"{args.ckpt} on {device}, "
              f"{'greedy policy (no search)' if args.sims <= 0 else f'{args.sims} sims/move'}")
    print(f"{len(openings)} openings, {args.games} games/pairing, {args.max_plies}-ply cap, "
          f"anchors at {args.nodes:,} nodes, seed {args.seed}\n", flush=True)

    if args.backend == "cutechess":
        return run_cutechess_backend(args, specs, openings, tag)

    csv_path = os.path.join(args.out, "eval.csv")
    results = []
    for player_spec, anchor_spec in specs:
        player = make_player(player_spec, args, args.concurrency, evaluator, seed=args.seed)
        opponent = make_player(anchor_spec, args, args.concurrency, evaluator, seed=args.seed + 1)
        try:
            match = play_match(player, opponent, args.games, openings, max_plies=args.max_plies,
                               concurrency=args.concurrency, progress=progress_printer(args.games))
        finally:
            player.close()
            opponent.close()

        if isinstance(player, SearchPlayer):
            match["net_evals"] = player.evals
        report(match, sprt=args.sprt)
        # In calibrate mode the same anchor is on the right of several pairings,
        # so the player has to be in the filename or the PGNs overwrite.
        stem = (f"{tag}_{match['player']}_vs_{match['opponent']}" if args.calibrate
                else f"{tag}_vs_{match['opponent']}")
        pgn = write_pgn(match, os.path.join(args.out, stem + ".pgn"), event=f"az eval {tag}",
                        tags={"AzSeed": args.seed, "AzSims": args.sims, "AzCkpt": args.ckpt or "",
                              "AnchorNodes": args.nodes})
        append_csv(csv_path, match, {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag,
            "sims": args.sims if player_spec["kind"] == "az" else "",
            "anchor_nodes": args.nodes, "seed": args.seed, "ckpt": args.ckpt or "", "pgn": os.path.basename(pgn)})
        with open(os.path.join(args.out, stem + ".json"), "w") as fh:
            json.dump(summary_json(match), fh, indent=2)
        results.append(match)
        print(f"  -> {pgn}\n", flush=True)

    final_summary(results, args, csv_path)
    return 0


def progress_printer(total, every=None):
    every = every or max(1, total // 10)
    def report_progress(done, n, elapsed):
        if done % every == 0 or done == n:
            rate = done / max(elapsed, 1e-6) * 3600
            print(f"  {done}/{n} games ({elapsed:.0f}s, {rate:,.0f} games/h)", flush=True)
    return report_progress


def final_summary(results, args, csv_path):
    if len(results) > 1:
        print("summary")
        for m in results:
            print(f"  {m['player']:>12} vs {m['opponent']:<10} {m['score']:6.1%}  "
                  f"Elo {m['elo']:+7.0f} [{m['elo_lo']:+.0f}, {m['elo_hi']:+.0f}]")
        if args.calibrate:
            print("  ^ the ladder's real spacing at this node count. Freeze --nodes here; "
                  "every Elo recorded later is relative to it.")
    elif results and not args.calibrate:
        print("only one anchor played — docs/06 wants two adjacent ones, because a single "
              "anchor saturates as the agent improves")
    print(f"\n{csv_path} appended. With several pairings on file, `ordo -Q -D -a 0 -A "
          f"{results[0]['opponent'] if results else 'anchor'} -p {args.out}/*.pgn` fits them jointly.")


def run_cutechess_backend(args, specs, openings, tag):
    """docs/06's preferred path: hand the whole tournament to cutechess-cli."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    book_path = os.path.join(args.out, "book.epd")
    with open(book_path, "w") as fh:
        fh.write("".join(f'{fen} ; id "{name}";\n' for name, _m, fen in openings))

    if not args.dry_run and shutil.which(args.cutechess_bin) is None:
        raise SystemExit(f"{args.cutechess_bin} not found on PATH — install cutechess-cli, or use "
                         "--dry-run to print the command, or --backend inproc to play in this process")

    csv_path = os.path.join(args.out, "eval.csv")
    for player_spec, anchor_spec in specs:
        if player_spec["kind"] != "az":
            raise SystemExit("the cutechess backend plays the checkpoint against anchors; "
                             "anchor-vs-anchor calibration is an --backend inproc job")
        pgn = os.path.join(args.out, f"{tag}_vs_{anchor_spec['name']}.pgn")
        cmd = cutechess_command(args, anchor_spec, book_path, pgn, repo_root)
        if args.dry_run:
            print("$ " + " ".join(cmd))
            continue
        match = run_cutechess(cmd)
        if match is None:
            print("cutechess produced no score line — see its output above", file=sys.stderr)
            continue
        report(match, sprt=args.sprt)
        append_csv(csv_path, match, {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "sims": args.sims,
            "anchor_nodes": args.nodes, "seed": args.seed, "ckpt": args.ckpt or "",
            "pgn": os.path.basename(pgn)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
