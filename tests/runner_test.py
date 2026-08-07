#!/usr/bin/env python
"""Integration test for the batched Runner's Python surface (docs/02).

The C++ side (tests/runner_test.cpp) proves the search: negamax signs, forced
mates with uniform priors, in-tree draw adjudication, the resign rule. What it
cannot see is the numpy boundary and the dump format as training will consume
it. So this drives a small fleet with a dummy uniform net and then audits the
books: every example dump must replay through Game move by move (push_u16) into
exactly the terminal position the results table claims, with visit counts,
mover-relative results and the flat legal_idx/visits layout all consistent.

Run:  python tests/runner_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce  # the built module at the repo root

failures = 0


def fail(what):
    global failures
    failures += 1
    if failures <= 20:
        print(f"  FAIL: {what}")


def check(ok, what):
    if not ok:
        fail(what)


TOTAL_GAMES = 6
SIMS = 24


def run_fleet():
    runner = ce.Runner(
        total_games=TOTAL_GAMES,
        n_parallel=3,
        sims=SIMS,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        temp_plies=8,
        resign_threshold=-1.0,  # disabled: a uniform net never reaches -0.95
        seed=7,
    )

    uniform = np.full((1, ce.POLICY_SIZE), 1.0 / ce.POLICY_SIZE, dtype=np.float32)
    cycles = 0
    # generous bound: one leaf per slot per cycle, MAX_PLIES moves of SIMS sims
    max_cycles = ce.MAX_PLIES * SIMS * 2 + 1000
    while True:
        batch = runner.pending()
        if not batch.size:
            break
        cycles += 1
        if cycles > max_cycles:
            fail("runner did not finish within the cycle budget")
            return None, None, None
        check(batch.dtype == np.uint8, "pending batch is uint8")
        check(batch.shape[1:] == (ce.NUM_PLANES, 8, 8), f"pending batch shape {batch.shape}")
        check(np.all(batch[:, 12] == 1), "the constant ones-plane survives the trip")
        runner.feed(np.repeat(uniform, batch.shape[0], axis=0), np.zeros(batch.shape[0], dtype=np.float32))

    check(runner.games_completed == TOTAL_GAMES, "all requested games completed")
    return runner.get_examples(), runner.get_results(), runner


def test_feed_validation():
    runner = ce.Runner(total_games=1, n_parallel=1, sims=4, seed=1)
    batch = runner.pending()
    check(batch.shape[0] == 1, "one pending leaf for one slot")
    try:
        runner.feed(np.zeros((2, ce.POLICY_SIZE), dtype=np.float32), np.zeros(2, dtype=np.float32))
        fail("mismatched batch size must raise")
    except ValueError:
        pass
    try:
        runner.feed(np.zeros((1, 7), dtype=np.float32), np.zeros(1, dtype=np.float32))
        fail("wrong policy width must raise")
    except ValueError:
        pass


def test_books(ex, res):
    n = len(res["game_id"])
    check(n == TOTAL_GAMES, f"{TOTAL_GAMES} result rows (got {n})")
    check(sorted(res["game_id"].tolist()) == list(range(TOTAL_GAMES)), "game ids are 0..N-1")

    draws = {
        int(ce.Termination.STALEMATE),
        int(ce.Termination.FIFTY_MOVE),
        int(ce.Termination.THREEFOLD),
        int(ce.Termination.INSUFFICIENT_MATERIAL),
        int(ce.Termination.PLY_CAP),
    }
    for i in range(n):
        term, result, plies = int(res["termination"][i]), int(res["result"][i]), int(res["plies"][i])
        check(0 < plies <= ce.MAX_PLIES, f"plies in range (game {i}: {plies})")
        check(int(res["resign_disabled"][i]) == 0, "no audit games when resignation is off")
        check(int(res["would_resign_side"][i]) == -1, "no resign triggers with threshold -1")
        if term == int(ce.Termination.CHECKMATE):
            check(abs(result) == 1, "checkmate is decisive")
        elif term in draws:
            check(result == 0, f"draw termination {term} scores 0")
        else:
            fail(f"unexpected termination {term} with resignation disabled")

    # flat layout: n_legal partitions legal_idx and visits exactly
    check(int(ex["n_legal"].sum()) == len(ex["legal_idx"]) == len(ex["visits"]),
          "n_legal sums to the flat array lengths")
    check(np.all(ex["legal_idx"] < ce.POLICY_SIZE), "legal indices are in the policy range")

    offsets = np.concatenate([[0], np.cumsum(ex["n_legal"], dtype=np.int64)])
    by_game = {}
    for r, gid in enumerate(ex["game_id"]):
        by_game.setdefault(int(gid), []).append(r)

    results_by_id = {int(res["game_id"][i]): i for i in range(n)}
    for gid, rows in by_game.items():
        i = results_by_id[gid]
        result_white, plies = int(res["result"][i]), int(res["plies"][i])
        check(len(rows) == plies, f"game {gid}: one example per ply")
        game = ce.Game()
        for r in rows:
            check(int(ex["ply"][r]) == game.ply, f"game {gid}: examples in ply order")
            lo, hi = offsets[r], offsets[r + 1]
            check(int(ex["visits"][lo:hi].sum()) == SIMS - 1,
                  f"game {gid} ply {game.ply}: root visits sum to sims-1")
            legal_here = set(ce.legal_move_indices(game.fen))
            check(set(ex["legal_idx"][lo:hi].tolist()) == legal_here,
                  f"game {gid} ply {game.ply}: legal_idx is exactly the legal move set")
            mover_white = game.white_to_move
            want = result_white if mover_white else -result_white
            check(int(ex["result"][r]) == want, f"game {gid} ply {game.ply}: mover-relative result")
            game.push_u16(int(ex["move"][r]))  # raises ValueError if the dump is corrupt
        out = game.outcome()
        check(int(out.reason) == int(res["termination"][i]),
              f"game {gid}: replay reaches the recorded termination "
              f"(replay {out.reason}, recorded {res['termination'][i]})")
        stm_white = game.white_to_move
        check((out.value if stm_white else -out.value) == result_white,
              f"game {gid}: replay outcome matches the recorded result")


def test_drained():
    # get_examples / get_results clear on return; a second call is empty
    runner = ce.Runner(total_games=1, n_parallel=1, sims=8, temp_plies=0, seed=3)
    uniform = None
    while True:
        batch = runner.pending()
        if not batch.size:
            break
        if uniform is None or uniform.shape[0] != batch.shape[0]:
            uniform = np.full((batch.shape[0], ce.POLICY_SIZE), 1.0 / ce.POLICY_SIZE, dtype=np.float32)
        runner.feed(uniform, np.zeros(batch.shape[0], dtype=np.float32))
    check(len(runner.get_examples()["game_id"]) > 0, "examples accumulated")
    check(len(runner.get_examples()["game_id"]) == 0, "get_examples clears on return")
    check(len(runner.get_results()["game_id"]) == 1, "results accumulated")
    check(len(runner.get_results()["game_id"]) == 0, "get_results clears on return")


def main():
    ex, res, _runner = run_fleet()
    if ex is not None:
        test_books(ex, res)
    test_feed_validation()
    test_drained()

    if failures:
        print(f"\n{failures} FAILURE(S)")
        return 1
    print("\nrunner python ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
