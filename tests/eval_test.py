#!/usr/bin/env python
"""Tests for the Elo evaluation harness (docs/06).

The search itself is proven in C++ (tests/runner_test.cpp: negamax signs,
forced mates, in-tree draw adjudication). What this covers is everything
docs/06 adds on top, and the two seams where a silent wrong answer is easiest:

* **Batch alignment.** `Searcher.pending()` returns one row per slot awaiting
  evaluation and `feed()` takes them back in that order. Swap two rows and
  nothing crashes — every game stays legal, every match finishes, and the Elo
  is quietly wrong. So the fleet is checked against single-slot searches of the
  same positions with a *position-dependent* dummy net, where a permuted batch
  changes the answer.
* **History.** A slot copies the Game it is given, so the tree sees the
  repetition window. A Searcher that rebuilt the position from a FEN would look
  identical until the day a won match got drawn by repetition.

Plus the harness's arithmetic (Elo, CI, LOS, SPRT), the opening book's
legality, an end-to-end match, and the UCI wrapper driven as a subprocess the
way cutechess drives it.

Run:  python tests/eval_test.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce  # the built module at the repo root

from az import book, eval as azeval

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

failures = 0


def fail(what):
    global failures
    failures += 1
    if failures <= 20:
        print(f"  FAIL: {what}")


def check(ok, what):
    if not ok:
        fail(what)


def section(name):
    print(f"\n{name}")


# --- a deterministic stand-in for the net ------------------------------------

class PositionNet:
    """Priors and value that depend only on the position, never on batch order.

    This is what makes the fleet tests bite: two slots evaluated in one batch
    must each receive *their own* row back. A uniform dummy net cannot tell the
    difference between correct alignment and a shuffle.
    """

    def __init__(self):
        self.calls = 0
        self.evals = 0

    def __call__(self, planes):
        self.calls += 1
        self.evals += planes.shape[0]
        flat = planes.reshape(planes.shape[0], -1).astype(np.uint64)
        keys = (flat * np.arange(1, flat.shape[1] + 1, dtype=np.uint64)).sum(1)
        policy = np.empty((planes.shape[0], ce.POLICY_SIZE), np.float32)
        value = np.empty(planes.shape[0], np.float32)
        for i, key in enumerate(keys):
            rng = np.random.default_rng(int(key))
            row = rng.random(ce.POLICY_SIZE, dtype=np.float32)
            policy[i] = row / row.sum()
            value[i] = rng.random() * 2 - 1
        return policy, value


def uniform_net(planes):
    n = planes.shape[0]
    return (np.full((n, ce.POLICY_SIZE), 1.0 / ce.POLICY_SIZE, np.float32),
            np.zeros(n, np.float32))


def drive(searcher, games, net=uniform_net):
    """Run a fleet to completion. -> (evals, [root dict per slot])."""
    for i, game in enumerate(games):
        searcher.set_position(i, game)
    evals = 0
    while True:
        batch = searcher.pending()
        if not batch.size:
            break
        evals += batch.shape[0]
        searcher.feed(*net(batch))
    return evals, [searcher.root(i) for i in range(len(games))]


# --- the Searcher ------------------------------------------------------------

MATE_IN_1 = "1k6/8/1K6/8/8/8/8/7R w - - 0 1"        # Rh8#
MATE_IN_2 = "3k4/8/R7/8/8/8/8/6RK w - - 0 1"        # 1.Ra7/Rg7 then mate on the 8th


def test_searcher_finds_mates():
    section("Searcher: forced mates with uniform priors")
    # Same positions and simulation budgets as tests/runner_test.cpp, so a
    # discrepancy here is the pybind layer's and not the tree's.
    one = ce.Searcher(n_slots=1, sims=800, seed=1)
    drive(one, [ce.Game(MATE_IN_1)])
    check(one.best_move(0) == "h1h8", f"mate in 1 played (got {one.best_move(0)})")
    check(one.root_q(0) > 0.8, f"root value near +1 with a mate in hand (got {one.root_q(0):.3f})")

    two = ce.Searcher(n_slots=1, sims=5000, seed=1)
    drive(two, [ce.Game(MATE_IN_2)])
    check(two.best_move(0) in ("a6a7", "g1g7"), f"mate in 2 cuts the 7th rank (got {two.best_move(0)})")
    check(two.root_q(0) > 0.6, f"mate in 2 root value strongly positive (got {two.root_q(0):.3f})")

    # The mated side must see it coming: -1 for the defender in the same tree.
    lost = ce.Searcher(n_slots=1, sims=400, seed=1)
    drive(lost, [ce.Game("6k1/5ppp/8/8/8/8/8/R5RK b - - 0 1")])
    check(lost.root_q(0) < 0.0, f"the side facing a back-rank mate is not optimistic (got {lost.root_q(0):.3f})")


def test_searcher_root_stats():
    section("Searcher: root statistics agree with the engine")
    game = ce.Game()
    s = ce.Searcher(n_slots=1, sims=200, seed=3)
    _, (root,) = drive(s, [game])

    legal = set(game.legal_moves())
    check(set(root["moves"]) == legal, "root edges are exactly the legal moves")
    check(len(root["moves"]) == len(root["visits"]) == len(root["q"]) == len(root["prior"]),
          "root arrays are aligned")
    # sims - 1, not sims: the first simulation evaluates the root itself and so
    # descends through no edge at all.
    check(int(root["visits"].sum()) == s.sims_done(0) - 1,
          f"visits sum to the descending simulations ({int(root['visits'].sum())} vs {s.sims_done(0)} - 1)")
    check(s.sims_done(0) == 200, f"exactly the requested sims ran (got {s.sims_done(0)})")
    check(abs(float(root["prior"].sum()) - 1.0) < 1e-4, "priors are renormalised over the legal set")
    check(all(int(idx) == ce.move_to_index(game.fen, uci)
              for idx, uci in zip(root["index"], root["moves"])),
          "root move indices match the encoder")
    check(not s.active(0), "the slot goes inactive when its simulations are done")
    check(s.fen(0) == game.fen, "the slot reports the position it searched")


def test_fleet_matches_single_slot():
    section("Searcher: a fleet gives each slot its own evaluations")
    games = [ce.Game(),
             ce.Game("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
             ce.Game(MATE_IN_2),
             ce.Game("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3")]

    fleet = ce.Searcher(n_slots=len(games), sims=200, seed=11)
    _, fleet_roots = drive(fleet, games, PositionNet())
    fleet_moves = [fleet.best_move(i) for i in range(len(games))]

    for i, game in enumerate(games):
        alone = ce.Searcher(n_slots=1, sims=200, seed=11)
        _, (root,) = drive(alone, [ce.Game(game.fen)], PositionNet())
        check(list(root["moves"]) == list(fleet_roots[i]["moves"]),
              f"slot {i}: same move order alone as in the fleet")
        check(np.array_equal(root["visits"], fleet_roots[i]["visits"]),
              f"slot {i}: identical visit counts alone and in a batch of {len(games)}")
        check(alone.best_move(0) == fleet_moves[i],
              f"slot {i}: same best move ({alone.best_move(0)} vs {fleet_moves[i]})")

    # The dummy net really does discriminate: a permuted feed must change things.
    # Without this, the assertions above would pass for a Searcher that ignored
    # the batch order entirely.
    shuffled = ce.Searcher(n_slots=len(games), sims=200, seed=11)
    net = PositionNet()
    for i, game in enumerate(games):
        shuffled.set_position(i, game)
    while True:
        batch = shuffled.pending()
        if not batch.size:
            break
        policy, value = net(batch)
        order = np.roll(np.arange(batch.shape[0]), 1)  # feed neighbours' rows
        shuffled.feed(policy[order], value[order])
    differs = any(not np.array_equal(shuffled.root(i)["visits"], fleet_roots[i]["visits"])
                  for i in range(len(games)))
    check(differs, "a misaligned feed changes the search — the test above is meaningful")


def test_searcher_determinism():
    section("Searcher: reproducible given a seed")
    games = [ce.Game(MATE_IN_2), ce.Game()]

    def run(seed, noise):
        s = ce.Searcher(n_slots=2, sims=200, dirichlet_eps=noise, seed=seed)
        _, roots = drive(s, [ce.Game(g.fen) for g in games], PositionNet())
        return [(list(r["moves"]), r["visits"].tolist()) for r in roots]

    check(run(5, 0.0) == run(5, 0.0), "match play (no noise) repeats exactly")
    check(run(5, 0.25) == run(5, 0.25), "same seed repeats exactly even with root noise")
    check(run(5, 0.25) != run(5, 0.0), "root noise actually perturbs the search")


def test_searcher_copies_history():
    section("Searcher: the slot copies the game, history included")
    # Seven reversible shuffling plies: the position to move from, and the one a
    # single move reaches, have each occurred twice. So Kd8 is an immediate
    # threefold draw — but only for a searcher that was given the history.
    start, shuffle = "3k4/8/8/8/8/8/8/3K1R2 w - - 0 1", ["f1f2", "d8d7", "f2f1", "d7d8", "f1f2", "d8d7", "f2f1"]
    played = ce.Game(start)
    for uci in shuffle:
        played.push(uci)
    check(not played.outcome().over, "the shuffled game is not over yet")

    def visits_on(game):
        s = ce.Searcher(n_slots=1, sims=400, seed=2)
        # Every leaf is a win for whoever moves there, so the mover at the root
        # sees losses everywhere; a draw by repetition is the one good option.
        evals, (root,) = drive(s, [game], lambda p: (
            np.full((p.shape[0], ce.POLICY_SIZE), 1.0 / ce.POLICY_SIZE, np.float32),
            np.ones(p.shape[0], np.float32)))
        return evals, dict(zip(root["moves"], root["visits"].tolist()))

    with_history, from_fen = visits_on(played), visits_on(ce.Game(played.fen))
    check(with_history[1]["d7d8"] > 4 * from_fen[1]["d7d8"],
          f"the repetition draw is preferred only with history "
          f"({with_history[1]['d7d8']} vs {from_fen[1]['d7d8']} visits)")
    check(with_history[0] < 400, f"threefold leaves need no network call (got {with_history[0]} evals)")
    check(from_fen[0] == 400, f"without history every leaf needs one (got {from_fen[0]} evals)")

    # And the copy is a copy: advancing the caller's game mid-search is harmless.
    game = ce.Game(MATE_IN_1)
    s = ce.Searcher(n_slots=1, sims=200, seed=1)
    s.set_position(0, game)
    game.push("h1h2")  # the caller moves on; the slot must not notice
    while True:
        batch = s.pending()
        if not batch.size:
            break
        s.feed(*uniform_net(batch))
    check(s.best_move(0) == "h1h8", f"the slot searched its own copy (got {s.best_move(0)})")
    check(s.fen(0) == MATE_IN_1, "the slot's position is untouched by the caller")


def test_searcher_validation():
    section("Searcher: bad input raises instead of corrupting a match")

    def raises(fn, exc=Exception):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    check(raises(lambda: ce.Searcher(n_slots=0, sims=100), ValueError), "n_slots=0 rejected")
    check(raises(lambda: ce.Searcher(n_slots=1, sims=1), ValueError), "sims<2 rejected")
    check(raises(lambda: ce.Searcher(n_slots=1, sims=10, dirichlet_eps=2.0), ValueError),
          "dirichlet_eps outside [0,1] rejected")

    s = ce.Searcher(n_slots=2, sims=10)
    check(raises(lambda: s.set_position(2, ce.Game()), IndexError), "slot out of range rejected")
    check(raises(lambda: s.set_position(0, ce.Game("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")), ValueError),
          "a finished game rejected")
    check(raises(lambda: s.best_move(0), RuntimeError), "reading a slot that never searched raises")

    s.set_position(0, ce.Game())
    batch = s.pending()
    check(batch.shape == (1, ce.NUM_PLANES, 8, 8), f"pending shape is (B,19,8,8) (got {batch.shape})")
    check(raises(lambda: s.feed(np.zeros((2, ce.POLICY_SIZE), np.float32), np.zeros(2, np.float32)), ValueError),
          "feed with the wrong batch size rejected")
    check(raises(lambda: s.feed(np.zeros((1, 10), np.float32), np.zeros(1, np.float32)), ValueError),
          "feed with the wrong policy width rejected")

    s.clear(0)
    check(not s.active(0), "clear() abandons the search")
    check(not raises(lambda: s.best_move(0)), "an abandoned search still reports its best move so far")


# --- the opening book --------------------------------------------------------

def test_book():
    section("book: legal, varied, and round-trips through EPD")
    entries = book.lines()
    check(len(entries) >= 30, f"the book has enough lines for a 100-game match (got {len(entries)})")
    check(len({fen for _n, _m, fen in entries}) == len(entries), "every opening reaches a distinct position")
    check(len({name for name, _m, _f in entries}) == len(entries), "opening names are unique")

    for name, moves, fen in entries:
        check(len(moves) == book.BOOK_PLIES, f"{name}: {book.BOOK_PLIES} plies")
        replay = ce.Game()
        for uci in moves:
            check(uci in replay.legal_moves(), f"{name}: {uci} is legal")
            replay.push(uci)
        check(replay.fen == fen, f"{name}: the recorded FEN matches the replay")
        check(replay.white_to_move, f"{name}: an even-ply book leaves white to move, so colour pairs are symmetric")

    check(len(book.lines(limit=5)) == 5, "--limit truncates the book")
    check(all(len(m) == 2 for _n, m, _f in book.lines(plies=2)), "--plies truncates each line")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "book.epd")
        with open(path, "w") as fh:
            fh.write(book.as_epd(entries))
        loaded = azeval.load_book(path)
        check(len(loaded) == len(entries), "the EPD book reloads with every line")
        check([n for n, _m, _f in loaded] == [n for n, _m, _f in entries], "EPD keeps the opening names")
        check([f for _n, _m, f in loaded] == [f for _n, _m, f in entries], "EPD keeps the positions")
        check(all(m is None for _n, m, _f in loaded), "an external book has no move list, only positions")


# --- the arithmetic ----------------------------------------------------------

def test_statistics():
    section("statistics: Elo, CI, LOS, SPRT")
    check(abs(azeval.score_to_elo(0.5)) < 1e-9, "an even score is 0 Elo")
    check(azeval.score_to_elo(0.75) > 0 > azeval.score_to_elo(0.25), "Elo tracks the score")
    check(abs(azeval.score_to_elo(0.75) + azeval.score_to_elo(0.25)) < 1e-9,
          "Elo is antisymmetric about an even score")
    check(azeval.score_to_elo(0.0) < -2000 and azeval.score_to_elo(1.0) > 2000,
          "a shutout is clamped, not infinite")

    elo, lo, hi = azeval.elo_from_wdl(25, 50, 25)
    check(abs(elo) < 1e-9, "an even match is 0 Elo")
    check(lo < 0 < hi, "the 95% band straddles 0 at an even match")
    check(azeval.elo_from_wdl(0, 0, 0) == (0.0, 0.0, 0.0), "no games is not a crash")

    def half_width(wins, draws, losses):
        _e, low, high = azeval.elo_from_wdl(wins, draws, losses)
        return (high - low) / 2

    # docs/06's ±35 at 100 games and ±17 at 400 are that many games at a
    # realistic draw rate; the all-decisive worst case is about twice as wide.
    check(abs(half_width(12, 76, 12) - 35) < 6, f"±35 Elo over 100 draw-heavy games (got ±{half_width(12, 76, 12):.0f})")
    check(abs(half_width(50, 300, 50) - 17) < 4, f"±17 Elo over 400 of them (got ±{half_width(50, 300, 50):.0f})")
    check(abs(half_width(50, 0, 50) - 69) < 3,
          f"±69 Elo over 100 all-decisive games, the worst case (got ±{half_width(50, 0, 50):.0f})")
    check(half_width(50, 300, 50) < half_width(12, 76, 12), "400 games give a tighter band than 100")
    check(half_width(12, 76, 12) < half_width(50, 0, 50), "draws tighten the band; the binomial ignores that")

    # A shutout is the least informative result there is, and the plain normal
    # approximation calls it exact — the case this harness will hit constantly
    # in early phases.
    _e, low, high = azeval.elo_from_wdl(0, 0, 10)
    check(high - low > 100, f"a 0-10 shutout does not claim a zero-width band (got {high - low:.0f} Elo wide)")
    check(low < high < 0, "and it still reports a clearly negative Elo")
    check(azeval.elo_from_wdl(0, 10, 0)[1] < 0 < azeval.elo_from_wdl(0, 10, 0)[2],
          "an all-draw match is uncertain, not exactly equal")
    check(half_width(0, 0, 10) > half_width(0, 0, 100), "more shutout games narrow the bound (rule of three)")

    check(abs(azeval.los(10, 10) - 0.5) < 1e-9, "LOS is 50% on an equal decisive split")
    check(azeval.los(30, 10) > 0.99, "LOS is high when wins dominate")
    check(azeval.los(0, 0) == 0.5, "LOS with no decisive games is 50%")

    low, high = azeval.sprt_bounds()
    check(low < 0 < high, "the SPRT bounds bracket zero")
    check(azeval.sprt_llr(60, 20, 20, 0, 50) > 0, "a clear improvement pushes the LLR toward H1")
    check(azeval.sprt_llr(20, 20, 60, 0, 50) < 0, "a clear regression pushes it toward H0")
    check(azeval.sprt_llr(0, 0, 0, 0, 50) == 0.0, "an empty match is no evidence either way")
    check(azeval.sprt_llr(300, 100, 100, 0, 50) > azeval.sprt_llr(30, 10, 10, 0, 50),
          "the same score with more games is stronger evidence")


def test_anchor_parsing():
    section("anchors: the docs/06 ladder")
    check(azeval.parse_anchor("random")["kind"] == "random", "random parses")
    check(azeval.parse_anchor("skill3") == {"kind": "stockfish", "skill": 3, "name": "skill3"}, "skill3 parses")
    check(azeval.parse_anchor("sf1320") == {"kind": "stockfish", "elo": 1320, "name": "sf1320"}, "sf1320 parses")
    check(azeval.parse_anchor("SF1700")["elo"] == 1700, "anchor names are case-insensitive")
    for bad in ("sf1000", "skill99", "stockfish", ""):
        try:
            azeval.parse_anchor(bad)
            fail(f"{bad!r} should not parse as an anchor")
        except ValueError:
            pass
    check(all(azeval.parse_anchor(name) for name in azeval.LADDER), "every ladder name parses")


# --- a match end to end ------------------------------------------------------

def test_match():
    section("match: a full random-vs-random pairing with all its outputs")
    games = 12
    openings = book.lines(limit=6)
    player, opponent = azeval.RandomPlayer(seed=1), azeval.RandomPlayer(seed=2)
    match = azeval.play_match(player, opponent, games, openings, max_plies=60, concurrency=4)

    check(match["games"] == games, f"every game finished ({match['games']}/{games})")
    check(match["wins"] + match["draws"] + match["losses"] == games, "results account for every game")
    check(abs(match["score"] - (match["wins"] + 0.5 * match["draws"]) / games) < 1e-9, "the score is the score")
    check(sum(match["terminations"].values()) == games, "every game has a termination reason")
    check(all(r["plies"] <= 60 for r in match["records"]), "the ply cap adjudicates as configured")
    check([r["id"] for r in match["records"]] == list(range(games)), "records come back in game order")

    colors = [r["our_color"] for r in match["records"]]
    check(colors == [i % 2 for i in range(games)], "colours alternate")
    pairs = [(match["records"][i]["opening"], match["records"][i + 1]["opening"]) for i in range(0, games, 2)]
    check(all(a == b for a, b in pairs), "each opening is played once with each colour")
    check(len({r["opening"] for r in match["records"]}) == games // 2, "the book is walked, not repeated")

    for rec in match["records"]:
        replay = ce.Game(rec["start_fen"], 60)
        for uci in rec["moves"]:
            replay.push(uci)  # raises if the harness ever recorded an illegal move
        check(replay.fen == rec["game"].fen, f"game {rec['id']} replays to its final position")
        out = replay.outcome()
        result_white = out.value if replay.white_to_move else -out.value
        check(out.over and result_white == rec["result_white"],
              f"game {rec['id']}: the recorded result is the position's")

    with tempfile.TemporaryDirectory() as tmp:
        pgn = azeval.write_pgn(match, os.path.join(tmp, "m.pgn"), event="test", tags={"AzSeed": 7})
        import chess.pgn

        with open(pgn) as fh:
            read = []
            while (g := chess.pgn.read_game(fh)) is not None:
                read.append(g)
        check(len(read) == games, f"the PGN holds every game ({len(read)}/{games})")
        check(all(g.headers["AzSeed"] == "7" for g in read), "reproducibility tags are written")
        check(all(g.headers["Result"] in ("1-0", "0-1", "1/2-1/2") for g in read), "no unfinished games in the PGN")
        results = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}
        check([results[g.headers["Result"]] for g in read] == [r["result_white"] for r in match["records"]],
              "PGN results match the match record, ply-cap draws included")
        check(all(len(list(g.mainline_moves())) == r["plies"]
                  for g, r in zip(read, match["records"])), "PGN move counts match")

        csv_path = os.path.join(tmp, "eval.csv")
        azeval.append_csv(csv_path, match, {"timestamp": "t", "tag": "test", "sims": 0,
                                            "anchor_nodes": 1, "seed": 7, "ckpt": "", "pgn": "m.pgn"})
        azeval.append_csv(csv_path, match, {"timestamp": "t2", "tag": "test", "sims": 0,
                                            "anchor_nodes": 1, "seed": 7, "ckpt": "", "pgn": "m.pgn"})
        import csv as csv_mod

        with open(csv_path) as fh:
            rows = list(csv_mod.DictReader(fh))
        check(len(rows) == 2, "eval.csv appends rather than overwrites")
        check(list(rows[0]) == azeval.CSV_COLUMNS, "eval.csv has the documented columns")
        check(int(rows[0]["games"]) == games, "eval.csv records the match")
        check(json.loads(json.dumps(azeval.summary_json(match)))["games"] == games,
              "the JSON summary serialises (no move logs in it)")


def test_search_player_plays_a_match():
    section("match: the agent's search player drives a fleet")
    net = PositionNet()
    player = azeval.SearchPlayer(net, sims=16, n_slots=4, seed=1)
    match = azeval.play_match(player, azeval.RandomPlayer(seed=2), 4, book.lines(limit=2),
                              max_plies=40, concurrency=4)
    check(match["games"] == 4, "the search player finished its games")
    check(player.evals > 0 and net.calls > 0, "the fleet actually called the net")
    check(net.evals == player.evals, "every position handed to the net is counted")
    check(net.calls < player.evals, "positions were batched, not evaluated one at a time")
    for rec in match["records"]:
        replay = ce.Game(rec["start_fen"], 40)
        for uci in rec["moves"]:
            replay.push(uci)


def test_cutechess_command():
    section("cutechess: the command docs/06 prefers")
    args = azeval.build_parser().parse_args(
        ["--ckpt", "runs/x/bootstrap.pt", "--sims", "800", "--seed", "9", "--games", "100",
         "--stockfish", "/usr/bin/stockfish", "--nodes", "12345", "--concurrency", "4",
         "--sprt", "0,50", "--cutechess-arg=-draw", "--cutechess-arg=movenumber=40"])
    args.sprt = tuple(float(x) for x in args.sprt.split(","))
    cmd = azeval.cutechess_command(args, azeval.parse_anchor("sf1320"), "/tmp/b.epd", "/tmp/o.pgn", REPO)
    joined = " ".join(cmd)

    check(cmd[0] == "cutechess-cli", "the binary leads")
    check(cmd.count("-engine") == 2, "two engines")
    check("arg=az.uci" in cmd and "arg=--sims" in cmd and "arg=800" in cmd, "our side runs az.uci at --sims")
    check(f"dir={REPO}" in cmd, "the engine runs from the repo root so `-m az.uci` resolves")
    check("name=az-800s-seed9" in cmd, "the seed rides in the engine name, so it lands in the PGN")
    check("option.UCI_Elo=1320" in cmd and "option.UCI_LimitStrength=true" in cmd, "the anchor is Elo-limited")
    check("nodes=12345" in cmd, "the anchor is node-limited at the frozen budget")
    check("tc=inf" in cmd, "no wall-clock control for either side")
    check("file=/tmp/b.epd" in cmd and "format=epd" in cmd, "the opening book is passed")
    check("-repeat" in cmd, "openings are replayed with colours swapped")
    check("-maxmoves" in cmd and cmd[cmd.index("-maxmoves") + 1] == str(ce.MAX_PLIES // 2),
          "the move cap mirrors self-play's ply cap")
    check("elo0=0" in joined and "elo1=50" in joined, "SPRT bounds are passed through")
    check(joined.endswith("-draw movenumber=40"), "--cutechess-arg lands verbatim at the end")

    rnd = azeval.cutechess_command(args, azeval.parse_anchor("random"), "/tmp/b.epd", "/tmp/o.pgn", REPO)
    check("arg=--random" in rnd, "the random anchor is the az.uci floor engine")

    parsed = azeval.run_cutechess.__doc__ and azeval.summarize_counts("a", "b", 60, 20, 20)
    check(parsed["games"] == 100 and abs(parsed["score"] - 0.7) < 1e-9, "cutechess counts turn into a summary")


# --- the UCI wrapper ---------------------------------------------------------

def run_engine(engine_args, commands, timeout=180):
    """Drive az.uci over a pipe exactly as cutechess would. -> stdout lines."""
    proc = subprocess.run([sys.executable, "-m", "az.uci", *engine_args],
                          input="".join(c + "\n" for c in commands) + "quit\n",
                          text=True, capture_output=True, cwd=REPO, timeout=timeout)
    if proc.returncode != 0:
        fail(f"az.uci {' '.join(engine_args)} exited {proc.returncode}: {proc.stderr.strip()[:400]}")
    return proc.stdout.splitlines()


def bestmoves(lines):
    return [ln.split()[1] for ln in lines if ln.startswith("bestmove")]


def test_uci_random():
    section("uci: the random-mover floor engine")
    lines = run_engine(["--random", "--seed", "3"],
                       ["uci", "isready", "position startpos moves e2e4 e7e5", "go",
                        f"position fen {MATE_IN_1}", "go nodes 100",
                        "position fen 7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "go"])
    check("uciok" in lines, "uciok answered")
    check("readyok" in lines, "readyok answered")
    check(any(ln.startswith("id name az-random") for ln in lines), "identifies itself")

    moves = bestmoves(lines)
    check(len(moves) == 3, f"one bestmove per go (got {len(moves)})")
    if len(moves) == 3:
        after_e4e5 = ce.push(ce.push(ce.STARTPOS, "e2e4"), "e7e5")
        check(ce.is_legal(after_e4e5, moves[0]), f"{moves[0]} is legal in the position sent")
        check(ce.is_legal(MATE_IN_1, moves[1]), f"{moves[1]} is legal")
        check(moves[2] == "0000", f"a finished position answers with the null move (got {moves[2]})")

    again = run_engine(["--random", "--seed", "3"], ["position startpos", "go"])
    check(bestmoves(again)[:1] == bestmoves(run_engine(["--random", "--seed", "3"],
                                                       ["position startpos", "go"]))[:1],
          "the same seed replays the same move")


def test_uci_search():
    section("uci: the agent engine (net + MCTS)")
    import torch

    from az.net import PolicyValueNet, save_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "tiny.pt")
        torch.manual_seed(0)
        save_checkpoint(PolicyValueNet(channels=16, blocks=2), ckpt)

        lines = run_engine(["--ckpt", ckpt, "--sims", "32", "--device", "cpu", "--seed", "5"],
                           ["uci", "isready",
                            "position startpos moves d2d4 d7d5", "go",
                            f"position fen {MATE_IN_1}", "go nodes 200",
                            "position startpos", "go",
                            "setoption name Sims value 24", "position startpos", "go wtime 1000 btime 1000"])
        check("uciok" in lines, "uciok answered")
        check(any(ln.startswith("id name az-mcts") for ln in lines), "identifies itself")
        check(any("option name Sims type spin" in ln for ln in lines), "Sims is a settable option")

        moves, infos = bestmoves(lines), [ln for ln in lines if ln.startswith("info depth")]
        check(len(moves) == 4, f"one bestmove per go (got {len(moves)})")
        check(len(infos) == 4, f"one info line per go (got {len(infos)})")
        if len(moves) == 4:
            after = ce.push(ce.push(ce.STARTPOS, "d2d4"), "d7d5")
            check(ce.is_legal(after, moves[0]), f"{moves[0]} is legal")
            check(moves[1] == "h1h8", f"search finds mate in 1 even with an untrained net (got {moves[1]})")
        if len(infos) == 4:
            check(" nodes 32 " in infos[0], f"the default sim budget is reported ({infos[0]})")
            check(" nodes 200 " in infos[1], f"`go nodes N` sets the budget ({infos[1]})")
            check(" nodes 32 " in infos[2], f"`go nodes N` was for that move only ({infos[2]})")
            check(" nodes 24 " in infos[3], f"setoption Sims changes the standing budget ({infos[3]})")
            check("score cp" in infos[0], "a real centipawn score is reported for adjudication")
            check(int(infos[1].split(" score cp ")[1].split()[0]) > 500,
                  f"the mate position scores as winning ({infos[1]})")
            check("wtime" not in " ".join(lines), "clock tokens are ignored, not echoed as an error")

        repeat = run_engine(["--ckpt", ckpt, "--sims", "32", "--device", "cpu", "--seed", "5"],
                            ["position startpos moves d2d4 d7d5", "go"])
        check(bestmoves(repeat) == moves[:1], "the same seed and checkpoint replay the same move")


def test_score_cp():
    section("uci: root value -> centipawns")
    from az.uci import score_cp

    check(score_cp(0.0) == 0, "an even position is 0 cp")
    check(score_cp(0.5) > 0 > score_cp(-0.5), "the sign follows the value")
    check(score_cp(0.5) == -score_cp(-0.5), "the mapping is antisymmetric")
    check(score_cp(1.0) == score_cp(2.0) < 5000, "certainty is clamped, not infinite")
    check(score_cp(0.9) > score_cp(0.3), "it is monotone")


def main():
    test_searcher_finds_mates()
    test_searcher_root_stats()
    test_fleet_matches_single_slot()
    test_searcher_determinism()
    test_searcher_copies_history()
    test_searcher_validation()
    test_book()
    test_statistics()
    test_anchor_parsing()
    test_match()
    test_search_player_plays_a_match()
    test_cutechess_command()
    test_score_cp()
    test_uci_random()
    test_uci_search()

    if failures:
        print(f"\n{failures} FAILURE(S)")
        return 1
    print("\neval ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
