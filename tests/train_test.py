#!/usr/bin/env python
"""Tests for the self-play training loop (docs/05).

The search is proven in C++ (tests/runner_test.cpp) and its Python surface in
tests/runner_test.py. What this covers is the layer docs/05 adds, and it is
almost entirely about *alignment* — the failure mode of a training loop is not
a crash, it is a row of planes married to another row's policy target. Nothing
downstream notices; the Elo curve just never goes up.

* **Replay alignment.** `examples_to_chunk` reconstructs positions by replaying
  moves breadth-first across the whole generation while writing rows back at
  the dump's own indices. Off by one ply, or one game's rows written under
  another's, and every check below still passes except these: each row's planes
  are compared against an independent depth-first replay of that game, and each
  row's legal_idx against the legal move set of the position it claims.
* **Buffer bookkeeping.** Eviction moves the flat legal/visit arrays under the
  per-row offsets. A trim that forgets to rebase them silently hands the
  trainer another position's policy target.
* **Decisive premium.** The sampler has to hit the docs/05 weighting; a premium
  that quietly does nothing looks exactly like one that works.

Plus the generation statistics, a gradient step that has to actually descend,
and one full generation end to end on the CPU.

Run:  python tests/train_test.py
"""

import contextlib
import csv as csv_mod
import io
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce  # the built module at the repo root

from az import dashboard, train
from az.buffer import ReplayBuffer, trim_front
from az.selfplay import chunk_stats, examples_to_chunk, play_generation, result_stats

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


class UniformEvaluator:
    """The dummy net of tests/runner_test.py: uniform priors, value 0."""

    def __init__(self):
        self.evals = 0

    def __call__(self, planes):
        n = planes.shape[0]
        self.evals += n
        return (np.full((n, ce.POLICY_SIZE), 1.0 / ce.POLICY_SIZE, dtype=np.float32),
                np.zeros(n, dtype=np.float32))


# --- 1. self-play dump -> buffer chunk ---------------------------------------

def test_replay(gen):
    """Every row's planes and legal set must match the position it claims."""
    ex = gen["examples"]
    chunk = examples_to_chunk(ex)
    n = len(ex["game_id"])
    check(len(chunk["z"]) == n, "one chunk row per example")
    check(chunk["planes"].shape == (n, ce.NUM_PLANES, 8, 8), f"planes shape {chunk['planes'].shape}")
    check(int(chunk["n_legal"].sum()) == len(chunk["legal_idx"]) == len(chunk["visits"]),
          "n_legal partitions the flat arrays")

    # An independent, depth-first replay: one game at a time, one encode per
    # position, nothing shared with the breadth-first path under test.
    by_game = {}
    for row in range(n):
        by_game.setdefault(int(ex["game_id"][row]), []).append(row)

    checked = 0
    for gid, rows in by_game.items():
        rows.sort(key=lambda r: int(ex["ply"][r]))
        game = ce.Game()
        for row in rows:
            check(int(ex["ply"][row]) == game.ply, f"game {gid}: rows are ply-ordered")
            want = ce.encode_batch([game])[0]
            check(np.array_equal(chunk["planes"][row], want),
                  f"game {gid} ply {game.ply}: planes match the position before the move")
            lo = int(chunk["legal_off"][row])
            hi = lo + int(chunk["n_legal"][row])
            check(set(chunk["legal_idx"][lo:hi].tolist()) == set(ce.legal_move_indices(game.fen)),
                  f"game {gid} ply {game.ply}: legal_idx is this position's legal move set")
            check(int(chunk["visits"][lo:hi].sum()) == SIMS - 1,
                  f"game {gid} ply {game.ply}: visits sum to sims-1")
            check(chunk["z"][row] == ex["result"][row], f"game {gid}: z is the mover-relative result")
            game.push_u16(int(ex["move"][row]))
            checked += 1
    check(checked == n, "every row was verified")

    # The value target must be zero-sum across a game's two sides.
    for gid, rows in by_game.items():
        signs = {int(ex["ply"][r]) % 2: float(chunk["z"][r]) for r in rows}
        if len(signs) == 2:
            check(signs[0] == -signs[1] or signs[0] == signs[1] == 0.0,
                  f"game {gid}: the two sides' z are opposite (or both 0 for a draw)")
    return chunk


def test_empty_dump():
    empty = {k: np.empty(0, dtype=t) for k, t in
             (("game_id", np.uint32), ("ply", np.uint16), ("move", np.uint16),
              ("result", np.int8), ("n_legal", np.uint16), ("legal_idx", np.uint16),
              ("visits", np.uint16))}
    chunk = examples_to_chunk(empty)
    check(len(chunk["z"]) == 0, "an empty dump makes an empty chunk")
    check(chunk_stats(chunk)["positions"] == 0, "chunk_stats survives an empty chunk")

    bad = dict(empty)
    bad["n_legal"] = np.array([3], dtype=np.uint16)
    bad["game_id"] = np.array([0], dtype=np.uint32)
    bad["ply"] = np.array([0], dtype=np.uint16)
    bad["move"] = np.array([0], dtype=np.uint16)
    bad["result"] = np.array([0], dtype=np.int8)
    try:
        examples_to_chunk(bad)
        fail("a dump whose n_legal does not sum to the flat arrays must raise")
    except ValueError:
        pass


# --- 2. generation statistics ------------------------------------------------

def test_result_stats():
    # 6 games: 2 white wins, 1 black win, 3 draws. Two are resign audits, and
    # one of those salvaged a draw after triggering — a 50% false-positive rate.
    results = {
        "game_id": np.arange(6, dtype=np.uint32),
        "result": np.array([1, 1, -1, 0, 0, 0], dtype=np.int8),
        "plies": np.array([40, 60, 80, 100, 120, 140], dtype=np.int32),
        "termination": np.array([int(ce.Termination.CHECKMATE), int(ce.Termination.RESIGN),
                                 int(ce.Termination.CHECKMATE), int(ce.Termination.THREEFOLD),
                                 int(ce.Termination.FIFTY_MOVE), int(ce.Termination.PLY_CAP)],
                                dtype=np.uint8),
        "resign_disabled": np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8),
        "would_resign_side": np.array([-1, 1, -1, 1, 0, -1], dtype=np.int8),
        "would_resign_ply": np.array([-1, 50, -1, 60, 70, -1], dtype=np.int32),
    }
    s = result_stats(results)
    check(s["games"] == 6, "game count")
    check(abs(s["avg_plies"] - 90.0) < 1e-9, f"avg plies {s['avg_plies']}")
    check(abs(s["draw_frac"] - 0.5) < 1e-9, f"draw fraction {s['draw_frac']}")
    check(abs(s["resign_frac"] - 1 / 6) < 1e-9, f"resign fraction {s['resign_frac']}")
    # (2 white wins - 1 black win) / 6 -> mean +1/6 -> score 0.583
    check(abs(s["white_score"] - (1 + 1 / 6) / 2) < 1e-9, f"white score {s['white_score']}")
    check(s["audit_games"] == 2 and s["resign_triggers"] == 2, "audited resign triggers counted")
    # Both audit games triggered and both ended in draws: the resigning side
    # salvaged every time, which is a 100% false-positive rate.
    check(abs(s["resign_fp_frac"] - 1.0) < 1e-9, f"resign false positives {s['resign_fp_frac']}")
    check(s["terminations"]["checkmate"] == 2, "terminations counted by name")

    empty = result_stats({k: v[:0] for k, v in results.items()})
    check(empty["games"] == 0, "result_stats survives an empty generation")


# --- 3. replay buffer --------------------------------------------------------

def make_chunk(tag, n, draw_every=3, rng=None):
    """A chunk whose every field encodes its row tag, so a mix-up is visible."""
    rng = rng or np.random.default_rng(tag)
    n_legal = rng.integers(1, 6, size=n).astype(np.int32)
    off = np.concatenate([[0], np.cumsum(n_legal.astype(np.int64))])
    ids = np.arange(n) + tag * 1000
    planes = np.zeros((n, ce.NUM_PLANES, 8, 8), np.uint8)
    planes[:, 0, 0, 0] = ids % 251
    planes[:, 0, 0, 1] = ids // 251
    legal_idx = np.concatenate([(ids[i] + np.arange(n_legal[i])) % ce.POLICY_SIZE for i in range(n)])
    visits = np.concatenate([np.full(n_legal[i], (ids[i] % 97) + 1) for i in range(n)])
    z = np.where(np.arange(n) % draw_every == 0, 0.0, 1.0).astype(np.float32)
    return {"planes": planes, "legal_off": off[:-1], "n_legal": n_legal,
            "legal_idx": legal_idx.astype(np.uint16), "visits": visits.astype(np.uint16),
            "z": z, "ply": np.arange(n, dtype=np.int32)}, ids


def row_id(planes_row):
    return int(planes_row[0, 0, 0]) + 251 * int(planes_row[0, 0, 1])


def test_buffer_fifo():
    buf = ReplayBuffer(capacity=250, decisive_premium=1.0)
    ids_all = []
    for tag in range(4):
        chunk, ids = make_chunk(tag, 100)
        buf.add(chunk)
        ids_all.append(ids)
    check(len(buf) == 250, f"buffer is capped at capacity (got {len(buf)})")

    # The newest 250 of the 400 added: the last 50 of chunk 1, then 2 and 3.
    want = np.concatenate([ids_all[1][50:], ids_all[2], ids_all[3]])
    got = buf.gather(np.arange(len(buf)))
    check([row_id(p) for p in got["planes"]] == want.tolist(),
          "FIFO keeps the newest positions, in order")
    check(len(got["legal"]) == 250, "gather returns every requested row")

    # ...and the ragged arrays came along with them: each row's legal indices
    # are still the ones its tag generates.
    ok = True
    for i, ident in enumerate(want):
        n = int(got["valid"][i].sum())
        expect = [(ident + k) % ce.POLICY_SIZE for k in range(n)]
        ok &= got["legal"][i][:n].tolist() == expect
        ok &= got["visits"][i][:n].tolist() == [float((ident % 97) + 1)] * n
        ok &= not got["valid"][i][n:].any()
        ok &= not got["visits"][i][n:].any()
    check(ok, "trimmed chunks keep every row's legal_idx/visits aligned")
    check(abs(buf.draw_frac - float((got["z"] == 0).mean())) < 1e-9,
          "draw_frac matches the buffered contents")


def test_buffer_trim():
    chunk, ids = make_chunk(7, 20)
    trimmed = trim_front(chunk, 6)
    check(len(trimmed["z"]) == 14, "trim drops the requested rows")
    check(trimmed["legal_off"][0] == 0, "trim rebases the flat offsets")
    check(int(trimmed["n_legal"].sum()) == len(trimmed["legal_idx"]), "trim keeps the flat arrays sized")
    for i in range(14):
        lo = int(trimmed["legal_off"][i])
        hi = lo + int(trimmed["n_legal"][i])
        ident = ids[i + 6]
        check(trimmed["legal_idx"][lo:hi].tolist() == [(ident + k) % ce.POLICY_SIZE
                                                       for k in range(hi - lo)],
              f"trimmed row {i} keeps its own legal indices")
    check(trimmed["planes"].base is None, "trim copies rather than viewing (a view pins the parent)")


def test_buffer_premium():
    buf = ReplayBuffer(capacity=10_000, decisive_premium=2.0)
    chunk, _ = make_chunk(1, 900, draw_every=3)      # 300 draws, 600 decisive
    buf.add(chunk)
    rng = np.random.default_rng(0)
    idx = buf.sample_indices(200_000, rng)
    got = buf.gather(np.sort(idx))
    frac = float((got["z"] != 0).mean())
    want = 2 * 600 / (2 * 600 + 300)                 # 0.8
    check(abs(frac - want) < 0.01, f"decisive premium: sampled {frac:.3f}, want {want:.3f}")

    flat = ReplayBuffer(capacity=10_000, decisive_premium=1.0)
    flat.add(make_chunk(1, 900, draw_every=3)[0])
    frac1 = float((flat.gather(np.sort(flat.sample_indices(200_000, rng)))["z"] != 0).mean())
    check(abs(frac1 - 2 / 3) < 0.01, f"premium 1.0 is uniform: {frac1:.3f}, want 0.667")

    single = ReplayBuffer(capacity=100)
    single.add(make_chunk(2, 10, draw_every=1)[0])   # all draws: no decisive pool
    check(len(single.sample(32, rng)["z"]) == 32, "a one-sided buffer still samples")


def test_buffer_sample_shape(chunk):
    buf = ReplayBuffer(capacity=1_000_000)
    buf.add(chunk)
    rng = np.random.default_rng(3)
    batch = buf.sample(64, rng)
    check(batch["planes"].dtype == np.uint8, "sampled planes stay uint8 for the bus")
    check(batch["legal"].dtype == np.int64, "legal indices are int64 for torch.gather")
    check(batch["visits"].shape == batch["legal"].shape == batch["valid"].shape,
          "ragged fields share one padded width")
    check(bool((batch["visits"].sum(1) > 0).all()), "every sampled row has visits")
    check(bool((batch["visits"][~batch["valid"]] == 0).all()), "padded slots carry no visits")
    check(bool(np.isin(batch["z"], [-1.0, 0.0, 1.0]).all()), "z is a game result")


# --- 4. the gradient step ----------------------------------------------------

def test_train_steps(chunk):
    import torch

    from az.net import PolicyValueNet

    torch.manual_seed(0)
    device = torch.device("cpu")
    net = PolicyValueNet(channels=16, blocks=1).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    buf = ReplayBuffer(capacity=100_000)
    buf.add(chunk)
    rng = np.random.default_rng(0)

    first = train.train_steps(net, opt, buf, 3, 32, device, np.random.default_rng(0))
    later = train.train_steps(net, opt, buf, 40, 32, device, rng)
    check(np.isfinite(first["loss_p"]) and np.isfinite(first["loss_v"]), "losses are finite")
    check(later["loss_p"] < first["loss_p"],
          f"policy loss descends ({first['loss_p']:.3f} -> {later['loss_p']:.3f})")
    check(later["loss_v"] < first["loss_v"] + 1e-6,
          f"value loss does not diverge ({first['loss_v']:.4f} -> {later['loss_v']:.4f})")
    check(0.0 <= later["policy_entropy"] < 6.0, f"target entropy in nats: {later['policy_entropy']}")
    check(0.0 <= later["value_sign"] <= 1.0, "value sign agreement is a fraction")

    # Steps scale with buffer fill (module docstring): a fresh buffer gets a
    # fraction of the budget, a full one gets all of it.
    args = train.build_parser().parse_args(["--run", "x", "--steps-per-gen", "600", "--min-steps", "50"])
    small = ReplayBuffer(capacity=1_000_000)
    small.add(chunk)
    check(train.steps_for(small, args) == 50, "a nearly empty buffer gets the floor")
    full = ReplayBuffer(capacity=len(chunk["z"]))
    full.add(chunk)
    check(train.steps_for(full, args) == 600, "a full buffer gets the whole budget")
    args.scale_steps = False
    check(train.steps_for(small, args) == 600, "--no-scale-steps takes the flat count")


# --- 5. one generation end to end -------------------------------------------

def test_end_to_end():
    run = tempfile.mkdtemp(prefix="az_train_test_")
    try:
        rc = train.main(["--run", run, "--generations", "2", "--games-per-gen", "4",
                         "--n-parallel", "4", "--sims", "8", "--temp-plies", "4",
                         "--buffer", "4000", "--steps-per-gen", "6", "--min-steps", "2",
                         "--batch", "16", "--channels", "16", "--blocks", "1",
                         "--anchor", "", "--device", "cpu", "--amp", "off",
                         "--allow-concurrent", "--seed", "5"])
        check(rc == 0, "az.train returns 0")
        for name in ("log.csv", "latest.pt", "opt.pt", "config.json", "gen_000.pt", "gen_001.pt"):
            check(os.path.exists(os.path.join(run, name)), f"{name} written")

        with open(os.path.join(run, "log.csv"), newline="") as fh:
            reader = csv_mod.DictReader(fh)
            check(reader.fieldnames == train.CSV_COLUMNS, "log.csv header is the documented schema")
            rows = list(reader)
        check(len(rows) == 2, f"one row per generation (got {len(rows)})")
        for gen, row in enumerate(rows):
            check(int(row["gen"]) == gen, "generations are numbered in order")
            check(int(row["games"]) == 4, "every requested game was played")
            check(int(row["positions"]) > 0 and int(row["buffer_size"]) > 0, "positions reached the buffer")
            check(int(row["steps"]) >= 2, "gradient steps were taken")
            check(0.0 <= float(row["draw_frac"]) <= 1.0, "draw fraction is a fraction")
            check(row["elo_vs_anchor_a"] == "", "no Elo column without anchors")
        check(int(rows[1]["buffer_size"]) > int(rows[0]["buffer_size"]), "the buffer grows")

        from az.net import load_checkpoint
        net = load_checkpoint(os.path.join(run, "latest.pt"))
        check(net.channels == 16 and net.blocks == 1, "the checkpoint knows its own architecture")

        # Restarting in place continues the numbering instead of overwriting
        # gen_000.pt with a later, differently-trained net of the same name.
        train.main(["--run", run, "--init", os.path.join(run, "latest.pt"),
                    "--init-opt", os.path.join(run, "opt.pt"), "--generations", "1",
                    "--games-per-gen", "2", "--n-parallel", "2", "--sims", "8",
                    "--buffer", "4000", "--steps-per-gen", "4", "--min-steps", "2",
                    "--batch", "16", "--anchor", "", "--device", "cpu", "--amp", "off",
                    "--allow-concurrent"])
        check(os.path.exists(os.path.join(run, "gen_002.pt")), "a restart continues the numbering")
        with open(os.path.join(run, "log.csv"), newline="") as fh:
            gens = [int(r["gen"]) for r in csv_mod.DictReader(fh)]
        check(gens == [0, 1, 2], f"log.csv is one lineage, in order (got {gens})")
        with open(os.path.join(run, "config.json")) as fh:
            check(len(json.load(fh)) == 2, "config.json keeps a record per start")

        # The dashboard has to read what the trainer writes.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            dashboard.main([run])
        check("3 generations" in out.getvalue(), "dashboard reads the run log")
    finally:
        shutil.rmtree(run, ignore_errors=True)


class FakeBuffer:
    def __init__(self, draw_frac):
        self.draw_frac = draw_frac


def test_watchlist():
    """warn() fires on docs/05's numbers and stays quiet on healthy ones."""
    args = train.build_parser().parse_args(["--run", "x", "--games-per-gen", "256",
                                            "--n-parallel", "256"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        train.warn({"resign_fp_frac": 0.2, "resign_triggers": 20}, FakeBuffer(0.5),
                   {"avg_batch": 100.0}, args)
    text = out.getvalue()
    check("of 256 slots" in text, "a fleet running empty in the tail is flagged")
    check("value head is starving" in text, "a 50% draw buffer is flagged")
    check("resign false positives" in text, "a hot resign threshold is flagged")

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        train.warn({"resign_fp_frac": 0.01, "resign_triggers": 20}, FakeBuffer(0.12),
                   {"avg_batch": 240.0}, args)
    check(quiet.getvalue() == "", f"healthy numbers say nothing (said {quiet.getvalue()!r})")

    # Not measured this generation: nan must not read as "0% and fine" or fire.
    nan = io.StringIO()
    with contextlib.redirect_stdout(nan):
        train.warn({"resign_fp_frac": float("nan"), "resign_triggers": 0}, FakeBuffer(0.12),
                   {"avg_batch": 240.0}, args)
    check(nan.getvalue() == "", "an unmeasured false-positive rate is not a warning")


def test_orphan_detection():
    """lesson #6, and the false positive that makes the check useless."""
    check(train.is_trainer(["/usr/bin/python3", "-m", "az.train", "--run", "x"]),
          "python -m az.train is a trainer")
    check(train.is_trainer(["python3.13", "/repo/az/train.py", "--run", "x"]),
          "the script path counts as well as the module")
    check(not train.is_trainer(["timeout", "900", "python", "-m", "az.train"]),
          "a `timeout` wrapper is not itself a trainer")
    check(not train.is_trainer(["/bin/bash", "-c", "python -m az.train --run x"]),
          "the launching shell is not a trainer")
    check(not train.is_trainer(["python", "tests/train_test.py"]),
          "a python process that merely mentions training is not a trainer")
    check(not train.is_trainer(["python"]), "a bare interpreter is not a trainer")
    check(isinstance(train.other_trainers(), list), "the /proc scan runs on this box")


def test_dashboard_lineage():
    """Two run dirs with Elo columns: concatenation, absolute scale, advice."""
    root = tempfile.mkdtemp(prefix="az_dash_test_")
    try:
        runs = []
        for r in range(2):
            path = os.path.join(root, f"hunt{r}")
            os.makedirs(path)
            runs.append(path)
            with open(os.path.join(path, "log.csv"), "w", newline="") as fh:
                w = csv_mod.DictWriter(fh, fieldnames=train.CSV_COLUMNS, extrasaction="ignore")
                w.writeheader()
                for g in range(8):
                    # hunt0 climbs, hunt1 is the plateau the playbook is for
                    elo = -100 + 5 * g if r == 0 else -60
                    w.writerow({"gen": g, "games": 256, "avg_plies": 120 - g, "draw_frac": 0.5,
                                "resign_frac": 0.1, "white_score": 0.51, "policy_loss": 2.0 - 0.01 * g,
                                "value_loss": 0.8, "buffer_size": 10000 * (g + 1), "steps": 600,
                                "sims": 400, "anchor_a": "sf1320", "elo_vs_anchor_a": elo,
                                "elo_a_lo": elo - 40, "elo_a_hi": elo + 40,
                                "resign_fp_frac": 0.2, "sp_seconds": 200, "train_seconds": 60,
                                "eval_seconds": 90})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            dashboard.main(runs + ["--width", "40"])
        text = out.getvalue()
        check("16 generations" in text, "runs are concatenated into one lineage")
        check("hunt0 (gen 0-7)" in text and "hunt1 (gen 8-15)" in text, "run boundaries are labelled")
        check("Elo vs sf1320" in text, "the Elo chart is drawn")
        check("1320" in text, "a rated anchor puts the curve on an absolute scale")
        check("value head is starving" in text, "a 50% draw rate raises the docs/05 flag")
        check("false positives" in text, "a hot resign threshold is flagged")
        check("Playbook step 1" in text, "a flat Elo curve suggests raising sims")
    finally:
        shutil.rmtree(root, ignore_errors=True)


TOTAL_GAMES = 4
SIMS = 8


def main():
    section("self-play generation (uniform dummy net)")
    ev = UniformEvaluator()
    gen = play_generation(ev, total_games=TOTAL_GAMES, sims=SIMS, n_parallel=TOTAL_GAMES,
                          temp_plies=4, resign_threshold=-1.0, seed=11)
    check(len(gen["results"]["game_id"]) == TOTAL_GAMES, "every requested game finished")
    check(gen["evals"] == ev.evals and gen["cycles"] > 0, "throughput counters agree with the net")
    check(0 < gen["avg_batch"] <= TOTAL_GAMES, f"avg batch {gen['avg_batch']} within the fleet")
    print(f"  {TOTAL_GAMES} games, {len(gen['examples']['game_id'])} positions, "
          f"{gen['cycles']} cycles, {gen['seconds']:.1f}s")

    section("dump -> chunk replay alignment")
    chunk = test_replay(gen)
    test_empty_dump()

    section("generation statistics")
    test_result_stats()

    section("replay buffer")
    test_buffer_fifo()
    test_buffer_trim()
    test_buffer_premium()
    test_buffer_sample_shape(chunk)

    section("gradient steps")
    test_train_steps(chunk)

    section("run management")
    test_watchlist()
    test_orphan_detection()

    section("end to end")
    test_end_to_end()
    test_dashboard_lineage()

    if failures:
        print(f"\n{failures} FAILURE(S)")
        return 1
    print("\ntrain ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
