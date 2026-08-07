#!/usr/bin/env python
"""Tests for the policy+value network (docs/03-net.md).

The net itself is boilerplate ResNet; what can actually be *wrong* here is the
wiring to everything around it, so that is what these tests are about:

1. Policy geometry. `spatial_to_policy` flattens the 73-plane head into 4672
   logits, and its index convention must be the C++ encoder's
   (`from_square * 73 + type`). Checked against `chess_engine.move_to_index`
   on real positions — including a black-to-move one, where every square is
   flipped — never against a Python restatement of the same formula.
2. Input scaling: the halfmove plane is the only non-binary channel, and the
   caller's uint8 buffer must survive the trip unmodified (the runner reuses
   it).
3. Losses: the ragged/padded form the runner's dumps demand must equal the
   dense masked form, and illegal logits must not enter the loss at all.
4. The runner seam: a real fleet driven to completion by an `Evaluator`, which
   is the only test that exercises numpy dtypes, the (B,4672)/(B,) contract
   and the C++ side in one piece.

Run:  python tests/net_test.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce  # the built module at the repo root
from az.net import (Evaluator, PolicyValueNet, POLICY_SIZE, MOVE_TYPES, NUM_PLANES,
                    dense_policy_loss, load_checkpoint, param_count, planes_to_tensor,
                    policy_loss, save_checkpoint, spatial_to_policy)

failures = 0


def fail(what):
    global failures
    failures += 1
    if failures <= 20:
        print(f"  FAIL: {what}")


def check(ok, what):
    if not ok:
        fail(what)


# Startpos, a black-to-move middlegame (orientation flip), a promotion-rich
# position, and one with a live en-passant square.
POSITIONS = [
    ce.STARTPOS,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 3 2",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3",
]


def test_constants():
    check(NUM_PLANES == ce.NUM_PLANES, f"NUM_PLANES {NUM_PLANES} != engine {ce.NUM_PLANES}")
    check(POLICY_SIZE == ce.POLICY_SIZE, f"POLICY_SIZE {POLICY_SIZE} != engine {ce.POLICY_SIZE}")


def test_architecture():
    net = PolicyValueNet()
    total = param_count(net)
    # docs/03: ~1.9M tower + heads ~= 2.0M total. A drift outside this band
    # means the architecture changed without the doc changing.
    check(1_800_000 <= total <= 2_100_000, f"parameter count {total} outside the ~2.0M budget")
    policy_params = sum(p.numel() for p in net.policy_conv.parameters())
    check(policy_params == 128 * MOVE_TYPES + MOVE_TYPES,
          f"spatial policy head is {policy_params} params (expected ~9.4k, not an FC head)")
    check(len(net.tower) == 6 and net.channels == 128, "v1 tower is 6 blocks of 128")
    print(f"  parameters: {total:,} (policy head {policy_params:,})")

    net.eval()
    x = torch.zeros(3, NUM_PLANES, 8, 8)
    with torch.no_grad():
        logits, v = net(x)
    check(logits.shape == (3, POLICY_SIZE), f"policy shape {tuple(logits.shape)}")
    check(v.shape == (3,), f"value shape {tuple(v.shape)}")
    check(bool((v.abs() <= 1).all()), "tanh keeps the value in [-1, 1]")
    check(torch.isfinite(logits).all(), "logits are finite")

    with torch.no_grad():
        again, v2 = net(x)
    check(torch.equal(logits, again) and torch.equal(v, v2), "eval-mode forward is deterministic")

    # Growth path (docs/05 plateau playbook step 4) must be a constructor arg.
    big = PolicyValueNet(channels=160, blocks=8)
    check(param_count(big) > total, "wider/deeper net builds and is bigger")


def test_policy_geometry():
    """logits[sq * 73 + t] must be head plane t at square sq — and `sq` must be
    where the moving piece actually stands in the mover-oriented planes."""
    b, n = 2, MOVE_TYPES * 64
    head_map = torch.arange(b * n, dtype=torch.float32).reshape(b, MOVE_TYPES, 8, 8)
    flat = spatial_to_policy(head_map)
    check(flat.shape == (b, POLICY_SIZE), f"flattened shape {tuple(flat.shape)}")
    ok = True
    for sq in range(64):
        for t in range(MOVE_TYPES):
            if flat[1, sq * MOVE_TYPES + t] != head_map[1, t, sq // 8, sq % 8]:
                ok = False
    check(ok, "spatial_to_policy indexes as from_square * 73 + move_type")

    # And the full forward agrees with the head map the conv produced.
    net = PolicyValueNet(channels=16, blocks=1).eval()
    captured = {}
    handle = net.policy_conv.register_forward_hook(lambda m, i, o: captured.update(map=o))
    planes = ce.encode_batch(POSITIONS)
    with torch.no_grad():
        logits, _ = net(planes_to_tensor(planes, "cpu", channels_last=False))
    handle.remove()
    check(torch.allclose(logits, spatial_to_policy(captured["map"])),
          "forward flattens the policy conv output with spatial_to_policy")

    # channels_last is a self-play default; a memory format must not reorder
    # the index space (a bug that would cancel out in any fp16-vs-fp32 check).
    with torch.no_grad():
        nhwc_logits, _ = net(planes_to_tensor(planes, "cpu", channels_last=True))
    check(torch.allclose(logits, nhwc_logits, atol=1e-5),
          "channels_last does not permute the policy index")

    # The alignment claim of docs/03: the logit for a legal move is produced by
    # the head cell sitting on that piece's own square. The encoder's index
    # says which cell that is; the input planes say which piece is there.
    for i, fen in enumerate(POSITIONS):
        own_pieces = planes[i, 0:6].reshape(6, 64).sum(axis=0)
        for uci in ce.legal_moves(fen):
            idx = ce.move_to_index(fen, uci)
            from_sq, move_type = divmod(idx, MOVE_TYPES)
            if not (0 <= move_type < MOVE_TYPES and own_pieces[from_sq] == 1):
                fail(f"{fen}: move {uci} -> index {idx} does not sit on a mover's piece")
                break


def test_input_scaling():
    planes = ce.encode_batch(POSITIONS)
    original = planes.copy()
    x = planes_to_tensor(planes, "cpu")
    check(x.shape == (len(POSITIONS), NUM_PLANES, 8, 8) and x.dtype == torch.float32,
          f"planes_to_tensor gives float32 {tuple(x.shape)}")
    check(np.array_equal(planes, original), "the caller's uint8 buffer is not modified in place")
    check(bool(((x >= 0) & (x <= 1)).all()), "every channel is normalized into [0, 1]")

    binary = torch.cat([x[:, :18], x[:, 19:]], dim=1)
    check(bool(((binary == 0) | (binary == 1)).all()), "all planes but the halfmove one stay binary")

    # The halfmove plane carries min(clock, 100); the scaling must land it in
    # [0, 1] with the same saturation the C++ encoder applies.
    clocks = planes_to_tensor(ce.encode_batch(["8/8/4k3/8/8/4K3/8/8 w - - 49 80",
                                               "8/8/4k3/8/8/4K3/8/8 w - - 100 80"]), "cpu")
    check(abs(float(clocks[0, 18, 0, 0]) - 0.49) < 1e-6,
          f"halfmove plane scaled ({float(clocks[0, 18, 0, 0])})")
    check(abs(float(clocks[1, 18, 0, 0]) - 1.0) < 1e-6, "halfmove plane saturates at 1.0")

    strided = planes_to_tensor(planes, "cpu", channels_last=True)
    check(strided.is_contiguous(memory_format=torch.channels_last), "channels_last is honored")
    check(torch.equal(strided, x), "channels_last changes layout, not values")

    try:
        planes_to_tensor(planes.astype(np.float32), "cpu")
        fail("float input must be rejected (it would alias the caller's buffer)")
    except TypeError:
        pass


def make_ragged_batch(rng, b=6, width=None):
    """Random logits plus a padded ragged legal set with visit-count targets."""
    fens = [ce.STARTPOS] * b
    legal = [ce.legal_move_indices(f) for f in fens]
    # Random subsets of varying size, so padding is actually exercised.
    legal = [sorted(rng.choice(l, size=int(rng.integers(2, len(l))), replace=False).tolist())
             for l in legal]
    width = width or max(len(l) for l in legal)

    idx = np.zeros((b, width), dtype=np.int64)
    tgt = np.zeros((b, width), dtype=np.float32)
    valid = np.zeros((b, width), dtype=bool)
    for i, l in enumerate(legal):
        idx[i, :len(l)] = l
        visits = rng.integers(0, 50, size=len(l)).astype(np.float32)
        visits[rng.integers(len(l))] += 1.0  # never an all-zero row
        tgt[i, :len(l)] = visits / visits.sum()
        valid[i, :len(l)] = True
    logits = torch.from_numpy(rng.normal(size=(b, POLICY_SIZE)).astype(np.float32))
    return (logits, torch.from_numpy(idx), torch.from_numpy(tgt), torch.from_numpy(valid), legal)


def test_losses():
    rng = np.random.default_rng(0)
    logits, idx, tgt, valid, legal = make_ragged_batch(rng)

    sparse = policy_loss(logits, idx, tgt, valid)

    mask = torch.zeros(logits.shape, dtype=torch.bool)
    dense_tgt = torch.zeros_like(logits)
    for i, l in enumerate(legal):
        mask[i, l] = True
        dense_tgt[i, idx[i, :len(l)]] = tgt[i, :len(l)]
    dense = dense_policy_loss(logits, dense_tgt, mask)
    check(torch.allclose(sparse, dense, atol=1e-5),
          f"ragged loss {float(sparse):.6f} == dense masked loss {float(dense):.6f}")

    # Illegal logits are outside the softmax, so they cannot move the loss.
    poisoned = logits.clone()
    poisoned[~mask] = 1e4
    check(torch.allclose(policy_loss(poisoned, idx, tgt, valid), sparse, atol=1e-5),
          "illegal logits do not enter the loss")

    # Extra padding must be inert, whatever garbage sits in the pad slots.
    pad = 7
    wide_idx = torch.cat([idx, torch.zeros(idx.shape[0], pad, dtype=torch.int64)], dim=1)
    wide_tgt = torch.cat([tgt, torch.zeros(tgt.shape[0], pad)], dim=1)
    wide_valid = torch.cat([valid, torch.zeros(valid.shape[0], pad, dtype=torch.bool)], dim=1)
    check(torch.allclose(policy_loss(logits, wide_idx, wide_tgt, wide_valid), sparse, atol=1e-5),
          "padded slots do not contribute")
    check(torch.isfinite(policy_loss(logits, wide_idx, wide_tgt, wide_valid)), "padding does not produce NaN")

    # Analytic anchors: uniform target on n equal logits costs log(n); a
    # confident one-hot prediction costs ~0.
    n = len(legal[0])
    flat_logits = torch.zeros(1, POLICY_SIZE)
    uni_idx = idx[:1, :n]
    uni_tgt = torch.full((1, n), 1.0 / n)
    check(abs(float(policy_loss(flat_logits, uni_idx, uni_tgt)) - np.log(n)) < 1e-4,
          "uniform target over n legal moves costs log(n)")
    hot_logits = torch.full((1, POLICY_SIZE), -30.0)
    hot_logits[0, uni_idx[0, 0]] = 30.0
    hot_tgt = torch.zeros(1, n)
    hot_tgt[0, 0] = 1.0
    check(float(policy_loss(hot_logits, uni_idx, hot_tgt)) < 1e-4, "a confident correct policy costs ~0")

    # Gradients reach the net through the whole path.
    net = PolicyValueNet(channels=16, blocks=1)
    x = planes_to_tensor(ce.encode_batch([ce.STARTPOS] * 2), "cpu", channels_last=False)
    out, v = net(x)
    legal0 = ce.legal_move_indices(ce.STARTPOS)
    gidx = torch.tensor([legal0, legal0], dtype=torch.int64)
    gtgt = torch.full((2, len(legal0)), 1.0 / len(legal0))
    loss = policy_loss(out, gidx, gtgt) + torch.nn.functional.mse_loss(v, torch.tensor([1.0, -1.0]))
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    check(len(grads) == len(list(net.parameters())), "every parameter gets a gradient")
    check(all(torch.isfinite(g).all() for g in grads), "gradients are finite")
    check(any(g.abs().sum() > 0 for g in grads), "gradients are non-zero")


def test_checkpoint():
    torch.manual_seed(0)
    net = PolicyValueNet(channels=32, blocks=2).eval()
    x = planes_to_tensor(ce.encode_batch(POSITIONS), "cpu", channels_last=False)
    with torch.no_grad():
        p0, v0 = net(x)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gen_007.pt")
        save_checkpoint(net, path, gen=7)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        check("state_dict" in raw, "checkpoint uses the RL_2048 {'state_dict': ...} convention")
        check(raw.get("gen") == 7, "extra fields ride along")

        loaded = load_checkpoint(path, "cpu")
        check(loaded.channels == 32 and loaded.blocks == 2, "architecture is restored from the checkpoint")
        check(not loaded.training, "loaded net comes back in eval mode")
        with torch.no_grad():
            p1, v1 = loaded(x)
        check(torch.equal(p0, p1) and torch.equal(v0, v1), "reloaded net reproduces the outputs bit-for-bit")


def test_evaluator():
    torch.manual_seed(0)
    net = PolicyValueNet(channels=32, blocks=2)
    planes = ce.encode_batch(POSITIONS)

    ev = Evaluator(net, device="cpu")
    p, v = ev(planes)
    check(p.dtype == np.float32 and v.dtype == np.float32, f"feed dtypes ({p.dtype}, {v.dtype}) must be float32")
    check(p.shape == (len(POSITIONS), POLICY_SIZE) and v.shape == (len(POSITIONS),),
          f"feed shapes {p.shape}, {v.shape}")
    check(p.flags["C_CONTIGUOUS"] and v.flags["C_CONTIGUOUS"], "arrays handed to feed() are C-contiguous")
    check(np.allclose(p.sum(axis=1), 1.0, atol=1e-4), "priors are a full-head softmax (rows sum to 1)")
    check(np.all(p >= 0) and np.all(np.abs(v) <= 1.0), "priors non-negative, values in [-1, 1]")

    empty, empty_v = ev(np.zeros((0, NUM_PLANES, 8, 8), dtype=np.uint8))
    check(empty.shape == (0, POLICY_SIZE) and empty_v.shape == (0,), "an empty batch round-trips")

    # Between generations the self-play copy is re-synced from the trainer.
    with torch.no_grad():
        for param in net.parameters():
            param.add_(0.05)
    check(np.allclose(ev(planes)[0], p, atol=1e-6), "training steps do not leak into the evaluator's copy")
    refreshed = ev.refresh(net)(planes)[0]
    check(not np.allclose(refreshed, p, atol=1e-6), "refresh() picks up the new weights")
    check(np.allclose(refreshed, Evaluator(net, device="cpu")(planes)[0], atol=1e-6),
          "refresh() matches a freshly constructed evaluator")

    # The evaluator must not convert the caller's net in place — the trainer
    # keeps fp32 master weights while self-play runs in fp16 off the same net.
    check(next(net.parameters()).dtype == torch.float32, "constructing an Evaluator leaves the net fp32")

    if torch.cuda.is_available():
        # A fresh net: the perturbed one above has no reason to stay inside
        # fp16's range, and this test is about precision, not about that.
        torch.manual_seed(1)
        fresh = PolicyValueNet(channels=32, blocks=2)
        half = Evaluator(fresh, device="cuda", dtype=torch.float16)
        check(next(fresh.parameters()).dtype == torch.float32 and
              next(fresh.parameters()).device.type == "cpu",
              "the fp16 evaluator did not move or truncate the training net")
        p32, v32 = Evaluator(fresh, device="cuda", dtype=torch.float32)(planes)
        p16, v16 = half(planes)
        check(np.abs(p32 - p16).max() < 5e-3, f"fp16 priors track fp32 (max diff {np.abs(p32 - p16).max():.2e})")
        check(np.abs(v32 - v16).max() < 5e-2, f"fp16 values track fp32 (max diff {np.abs(v32 - v16).max():.2e})")
        check(np.isfinite(p16).all() and np.isfinite(v16).all(), "fp16 forward has no NaN/Inf")
    else:
        print("  (no CUDA: skipping the fp16 comparison)")


def test_runner_integration():
    """The whole 02<->03 seam: a real fleet played out by a real net."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ev = Evaluator(PolicyValueNet(channels=16, blocks=2), device=device)

    total_games, sims = 2, 8
    runner = ce.Runner(total_games=total_games, n_parallel=2, sims=sims, temp_plies=4, seed=11)
    cycles, max_cycles = 0, ce.MAX_PLIES * sims * 2 + 1000
    while True:
        batch = runner.pending()
        if not batch.size:
            break
        cycles += 1
        if cycles > max_cycles:
            fail("self-play with the real net did not finish within the cycle budget")
            return
        runner.feed(*ev(batch))

    check(runner.games_completed == total_games, "the fleet finished every game")
    ex, res = runner.get_examples(), runner.get_results()
    check(len(res["game_id"]) == total_games, "one result row per game")
    check(len(ex["game_id"]) == int(res["plies"].sum()), "one example per ply played")

    # Replaying the dump is the check that the net's priors produced *legal*
    # moves all the way down: push_u16 raises on anything else.
    offsets = np.concatenate([[0], np.cumsum(ex["n_legal"], dtype=np.int64)])
    game = ce.Game()
    for r in range(len(ex["game_id"])):
        if int(ex["game_id"][r]) != int(ex["game_id"][0]):
            break
        check(int(ex["visits"][offsets[r]:offsets[r + 1]].sum()) == sims - 1,
              f"ply {r}: root visits sum to sims-1")
        game.push_u16(int(ex["move"][r]))
    print(f"  self-play with the net: {total_games} games, {int(res['plies'].sum())} plies, "
          f"{cycles} forwards on {device}")


def main():
    test_constants()
    test_architecture()
    test_policy_geometry()
    test_input_scaling()
    test_losses()
    test_checkpoint()
    test_evaluator()
    test_runner_integration()

    if failures:
        print(f"\n{failures} FAILURE(S)")
        return 1
    print("\nnet ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
