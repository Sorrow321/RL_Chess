#!/usr/bin/env python
"""Forward-pass latency of the net (docs/03-net.md, "Inference budget").

One GPU forward per runner cycle serves every game in the fleet, so this
number sets self-play throughput for the whole project. docs/03 budgets
~1.5-2.5 ms at B=256 fp16 for the 6x128 v1 net on a 5090, and ~4-6 ms for a
10x192 — "pay only for measured Elo". This script is how that gets measured
instead of assumed, including the parts docs/03 explicitly refuses to take on
faith: channels_last and torch.compile.

    python bench/net_bench.py                        # v1 net, the default sweep
    python bench/net_bench.py --batch 256 --dtype fp16 --compile
    python bench/net_bench.py --channels 192 --blocks 10   # the growth-path net

Timings are end-to-end through az.net.Evaluator: uint8 planes in, numpy
priors and values out, exactly what the runner pays per cycle. That includes
the host-to-device copy and the softmax, which a bare forward-pass timing
would flatter away.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from az.net import Evaluator, PolicyValueNet, NUM_PLANES, param_count

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# Self-play shape from docs/02/05, used to turn milliseconds into games/hour.
SIMS_PER_MOVE = 400
PLIES_PER_GAME = 120


def time_evaluator(ev, planes, iters, warmup):
    for _ in range(warmup):
        ev(planes)
    if ev.device.type == "cuda":
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ev(planes)
        if ev.device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="64,128,256,512", help="comma-separated batch sizes")
    ap.add_argument("--dtype", default="fp16,fp32", help="comma-separated: fp16, bf16, fp32")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--compile", action="store_true", help="also time a torch.compile'd copy")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    net = PolicyValueNet(channels=args.channels, blocks=args.blocks)
    device = torch.device(args.device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"net {args.blocks}x{args.channels}, {param_count(net):,} parameters | {name} | "
          f"torch {torch.__version__}")
    print(f"channels_last={not args.no_channels_last}, {args.iters} iters after {args.warmup} warmup\n")

    print(f"{'batch':>6} {'dtype':>6} {'compile':>8} {'median ms':>10} {'best ms':>9} "
          f"{'pos/s':>10} {'games/h':>9}")
    rng = np.random.default_rng(0)
    for batch in [int(b) for b in args.batch.split(",")]:
        # Plane *content* does not affect timing; the shape and dtype do.
        planes = rng.integers(0, 2, size=(batch, NUM_PLANES, 8, 8), dtype=np.uint8)
        for dt in args.dtype.split(","):
            variants = [("no", False)] + ([("yes", True)] if args.compile else [])
            for label, use_compile in variants:
                ev = Evaluator(net, device=device, dtype=DTYPES[dt],
                               channels_last=not args.no_channels_last)
                if use_compile:
                    ev.net = torch.compile(ev.net)
                median, best = time_evaluator(ev, planes, args.iters, args.warmup)
                pos_per_s = batch / (median / 1e3)
                games_per_h = pos_per_s * 3600 / (SIMS_PER_MOVE * PLIES_PER_GAME)
                print(f"{batch:>6} {dt:>6} {label:>8} {median:>10.2f} {best:>9.2f} "
                      f"{pos_per_s:>10,.0f} {games_per_h:>9,.0f}")

    print(f"\ngames/h is the GPU-bound ceiling: {SIMS_PER_MOVE} sims x {PLIES_PER_GAME} plies per game "
          "(docs/02/05),\nassuming every sim needs an evaluation and the C++ descent hides under the "
          "forward.\nMeasured self-play throughput below ~half of it means profiling, not more GPU.")


if __name__ == "__main__":
    main()
