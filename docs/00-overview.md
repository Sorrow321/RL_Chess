# 00 — Overview

## Goal and success criteria

- **Milestone 1:** after Phase 0 + one supervised overnight, the agent
  (net + MCTS) beats Stockfish limited to ~1000 Elo in a 100-game match
  with statistical significance.
- **Milestone 2:** after several self-play nights, ~1800 Elo by the same
  harness.
- Non-goal: superhuman play. That is a different compute regime and none
  of the designs here should be contorted for it.

## Architecture (inherited from RL_2048, adapted)

```
┌────────────────────────── Python ──────────────────────────┐
│ torch net (fp16)   training loop   replay buffer   eval    │
│        ▲    │                                              │
│  planes │    │ priors (4672) + values                      │
└─────────┼────┼─────────────────────────────────────────────┘
          │    ▼
┌──────── C++ (pybind11 module) ─────────────────────────────┐
│ Runner: N concurrent games, each an arena-allocated PUCT   │
│ tree; pauses at leaves needing eval; batches them.         │
│ Movegen: embedded library, perft-validated.                │
│ Encoders: position→planes, move↔index (shared with Python).│
└────────────────────────────────────────────────────────────┘
```

One GPU forward per cycle serves every game's pending leaf. This pattern
delivered ~10k moves/s (dummy net) and ~2.3k moves/s (real net) on 2048;
chess will be slower per move (bigger net, more sims) but the shape is
identical.

## Phases

| Phase | What | Effort | Exit criterion |
|---|---|---|---|
| 0 | Engine: movegen + encodings ([01](01-engine.md)) | 1–2 days | perft exact to depth 6; encoding tests green |
| 0.5 | Eval harness ([06](06-eval.md)) | 0.5 day | Elo of random-mover measured with CI |
| 1 | Supervised bootstrap ([04](04-bootstrap.md)) | 1 night | greedy policy > 1000 Elo behavior |
| 2 | Batched runner + net ([02](02-mcts-runner.md), [03](03-net.md)) | 1–2 days | self-play games complete, legal, ~5k games/h |
| 3 | Self-play loop ([05](05-selfplay-training.md)) | nights | Elo curve up and to the right |

## Hardware assumptions

RTX 5090 (32 GB), 16-core CPU. All throughput estimates in these docs
assume this box and should be re-derived if it changes.

## Lessons imported from RL_2048 (binding unless contradicted by data)

1. **Verify every low-level component against a reference before trusting
   it.** Perft and encoding round-trips are the chess equivalents of the
   2048 engine equivalence tests, which caught real bugs.
2. **Hard labels for bootstrap policy.** Weak-search visit distributions
   were near-uniform noise in 2048. Human moves are already hard labels.
3. **Sims per move gate target quality.** When self-play plateaus, raise
   sims before touching anything else. (2048: 160→256 sims broke a
   6-generation plateau.)
4. **Search amplification at eval is the cheap milestone-chaser.**
   Evaluate at 2–4× self-play sims when hunting a target. (2048: the 8192
   came from a 1536-sim eval on a net trained at 256.)
5. **Build the measurement harness before the thing it measures.**
6. **Kill processes by exact PID and verify with pgrep** — an orphaned
   trainer silently halved throughput for 40 minutes once.
7. **Log per-generation CSV from day one** and keep the dashboard
   renderer; plateau decisions need the curve, not memory of it.
