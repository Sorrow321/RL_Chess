# 05 — Self-play training loop

Structurally the RL_2048 `az/train.py` generation loop with chess metrics.
Per generation: self-play → buffer → gradient steps → checkpoint → eval →
CSV row. Keep the dashboard renderer.

## Hyperparameters v1 (starting points, not commitments)

| Knob | Value | Notes |
|---|---|---|
| games/generation | 256 | |
| n_parallel | 256 | one slot per game |
| sims/move (self-play) | 400 | raise first at plateaus (lesson #3) |
| c_puct | 1.5 | |
| Dirichlet α / ε | 0.3 / 0.25 | α=0.3 standard for chess branching |
| temperature plies | 15 | τ=1 sampling, then argmax |
| resign threshold | -0.95 × 8 moves | with the 10% no-resign audit |
| ply cap | 250 → draw | |
| buffer | 1.5M positions | FIFO, ~15–20 generations deep |
| steps/generation | 600 × batch 2048 | ≈ 1 buffer epoch per generation |
| lr | 1.5e-4 Adam | bootstrap-initialized net; decay at plateaus |
| eval | 100 games vs 2 Stockfish anchors | see [06](06-eval.md) |

Throughput target: ~5–8k games/h ⇒ a generation ≈ 3–5 min self-play +
~1 min train + eval. Comparable cadence to the 2048 runs, which made the
plateau playbook responsive.

## Buffer sampling

- Uniform over the buffer as baseline.
- **Decisive-game premium ×2** (draws dominate as play strengthens and
  flatten the value signal — the chess analog of 2048's late-game
  rarity problem). Log the buffer's draw fraction per generation.
- Open lever, not v1: endgame-ply upweighting; opening dedup by
  position-hash count cap.

## Plateau playbook (imported, in order)

1. Self-play sims 400 → 640 → 800 (restart run from latest checkpoint;
   buffer refills in ~5 generations — accept it, as in 2048).
2. lr decay ×0.5.
3. Sampling reweights (draw premium ↑, endgame upweight).
4. Net one size up (6×128 → 8×160), reinitialize optimizer, keep weights
   via net2net-style widening or just retrain heads first — decide when
   there.

## Metrics per generation (CSV)

`gen, games, avg_plies, draw_frac, resign_frac, white_score,
policy_loss, value_loss, buffer_size, elo_vs_anchor_a, elo_vs_anchor_b,
sp_seconds, train_seconds, eval_seconds`

Watch specifically:
- **draw_frac** → buffer reweighting trigger (healthy weak-play range is
  ~5–15%; >40% means the value head is starving).
- **avg_plies** falling + resign_frac rising = resign threshold too hot.
- Elo vs *two* anchors (e.g. 1300 and 1700) — a single anchor saturates
  and the curve goes blind, the exact failure the 2048 tile-count metric
  had before we added higher-sims evals.

## Run management

- `--target-elo` stop condition mirroring 2048's `--target-tile`.
- Restarts always from `latest.pt` into a fresh run dir (`runs/huntN`),
  dashboard concatenates run logs — the lineage bookkeeping from 2048
  applies verbatim, including the orphaned-process check on every
  restart.
