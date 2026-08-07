# 03 — Network

## Architecture v1

Small AlphaZero-style ResNet. Start small; the plateau playbook widens it
only when self-play stalls with sims already raised.

```
input  (19, 8, 8)
stem   conv3x3 19→128, BN, ReLU
tower  6 × residual block [conv3x3 128→128, BN, ReLU, conv3x3, BN, +skip, ReLU]
policy head:  conv1x1 128→73, reshape → 4672 logits      (AZ-style spatial head)
value head:   conv1x1 128→8, BN, ReLU, flatten(512) → fc 256 → ReLU → fc 1 → tanh
```

- **Spatial policy head** (73 planes over 8×8) instead of a giant FC:
  ~9k params vs ~2.4M, and the from-square structure of the move encoding
  ([01](01-engine.md)) aligns with it by construction.
- Parameter count ≈ **1.9M tower + heads ≈ 2.0M total** — deliberately
  ~2.5× the 2048 net, not 20×. Growth path: 6×128 → 8×160 → 10×192,
  changing one axis at a time.
- BatchNorm is fine (training batches are large); if train/inference
  drift ever bites, switch to GroupNorm and retrain — noted, not expected.

## Targets and losses

- Policy: cross-entropy against the sparse visit distribution (self-play)
  or hard human move label (bootstrap). Logits outside the legal set are
  excluded from the softmax at *training* time too (mask with -inf) so
  the net never wastes capacity suppressing illegal moves.
- Value: MSE against game result z ∈ {-1, 0, +1} from the mover's
  perspective. tanh output keeps it bounded; no reward shaping — chess
  has no dense reward and we do not invent one.
- Total: `L = CE + λ·MSE`, λ=1 to start; watch the value head for draw
  collapse (predicting ~0 everywhere) — if it happens, upweight decisive
  games in sampling before touching λ.

## Inference budget (the number that rules the system)

The Runner cycle time is GPU-bound. Targets on the 5090, fp16, B=256:

| Net | Est. forward | Cycle share |
|---|---|---|
| 6×128 (v1) | ~1.5–2.5 ms | fine |
| 10×192 | ~4–6 ms | halves self-play throughput — pay only for measured Elo |

Practices: `torch.inference_mode()`, fp16 (or bf16) weights for
self-play, channels_last, optionally `torch.compile` (measure, don't
assume). Keep a single copy of the net on GPU shared by self-play and
eval; checkpoints in fp32.

## Checkpoint format

Same convention as RL_2048 (`{"state_dict": ...}` + `gen_NNN.pt` /
`latest.pt` naming) so run-management tooling ports over unchanged.
