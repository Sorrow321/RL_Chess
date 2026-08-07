# Chess AlphaZero

**Status: design phase.** No code yet — this repo currently holds the
component design docs for an AlphaZero-style chess agent, built on the
architecture proven in the sibling project
[`RL_2048`](../RL_2048) (batched C++ MCTS bridge + PyTorch in Python).

Goal: an agent that clearly clears ~1000 Elo after one overnight run on a
single RTX 5090, with a path to 1800+ over subsequent nights.

## Design docs

| Doc | Component |
|---|---|
| [00-overview](docs/00-overview.md) | Architecture, phases, goals, inherited lessons |
| [01-engine](docs/01-engine.md) | C++ movegen integration, encodings, perft validation |
| [02-mcts-runner](docs/02-mcts-runner.md) | Batched two-player PUCT runner (the bridge) |
| [03-net](docs/03-net.md) | Network architecture and inference budget |
| [04-bootstrap](docs/04-bootstrap.md) | Supervised pretraining on Lichess games |
| [05-selfplay-training](docs/05-selfplay-training.md) | Self-play loop, buffer, hyperparameters |
| [06-eval](docs/06-eval.md) | Elo measurement harness |

Build order: 01 → 06 → 04 → 02/03 → 05. The eval harness comes early on
purpose — every later phase needs it to know whether it worked.
