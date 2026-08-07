# Chess AlphaZero

**Status: Phase 0 in progress.** The engine (movegen, rules, perft) is in;
the encoders, the runner and everything above them are still design docs.
An AlphaZero-style chess agent, built on the architecture proven in the
sibling project [`RL_2048`](../RL_2048) (batched C++ MCTS bridge + PyTorch
in Python).

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

## Build

```bash
git submodule update --init          # pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)       # -> chess_engine*.so at the repo root
```

`tests/run_perft.sh` is the Phase 0 gate: the full perft suite (442M nodes,
under a second) plus the rules tests. It must be green before anything
downstream of the movegen is worth running.

```python
import chess_engine as ce

g = ce.Game()                        # or ce.Game(fen, max_plies=250)
g.push("e2e4")
g.legal_moves()                      # UCI strings
g.outcome()                          # <Outcome none value=0>, .reason is a Termination
ce.perft(ce.STARTPOS, 6, threads=8)  # 119060324
```
