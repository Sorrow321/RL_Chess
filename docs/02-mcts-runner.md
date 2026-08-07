# 02 — Batched MCTS Runner (the bridge)

Port of `RL_2048/cpp/az_engine.cpp`'s Runner to a two-player,
deterministic game. The public API is intentionally identical:

```python
runner = chess_az.Runner(total_games, n_parallel, sims,
                         c_puct, dirichlet_alpha, dirichlet_eps,
                         temp_plies, resign_threshold, seed)
while (batch := runner.pending()).size:      # (B,19,8,8) uint8 planes
    p, v = net(gpu(batch))                   # p: (B,4672) softmax, v: (B,) tanh
    runner.feed(p, v)
examples = runner.get_examples()
results  = runner.get_results()              # per game: result, plies, termination
```

## What changes vs the 2048 tree

| Aspect | 2048 | Chess |
|---|---|---|
| Chance nodes | sampled spawns + progressive widening | **none — delete entirely** |
| Value range | [0, ~1.2], death = 0 | **[-1, 1] from side-to-move perspective** |
| Backup | same-sign up the path | **negate at every ply (negamax)** |
| Terminal values | 0 | mate = **-1 for the mated mover**; all draws = 0 |
| Actions | 4 | up to ~218 legal, indices into 4672 |
| Move choice | argmax visits | **τ=1 visit-sampling for first `temp_plies` (~15), then argmax** |

The negamax convention is the classic bug farm: Q stored at an action
edge is from the perspective of the player *making* that move; when
backing up value `v` evaluated at a node, flip sign each step upward.
Write a unit test with a mate-in-2 position: the tree must find it with
enough sims and priors set uniform.

## Node layout (arena, per game, reset per move)

```cpp
struct ActionEdge {           // one per legal move of the parent
    uint16_t move_index;      // 0..4671
    uint16_t move;            // library-native move for push()
    float    prior;
    float    acc_value;       // sum of backed-up values, mover's perspective
    int32_t  n_visits;
    StateNode *child;         // nullptr until first descent
};
struct StateNode {
    ActionEdge *edges;        // arena array
    ActionEdge *parent_edge;
    int32_t  n_visits;
    int16_t  n_edges;         // -1 unexpanded, 0 terminal
    int8_t   terminal_value;  // set when n_edges == 0
};
```

Positions are NOT stored in nodes — each slot keeps one `Board` and
makes/unmakes moves during descent (movegen libraries support this
cheaply). Arena sized as `sims * (sizeof(StateNode) + avg_branching *
sizeof(ActionEdge))` with generous slack; abort on overflow like 2048.

## Search per sim

1. Descend by PUCT: `argmax_a Q(a) + c_puct * P(a) * sqrt(N) / (1 + n_a)`,
   Q from the mover's perspective at that node (flip acc sign as needed),
   unvisited Q = 0. Make each move on the slot board.
2. At an unexpanded node: generate legal moves (structure expansion).
   No legal moves → terminal (mate/stalemate via in-check test), back up
   ±1/0 immediately, no net call. Draw rules checked here too
   (50-move, repetition against the game's hash history + path hashes).
3. Otherwise → pending: encode planes into the batch buffer, pause slot.
4. `feed`: attach priors (mask to legal indices, renormalize; Dirichlet
   mix at root), back up value with sign flips, unmake to root, next sim.

## Game loop per slot

After `sims` simulations: record example, pick move (τ-sample or argmax),
push on the *game* board, append Zobrist key to the game's repetition
history, check game end (incl. 250-ply cap), start next move search or
flush the finished game and start a new one — all identical in shape to
2048's `finish_move`.

**Resign rule (new, big throughput lever):** if root value < `-0.95` for
8 consecutive own moves, resign (loss). Cuts dead-lost tails off games.
Validation guard: run 10% of self-play games with resignation disabled
and log how often the "resigned" side would have salvaged a draw/win —
keep the false-positive rate under ~5% or raise the threshold.

## Example record (what training consumes)

Per move, mirroring the 2048 dump philosophy (raw facts, small):

```
game_id      u32
ply          u16
move_played  u16      (library move)
result       i8       (+1/0/-1 from THIS mover's perspective, filled at game end)
n_legal      u16
legal_idx    u16[n_legal]     (indices into 4672)
visits       u16[n_legal]     (aligned with legal_idx — sparse policy target)
```

Positions are reconstructed for training by replaying `move_played`
sequences through the C++ encoder (`encode_batch`) — dumps stay tiny
(~1 KB/move) and the encoder stays the single source of truth.

## Threading

Same as 2048: OpenMP parallel-for over slots inside `pending()` with the
GIL released; mutex-guarded flush of finished games. Slot boards make
this embarrassingly parallel.

## Throughput budget (sanity math, 5090)

256 slots × ~120 plies × 400 sims, net forward ~2–3 ms at B≈256 fp16,
C++ descent+movegen well under the GPU time → ≈ 45k sims/s/game-slot
aggregate ⇒ **~5–8k games/hour**. If measured throughput lands under
half of this, profile before proceeding (likely suspects: encode copies,
GIL round-trip overhead, movegen in descent).
