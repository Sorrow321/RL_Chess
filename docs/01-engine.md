# 01 — Engine: movegen, rules, encodings

## Movegen: embed, don't write

Primary candidate: **[Disservin/chess-library](https://github.com/Disservin/chess-library)**
— header-only C++17, clean API (`Board`, `Movelist`, `movegen::legalmoves`),
built-in Zobrist hashing, FEN I/O, and it's fast enough (tens of millions
of nodes/s in perft). Fallbacks if benchmarks disappoint: `surge`, or
extracting Stockfish's movegen (fastest, most integration work).

Selection procedure: wire up perft for 2–3 candidates, compare nodes/s and
API friction, pick one, delete the others. Half a day, do not overthink.

## Rules coverage checklist

All of these must be handled and *tested* before any RL:

- [ ] Castling (rights tracking through rook/king moves and captures)
- [ ] En passant (including the perft-famous pin edge cases)
- [ ] Promotions (Q/R/B/N — underpromotions matter for the move encoding)
- [ ] Fifty-move rule → draw
- [ ] Threefold repetition → draw (needs position-hash history per game;
      the library's Zobrist keys + a small per-game hash list)
- [ ] Stalemate, checkmate detection
- [ ] Insufficient material → draw
- [ ] Self-play ply cap: **hard draw at 250 plies** (adjudication; keeps
      degenerate early-training games from running forever)

## Perft validation (the trust anchor)

Standard suite, exact node counts, run as a test binary + CI-style script:

| Position | Depth | Expected nodes |
|---|---|---|
| startpos | 6 | 119,060,324 |
| Kiwipete (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -`) | 5 | 193,690,690 |
| Position 3 (`8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -`) | 6 | 11,030,083 |
| Position 4 (`r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -`) | 5 | 15,833,292 |
| Position 5 (`rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -`) | 5 | 89,941,194 |

Any mismatch anywhere = stop, fix, re-run all.

## Input encoding: position → planes

**19 binary planes of 8×8, always oriented to the side to move** (flip
ranks and swap piece colors when black moves). Orientation-to-mover halves
what the net must learn and is the single most important encoding decision.

| Planes | Content |
|---|---|
| 0–5 | own P, N, B, R, Q, K |
| 6–11 | opponent P, N, B, R, Q, K |
| 12 | ones (side-to-move constant / bias plane) |
| 13–16 | castling rights: own K-side, own Q-side, opp K-side, opp Q-side |
| 17 | en-passant file (file column set, else zero) |
| 18 | halfmove clock / 100 (the one non-binary plane) |

No history planes in v1 (AZ used 8 half-moves; at our scale they cost
input width for unclear gain). Recorded as an open lever in
[05](05-selfplay-training.md).

Implementation note: the C++ encoder writes `uint8` planes (plane 18
scaled ×100 into 0–100) directly into a caller-provided numpy buffer;
Python converts to fp16 on GPU. The same encoder is exposed to Python
(`encode_batch(fens_or_game_refs) -> (B,19,8,8) uint8`) so training and
inference share one implementation — one source of truth, like the 2048
`row_tbl`.

## Move encoding: move ↔ index (AlphaZero 8×8×73)

Index = `from_square * 73 + move_type`, from the mover's perspective
(same orientation as planes):

- types 0–55: queen-like moves — 8 directions × 7 distances
- types 56–63: knight moves — 8 offsets
- types 64–72: underpromotions — {N, B, R} × {forward, capture-left,
  capture-right} (queen promotions ride the queen-move planes)

Total 4672 logits. Both directions (`move_to_index`, `index_to_move`)
live next to the movegen in C++ and are exposed to Python.

**Test:** for every legal move in every position of a depth-4 perft walk
from the five suite positions, assert
`index_to_move(move_to_index(m)) == m` and that no two legal moves in a
position collide. This is the encoding's perft.

## Deliverables of this phase

- `cpp/chess_engine.cpp` (or split headers): movegen wrapper, rules/draw
  logic, both encoders, perft.
- `tests/perft_test` binary + a script that runs the full suite.
- pybind11 exposure of: `legal_moves(fen)`, `push(fen, idx) -> fen'`,
  `encode_batch`, `move_to_index`/`index_to_move` — enough for Python-side
  tooling and the eval harness even before the Runner exists.
