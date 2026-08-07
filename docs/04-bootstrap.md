# 04 — Supervised bootstrap from Lichess

The chess analog of RL_2048's heuristic bootstrap, with a stronger
theoretical backing: Maia showed a policy-only net trained on human games
plays at roughly the level of its training population. One supervised
overnight should put the greedy policy at or above the 1000-Elo milestone
before any self-play happens.

## Data source

[database.lichess.org](https://database.lichess.org) monthly PGN dumps
(`.pgn.zst`, ~100M games/month recently). One month is more than enough.

Filtering (applied while streaming the zst, never fully decompressed):

- time controls: rapid + classical + blitz ≥ 3+0 (bullet moves are noise)
- both players rated **1400–2200** (the band we want the prior to imitate;
  below 1400 teaches blunders, above 2200 is sparse)
- rated games only, standard variant, no abandoned/cheat-flagged
- game length ≥ 20 plies
- drop the final 4 plies of games lost on time (panic moves)

Expected yield from one month: ~10–20M games ⇒ **~500M–1B positions**;
we will sample, not exhaust.

## Preprocessing pipeline

`python-chess` for PGN parsing is slow but this is offline and
parallelizes trivially: shard the zst stream across 16 workers, each
emits binary records. Budget ~a few hours once; if it's painful,
`pgn-extract` prefilters faster.

Record per position (compact, planes derived later by the C++ encoder):

```
moves_so_far  — stored once per game as u16 move list
played        u16      (the human's move -> hard policy label)
result        i8       (+1/0/-1 from mover's perspective)
mover_elo     u16      (kept for later filtering/weighting experiments)
```

Then a packing step replays games through the C++ encoder into training
shards of `(planes uint8 (19,8,8), label u16, z i8)` — 1.2 KB/position,
so a 100M-position training set ≈ 120 GB. If disk is annoying, encode
on the fly from the move lists in the dataloader (CPU has 16 cores idle
during training; measure which is faster).

Position sampling: at most **4 positions per game** (random plies) to
fight opening duplication — the first 10 plies of chess are a handful of
positions repeated millions of times; without capping, the net memorizes
openings and starves on middlegames.

## Training

- Net from [03](03-net.md), hard-label CE on the played move + MSE on z.
- ~100M sampled positions, batch 4096, Adam 1e-3 with cosine decay,
  2–3 epochs. On the 5090: a few hours.
- Expected top-1 move agreement: **45–55%** on held-out games (human
  agreement numbers from the Maia line of work). Below 40% = bug hunt
  (most likely: orientation flip inconsistency between encoder uses, or
  move-index mismatch — test via [01](01-engine.md) round-trips first).

## Sanity gates (before building anything else on top)

1. Greedy policy (argmax, no search) vs random-mover: must win ~100%.
2. Greedy vs Stockfish `UCI_Elo=1320` (its floor): should score points.
3. Eyeball 5 games as PGN: openings sane, no one-move piece hangs every
   game, promotions/castling actually used.

Deliverable: `bootstrap.pt` checkpoint + a short report of the three
gates, wired into the [06](06-eval.md) harness.
