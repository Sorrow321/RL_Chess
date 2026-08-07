# 06 — Elo evaluation harness

Built **before** bootstrap and self-play (phase 0.5): every later phase's
exit criterion is a number this harness produces. "1000 Elo" is only
meaningful against calibrated opponents.

## Design: UCI wrapper + cutechess-cli

Wrap the agent as a UCI engine (thin Python script: reads `position`/`go`,
runs net+MCTS via the Runner with `total_games=1`-style single search or a
direct search entry point, prints `bestmove`). Then reuse mature tooling
instead of writing a tournament manager:

- **cutechess-cli** runs the matches: openings book, time/node control,
  draw/resign adjudication, PGN output, SPRT support.
- Opponents: **Stockfish with `UCI_LimitStrength=true, UCI_Elo=N`**
  (calibrated floor 1320) and **`Skill Level` 0–5** for below-floor
  anchors; a random-mover UCI script as the absolute floor.

Fallback if the UCI wrapper is annoying: in-process orchestration with
python-chess talking to Stockfish. Slower and more code to trust — the
wrapper is preferred precisely because cutechess is battle-tested.

## Match protocol

- 100 games per pairing minimum (±~35 Elo at 95% CI near even scores);
  400 games (±~17) for milestone claims.
- Openings: a small varied book (e.g. 2–4 moves deep, cutechess `-openings`)
  with color-swapped pairs, so results aren't one opening's opinion.
- Our side: fixed **sims/move** (e.g. 400 for tracking, 1600 for
  milestone attempts — search amplification, lesson #4). No wall-clock
  time controls for the agent; node-limited Stockfish
  (`go nodes N`) keeps its strength stable across machines.
- Elo from match score with binomial CI; `ordo` over the PGNs when
  multiple anchors/pairings accumulate.

## Anchor ladder

| Anchor | Purpose |
|---|---|
| random-mover | floor sanity, must be ~100% win from day one |
| Stockfish Skill 0–3 (node-limited) | sub-1320 gradations |
| SF `UCI_Elo` 1320 | **Milestone 1 anchor** |
| SF `UCI_Elo` 1700 | tracking anchor for self-play phase |
| SF `UCI_Elo` 2000+ | Milestone 2 territory |

Report vs *two adjacent* anchors at all times (see [05](05-selfplay-training.md)
metrics) — one anchor saturates as the agent improves.

## Caveats to write down now, not rediscover

- Stockfish `UCI_Elo` calibration assumes its own time management; with
  node limits the effective strength shifts — calibrate once by playing
  anchors against each other, then freeze the node counts.
- Draw adjudication (cutechess `-draw movenumber=40 movecount=8 score=8`)
  keeps weak-play marathons short but must match self-play's cap
  philosophy, or eval Elo and self-play behavior drift apart.
- The UCI wrapper must be deterministic given a seed except for search
  randomness — log seeds in PGN tags for reproducibility.
