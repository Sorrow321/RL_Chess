"""Stage 2 of the supervised bootstrap (docs/04): game records -> training shards.

    python -m az.pack --games data/games --out data/shards --workers 16

Replays the stage-1 games, samples at most `--per-game` plies from each, and
runs those positions through the C++ encoder into fixed-size records:

    planes u8[19*64] | label u16 | label_pos u16 | n_legal u16
                     | legal_off u64 | z i8 | elo u16 | ply u16 | pad -> 1240 B

Why the sampling cap (docs/04): the first ten plies of chess are a handful of
positions repeated millions of times. Without a per-game cap the net memorizes
openings and starves on middlegames.

Why the legal move set is stored: docs/03 masks illegal logits out of the
softmax at *training* time, so the loss needs to know what was legal. The
runner's ragged layout is reused verbatim — `n_legal` entries at `legal_off`
in the shard's side file — which costs ~6% over the planes and lets the
bootstrap and self-play share one loss function. `label_pos` is the played
move's slot inside that list, so the hard label is a plain index.

Disk: 1240 B/position, i.e. ~124 GB for the doc's 100M positions. Cap with
--max-positions and check `df` first; the intermediate game records are 30x
smaller, so re-packing a different sample is cheap.
"""
import argparse
import collections
import multiprocessing as mp
import os
import sys
import time

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

from az.pgn import iter_records, log_flushed, unpack_move

PLANE_BYTES = ce.NUM_PLANES * 64

POS_DTYPE = np.dtype([
    ("planes", "u1", (PLANE_BYTES,)),
    ("label", "<u2"),      # policy index (0..4671) of the human's move
    ("label_pos", "<u2"),  # where that index sits in the legal list below
    ("n_legal", "<u2"),
    ("legal_off", "<u8"),  # into the shard's .legal file (u16 entries)
    ("z", "i1"),           # game result from THIS mover's perspective
    ("elo", "<u2"),        # mover's rating, for later filtering experiments
    ("ply", "<u2"),
    ("pad", "u1", (5,)),
])
assert POS_DTYPE.itemsize == 1240

LEGAL_DTYPE = np.dtype("<u2")


def sample_plies(n_plies, per_game, rng):
    """Uniform plies without replacement, sorted so the replay is one pass."""
    k = min(per_game, n_plies)
    return np.sort(rng.choice(n_plies, size=k, replace=False))


def positions_from_game(rec, per_game, rng):
    """Replay one game, yielding (fen, uci, z, elo, ply) for the sampled plies."""
    board = chess.Board()
    wanted = set(int(p) for p in sample_plies(len(rec.moves), per_game, rng))
    out = []
    for ply, code in enumerate(rec.moves):
        move = unpack_move(int(code))
        if ply in wanted:
            white_to_move = ply % 2 == 0
            out.append((board.fen(), move.uci(),
                        rec.result if white_to_move else -rec.result,
                        rec.white_elo if white_to_move else rec.black_elo,
                        ply))
        board.push(move)
    return out


def pack_games(games, per_game, seed):
    """Worker: game records -> (positions array, legal array) with local offsets.

    Offsets are relative to the returned legal array; the parent rebases them
    onto the shard file it is currently writing.
    """
    rng = np.random.default_rng(seed)
    sampled = []
    for rec in games:
        sampled.extend(positions_from_game(rec, per_game, rng))
    if not sampled:
        return np.empty(0, dtype=POS_DTYPE), np.empty(0, dtype=LEGAL_DTYPE)

    fens = [s[0] for s in sampled]
    planes = ce.encode_batch(fens).reshape(len(sampled), PLANE_BYTES)

    out = np.zeros(len(sampled), dtype=POS_DTYPE)
    legal_parts = []
    offset = 0
    for i, (fen, uci, z, elo, ply) in enumerate(sampled):
        legal = np.asarray(ce.legal_move_indices(fen), dtype=LEGAL_DTYPE)
        label = ce.move_to_index(fen, uci)
        where = np.flatnonzero(legal == label)
        if len(where) != 1:
            # The played move must be in the legal set the net will see; if it
            # is not, the encoders disagree with the movegen and everything
            # downstream is meaningless.
            raise AssertionError(f"played move {uci} not in the legal set of {fen}")
        out[i]["label"] = label
        out[i]["label_pos"] = where[0]
        out[i]["n_legal"] = len(legal)
        out[i]["legal_off"] = offset
        out[i]["z"] = z
        out[i]["elo"] = min(elo, 65535)
        out[i]["ply"] = min(ply, 65535)
        legal_parts.append(legal)
        offset += len(legal)
    out["planes"] = planes
    return out, np.concatenate(legal_parts)


class ShardWriter:
    """Paired .bin (fixed records) + .legal (ragged move indices) shards."""

    def __init__(self, out_dir, prefix="pos", shard_positions=200_000):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix = out_dir, prefix
        self.shard_positions = shard_positions
        self.index, self.count, self.legal_used = 0, 0, 0
        self.pos_fh = self.legal_fh = None

    def write(self, positions, legal):
        if self.pos_fh is None or self.count >= self.shard_positions:
            self._rotate()
        positions = positions.copy()
        positions["legal_off"] += self.legal_used
        self.pos_fh.write(positions.tobytes())
        self.legal_fh.write(legal.tobytes())
        self.count += len(positions)
        self.legal_used += len(legal)

    def _rotate(self):
        was_open = self.pos_fh is not None
        self.close()
        if was_open:
            self.index += 1
        base = os.path.join(self.out_dir, f"{self.prefix}.{self.index:03d}")
        self.pos_fh = open(base + ".bin", "wb")
        self.legal_fh = open(base + ".legal", "wb")
        self.count, self.legal_used = 0, 0

    def close(self):
        for fh in (self.pos_fh, self.legal_fh):
            if fh is not None:
                fh.close()
        self.pos_fh = self.legal_fh = None


def _worker(args):
    games, per_game, seed = args
    return pack_games(games, per_game, seed)


def chunks_of_games(games_path, chunk_games, max_games=None):
    chunk, seen = [], 0
    for rec in iter_records(games_path):
        chunk.append(rec)
        seen += 1
        if len(chunk) == chunk_games:
            yield chunk
            chunk = []
        if max_games and seen >= max_games:
            break
    if chunk:
        yield chunk


def build(games_path, out_dir, per_game=4, workers=None, seed=0, chunk_games=2000,
          shard_positions=200_000, max_positions=None, max_games=None, log=log_flushed):
    workers = workers or min(16, os.cpu_count() or 1)
    writer = ShardWriter(out_dir, shard_positions=shard_positions)
    stats = collections.Counter()
    t0 = time.time()

    tasks = ((chunk, per_game, seed + i)
             for i, chunk in enumerate(chunks_of_games(games_path, chunk_games, max_games)))
    try:
        # imap (ordered, small chunksize) keeps memory bounded and the output
        # deterministic given --seed: shard N holds the same positions on a
        # re-run, which matters because the val split is by shard.
        with mp.Pool(workers) as pool:
            for positions, legal in pool.imap(_worker, tasks, chunksize=1):
                writer.write(positions, legal)
                before = stats["positions"]
                stats["positions"] += len(positions)
                stats["draws"] += int((positions["z"] == 0).sum())
                stats["legal_total"] += int(positions["n_legal"].sum())
                if stats["positions"] // 1_000_000 != before // 1_000_000:
                    log(f"  {stats['positions']:,} positions, {time.time() - t0:.0f}s")
                if max_positions and stats["positions"] >= max_positions:
                    pool.terminate()
                    break
    finally:
        writer.close()

    n = max(stats["positions"], 1)
    log(f"packed {stats['positions']:,} positions into {writer.index + 1} shard(s) "
        f"in {time.time() - t0:.0f}s "
        f"({stats['positions'] / max(time.time() - t0, 1e-9):,.0f}/s)")
    log(f"  draws {stats['draws'] / n:.1%}, mean legal moves {stats['legal_total'] / n:.1f}, "
        f"{stats['positions'] * POS_DTYPE.itemsize / 1e9:.1f} GB")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", required=True, help="stage-1 shard dir or file (az.pgn)")
    ap.add_argument("--out", required=True, help="output directory for training shards")
    ap.add_argument("--per-game", type=int, default=4, help="max positions sampled per game")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-positions", type=int, default=200_000,
                    help="shards rotate at this many positions, rounded up to a chunk")
    ap.add_argument("--chunk-games", type=int, default=2000, help="games per worker task")
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    build(args.games, args.out, per_game=args.per_game, workers=args.workers,
          seed=args.seed, chunk_games=args.chunk_games, shard_positions=args.shard_positions,
          max_positions=args.max_positions, max_games=args.max_games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
