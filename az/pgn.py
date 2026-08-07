"""Stage 1 of the supervised bootstrap (docs/04): Lichess PGN -> game records.

    python -m az.pgn --input data/lichess_db_standard_rated_2025-07.pgn.zst \
                     --out data/games --workers 16

Streams a monthly dump from https://database.lichess.org straight out of the
zstd frame — the file is never fully decompressed — applies the docs/04
filters, and writes one compact record per surviving game:

    n_plies u16 | result i8 | white_elo u16 | black_elo u16 | termination u8
    moves   u16[n_plies]

~2 bytes per ply, so a whole month of filtered games is a few GB rather than
the ~100 GB the packed planes would take. Positions are *not* materialized
here: docs/04 keeps parsing (hours, once) separate from packing (minutes,
whenever the sampling rule changes), and az/pack.py is the second stage.

Moves are stored as `from | to << 6 | promotion << 12`, which is python-chess's
own numbering and reconstructs a `chess.Move` with no string parsing. The
policy index cannot be used here: it is position-dependent, so it would need a
board to be resolved, which is exactly what the replay in stage 2 provides.

Parallelism: the parent decompresses and splits the stream into game texts
(cheap string work), workers do the SAN parsing (the real cost, ~1 ms/game).
Every filter lives in the worker, so there is exactly one place where the
docs/04 rules are stated, at the cost of shipping rejected games over the
pipe too.
"""
import argparse
import collections
import functools
import io
import multiprocessing as mp
import os
import re
import struct
import sys
import threading
import time

import chess
import chess.pgn
import numpy as np

# --- record format -----------------------------------------------------------

GAME_HEADER = struct.Struct("<HbHHB")  # n_plies, result, white_elo, black_elo, termination

TERM_NORMAL = 0
TERM_TIME_FORFEIT = 1

GameRecord = collections.namedtuple(
    "GameRecord", "moves result white_elo black_elo termination")


def pack_move(move):
    """chess.Move -> u16: from | to << 6 | promotion << 12 (promotion 0 = none)."""
    return move.from_square | (move.to_square << 6) | ((move.promotion or 0) << 12)


def unpack_move(code):
    promo = code >> 12
    return chess.Move(code & 63, (code >> 6) & 63, promotion=promo or None)


def write_record(rec, out):
    out.write(GAME_HEADER.pack(len(rec.moves), rec.result, rec.white_elo,
                               rec.black_elo, rec.termination))
    out.write(np.asarray(rec.moves, dtype="<u2").tobytes())


def iter_records(path):
    """Yield GameRecords from a shard file or a directory of them."""
    for shard in shard_paths(path):
        with open(shard, "rb") as f:
            while True:
                head = f.read(GAME_HEADER.size)
                if not head:
                    break
                n, result, w_elo, b_elo, term = GAME_HEADER.unpack(head)
                moves = np.frombuffer(f.read(2 * n), dtype="<u2")
                if len(moves) != n:
                    raise EOFError(f"{shard}: truncated game record")
                yield GameRecord(moves, result, w_elo, b_elo, term)


def shard_paths(path):
    if os.path.isdir(path):
        shards = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
        if not shards:
            raise FileNotFoundError(f"no *.bin game shards in {path}")
        return [os.path.join(path, f) for f in shards]
    return [path]


# --- filters (docs/04) -------------------------------------------------------

class Filters:
    """The docs/04 filter set. Defaults are the doc's; every one is a knob."""

    def __init__(self, min_elo=1400, max_elo=2200, min_estimate_seconds=180,
                 min_plies=20, drop_plies_on_time=4, rated_only=True):
        self.min_elo = min_elo
        self.max_elo = max_elo
        # Lichess' own bullet/blitz boundary: base + 40*increment. "blitz 3+0"
        # in the doc is this number at 180, and it also lets 2+2 (=200s) in
        # while keeping 1+0 and 2+1 out.
        self.min_estimate_seconds = min_estimate_seconds
        self.min_plies = min_plies
        self.drop_plies_on_time = drop_plies_on_time
        self.rated_only = rated_only


RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
TIME_CONTROL = re.compile(r"^(\d+)\+(\d+)$")


def header_reject_reason(h, cfg):
    """None if the headers pass, else a short reason string (for the stats)."""
    if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
        return "variant"          # 960, from-position, atomic, ...
    if cfg.rated_only and "Rated" not in h.get("Event", ""):
        return "unrated"
    term = h.get("Termination", "Normal")
    if term not in ("Normal", "Time forfeit"):
        return "termination"      # Abandoned, Rules infraction (cheat-flagged)
    if h.get("Result") not in RESULTS:
        return "result"

    tc = TIME_CONTROL.match(h.get("TimeControl", "-"))
    if not tc:
        return "time_control"     # correspondence ("-") or malformed
    if int(tc.group(1)) + 40 * int(tc.group(2)) < cfg.min_estimate_seconds:
        return "time_control"     # bullet

    for tag in ("WhiteElo", "BlackElo"):
        try:
            elo = int(h.get(tag, "?"))
        except ValueError:
            return "elo"          # unrated/provisional "?"
        if not (cfg.min_elo <= elo <= cfg.max_elo):
            return "elo"
    return None


class _Collector(chess.pgn.BaseVisitor):
    """Header-first visitor: rejected games never get their movetext parsed.

    Returning SKIP from end_headers() puts python-chess on its fast path,
    which is the whole reason the filters are checked here and not after
    building a game object.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.headers = {}
        self.moves = []
        self.reason = None

    def begin_game(self):
        self.headers = {}
        self.moves = []
        self.reason = None

    def visit_header(self, name, value):
        self.headers[name] = value

    def end_headers(self):
        self.reason = header_reject_reason(self.headers, self.cfg)
        if self.reason:
            return chess.pgn.SKIP

    def visit_move(self, board, move):
        self.moves.append(pack_move(move))

    def handle_error(self, error):
        self.reason = "parse_error"

    def result(self):
        if self.reason:
            return None, self.reason
        moves = self.moves
        # docs/04: the last few plies of a game lost on time are panic moves,
        # not the human policy we want to imitate.
        if self.headers.get("Termination") == "Time forfeit" and self.cfg.drop_plies_on_time:
            moves = moves[:-self.cfg.drop_plies_on_time]
        if len(moves) < self.cfg.min_plies:
            return None, "short"
        term = TERM_TIME_FORFEIT if self.headers.get("Termination") == "Time forfeit" else TERM_NORMAL
        return GameRecord(np.asarray(moves, dtype="<u2"),
                          RESULTS[self.headers["Result"]],
                          int(self.headers["WhiteElo"]), int(self.headers["BlackElo"]),
                          term), None


# --- streaming ---------------------------------------------------------------

def open_pgn(path):
    """Text handle over a .pgn or .pgn.zst, decompressed on the fly."""
    if path.endswith(".zst"):
        import zstandard
        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_game_texts(handle):
    """Split a PGN stream into per-game texts without parsing any chess.

    A game ends where the next one's header block begins: a '[' line that
    follows movetext. Cheap enough to keep up with zstd on one core.
    """
    buf, seen_movetext = [], False
    for line in handle:
        if seen_movetext and line.startswith("["):
            yield "".join(buf)
            buf, seen_movetext = [], False
        if not seen_movetext and line.strip() and not line.startswith(("[", "%", ";")):
            seen_movetext = True
        buf.append(line)
    if seen_movetext:
        yield "".join(buf)


def chunked(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def throttled(iterable, limit):
    """Bound how far a Pool's feeder thread may run ahead of the workers.

    multiprocessing.Pool drains its input generator as fast as it can; on a
    30 GB stream that is a memory bomb. `release` is called by the consumer
    for each result it takes.
    """
    sem = threading.Semaphore(limit)
    def gen():
        for item in iterable:
            sem.acquire()
            yield item
    return gen(), sem.release


_CFG = None


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def parse_chunk(texts, cfg=None):
    """Worker: game texts -> (record bytes, reject-reason counts, kept)."""
    cfg = cfg or _CFG
    out = io.BytesIO()
    stats = collections.Counter()
    visitor = functools.partial(_Collector, cfg)
    for text in texts:
        parsed = chess.pgn.read_game(io.StringIO(text), Visitor=visitor)
        if parsed is None:
            stats["empty"] += 1
            continue
        rec, reason = parsed
        if rec is None:
            stats[reason] += 1
            continue
        stats["kept"] += 1
        write_record(rec, out)
    return out.getvalue(), stats


class ShardWriter:
    """Rotating output files, so no single shard grows past `max_bytes`."""

    def __init__(self, out_dir, prefix="games", max_bytes=1 << 30):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.max_bytes = out_dir, prefix, max_bytes
        self.index, self.written, self.fh = 0, 0, None

    def write(self, payload):
        if self.fh is None or self.written + len(payload) > self.max_bytes:
            self._rotate()
        self.fh.write(payload)
        self.written += len(payload)

    def _rotate(self):
        if self.fh is not None:
            self.fh.close()
            self.index += 1
        path = os.path.join(self.out_dir, f"{self.prefix}.{self.index:03d}.bin")
        self.fh = open(path, "wb")
        self.written = 0

    def close(self):
        if self.fh is not None:
            self.fh.close()


def log_flushed(*args):
    # These runs are an hour long behind a pipe or a nohup; unflushed progress
    # is progress nobody can see.
    print(*args, flush=True)


def build(inputs, out_dir, cfg, workers=None, max_games=None, chunk_games=400,
          shard_bytes=1 << 30, log=log_flushed):
    """Run stage 1 over one or more PGN files. Returns the reject/keep counts."""
    workers = workers or min(16, os.cpu_count() or 1)
    writer = ShardWriter(out_dir, max_bytes=shard_bytes)
    totals = collections.Counter()
    t0 = time.time()

    def texts():
        seen = 0
        for path in inputs:
            with open_pgn(path) as handle:
                for text in iter_game_texts(handle):
                    yield text
                    seen += 1
                    if max_games and seen >= max_games:
                        return

    source, release = throttled(chunked(texts(), chunk_games), workers * 4)
    try:
        with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for payload, stats in pool.imap_unordered(parse_chunk, source):
                release()
                writer.write(payload)
                before = totals["kept"]
                totals.update(stats)
                if totals["kept"] // 100_000 != before // 100_000:
                    seen = sum(totals.values())
                    log(f"  {totals['kept']:,} kept / {seen:,} read "
                        f"({totals['kept'] / max(seen, 1):.1%}), {time.time() - t0:.0f}s")
    finally:
        writer.close()
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", nargs="+", required=True, help="*.pgn or *.pgn.zst")
    ap.add_argument("--out", required=True, help="output directory for game shards")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--max-games", type=int, default=None, help="stop after reading this many")
    ap.add_argument("--min-elo", type=int, default=1400)
    ap.add_argument("--max-elo", type=int, default=2200)
    ap.add_argument("--min-seconds", type=int, default=180,
                    help="minimum base + 40*increment (docs/04: no bullet)")
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--shard-mb", type=int, default=1024)
    args = ap.parse_args()

    cfg = Filters(min_elo=args.min_elo, max_elo=args.max_elo,
                  min_estimate_seconds=args.min_seconds, min_plies=args.min_plies)
    totals = build(args.input, args.out, cfg, workers=args.workers,
                   max_games=args.max_games, shard_bytes=args.shard_mb << 20)

    seen = sum(totals.values())
    print(f"\nread {seen:,} games, kept {totals['kept']:,} ({totals['kept'] / max(seen, 1):.1%})")
    for reason, n in totals.most_common():
        if reason != "kept":
            print(f"  rejected {reason:14s} {n:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
