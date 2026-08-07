"""Reading the bootstrap training shards written by az.pack (docs/04).

Shards are memory-mapped, never loaded: at 1240 B/position the doc's corpus is
far larger than RAM, and the OS page cache is better at this than we are.
A batch is a fancy-index into the memmap plus a gather of the ragged legal
move lists, which is why `legal_off` is stored per record — no cumsum over
100M rows at startup.

The train/val split is **by shard**, not by position. Positions sampled from
one game are adjacent in a shard, so a random position-level split would put
sibling positions on both sides and inflate held-out move agreement — the one
number docs/04 uses to decide whether the bootstrap worked.
"""
import os
import queue
import threading

import numpy as np

from az.pack import LEGAL_DTYPE, POS_DTYPE


class PositionShards:
    """A concatenated, memory-mapped view over a set of az.pack shards."""

    def __init__(self, paths):
        self.paths = list(paths)
        if not self.paths:
            raise FileNotFoundError("no shards")
        self.pos = [np.memmap(p, dtype=POS_DTYPE, mode="r") for p in self.paths]
        self.legal = [np.memmap(p[:-4] + ".legal", dtype=LEGAL_DTYPE, mode="r")
                      for p in self.paths]
        counts = np.array([len(p) for p in self.pos], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(counts)])

    @staticmethod
    def shard_files(path):
        if os.path.isdir(path):
            names = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
            return [os.path.join(path, f) for f in names]
        return [path]

    @classmethod
    def split(cls, path, val_shards=1):
        """(train, val) over a shard directory; val is the last `val_shards`."""
        files = cls.shard_files(path)
        if not val_shards:
            return cls(files), None
        if len(files) <= val_shards:
            raise ValueError(f"{len(files)} shard(s) is too few to hold out {val_shards}")
        return cls(files[:-val_shards]), cls(files[-val_shards:])

    def __len__(self):
        return int(self.starts[-1])

    def batch(self, idx):
        """Global record indices -> dict of numpy arrays, legal lists padded.

        planes (B,19,8,8) u8 | legal (B,L) i64 | valid (B,L) bool
        label_pos (B,) i64   | z (B,) f32      | label/elo/ply (B,) i64
        """
        idx = np.asarray(idx, dtype=np.int64)
        shard_of = np.searchsorted(self.starts, idx, side="right") - 1

        recs = np.empty(len(idx), dtype=POS_DTYPE)
        groups = []
        for s in np.unique(shard_of):
            rows = np.flatnonzero(shard_of == s)
            local = idx[rows] - self.starts[s]
            recs[rows] = self.pos[s][local]
            groups.append((int(s), rows))

        width = int(recs["n_legal"].max()) if len(recs) else 1
        span = np.arange(width, dtype=np.int64)
        valid = span[None, :] < recs["n_legal"][:, None].astype(np.int64)
        legal = np.zeros((len(idx), width), dtype=np.int64)
        for s, rows in groups:
            flat = recs["legal_off"][rows][:, None].astype(np.int64) + span[None, :]
            # Pad slots read slot 0 and are masked off by `valid`.
            legal[rows] = self.legal[s][np.where(valid[rows], flat, 0)]

        return {
            "planes": recs["planes"].reshape(-1, 19, 8, 8),
            "legal": legal,
            "valid": valid,
            "label_pos": recs["label_pos"].astype(np.int64),
            "label": recs["label"].astype(np.int64),
            "z": recs["z"].astype(np.float32),
            "elo": recs["elo"].astype(np.int64),
            "ply": recs["ply"].astype(np.int64),
        }


def iter_batches(shards, batch_size, rng=None, shuffle=True, drop_last=True, prefetch=3,
                 block=128, mix=16):
    """Batches of `shards`, gathered on a background thread.

    Sampling is by **contiguous block, then shuffled in a buffer**, not by
    independent random record, and the difference is the whole ballgame at
    corpus scale. Measured on the 99-shard/122 GB bootstrap corpus, batch 4096:

        random records, kernel readahead on      9,846 pos/s   50.9 KB/record
        random records, MADV_RANDOM             17,166 pos/s    2.8 KB/record
        blocks of 128, kernel readahead on     372,370 pos/s    1.4 KB/record

    A 1240-byte record read at random faults a whole 128 KB readahead window,
    so the loader moved 40x more bytes than it used and starved the GPU (0%
    utilization, ~3 h/epoch). Turning readahead off with MADV_RANDOM fixes the
    amplification but leaves one thread serializing on page-fault latency.
    Contiguous blocks fix both and *want* the readahead: 2.8x the ~131k pos/s
    the GPU can consume, so training is GPU-bound again.

    The cost of blocks is correlation — 4096 records as 32 blocks samples only
    32 regions of the corpus. So `mix` batches' worth are gathered at once and
    shuffled together before being handed out, which puts ~512 regions in
    every batch while keeping each read sequential. `block=1` degenerates to
    fully random sampling, which is what small corpora (that fit in cache
    anyway) get automatically.
    """
    n = len(shards)
    block = max(1, min(block, n // max(batch_size, 1)))
    n_blocks = n // block
    order = rng.permutation(n_blocks) if shuffle else np.arange(n_blocks)
    blocks_per_super = max(1, (mix * batch_size) // block)
    span = np.arange(block, dtype=np.int64)

    q = queue.Queue(maxsize=prefetch)
    def produce():
        try:
            for i in range(0, len(order), blocks_per_super):
                starts = order[i:i + blocks_per_super].astype(np.int64) * block
                idx = (starts[:, None] + span[None, :]).ravel()
                # Gather sorted (sequential reads), then shuffle the rows so a
                # batch is not a handful of neighbourhoods.
                big = shards.batch(np.sort(idx))
                perm = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
                stop = len(idx) - batch_size + 1 if drop_last else len(idx)
                for j in range(0, stop, batch_size):
                    rows = perm[j:j + batch_size]
                    q.put({k: v[rows] for k, v in big.items()})
        except Exception as exc:  # surface it on the consumer side
            q.put(exc)
            return
        q.put(None)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def shard_stats(shards, sample=100_000, block=128, rng=None):
    """Cheap corpus summary for the run log: draw rate, elo, plies, legal moves.

    Blocks here too — 100k independent random records cost gigabytes of
    readahead and made this "cheap" summary the slowest part of startup.
    """
    rng = rng or np.random.default_rng(0)
    n = len(shards)
    block = max(1, min(block, n // 8))
    starts = rng.integers(0, max(n - block, 1), size=max(1, min(sample, n) // block))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
    b = shards.batch(np.sort(idx[idx < n]))
    return {
        "positions": n,
        "draw_frac": float((b["z"] == 0).mean()),
        "white_frac": float((b["ply"] % 2 == 0).mean()),
        "elo_mean": float(b["elo"].mean()),
        "ply_mean": float(b["ply"].mean()),
        "legal_mean": float(b["valid"].sum(1).mean()),
    }
