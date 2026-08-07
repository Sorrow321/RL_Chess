"""The self-play replay buffer (docs/05): FIFO over generations, sampled with
a decisive-game premium.

    buf = ReplayBuffer(capacity=1_500_000, decisive_premium=2.0)
    buf.add(examples_to_chunk(gen["examples"]))
    batch = buf.sample(2048, rng)      # planes / legal / valid / visits / z

**Chunked, not flat.** A generation arrives as one block and leaves as one
block, so the buffer is a deque of blocks and eviction is a pop, not a 1.8 GB
`concatenate` every generation. Sampling maps a global row index to a block
with `searchsorted` — the same shape as `az.data.PositionShards`, for the same
reason: the ragged legal-move lists live in flat side arrays and a block knows
its own offsets into them.

**Decisive premium.** docs/05: draws dominate as play strengthens and flatten
the value signal, the chess analog of 2048's late-game rarity problem, so a
position from a decisive game is drawn twice as often as one from a draw. With
only two weights there is no need for a weighted-sampling structure: split the
rows into two pools once per add, then draw how many come from each pool
(binomial on the weighted mass) and sample uniformly inside each. Set
`decisive_premium=1.0` to turn it off and get plain uniform sampling.

Memory: a position is 19x8x8 planes (1216 B) plus ~4 B per legal move, so the
doc's 1.5M-position buffer is ~2 GB of RAM. Planes are kept as uint8 all the
way to the GPU (`az.net.planes_to_tensor`).
"""
import numpy as np

CHUNK_KEYS = ("planes", "legal_off", "n_legal", "legal_idx", "visits", "z", "ply")


class ReplayBuffer:
    def __init__(self, capacity, decisive_premium=2.0):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if decisive_premium <= 0:
            raise ValueError("decisive_premium must be > 0")
        self.capacity = int(capacity)
        self.decisive_premium = float(decisive_premium)
        self.chunks = []
        self.generations_seen = 0
        self._reindex()

    # --- contents -------------------------------------------------------------

    def __len__(self):
        return int(self.starts[-1]) if len(self.starts) else 0

    def add(self, chunk):
        """Append one generation's positions, evicting the oldest over capacity."""
        missing = [k for k in CHUNK_KEYS if k not in chunk]
        if missing:
            raise KeyError(f"chunk is missing {missing}")
        self.generations_seen += 1
        if len(chunk["z"]):
            self.chunks.append(chunk)
        self._evict()
        self._reindex()
        return self

    def _evict(self):
        total = sum(len(c["z"]) for c in self.chunks)
        while total > self.capacity and self.chunks:
            oldest = len(self.chunks[0]["z"])
            excess = total - self.capacity
            if excess >= oldest:
                self.chunks.pop(0)
                total -= oldest
            else:
                # Partial eviction. The slices are copied rather than kept as
                # views: a view holds the whole parent array alive, which is
                # the one way a capped buffer still grows without bound.
                self.chunks[0] = trim_front(self.chunks[0], excess)
                total -= excess

    def _reindex(self):
        counts = np.array([len(c["z"]) for c in self.chunks], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(counts)])
        z = np.concatenate([c["z"] for c in self.chunks]) if self.chunks else np.empty(0, np.float32)
        self._decisive = np.flatnonzero(z != 0)
        self._drawn = np.flatnonzero(z == 0)

    @property
    def draw_frac(self):
        """Fraction of buffered positions that came from drawn games.

        docs/05 wants this in the log every generation: healthy weak play is
        ~5-15%, and above ~40% the value head is starving.
        """
        n = len(self)
        return len(self._drawn) / n if n else 0.0

    def stats(self):
        return {"size": len(self), "chunks": len(self.chunks), "capacity": self.capacity,
                "draw_frac": self.draw_frac, "fill": len(self) / self.capacity,
                "generations_seen": self.generations_seen}

    # --- sampling -------------------------------------------------------------

    def sample_indices(self, n, rng):
        """`n` global row indices, with the decisive premium applied."""
        if not len(self):
            raise ValueError("buffer is empty")
        n_dec, n_draw = len(self._decisive), len(self._drawn)
        if n_dec == 0 or n_draw == 0 or self.decisive_premium == 1.0:
            return rng.integers(0, len(self), size=n)
        mass = self.decisive_premium * n_dec
        k = int(rng.binomial(n, mass / (mass + n_draw)))
        return np.concatenate([self._decisive[rng.integers(0, n_dec, size=k)],
                               self._drawn[rng.integers(0, n_draw, size=n - k)]])

    def sample(self, n, rng):
        """A training batch. Sampling is with replacement, as in RL_2048.

        planes (B,19,8,8) u8 | legal (B,L) i64 | valid (B,L) bool
        visits (B,L) f32 (raw counts — the trainer normalizes) | z (B,) f32
        """
        return self.gather(np.sort(self.sample_indices(n, rng)))

    def gather(self, idx):
        """Global row indices -> a padded batch. Rows may span several chunks."""
        if not self.chunks:
            raise ValueError("buffer is empty")
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("row index out of range")
        chunk_of = np.searchsorted(self.starts, idx, side="right") - 1

        planes = np.empty((len(idx), *self.chunks[0]["planes"].shape[1:]), np.uint8)
        z = np.empty(len(idx), np.float32)
        n_legal = np.empty(len(idx), np.int64)

        groups = []
        for c in np.unique(chunk_of):
            rows = np.flatnonzero(chunk_of == c)
            local = idx[rows] - self.starts[c]
            chunk = self.chunks[c]
            planes[rows] = chunk["planes"][local]
            z[rows] = chunk["z"][local]
            n_legal[rows] = chunk["n_legal"][local]
            groups.append((int(c), rows, local))

        width = int(n_legal.max()) if len(idx) else 1
        span = np.arange(width, dtype=np.int64)
        valid = span[None, :] < n_legal[:, None]
        legal = np.zeros((len(idx), width), dtype=np.int64)
        visits = np.zeros((len(idx), width), dtype=np.float32)
        for c, rows, local in groups:
            chunk = self.chunks[c]
            # Pad slots read slot 0 of the flat arrays and are masked off by
            # `valid` — the same trick az.data uses on the bootstrap shards.
            flat = chunk["legal_off"][local][:, None] + span[None, :]
            take = np.where(valid[rows], flat, 0)
            legal[rows] = chunk["legal_idx"][take]
            visits[rows] = chunk["visits"][take]
        visits *= valid

        return {"planes": planes, "legal": legal, "valid": valid, "visits": visits, "z": z}


def trim_front(chunk, k):
    """Drop the first `k` rows of a chunk, flat side arrays included."""
    if k <= 0:
        return chunk
    lo = int(chunk["legal_off"][k]) if k < len(chunk["z"]) else len(chunk["legal_idx"])
    return {"planes": chunk["planes"][k:].copy(),
            "legal_off": (chunk["legal_off"][k:] - lo).copy(),
            "n_legal": chunk["n_legal"][k:].copy(),
            "legal_idx": chunk["legal_idx"][lo:].copy(),
            "visits": chunk["visits"][lo:].copy(),
            "z": chunk["z"][k:].copy(),
            "ply": chunk["ply"][k:].copy()}
