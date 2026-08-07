#!/usr/bin/env python
"""Tests for the supervised bootstrap pipeline (docs/04).

The pipeline is four stages that each hand a compact binary format to the next
one, and every stage is a place where a silent off-by-one turns into a net that
trains happily on garbage. So each stage is checked against an *independent*
reconstruction rather than against itself:

1. az.pgn: a synthetic corpus with one game per rejection reason proves each
   docs/04 filter fires for the right reason, that time-forfeit games lose
   exactly their last 4 plies, and that the packed u16 move list replays to
   the same final position python-chess reached.
2. az.pack: every packed record is re-derived from scratch — replay the game
   to `ply`, then ask the C++ encoder directly for the planes, the label and
   the legal set. This is the test that would catch planes taken *after* the
   played move, or a z with the wrong sign for black.
3. az.data: the padded ragged batch must equal the raw records it came from,
   and the train/val split must not put two positions from one game on
   opposite sides.
4. az.bootstrap / az.gates: the real training step and the real match loop,
   run small — a tiny net must be able to overfit a few hundred positions,
   and a match's bookkeeping must agree with a replay of its own PGN.

Run:  python tests/bootstrap_test.py
"""

import io
import os
import sys
import tempfile

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

from az import pgn as azpgn
from az import pack as azpack
from az.bootstrap import losses, to_device
from az.data import PositionShards, iter_batches
from az.gates import GreedyPlayer, elo_from_score, play_match, random_moves, write_pgn
from az.net import Evaluator, PolicyValueNet

failures = 0


def fail(what):
    global failures
    failures += 1
    if failures <= 20:
        print(f"  FAIL: {what}")


def check(ok, what):
    if not ok:
        fail(what)


# --- synthetic corpus --------------------------------------------------------

def random_game(rng, plies):
    """A random-move game of exactly `plies` plies, biased to castle and promote.

    Random play almost never castles or promotes, and those are exactly the
    moves whose encoding is special-cased (docs/01), so a corpus without them
    would test the easy half of the encoder. Games that end early are retried
    so the filter arithmetic below is exact rather than approximately right.
    """
    for _ in range(50):
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            special = [m for m in moves if board.is_castling(m) or m.promotion]
            move = special[rng.integers(len(special))] if special and rng.random() < 0.9 \
                else moves[rng.integers(len(moves))]
            board.push(move)
        if len(board.move_stack) == plies:
            return board
    raise RuntimeError(f"could not generate a {plies}-ply random game")


GOOD = {"Event": "Rated Blitz game", "TimeControl": "300+0", "WhiteElo": "1500",
        "BlackElo": "1600", "Termination": "Normal", "Result": "1-0"}


def corpus(rng):
    """(pgn text, expected reject reasons) — one game per docs/04 filter."""
    cases = [
        (dict(GOOD), 60, "kept"),
        (dict(GOOD, Result="0-1"), 44, "kept"),
        (dict(GOOD, Result="1/2-1/2", TimeControl="600+5"), 30, "kept"),
        (dict(GOOD, Termination="Time forfeit"), 40, "kept"),      # truncated by 4
        (dict(GOOD, TimeControl="60+0"), 40, "time_control"),      # bullet
        (dict(GOOD, TimeControl="120+1"), 40, "time_control"),     # 120+40 = 160 < 180
        (dict(GOOD, TimeControl="-"), 40, "time_control"),         # correspondence
        (dict(GOOD, WhiteElo="1200"), 40, "elo"),
        (dict(GOOD, BlackElo="2400"), 40, "elo"),
        (dict(GOOD, WhiteElo="?"), 40, "elo"),
        (dict(GOOD, Event="Casual Blitz game"), 40, "unrated"),
        (dict(GOOD, Variant="Crazyhouse"), 40, "variant"),
        (dict(GOOD, Termination="Abandoned"), 40, "termination"),
        (dict(GOOD, Result="*"), 40, "result"),
        (dict(GOOD), 18, "short"),                                 # under 20 plies
        (dict(GOOD, Termination="Time forfeit"), 22, "short"),     # 22 - 4 = 18
    ]
    out, meta = io.StringIO(), []
    for headers, plies, verdict in cases:
        board = random_game(rng, plies)
        game = chess.pgn.Game.from_board(board)
        for key, value in headers.items():
            game.headers[key] = value
        print(game, file=out, end="\n\n")
        meta.append((headers, board, verdict))
    return out.getvalue(), meta


def write_corpus(text, path):
    if path.endswith(".zst"):
        import zstandard
        with open(path, "wb") as fh:
            fh.write(zstandard.ZstdCompressor().compress(text.encode()))
    else:
        with open(path, "w") as fh:
            fh.write(text)


# --- stage 1 -----------------------------------------------------------------

def expected_record(headers, board):
    """The record a kept game must produce: moves, minus the time-forfeit tail."""
    moves = list(board.move_stack)
    if headers.get("Termination") == "Time forfeit":
        moves = moves[:-azpgn.Filters().drop_plies_on_time]
    replay = chess.Board()
    for move in moves:
        replay.push(move)
    return len(moves), replay.fen(), azpgn.RESULTS[headers["Result"]]


def test_filters(tmp, text, meta):
    for suffix in (".pgn", ".pgn.zst"):
        src = os.path.join(tmp, "corpus" + suffix)
        out = os.path.join(tmp, "games" + suffix.replace(".", "_"))
        write_corpus(text, src)
        totals = azpgn.build([src], out, azpgn.Filters(), workers=2, log=lambda *a: None)

        want = {}
        for _, _, verdict in meta:
            want[verdict] = want.get(verdict, 0) + 1
        check(dict(totals) == want, f"{suffix}: filter counts {dict(totals)} != {want}")

        records = list(azpgn.iter_records(out))
        check(len(records) == want["kept"], f"{suffix}: {len(records)} records written")

        # Every kept game must come back with its exact move list — replayed
        # through the u16 packing to the same position python-chess reached —
        # and time-forfeit games must have lost exactly their last 4 plies.
        expected = sorted(expected_record(h, b) for h, b, v in meta if v == "kept")
        got = []
        for rec in records:
            board = chess.Board()
            for code in rec.moves:
                board.push(azpgn.unpack_move(int(code)))
            got.append((len(rec.moves), board.fen(), rec.result))
        check(sorted(got) == expected, f"{suffix}: kept records != the games that should survive")

        forfeits = [r for r in records if r.termination == azpgn.TERM_TIME_FORFEIT]
        check(len(forfeits) == 1, f"{suffix}: one time-forfeit game survived (got {len(forfeits)})")
        elos = {(r.white_elo, r.black_elo) for r in records}
        check(elos == {(1500, 1600)}, f"{suffix}: ratings survive the round trip ({elos})")


def test_move_packing():
    board = chess.Board("4k3/P7/8/8/8/8/8/R3K2R w KQ - 0 1")
    for move in board.legal_moves:
        code = azpgn.pack_move(move)
        check(0 <= code < 65536, f"{move.uci()} packs into u16")
        check(azpgn.unpack_move(code) == move, f"{move.uci()} survives the u16 round trip")
    promos = [m for m in board.legal_moves if m.promotion]
    castles = [m for m in board.legal_moves if board.is_castling(m)]
    check(len(promos) == 4 and len(castles) == 2, "the packing case covers promotions and castling")


# --- stage 2 -----------------------------------------------------------------

def test_pack(games_dir):
    """Re-derive every packed record from the game records, independently."""
    games = list(azpgn.iter_records(games_dir))
    per_game, seed = 4, 3
    positions, legal_flat = azpack.pack_games(games, per_game, seed)

    # Same sampling the packer used, so the *contents* can be checked; the
    # sampling rule itself is checked on the side.
    rng = np.random.default_rng(seed)
    expected = []
    for rec in games:
        expected.extend([(rec, *p) for p in azpack.positions_from_game(rec, per_game, rng)])
    check(len(positions) == len(expected), f"{len(positions)} records for {len(expected)} samples")

    plies_per_game = {}
    for i, (rec, fen, uci, z, elo, ply) in enumerate(expected):
        row = positions[i]
        plies_per_game.setdefault(id(rec), []).append(ply)

        # Planes: the position *before* the played move, from the C++ encoder.
        want_planes = ce.encode_batch([fen]).reshape(-1)
        check(np.array_equal(row["planes"], want_planes), f"record {i}: planes match the encoder")

        # An independent replay of the same game reaches the same position.
        replay = ce.Game()
        for code in rec.moves[:ply]:
            replay.push(azpgn.unpack_move(int(code)).uci())
        check(replay.fen == fen, f"record {i}: ply {ply} replays to the packed position")

        # Label and legal set.
        want_legal = ce.legal_move_indices(fen)
        got_legal = legal_flat[row["legal_off"]:row["legal_off"] + row["n_legal"]]
        check(list(got_legal) == list(want_legal), f"record {i}: legal set matches the movegen")
        check(row["label"] == ce.move_to_index(fen, uci), f"record {i}: label is the played move")
        check(got_legal[row["label_pos"]] == row["label"], f"record {i}: label_pos points at the label")
        check(ce.index_to_move(fen, int(row["label"])) == uci, f"record {i}: label decodes to {uci}")

        # z is the result from THIS mover's perspective (docs/02 convention).
        white_to_move = chess.Board(fen).turn == chess.WHITE
        check(row["z"] == (rec.result if white_to_move else -rec.result),
              f"record {i}: z sign for {'white' if white_to_move else 'black'} to move")
        check(row["elo"] == (rec.white_elo if white_to_move else rec.black_elo),
              f"record {i}: elo is the mover's")
        check(row["ply"] == ply, f"record {i}: ply recorded")

    for plies in plies_per_game.values():
        check(len(plies) <= per_game, f"at most {per_game} positions per game (got {len(plies)})")
        check(len(set(plies)) == len(plies), "sampled plies within a game are distinct")

    again, _ = azpack.pack_games(games, per_game, seed)
    check(np.array_equal(again["label"], positions["label"]), "packing is deterministic given a seed")
    other, _ = azpack.pack_games(games, per_game, seed + 1)
    check(not np.array_equal(other["ply"], positions["ply"]), "a different seed samples different plies")


# --- stage 3 -----------------------------------------------------------------

def test_shards(tmp, games_dir):
    shard_dir = os.path.join(tmp, "shards")
    azpack.build(games_dir, shard_dir, per_game=4, workers=2, seed=5,
                 chunk_games=1, shard_positions=8, log=lambda *a: None)
    files = PositionShards.shard_files(shard_dir)
    check(len(files) > 1, f"the corpus produced {len(files)} shards (want several)")

    shards = PositionShards(files)
    raw = [np.memmap(f, dtype=azpack.POS_DTYPE, mode="r") for f in files]
    check(len(shards) == sum(len(r) for r in raw), "concatenated length")

    # A batch spanning every shard must equal the raw records it indexes.
    idx = np.arange(len(shards))
    batch = shards.batch(idx)
    flat = np.concatenate(raw)
    check(np.array_equal(batch["label"], flat["label"].astype(np.int64)), "batch labels")
    check(np.array_equal(batch["planes"].reshape(len(flat), -1), flat["planes"]), "batch planes")
    check(np.array_equal(batch["z"], flat["z"].astype(np.float32)), "batch z")
    check(batch["valid"].sum() == flat["n_legal"].sum(), "valid mask counts the legal moves")

    legal_mm = {i: np.memmap(f[:-4] + ".legal", dtype=azpack.LEGAL_DTYPE, mode="r")
                for i, f in enumerate(files)}
    ok = True
    row = 0
    for s, arr in enumerate(raw):
        for r in arr:
            n = int(r["n_legal"])
            want = legal_mm[s][r["legal_off"]:r["legal_off"] + n]
            if not np.array_equal(batch["legal"][row][:n], want.astype(np.int64)):
                ok = False
            if not batch["valid"][row][:n].all() or batch["valid"][row][n:].any():
                ok = False
            row += 1
    check(ok, "padded legal rows equal the ragged records, pads masked off")

    for i in range(len(batch["label"])):
        n = int(batch["valid"][i].sum())
        check(batch["label"][i] in batch["legal"][i][:n], f"row {i}: the label is legal")

    train, val = PositionShards.split(shard_dir, val_shards=1)
    check(len(train) + len(val) == len(shards), "split covers the corpus")
    check(len(val.paths) == 1 and train.paths[-1] != val.paths[0], "val is the held-out shard")

    seen = sum(len(b["label"]) for b in iter_batches(train, 4, rng=np.random.default_rng(0)))
    check(seen == (len(train) // 4) * 4, f"iter_batches yields whole batches ({seen})")

    # Block sampling reads contiguous runs and shuffles them in a buffer (the
    # loader rewrite that took the corpus from 9.8k to 372k positions/s). It
    # must still hand out a permutation of the corpus: every position exactly
    # once per epoch, none twice.
    def keys(batches):
        out = []
        for b in batches:
            out.extend(zip(b["label"].tolist(), b["ply"].tolist(), b["z"].tolist()))
        return sorted(out)

    raw_train = np.concatenate([np.memmap(f, dtype=azpack.POS_DTYPE, mode="r") for f in train.paths])
    want = sorted(zip(raw_train["label"].astype(np.int64).tolist(),
                      raw_train["ply"].astype(np.int64).tolist(),
                      raw_train["z"].astype(np.float32).tolist()))
    check(len(train) % 4 == 0, f"test corpus ({len(train)}) divides into whole batches")
    for block in (1, 2, 4):
        got = keys(iter_batches(train, 4, rng=np.random.default_rng(1), block=block, mix=2))
        check(got == want, f"block={block}: an epoch covers every position exactly once")
    ordered = keys(iter_batches(train, 4, rng=np.random.default_rng(1), shuffle=False, block=2))
    check(ordered == want, "shuffle=False still covers the corpus")
    return shard_dir


# --- stage 4 -----------------------------------------------------------------

def test_training(shard_dir):
    """The real training step: a tiny net must overfit a few hundred positions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards = PositionShards(PositionShards.shard_files(shard_dir))
    n = min(len(shards), 256)
    batch = shards.batch(np.arange(n))
    x, legal, valid, label_pos, z = to_device(batch, device)

    torch.manual_seed(0)
    net = PolicyValueNet(channels=32, blocks=2).to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    first = last = None
    for step in range(120):
        loss, loss_p, loss_v, logits, _ = losses(net, x, legal, valid, label_pos, z, 1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss_p.detach())
        last = float(loss_p.detach())

    with torch.no_grad():
        logits, v = net(x)
        gathered = logits.float().gather(1, legal).masked_fill(~valid, float("-inf"))
        agree = float((gathered.argmax(1) == label_pos).float().mean())
    check(last < first * 0.5, f"policy loss falls while overfitting ({first:.3f} -> {last:.3f})")
    check(agree > 0.5, f"a tiny net overfits {n} positions to >50% agreement (got {agree:.1%})")
    check(bool((v.abs() <= 1).all()), "value stays in [-1, 1] through training")
    check(logits.shape[1] == ce.POLICY_SIZE, "the loss trains the whole 4672-wide head")


def test_gates(tmp):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    player = GreedyPlayer(Evaluator(PolicyValueNet(channels=16, blocks=1), device=device))
    rng = np.random.default_rng(0)

    n = 4
    match = play_match(player, lambda games, r: random_moves(games, r), n, rng,
                       max_plies=60, label="smoke")
    check(match["wins"] + match["draws"] + match["losses"] == n, "every game is scored once")
    check(0.0 <= match["score"] <= 1.0, "score in [0, 1]")
    check(all(t is not None for t in match["terminations"]), "every game has a termination")

    # Replaying a game's own move log must reproduce its recorded result.
    for i in range(n):
        board = chess.Board()
        game = ce.Game(ce.STARTPOS, 60)
        for uci in match["moves"][i]:
            board.push(chess.Move.from_uci(uci))  # raises if the log is not legal
            game.push(uci)
        outcome = game.outcome()
        check(outcome.over, f"game {i}: the match stopped at a real termination")
        white_result = outcome.value if game.white_to_move else -outcome.value
        check(match["results"][i] == white_result,
              f"game {i}: recorded result {match['results'][i]} != replay {white_result}")

    path = os.path.join(tmp, "gate_games.pgn")
    write_pgn(match, path, "greedy", "random")
    with open(path) as fh:
        read_back = [g for g in iter(lambda: chess.pgn.read_game(fh), None)]
    check(len(read_back) == min(5, n), f"{len(read_back)} games written as PGN")

    elo, lo, hi = elo_from_score(0.5, 100)
    check(abs(elo) < 1e-6 and lo < 0 < hi, f"an even score is 0 Elo with a band ({lo:.0f}, {hi:.0f})")
    elo, _, _ = elo_from_score(0.75, 100)
    check(abs(elo - 191) < 2, f"75% is ~+191 Elo (got {elo:.0f})")


def main():
    rng = np.random.default_rng(7)
    text, meta = corpus(rng)

    with tempfile.TemporaryDirectory() as tmp:
        test_move_packing()
        test_filters(tmp, text, meta)
        games_dir = os.path.join(tmp, "games_pgn")
        test_pack(games_dir)
        shard_dir = test_shards(tmp, games_dir)
        test_training(shard_dir)
        test_gates(tmp)

    if failures:
        print(f"\n{failures} FAILURE(S)")
        return 1
    print("\nbootstrap ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
