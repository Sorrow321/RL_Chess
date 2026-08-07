#!/usr/bin/env python
"""The plane encoder's oracle tests (docs/01-engine.md).

The C++ move round-trip walk (tests/encoding_test.cpp) is blind to plane bugs,
so the planes get two tests of their own here, with python-chess as the
independent oracle:

1. Reference equivalence: a pure-Python encoder written straight from the
   plane table in docs/01-engine.md, sharing no code with the C++ one, must
   match encode_batch byte-for-byte on every position of a depth-3 walk from
   the five perft suite positions, plus hand-picked FENs covering all 16
   castling-rights combinations, en passant on each file (both colors), and
   halfmove clocks {0, 49, 99, 100, 150}.

2. Color-mirror invariance: encode(P) == encode(mirror(P)) byte-for-byte with
   python-chess Board.mirror() as the oracle, and for every legal move m,
   move_to_index(m) == move_to_index(mirror(m)). This is the only test forcing
   "the mover's perspective" to mean the same flip in both encoders.

Run:  python tests/encoding_test.py [--depth 3]
"""

import argparse
import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce  # the built module at the repo root

SUITE = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1",
]

failures = 0


def fail(what):
    global failures
    failures += 1
    if failures <= 20:
        print(f"  FAIL: {what}")


# --- the ~30-line reference encoder, straight from the table -----------------

def encode_ref(fen):
    board = chess.Board(fen)
    stm = board.turn
    planes = np.zeros((19, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        if stm == chess.BLACK:
            sq = chess.square_mirror(sq)  # flip ranks, keep files
        base = 0 if piece.color == stm else 6
        planes[base + piece.piece_type - 1, sq // 8, sq % 8] = 1
    planes[12] = 1
    planes[13] = board.has_kingside_castling_rights(stm)
    planes[14] = board.has_queenside_castling_rights(stm)
    planes[15] = board.has_kingside_castling_rights(not stm)
    planes[16] = board.has_queenside_castling_rights(not stm)
    if board.has_legal_en_passant():
        planes[17, :, chess.square_file(board.ep_square)] = 1
    planes[18] = min(board.halfmove_clock, 100)
    return planes


# --- position generation ------------------------------------------------------

def fen_of(board):
    # "legal" en-passant semantics match the movegen library, which keeps the
    # ep square only when an en-passant capture is actually legal.
    return board.fen(en_passant="legal")


def collect_walk(root_fen, depth, out):
    board = chess.Board(root_fen)

    def rec(d):
        out.add(fen_of(board))
        if d == 0:
            return
        for move in board.legal_moves:
            board.push(move)
            rec(d - 1)
            board.pop()

    rec(depth)


def hand_picked():
    fens = []
    # All 16 castling-rights combinations, both sides to move.
    for stm in "wb":
        for i in range(16):
            rights = "".join(c for c, bit in zip("KQkq", (8, 4, 2, 1)) if i & bit) or "-"
            fens.append(f"r3k2r/8/8/8/8/8/8/R3K2R {stm} {rights} - 0 1")

    # A legal en-passant capture on each file, for both capturing colors.
    def row(files_to_chars):
        out, empty = "", 0
        for f in range(8):
            if f in files_to_chars:
                out += str(empty) if empty else ""
                out += files_to_chars[f]
                empty = 0
            else:
                empty += 1
        return out + (str(empty) if empty else "")

    for f in range(8):
        adj = f - 1 if f > 0 else f + 1
        sq = "abcdefgh"[f]
        fens.append(f"4k3/8/8/8/{row({f: 'P', adj: 'p'})}/8/8/4K3 b - {sq}3 0 1")
        fens.append(f"4k3/8/8/{row({f: 'p', adj: 'P'})}/8/8/8/4K3 w - {sq}6 0 1")

    # Halfmove clocks, including the >100 clamp.
    for clock in (0, 49, 99, 100, 150):
        fens.append(f"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - {clock} 1")
    return fens


# --- mirror helpers -----------------------------------------------------------

def mirror_fen(fen):
    board = chess.Board(fen)
    mirrored = board.mirror()
    mirrored.halfmove_clock = board.halfmove_clock  # mirror() must not touch it
    mirrored.fullmove_number = board.fullmove_number
    return fen_of(mirrored)


_RANK_FLIP = str.maketrans("12345678", "87654321")


def mirror_uci(uci):
    return uci.translate(_RANK_FLIP)


# --- the tests ------------------------------------------------------------

def check_chunk(fens):
    cpp = np.asarray(ce.encode_batch(fens))
    ref = np.stack([encode_ref(f) for f in fens])
    for i in np.nonzero((cpp != ref).any(axis=(1, 2, 3)))[0]:
        plane = int(np.nonzero((cpp[i] != ref[i]).any(axis=(1, 2)))[0][0])
        fail(f"planes differ from reference at plane {plane} for {fens[i]}")

    mirrored = [mirror_fen(f) for f in fens]
    cpp_m = np.asarray(ce.encode_batch(mirrored))
    for i in np.nonzero((cpp != cpp_m).any(axis=(1, 2, 3)))[0]:
        fail(f"encode(P) != encode(mirror(P)) for {fens[i]} vs {mirrored[i]}")

    for fen, mfen in zip(fens, mirrored):
        moves, idxs = ce.legal_moves(fen), ce.legal_move_indices(fen)
        if len(set(idxs)) != len(moves):
            fail(f"move index collision in {fen}")
        m_index = dict(zip(ce.legal_moves(mfen), ce.legal_move_indices(mfen)))
        for uci, idx in zip(moves, idxs):
            if m_index.get(mirror_uci(uci)) != idx:
                fail(f"move_to_index({uci}) not mirror-invariant in {fen}")


def convention_pins():
    castle = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    for fen, uci, want in [
        (ce.STARTPOS, "e2e4", 877),
        (ce.STARTPOS, "g1f3", 501),
        (castle, "e1g1", 307),
        (castle, "e1c1", 335),
        ("8/P6k/8/8/8/8/7K/8 w - - 0 1", "a7a8n", 3568),
    ]:
        got = ce.move_to_index(fen, uci)
        if got != want:
            fail(f"convention pin {uci}: got {got}, want {want}")
        if ce.index_to_move(fen, want) != uci:
            fail(f"index_to_move({want}) != {uci}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3, help="walk depth from each suite position")
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()

    assert ce.POLICY_SIZE == 4672 and ce.NUM_PLANES == 19
    convention_pins()

    game = ce.Game()
    if not np.array_equal(*np.asarray(ce.encode_batch([game, game.fen]))):
        fail("encode_batch(Game) != encode_batch(fen)")

    positions = set()
    for fen in SUITE:
        collect_walk(fen, args.depth, positions)
    positions.update(hand_picked())
    fens = sorted(positions)
    print(f"checking {len(fens)} positions (depth-{args.depth} walk + hand-picked)")

    for start in range(0, len(fens), args.chunk):
        check_chunk(fens[start:start + args.chunk])
        done = min(start + args.chunk, len(fens))
        if done % (args.chunk * 8) == 0 or done == len(fens):
            print(f"  {done}/{len(fens)}")

    if failures:
        print(f"ENCODING TEST FAILED: {failures} failure(s)")
        return 1
    print("encoding test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
