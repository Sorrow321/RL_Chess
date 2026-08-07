"""Opening book for the eval harness (docs/06-eval.md).

    python -m az.book                       # print the book as EPD
    python -m az.book --out books/openings.epd
    python -m az.book --out books/openings.pgn --format pgn

Forty two-move openings, each played from both sides so a pairing is a set of
colour-swapped pairs. The point docs/06 makes is narrow and worth restating:
*results must not be one opening's opinion*. Two engines that are deterministic
from the initial position play the same game a hundred times, and a hundred
copies of one game is a one-game match with a misleadingly tight confidence
interval on it.

Two moves deep is the shallow end of the doc's 2–4 range on purpose. It is
enough to separate the major structures — open, semi-open, closed, Indian,
flank — while leaving the agent the whole middlegame to be judged on. Deeper
books measure how well the book was chosen.

The lines are stored as UCI and validated against the engine, so a typo is a
loud error at import time rather than a silently skipped opening. `format=epd`
is what cutechess-cli's `-openings` wants; the in-process harness in az.eval
uses `lines()` directly and keeps the move list, which is why its PGNs show the
opening moves and cutechess's show a FEN header.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

# (name, four plies of UCI). Ordered by first move so a truncated book (--limit)
# is still varied rather than all-1.e4.
BOOK = [
    ("Open Game",                "e2e4 e7e5 g1f3 b8c6"),
    ("Petroff Defence",          "e2e4 e7e5 g1f3 g8f6"),
    ("Philidor Defence",         "e2e4 e7e5 g1f3 d7d6"),
    ("Vienna Game",              "e2e4 e7e5 b1c3 g8f6"),
    ("Bishop's Opening",         "e2e4 e7e5 f1c4 g8f6"),
    ("King's Gambit Accepted",   "e2e4 e7e5 f2f4 e5f4"),
    ("Centre Game",              "e2e4 e7e5 d2d4 e5d4"),
    ("Sicilian, Classical",      "e2e4 c7c5 g1f3 d7d6"),
    ("Sicilian, Old Sicilian",   "e2e4 c7c5 g1f3 b8c6"),
    ("Sicilian, Paulsen",        "e2e4 c7c5 g1f3 e7e6"),
    ("Sicilian, Closed",         "e2e4 c7c5 b1c3 b8c6"),
    ("Sicilian, Alapin",         "e2e4 c7c5 c2c3 d7d5"),
    ("French Defence",           "e2e4 e7e6 d2d4 d7d5"),
    ("French, Franco-Benoni",    "e2e4 e7e6 d2d4 c7c5"),
    ("Caro-Kann Defence",        "e2e4 c7c6 d2d4 d7d5"),
    ("Scandinavian Defence",     "e2e4 d7d5 e4d5 d8d5"),
    ("Pirc Defence",             "e2e4 d7d6 d2d4 g8f6"),
    ("Modern Defence",           "e2e4 g7g6 d2d4 f8g7"),
    ("Alekhine's Defence",       "e2e4 g8f6 e4e5 f6d5"),
    ("Nimzowitsch Defence",      "e2e4 b8c6 d2d4 d7d5"),
    ("Owen's Defence",           "e2e4 b7b6 d2d4 c8b7"),
    ("Queen's Gambit Declined",  "d2d4 d7d5 c2c4 e7e6"),
    ("Queen's Gambit Accepted",  "d2d4 d7d5 c2c4 d5c4"),
    ("Slav Defence",             "d2d4 d7d5 c2c4 c7c6"),
    ("Queen's Pawn Game",        "d2d4 d7d5 g1f3 g8f6"),
    ("Indian Defence, ...e6",    "d2d4 g8f6 c2c4 e7e6"),
    ("King's Indian / Gruenfeld", "d2d4 g8f6 c2c4 g7g6"),
    ("Benoni Defence",           "d2d4 g8f6 c2c4 c7c5"),
    ("Budapest Gambit",          "d2d4 g8f6 c2c4 e7e5"),
    ("Trompowsky Attack",        "d2d4 g8f6 c1g5 f6e4"),
    ("Queen's Pawn, Torre",      "d2d4 g8f6 g1f3 e7e6"),
    ("Dutch Defence",            "d2d4 f7f5 g2g3 g8f6"),
    ("Benoni, Old",              "d2d4 c7c5 d4d5 e7e6"),
    ("English Opening",          "c2c4 e7e5 b1c3 g8f6"),
    ("English, Symmetrical",     "c2c4 c7c5 g1f3 g8f6"),
    ("English, Anglo-Indian",    "c2c4 g8f6 b1c3 e7e6"),
    ("Reti Opening",             "g1f3 d7d5 c2c4 e7e6"),
    ("King's Indian Attack",     "g1f3 g8f6 g2g3 d7d5"),
    ("Bird's Opening",           "f2f4 d7d5 g1f3 g8f6"),
    ("Larsen's Opening",         "b2b3 e7e5 c1b2 b8c6"),
]

BOOK_PLIES = 4


def lines(limit=None, plies=BOOK_PLIES):
    """-> [(name, [uci, ...], fen_after)], validated against the engine.

    `plies` truncates each line; the engine replay is what proves the moves are
    legal, so a shortened book is checked as thoroughly as the full one.
    """
    out = []
    for name, moves in BOOK[:limit]:
        ucis = moves.split()[:plies]
        game = ce.Game()
        for uci in ucis:
            # Raises ValueError naming the move and the position it was illegal
            # in — which is the entire error message a book typo deserves.
            game.push(uci)
        if game.outcome().over:
            raise ValueError(f"book line {name!r} is already a finished game")
        out.append((name, ucis, game.fen))
    return out


def as_epd(entries):
    """cutechess `-openings format=epd`: one FEN per line, `id` naming the line."""
    return "".join(f'{fen} ; id "{name}";\n' for name, _moves, fen in entries)


def as_pgn(entries):
    """cutechess `-openings format=pgn`. Needs python-chess for SAN movetext."""
    import chess
    import chess.pgn

    chunks = []
    for name, ucis, _fen in entries:
        board = chess.Board()
        for uci in ucis:
            board.push(chess.Move.from_uci(uci))
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "az eval book"
        game.headers["Opening"] = name
        game.headers["Result"] = "*"
        chunks.append(str(game))
    return "\n\n".join(chunks) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None, help="file to write (default: stdout)")
    ap.add_argument("--format", default=None, choices=["epd", "pgn"],
                    help="default: inferred from --out's extension, else epd")
    ap.add_argument("--plies", type=int, default=BOOK_PLIES, help="truncate each line")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N openings")
    args = ap.parse_args(argv)

    fmt = args.format or ("pgn" if (args.out or "").endswith(".pgn") else "epd")
    entries = lines(limit=args.limit, plies=args.plies)
    text = as_pgn(entries) if fmt == "pgn" else as_epd(entries)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"{len(entries)} openings ({args.plies} plies) -> {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
