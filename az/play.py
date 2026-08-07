"""Play a game against a checkpoint from the terminal.

    python -m az.play --ckpt runs/bootstrap_2026-07/bootstrap.pt
    python -m az.play --ckpt ... --color black --fen "<position>"

This is the **policy only**, argmax over the legal moves, no search — the same
GreedyPlayer the docs/04 gates measure. MCTS needs a single-position search
entry point the Runner does not expose to Python yet (docs/02: self-play always
starts from the initial position), which arrives with the UCI wrapper in
docs/06. Expect a bot that opens sensibly and punishes loose pieces, then
misses anything that needs more than one move of foresight.

The engine module is the source of truth for legality and for when the game is
over; python-chess is here to draw the board and to let you type SAN.

Commands at the prompt: a move (SAN like `Nf3` or UCI like `g1f3`), `moves`,
`eval`, `fen`, `undo`, `quit`.
"""
import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

from az.net import Evaluator


def net_view(evaluator, game, board, top=3):
    """(value, [(san, prob), ...]) for the side to move, best first."""
    priors, value = evaluator(ce.encode_batch([game]))
    legal = np.asarray(ce.legal_move_indices(game.fen), dtype=np.int64)
    probs = priors[0][legal]
    probs = probs / max(probs.sum(), 1e-9)          # renormalize over legal moves
    order = np.argsort(-probs)[:top]
    moves = [(board.san(chess.Move.from_uci(ce.index_to_move(game.fen, int(legal[i])))),
              float(probs[i])) for i in order]
    return float(value[0]), moves


def show(board, human_white, value=None):
    print()
    print(board.unicode(borders=True, empty_square=".",
                        orientation=chess.WHITE if human_white else chess.BLACK))
    if value is not None:
        # Value is from the side to move; report it from the human's side so
        # the sign does not flip under you every ply.
        signed = value if board.turn == (chess.WHITE if human_white else chess.BLACK) else -value
        verdict = "you are better" if signed > 0.15 else \
                  "the bot is better" if signed < -0.15 else "roughly balanced"
        print(f"  net value for you: {signed:+.2f}  ({verdict})")


def parse_move(board, text):
    """SAN or UCI -> a legal chess.Move, or None."""
    for parse in (board.parse_san, board.parse_uci):
        try:
            move = parse(text)
            if move in board.legal_moves:
                return move
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--color", default="white", choices=["white", "black"],
                    help="the colour YOU play")
    ap.add_argument("--fen", default=ce.STARTPOS)
    ap.add_argument("--max-plies", type=int, default=ce.MAX_PLIES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    evaluator = Evaluator.from_checkpoint(args.ckpt, device=args.device)
    game = ce.Game(args.fen, args.max_plies)
    board = chess.Board(args.fen)
    human_white = args.color == "white"
    history = []

    print(f"{args.ckpt} on {args.device} — greedy policy, no search.")
    print("You are " + ("White" if human_white else "Black") +
          ". Type a move (Nf3 or g1f3), or: moves, eval, fen, undo, quit.")
    show(board, human_white)

    while True:
        outcome = game.outcome()
        if outcome.over:
            break

        if board.turn == (chess.WHITE if human_white else chess.BLACK):
            try:
                text = input("\nyour move> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in ("quit", "q", "exit"):
                return 0
            if text == "moves":
                print("  " + " ".join(board.san(m) for m in board.legal_moves))
                continue
            if text == "fen":
                print("  " + game.fen)
                continue
            if text == "eval":
                value, moves = net_view(evaluator, game, board, top=5)
                print("  the bot would consider: " +
                      ", ".join(f"{san} {p:.0%}" for san, p in moves))
                show(board, human_white, value)
                continue
            if text == "undo":
                if len(history) < 2:
                    print("  nothing to undo")
                    continue
                history = history[:-2]
                game = ce.Game(args.fen, args.max_plies)
                board = chess.Board(args.fen)
                for uci in history:
                    game.push(uci)
                    board.push(chess.Move.from_uci(uci))
                show(board, human_white)
                continue

            move = parse_move(board, text)
            if move is None:
                print(f"  '{text}' is not a legal move here — try `moves`")
                continue
        else:
            value, moves = net_view(evaluator, game, board)
            best = moves[0]
            move = board.parse_san(best[0])
            alternatives = ", ".join(f"{san} {p:.0%}" for san, p in moves[1:])
            print(f"\nbot plays {best[0]} ({best[1]:.0%} of its policy; "
                  f"also liked {alternatives})")

        uci = move.uci()
        game.push(uci)
        board.push(move)
        history.append(uci)
        if board.turn == (chess.WHITE if human_white else chess.BLACK):
            value, _ = net_view(evaluator, game, board)
            show(board, human_white, value)

    outcome = game.outcome()
    white_result = outcome.value if game.white_to_move else -outcome.value
    if white_result == 0:
        verdict = "Draw"
    elif (white_result == 1) == human_white:
        verdict = "You win"
    else:
        verdict = "The bot wins"
    show(board, human_white)
    print(f"\n{verdict} — {ce.Termination(int(outcome.reason)).name.lower()} "
          f"after {game.ply} plies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
