"""The three docs/04 sanity gates for a bootstrap checkpoint.

    python -m az.gates --ckpt runs/bootstrap/bootstrap.pt --games 100
    python -m az.gates --ckpt runs/bootstrap/bootstrap.pt --stockfish /usr/bin/stockfish

1. Greedy policy (argmax, no search) vs a random mover: must win ~100%.
2. Greedy vs Stockfish at its `UCI_Elo` floor (1320): should score points.
3. Five games written out as PGN to eyeball: sane openings, no piece hung
   every game, castling and promotions actually used.

This is a *gate*, not the Elo harness. docs/06 builds that one on cutechess
with an opening book, node-limited anchors and SPRT; what is here is the
in-process python-chess fallback that doc names, kept deliberately small so
the bootstrap can be judged before phase 0.5 exists. Numbers from here are
sanity signals, not Elo claims.

Play is greedy on purpose: it measures the *policy*, which is what the
bootstrap trained. Search comes later and only makes it stronger.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

from az.net import Evaluator

WHITE, BLACK = 0, 1


def elo_from_score(score, n):
    """Elo difference implied by a match score, with a binomial 95% band."""
    if n == 0:
        return 0.0, 0.0, 0.0
    def to_elo(s):
        s = min(max(s, 1e-6), 1 - 1e-6)
        return -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(max(score * (1 - score), 1e-9) / n)
    return to_elo(score), to_elo(score - 1.96 * se), to_elo(score + 1.96 * se)


class GreedyPlayer:
    """argmax over the legal slice of the policy — no search, no temperature."""

    def __init__(self, evaluator):
        self.ev = evaluator

    def moves(self, games):
        """One batched forward for every game whose turn it is."""
        if not games:
            return []
        priors, _ = self.ev(ce.encode_batch(games))
        out = []
        for game, row in zip(games, priors):
            legal = np.asarray(ce.legal_move_indices(game.fen), dtype=np.int64)
            out.append(ce.index_to_move(game.fen, int(legal[np.argmax(row[legal])])))
        return out


def random_moves(games, rng):
    return [rng.choice(g.legal_moves()) for g in games]


def play_match(player, opponent_moves, n_games, rng, max_plies=None, label=""):
    """`n_games` games, colors alternating, all games stepped in lockstep.

    Batching matters: the net side is one GPU forward per *ply of the fleet*,
    not per game, so 100 games cost about what one game would. The ply cap is
    the Game's own (docs/02: 250 plies -> draw), so eval and self-play agree on
    when a marathon is a draw.
    """
    games = [ce.Game(ce.STARTPOS, max_plies or ce.MAX_PLIES) for _ in range(n_games)]
    our_color = [i % 2 for i in range(n_games)]  # even ids: we are white
    results, terminations, plies = [None] * n_games, [None] * n_games, [0] * n_games
    move_log = [[] for _ in range(n_games)]

    live = list(range(n_games))
    while live:
        still = []
        for i in live:
            outcome = games[i].outcome()
            if outcome.over:
                # Outcome.value is from the side to move; store it for white.
                results[i] = outcome.value if games[i].white_to_move else -outcome.value
                terminations[i] = int(outcome.reason)
                plies[i] = games[i].ply
            else:
                still.append(i)
        live = still
        if not live:
            break

        ours = [i for i in live if (WHITE if games[i].white_to_move else BLACK) == our_color[i]]
        ours_set = set(ours)
        theirs = [i for i in live if i not in ours_set]
        for i, uci in zip(ours, player.moves([games[i] for i in ours])):
            games[i].push(uci)
            move_log[i].append(uci)
        for i, uci in zip(theirs, opponent_moves([games[i] for i in theirs], rng)):
            games[i].push(uci)
            move_log[i].append(uci)

    wins = sum(1 for i in range(n_games) if results[i] == (1 if our_color[i] == WHITE else -1))
    losses = sum(1 for i in range(n_games) if results[i] == (-1 if our_color[i] == WHITE else 1))
    draws = n_games - wins - losses
    score = (wins + 0.5 * draws) / n_games
    elo, lo, hi = elo_from_score(score, n_games)
    return {"label": label, "games": n_games, "wins": wins, "draws": draws, "losses": losses,
            "score": score, "elo": elo, "elo_lo": lo, "elo_hi": hi,
            "mean_plies": float(np.mean(plies)), "moves": move_log, "results": results,
            "our_color": our_color, "terminations": terminations}


def report(match):
    print(f"{match['label']}: +{match['wins']} ={match['draws']} -{match['losses']} "
          f"({match['score']:.1%}, Elo {match['elo']:+.0f} "
          f"[{match['elo_lo']:+.0f}, {match['elo_hi']:+.0f}] 95% CI), "
          f"{match['mean_plies']:.0f} plies avg")
    # How the games ended, because "why did we draw" is the whole diagnosis: a
    # weak policy draws won positions by 50-move and repetition, which reads as
    # a failed gate but is a conversion problem, not a legality one.
    counts = {}
    for term in match["terminations"]:
        name = ce.Termination(term).name
        counts[name] = counts.get(name, 0) + 1
    print("  terminations: " + ", ".join(f"{k.lower()} {v}" for k, v in
                                         sorted(counts.items(), key=lambda kv: -kv[1])))


# --- gate 2: Stockfish -------------------------------------------------------

def stockfish_opponent(path, elo=1320, nodes=None, skill=None):
    """A move function backed by one Stockfish process (docs/06's fallback path).

    Node-limited rather than time-limited so the result does not depend on how
    busy this box is — the same reason docs/06 freezes node counts.
    """
    import chess
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(path)
    options = {}
    if skill is not None:
        options["Skill Level"] = skill
    else:
        options.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
    engine.configure(options)
    limit = chess.engine.Limit(nodes=nodes) if nodes else chess.engine.Limit(depth=8)

    def moves(games, _rng):
        out = []
        for game in games:
            board = chess.Board(game.fen)
            out.append(engine.play(board, limit).move.uci())
        return out

    moves.close = engine.close
    return moves


# --- gate 3: PGN -------------------------------------------------------------

def write_pgn(match, path, white_name, black_name, event="bootstrap gate"):
    import chess
    import chess.pgn

    with open(path, "w") as fh:
        for i in range(min(5, match["games"])):
            board = chess.Board()
            for uci in match["moves"][i]:
                board.push(chess.Move.from_uci(uci))
            game = chess.pgn.Game.from_board(board)
            we_are_white = match["our_color"][i] == WHITE
            game.headers["Event"] = event
            game.headers["White"] = white_name if we_are_white else black_name
            game.headers["Black"] = black_name if we_are_white else white_name
            # from_board() reads the result off the board, which says "*" for a
            # game the ply cap ended; the match knows better.
            game.headers["Result"] = {1: "1-0", 0: "1/2-1/2", -1: "0-1"}[match["results"][i]]
            game.headers["Termination"] = ce.Termination(match["terminations"][i]).name
            print(game, file=fh, end="\n\n")
    print(f"gate 3: wrote {min(5, match['games'])} games to {path} — eyeball them "
          "(openings sane? pieces hung? castling and promotions used?)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="bootstrap checkpoint")
    ap.add_argument("--games", type=int, default=100, help="games per gate")
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH"),
                    help="path to a stockfish binary (gate 2; skipped if absent)")
    ap.add_argument("--sf-elo", type=int, default=1320, help="UCI_Elo (1320 is its floor)")
    ap.add_argument("--sf-nodes", type=int, default=None, help="node limit per move")
    ap.add_argument("--out", default=None, help="dir for the PGN (default: next to the ckpt)")
    ap.add_argument("--max-plies", type=int, default=ce.MAX_PLIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    player = GreedyPlayer(Evaluator.from_checkpoint(args.ckpt, device=args.device))
    print(f"{args.ckpt} on {args.device}, greedy policy (no search), {args.games} games/gate\n")

    t0 = time.time()
    vs_random = play_match(player, lambda games, r: random_moves(games, r), args.games,
                           rng, args.max_plies, label="gate 1: vs random mover")
    report(vs_random)
    passed = vs_random["score"] >= 0.99
    print(f"  {'PASS' if passed else 'FAIL'} — docs/04 wants ~100%  ({time.time() - t0:.0f}s)\n")

    if args.stockfish:
        opponent = stockfish_opponent(args.stockfish, elo=args.sf_elo, nodes=args.sf_nodes)
        try:
            vs_sf = play_match(player, opponent, args.games, rng, args.max_plies,
                               label=f"gate 2: vs Stockfish UCI_Elo={args.sf_elo}")
        finally:
            opponent.close()
        report(vs_sf)
        print(f"  {'PASS' if vs_sf['score'] > 0.0 else 'FAIL'} — docs/04 wants points on the "
              "board, not a win\n")
    else:
        print("gate 2: SKIPPED — pass --stockfish /path/to/stockfish (or set $STOCKFISH)\n")

    write_pgn(vs_random, os.path.join(out_dir, "gate_games.pgn"), "bootstrap-greedy", "random")
    return 0


if __name__ == "__main__":
    sys.exit(main())
