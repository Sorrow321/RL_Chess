"""The agent as a UCI engine (docs/06-eval.md).

    python -m az.uci --ckpt runs/bootstrap/bootstrap.pt --sims 400
    python -m az.uci --random                # the absolute-floor anchor

This is the wrapper that lets cutechess-cli — or any GUI — play the agent
without knowing anything about torch, planes or PUCT. It reads `position` and
`go` on stdin, runs `sims` simulations through `chess_engine.Searcher`, and
prints `bestmove`. Everything cutechess is good at (openings, adjudication,
PGN, SPRT) then comes for free, which is the whole reason docs/06 chose this
shape over writing a tournament manager.

Two brains behind one protocol loop:

* `SearchBrain` — the real agent: net + MCTS at a **fixed simulation count**,
  never a wall-clock budget. Node-limited play is what makes an Elo number
  portable across machines and across how busy this box happens to be, so
  clock tokens in `go` are read and ignored. `go nodes N` *is* honoured — it
  lets the harness set search amplification per match (lesson #4) from the
  command line instead of restarting the engine.
* `RandomBrain` — uniform random legal moves, the floor of the anchor ladder.
  It imports no torch and loads no checkpoint, so it also serves as the
  dependency-free way to test this protocol loop.

Determinism: with `dirichlet_eps=0` (the default, and what a match wants) the
search is a pure function of the network's outputs, so a game replays move for
move. The seed is reported in `info string` at startup and is what az.eval
bakes into the engine name so it lands in the PGN.

Not implemented on purpose: pondering, `stop` during a search (the search is
synchronous and short), multi-PV. Nothing in docs/06's protocol needs them.
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

AUTHOR = "chess_alphazero (docs/06)"

# The engine must never adjudicate a game itself: the GUI or the harness owns
# that, and an engine that refuses to move because of its own ply cap looks
# exactly like a crash. So the Game we track is given a cap it cannot reach.
NO_PLY_CAP = 1 << 20


def score_cp(q):
    """Root value in [-1, 1] -> centipawns, the standard logistic inversion.

    cutechess reads this for draw/resign adjudication, so it has to be a real
    score and not a placeholder. Clamped short of certainty: `q = 1` maps to
    infinity, and an engine reporting `inf` upsets tooling more than it informs.
    """
    w = min(max((q + 1.0) / 2.0, 1e-4), 1 - 1e-4)
    return int(round(-400.0 * math.log10(1.0 / w - 1.0)))


# --- brains ------------------------------------------------------------------

class RandomBrain:
    """Uniform random legal move. The floor anchor, and no torch in sight."""

    name = "az-random"
    options = {"Seed": ("spin", 0, 0, 2**31 - 1)}

    def __init__(self, seed=0):
        self.seed = seed
        self.rng = random.Random(seed)

    def setoption(self, name, value):
        if name.lower() == "seed":
            self.seed = int(value)
            self.rng = random.Random(self.seed)

    def config(self):
        return f"random mover, seed={self.seed}"

    def newgame(self):
        pass

    def search(self, game, nodes=None):
        """-> (uci, [info lines]). `nodes` is meaningless here; a move is a move."""
        return self.rng.choice(game.legal_moves()), ["info depth 1 nodes 1 score cp 0"]


class SearchBrain:
    """Net + MCTS at a fixed simulation count — the agent under evaluation.

    The evaluator and the searcher are built lazily, at the first `isready`,
    so that `uci` answers instantly and a bad checkpoint path fails with a
    message rather than during option negotiation.
    """

    name = "az-mcts"
    options = {
        "Sims": ("spin", 400, 2, 1 << 20),
        "Seed": ("spin", 0, 0, 2**31 - 1),
        "Checkpoint": ("string", ""),
        "Device": ("string", ""),
        "CPuct": ("string", "1.5"),
        "DirichletEps": ("string", "0.0"),
    }

    def __init__(self, ckpt, sims=400, device=None, c_puct=1.5, dirichlet_eps=0.0, seed=0):
        self.ckpt = ckpt
        self.sims = sims
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_eps = dirichlet_eps
        self.seed = seed
        self.ev = None
        self.searcher = None

    def setoption(self, name, value):
        key = name.lower()
        if key == "sims":
            self.sims, self.searcher = int(value), None
        elif key == "seed":
            self.seed, self.searcher = int(value), None
        elif key == "cpuct":
            self.c_puct, self.searcher = float(value), None
        elif key == "dirichleteps":
            self.dirichlet_eps, self.searcher = float(value), None
        elif key == "checkpoint" and value:
            self.ckpt, self.ev, self.searcher = value, None, None
        elif key == "device" and value:
            self.device, self.ev, self.searcher = value, None, None

    def config(self):
        dev = self.device or "auto"
        return (f"ckpt={self.ckpt} sims={self.sims} c_puct={self.c_puct} "
                f"dirichlet_eps={self.dirichlet_eps} device={dev} seed={self.seed}")

    def load(self, sims=None):
        if self.ev is None:
            import torch  # deferred: `uci`/`--random` must not pay for the import

            from az.net import Evaluator
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if not self.ckpt:
                raise SystemExit("az.uci: no checkpoint — pass --ckpt or setoption name Checkpoint value <path>")
            self.ev = Evaluator.from_checkpoint(self.ckpt, device=device)
            self.device = str(self.ev.device)
        sims = sims or self.sims
        if self.searcher is None or self.searcher.sims != sims:
            self.searcher = ce.Searcher(n_slots=1, sims=sims, c_puct=self.c_puct,
                                        dirichlet_eps=self.dirichlet_eps, seed=self.seed)
        return self

    def newgame(self):
        self.searcher = None  # a fresh tree and a fresh noise stream per game

    def search(self, game, nodes=None):
        # `go nodes N` overrides this move only; the Sims option stays the
        # standing budget, so one odd `go` cannot silently reconfigure a match.
        self.load(max(2, int(nodes)) if nodes is not None else None)

        t0 = time.time()
        self.searcher.set_position(0, game)
        while True:
            batch = self.searcher.pending()
            if not batch.size:
                break
            self.searcher.feed(*self.ev(batch))

        move = self.searcher.best_move(0)
        root = self.searcher.root(0)
        best = max(range(len(root["moves"])), key=lambda i: root["visits"][i])
        elapsed = max(time.time() - t0, 1e-6)
        sims = self.searcher.sims_done(0)
        info = [f"info depth 1 seldepth 1 nodes {sims} nps {int(sims / elapsed)} "
                f"time {int(elapsed * 1000)} score cp {score_cp(self.searcher.root_q(0))} "
                f"pv {move}",
                f"info string visits {int(root['visits'][best])}/{sims} "
                f"q {float(root['q'][best]):+.3f} legal {len(root['moves'])}"]
        return move, info


# --- protocol ----------------------------------------------------------------

def parse_position(tokens, max_plies=NO_PLY_CAP):
    """`position [startpos | fen <fen>] [moves <uci>...]` -> a replayed Game."""
    if not tokens:
        raise ValueError("position: nothing to set")
    if tokens[0] == "startpos":
        fen, rest = ce.STARTPOS, tokens[1:]
    elif tokens[0] == "fen":
        cut = tokens.index("moves") if "moves" in tokens else len(tokens)
        fen, rest = " ".join(tokens[1:cut]), tokens[cut:]
    else:
        raise ValueError(f"position: expected 'startpos' or 'fen', got {tokens[0]!r}")

    game = ce.Game(fen, max_plies)
    # Replaying rather than jumping to the final FEN is the point: the Game
    # accumulates the repetition window the search reads, so the tree scores a
    # third repetition as the draw it is.
    for uci in (rest[1:] if rest and rest[0] == "moves" else []):
        game.push(uci)
    return game


def parse_go(tokens):
    """Only `nodes` survives. Clocks are read and dropped — see the module docstring."""
    for i, tok in enumerate(tokens):
        if tok == "nodes" and i + 1 < len(tokens):
            return int(tokens[i + 1])
    return None


def option_lines(brain):
    out = []
    for name, spec in brain.options.items():
        if spec[0] == "spin":
            out.append(f"option name {name} type spin default {spec[1]} min {spec[2]} max {spec[3]}")
        else:
            out.append(f"option name {name} type string default {spec[1] or '<empty>'}")
    return out


def uci_loop(brain, stream=sys.stdin, out=sys.stdout):
    """The whole protocol. Returns the exit code; `quit` or EOF ends it."""
    def say(line):
        print(line, file=out, flush=True)

    game = ce.Game(ce.STARTPOS, NO_PLY_CAP)
    for raw in stream:
        parts = raw.split()
        if not parts:
            continue
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            say(f"id name {brain.name}")
            say(f"id author {AUTHOR}")
            say(f"info string {brain.config()}")
            for line in option_lines(brain):
                say(line)
            say("uciok")
        elif cmd == "isready":
            if hasattr(brain, "load"):
                brain.load()
            say("readyok")
        elif cmd == "setoption":
            # setoption name <words...> [value <words...>]
            if "name" in args:
                cut = args.index("value") if "value" in args else len(args)
                name = " ".join(args[args.index("name") + 1:cut])
                value = " ".join(args[cut + 1:]) if cut < len(args) else ""
                brain.setoption(name, value)
        elif cmd == "ucinewgame":
            brain.newgame()
            game = ce.Game(ce.STARTPOS, NO_PLY_CAP)
        elif cmd == "position":
            game = parse_position(args)
        elif cmd == "go":
            if game.outcome().over:
                # Nothing legal to play. `0000` is the UCI null move, which is
                # what a GUI expects here rather than silence.
                say("bestmove 0000")
                continue
            move, info = brain.search(game, parse_go(args))
            for line in info:
                say(line)
            say(f"bestmove {move}")
        elif cmd in ("stop", "ponderhit", "debug", "setvalue"):
            pass  # the search is synchronous; there is nothing to interrupt
        elif cmd == "quit":
            return 0
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=os.environ.get("AZ_CKPT"), help="checkpoint to play (az.net format)")
    ap.add_argument("--sims", type=int, default=400, help="simulations per move; `go nodes N` overrides")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dirichlet-eps", type=float, default=0.0,
                    help="root exploration noise; 0 (default) is match play")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--random", action="store_true", help="play uniform random moves (the floor anchor)")
    args = ap.parse_args(argv)

    if args.random:
        brain = RandomBrain(seed=args.seed)
    else:
        if not args.ckpt:
            ap.error("--ckpt is required (or use --random)")
        brain = SearchBrain(args.ckpt, sims=args.sims, device=args.device, c_puct=args.c_puct,
                            dirichlet_eps=args.dirichlet_eps, seed=args.seed)
    return uci_loop(brain)


if __name__ == "__main__":
    sys.exit(main())
