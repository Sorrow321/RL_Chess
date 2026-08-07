"""Self-play training loop (docs/05) — the phase-3 driver.

    python -m az.train --run runs/hunt1 --init runs/bootstrap_2026-07/bootstrap.pt \
        --anchor sf1320,sf1700 --stockfish /usr/bin/stockfish

    python -m az.train --run runs/hunt2 --init runs/hunt1/latest.pt --sims 640   # plateau step 1
    python -m az.train --run runs/smoke --generations 1 --games-per-gen 8 \
        --sims 32 --anchor "" --device cpu                                       # smoke test

One generation is: self-play with the current net (root Dirichlet noise on) →
replay buffer → gradient steps on sampled batches → checkpoint → Elo against
two anchors → one row of `<run>/log.csv`. Render the accumulated curve with
`python -m az.dashboard <run>`.

Three things here are not obvious and are deliberate:

* **Steps scale with buffer fill.** docs/05 asks for 600 steps x batch 2048
  because that is "≈ 1 buffer epoch per generation" at a full 1.5M buffer. A
  fresh buffer holds one generation, so a flat 600 steps would be ~40 epochs
  over it and the first generations would overfit their own noise — right when
  the net is the bootstrap's and worth the most. So the step count is
  `steps_per_gen x fill`, floored at `--min-steps`, which *is* the doc's
  intent. `--no-scale-steps` restores the flat count.
* **The evaluator is one fp16 copy of the net**, shared by self-play and eval
  (docs/03) and refreshed from the fp32 trainer after each generation's steps.
* **`--target-elo` uses the lower end of the 95% band**, not the point
  estimate: milestone 1 is "beats ~1000 Elo with statistical significance"
  (docs/00), and a point estimate crossing a line is not that.

Run management (docs/05, lesson #6): restarts go into a *fresh* run dir from
the previous `latest.pt`, and every start refuses to run beside another
az.train unless told otherwise — an orphaned trainer once silently halved
throughput for 40 minutes.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chess_engine as ce

from az import eval as azeval
from az.buffer import ReplayBuffer
from az.net import Evaluator, PolicyValueNet, load_checkpoint, planes_to_tensor, policy_loss, save_checkpoint
from az.selfplay import chunk_stats, examples_to_chunk, play_generation, result_stats

# docs/05's metrics, in its order, plus the diagnostics the doc asks to watch
# but did not name columns for. Appended to forever (lesson #7).
CSV_COLUMNS = ["gen", "games", "avg_plies", "draw_frac", "resign_frac", "white_score",
               "policy_loss", "value_loss", "buffer_size",
               "anchor_a", "elo_vs_anchor_a", "elo_a_lo", "elo_a_hi",
               "anchor_b", "elo_vs_anchor_b", "elo_b_lo", "elo_b_hi",
               "sp_seconds", "train_seconds", "eval_seconds",
               "buffer_draw_frac", "resign_fp_frac", "visit_entropy", "policy_entropy",
               "value_sign", "steps", "lr", "sims", "positions", "avg_batch",
               "games_per_h", "evals_per_s", "timestamp"]


# --- training ----------------------------------------------------------------

def train_steps(net, opt, buffer, n_steps, batch_size, device, rng, lambda_value=1.0,
                amp_dtype=None, scaler=None):
    """Gradient steps on buffer batches. -> mean losses and diagnostics.

    The policy target is the root visit distribution, normalized over the legal
    moves and masked to them (docs/03) — the same `az.net.policy_loss` the
    bootstrap uses with a one-hot target, which is just the degenerate case.
    """
    net.train()
    p_sum = v_sum = ent_sum = sign_sum = 0.0
    for _ in range(n_steps):
        batch = buffer.sample(batch_size, rng)
        x = planes_to_tensor(batch["planes"], device, torch.float32)
        legal = torch.from_numpy(batch["legal"]).to(device, non_blocking=True)
        valid = torch.from_numpy(batch["valid"]).to(device, non_blocking=True)
        visits = torch.from_numpy(batch["visits"]).to(device, non_blocking=True)
        z = torch.from_numpy(batch["z"]).to(device, non_blocking=True)
        target = visits / visits.sum(1, keepdim=True).clamp_min(1.0)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits, v = net(x)
        logits, v = logits.float(), v.float()
        loss_p = policy_loss(logits, legal, target, valid)
        loss_v = F.mse_loss(v, z)

        opt.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_p + lambda_value * loss_v).backward()
            scaler.step(opt)
            scaler.update()
        else:
            (loss_p + lambda_value * loss_v).backward()
            opt.step()

        with torch.no_grad():
            p_sum += float(loss_p)
            v_sum += float(loss_v)
            # Entropy of the *target*: how sharp the search's opinion is. It
            # bounds the policy loss from below, so a policy loss that stops
            # falling while this falls too is the net keeping up, not stalling.
            ent_sum += float(-(target * torch.log(target.clamp_min(1e-12))).sum(1).mean())
            # Value head sign agreement on decisive positions only — draws have
            # no sign and would drag it to a meaningless 50%.
            decisive = z != 0
            if bool(decisive.any()):
                sign_sum += float((torch.sign(v[decisive]) == z[decisive]).float().mean())
    n = max(n_steps, 1)
    return {"loss_p": p_sum / n, "loss_v": v_sum / n, "policy_entropy": ent_sum / n,
            "value_sign": sign_sum / n, "steps": n_steps}


def steps_for(buffer, args):
    """How many gradient steps this generation gets (see the module docstring)."""
    if not args.scale_steps:
        return args.steps_per_gen
    scaled = round(args.steps_per_gen * len(buffer) / buffer.capacity)
    return int(min(args.steps_per_gen, max(args.min_steps, scaled)))


# --- evaluation --------------------------------------------------------------

def make_anchor(spec, args, seed):
    if spec["kind"] == "random":
        return azeval.RandomPlayer(seed=seed)
    if spec["kind"] == "stockfish":
        return azeval.StockfishPlayer(args.stockfish, elo=spec.get("elo"), skill=spec.get("skill"),
                                      nodes=args.anchor_nodes, threads=args.sf_threads, label=spec["name"])
    raise SystemExit(f"{spec['name']} is not usable as an anchor here")


def run_eval(evaluator, anchors, args, openings, gen, seed):
    """Play the anchor matches. -> one dict per anchor, in `anchors` order.

    Reuses az.eval end to end — the same match loop, the same Elo arithmetic,
    the same eval.csv and PGNs — so a generation's Elo and a standalone
    `python -m az.eval` are the same measurement, not two implementations of it.
    """
    out = []
    csv_path = os.path.join(args.run, "eval.csv")
    for i, spec in enumerate(anchors):
        player = azeval.SearchPlayer(evaluator, args.eval_sims, args.eval_concurrency,
                                     c_puct=args.c_puct, noise=0.0, seed=seed,
                                     label=f"az-gen{gen:03d}-{args.eval_sims}s")
        opponent = make_anchor(spec, args, seed + 1 + i)
        try:
            match = azeval.play_match(player, opponent, args.eval_games, openings,
                                      max_plies=args.max_plies, concurrency=args.eval_concurrency)
        finally:
            player.close()
            opponent.close()
        match["net_evals"] = player.evals

        pgn = os.path.join(args.run, f"gen_{gen:03d}_vs_{spec['name']}.pgn")
        if args.eval_pgn:
            azeval.write_pgn(match, pgn, event=f"az selfplay {os.path.basename(args.run)}",
                             tags={"AzSeed": seed, "AzSims": args.eval_sims, "AzGen": gen,
                                   "AnchorNodes": args.anchor_nodes})
        azeval.append_csv(csv_path, match, {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": f"gen{gen:03d}",
            "sims": args.eval_sims, "anchor_nodes": args.anchor_nodes, "seed": seed,
            "ckpt": os.path.join(args.run, f"gen_{gen:03d}.pt"),
            "pgn": os.path.basename(pgn) if args.eval_pgn else ""})

        out.append({"name": spec["name"], "nominal": spec.get("elo"), "match": match,
                    "elo": match["elo"], "elo_lo": match["elo_lo"], "elo_hi": match["elo_hi"],
                    "score": match["score"]})
    return out


def absolute_elo(row):
    """(estimate, lower bound) on an absolute scale, or None for an unrated anchor.

    Only `sf<N>` anchors carry a nominal number. It is Stockfish's own claim at
    *its* time management, not at our frozen node count (docs/06 caveat 1), so
    this is a label for the curve, never a milestone claim on its own.
    """
    if row["nominal"] is None:
        return None
    return row["nominal"] + row["elo"], row["nominal"] + row["elo_lo"]


# --- run management ----------------------------------------------------------

def is_trainer(argv):
    """Does this argv belong to a python process running az.train?

    Matching the whole command line for "az.train" is what an eyeball does and
    it is wrong: the shell, `timeout`, `nohup` and this very process's launcher
    all carry the string too, and a check that cries wolf on its own wrapper is
    a check that gets passed --allow-concurrent forever. So: argv[0] must be a
    python, and the module or script must be an argument in its own right.
    """
    if len(argv) < 2 or not os.path.basename(argv[0]).startswith("python"):
        return False
    return any(a == "az.train" or a.replace("\\", "/").endswith("az/train.py") for a in argv[1:])


def other_trainers():
    """(pid, cmdline) of every other az.train on this box (lesson #6)."""
    if not os.path.isdir("/proc"):
        return []
    me, out = os.getpid(), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = fh.read().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if is_trainer([a for a in argv if a]):
            out.append((int(entry), " ".join(a for a in argv if a)))
    return out


def check_orphans(allow):
    """Refuse to start beside another trainer, and say exactly how to fix it."""
    others = other_trainers()
    if not others:
        return
    print("another az.train is already running:", file=sys.stderr)
    for pid, cmd in others:
        print(f"  pid {pid}  {cmd}", file=sys.stderr)
    if allow:
        print("  --allow-concurrent given; continuing anyway\n", file=sys.stderr)
        return
    pids = " ".join(str(p) for p, _ in others)
    raise SystemExit(
        "\nAn orphaned trainer silently halves throughput and neither run notices (lesson #6).\n"
        f"Kill it by exact pid and verify:\n  kill {pids} && sleep 2 && pgrep -af az.train\n"
        "Pass --allow-concurrent if two trainers on this box is deliberate.")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, cwd=os.path.dirname(os.path.abspath(__file__))
                              ).stdout.strip() or None
    except Exception:
        return None


def first_generation(log_path):
    """Where to resume numbering. A restart in place must not overwrite gen_000.pt."""
    if not os.path.exists(log_path):
        return 0
    last = -1
    with open(log_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                last = max(last, int(float(row["gen"])))
            except (KeyError, TypeError, ValueError):
                continue
    return last + 1


def write_config(args, anchors):
    """Lineage bookkeeping: what this run is and what it came from.

    A list, appended to on every start into this dir, because "which sims did
    generation 24 run at" is exactly the question a plateau decision asks and
    exactly the one a config.json overwritten by the restart cannot answer.
    """
    path = os.path.join(args.run, "config.json")
    history = []
    if os.path.exists(path):
        try:
            with open(path) as fh:
                loaded = json.load(fh)
            history = loaded if isinstance(loaded, list) else [loaded]
        except (OSError, ValueError):
            history = []
    history.append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
                    "commit": git_commit(), "anchors": [a["name"] for a in anchors],
                    "args": dict(vars(args))})
    with open(path, "w") as fh:
        json.dump(history, fh, indent=2)


# --- entry point -------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir: checkpoints, log.csv, eval.csv, PGNs")
    ap.add_argument("--init", default=None,
                    help="checkpoint to start from (az.bootstrap's, or a previous run's latest.pt)")
    ap.add_argument("--init-opt", default=None,
                    help="optimizer state to continue (a run's opt.pt); default is a fresh Adam")
    ap.add_argument("--generations", type=int, default=100)

    # self-play (docs/05 hyperparameters v1)
    ap.add_argument("--games-per-gen", type=int, default=256)
    ap.add_argument("--n-parallel", type=int, default=256, help="game slots in flight")
    ap.add_argument("--sims", type=int, default=400, help="simulations per move; raise first at plateaus")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--temp-plies", type=int, default=15, help="plies of tau=1 visit sampling, then argmax")
    ap.add_argument("--resign-threshold", type=float, default=-0.95,
                    help="root value for 8 consecutive own moves resigns; -1 disables (docs/02)")
    ap.add_argument("--max-plies", type=int, default=ce.MAX_PLIES)

    # buffer and optimization
    ap.add_argument("--buffer", type=int, default=1_500_000, help="replay buffer capacity in positions")
    ap.add_argument("--decisive-premium", type=float, default=2.0,
                    help="sampling weight of positions from decisive games (1 = uniform)")
    ap.add_argument("--steps-per-gen", type=int, default=600)
    ap.add_argument("--min-steps", type=int, default=50, help="floor while the buffer is filling")
    ap.add_argument("--no-scale-steps", dest="scale_steps", action="store_false",
                    help="take the flat --steps-per-gen instead of scaling it by buffer fill")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--lambda-value", type=float, default=1.0, help="weight on the value MSE")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--channels", type=int, default=128, help="net width when starting from scratch")
    ap.add_argument("--blocks", type=int, default=6)

    # evaluation (docs/06)
    ap.add_argument("--anchor", default="sf1320,sf1700",
                    help="comma-separated anchors, docs/05 wants two adjacent ones; empty disables eval")
    ap.add_argument("--eval-games", type=int, default=100, help="games per anchor per evaluation")
    ap.add_argument("--eval-every", type=int, default=1, help="generations between evaluations; 0 disables")
    ap.add_argument("--eval-sims", type=int, default=0, help="sims per move at eval (0 = --sims)")
    ap.add_argument("--eval-concurrency", type=int, default=32)
    ap.add_argument("--eval-pgn", action="store_true", help="write a PGN per evaluated pairing")
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH"), help="path to a stockfish binary")
    ap.add_argument("--anchor-nodes", type=int, default=azeval.ANCHOR_NODES,
                    help="frozen node budget per anchor move (docs/06 caveat 1)")
    ap.add_argument("--sf-threads", type=int, default=1)
    ap.add_argument("--book", default=None, help="EPD opening book for eval (default: az.book's lines)")
    ap.add_argument("--target-elo", type=float, default=0.0,
                    help="stop once the 95%% lower bound against a rated anchor reaches this absolute Elo")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda, and it is an error if unavailable")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="start even though another az.train is running (see lesson #6)")
    return ap


def resolve_device(args):
    # Same refusal as az.bootstrap: a night of self-play that quietly ran 100x
    # slower on the CPU is a night gone, and the only warning was one line at
    # startup nobody read.
    if args.device:
        return torch.device(args.device)
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available — refusing to self-play on CPU by accident.\n"
            "Check `nvidia-smi` and retry; pass --device cpu to do it on purpose.")
    return torch.device("cuda")


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.eval_sims = args.eval_sims or args.sims
    check_orphans(args.allow_concurrent)

    anchors = [azeval.parse_anchor(a) for a in args.anchor.split(",") if a.strip()]
    if any(a["kind"] == "az" for a in anchors):
        raise SystemExit("--anchor takes opponents (random, skill<N>, sf<elo>), not 'az'")
    if not args.eval_every or not args.eval_games:
        anchors = []
    # Fail now, not in an hour: a night of training whose Elo column is empty
    # is a night with no way to tell whether it worked (lesson #5).
    if any(a["kind"] == "stockfish" for a in anchors) and not args.stockfish:
        raise SystemExit("Stockfish anchors need a binary: pass --stockfish /path/to/stockfish, "
                         "set $STOCKFISH, or pass --anchor '' to train blind")
    if len(anchors) == 1:
        print("only one anchor — docs/05 wants two adjacent ones, a single anchor saturates", flush=True)
    if len(anchors) > 2:
        print(f"{len(anchors)} anchors: log.csv carries the first two, eval.csv carries them all", flush=True)
    if not anchors:
        print("eval disabled — the Elo columns will be empty, and the Elo curve is the "
              "point of the exercise (docs/05)", flush=True)
    if args.target_elo and not any(a.get("elo") for a in anchors):
        raise SystemExit("--target-elo needs an anchor with a nominal rating (sf<elo>); "
                         "skill/random anchors have no absolute scale")

    device = resolve_device(args)
    os.makedirs(args.run, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    write_config(args, anchors)
    with open(os.path.join(args.run, "train.pid"), "w") as fh:
        fh.write(f"{os.getpid()}\n")

    if args.init:
        net = load_checkpoint(args.init, device)
        print(f"initialized from {args.init}", flush=True)
    else:
        net = PolicyValueNet(args.channels, args.blocks).to(device)
        print("random initialization — docs/00 phase 1 wants a bootstrap checkpoint first; "
              "self-play from noise will be very slow to move", flush=True)
    net = net.to(memory_format=torch.channels_last).train()

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.init_opt:
        opt.load_state_dict(torch.load(args.init_opt, map_location=device, weights_only=True))
        print(f"optimizer state continued from {args.init_opt}", flush=True)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    scaler = torch.amp.GradScaler(enabled=(args.amp == "fp16"))

    evaluator = Evaluator(net, device=device)
    buffer = ReplayBuffer(args.buffer, decisive_premium=args.decisive_premium)
    openings = azeval.load_book(args.book) if anchors else []

    log_path = os.path.join(args.run, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.DictWriter(log_f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    if new_log:
        log.writeheader()
    start_gen = first_generation(log_path)
    if start_gen:
        print(f"{log_path} already has {start_gen} generations; continuing the numbering "
              f"at gen {start_gen} (docs/05 prefers a fresh run dir per restart)", flush=True)

    print(f"{args.run} on {device}: {args.generations} generations x {args.games_per_gen} games "
          f"@ {args.sims} sims, buffer {args.buffer:,}, "
          f"{'eval ' + ','.join(a['name'] for a in anchors) if anchors else 'no eval'}\n", flush=True)

    t_run = time.time()
    for gen in range(start_gen, start_gen + args.generations):
        try:
            row = run_generation(gen, args, net, opt, evaluator, buffer, openings, anchors,
                                 device, rng, amp_dtype, scaler)
        except KeyboardInterrupt:
            save_checkpoint(net, os.path.join(args.run, "latest.pt"), gen=gen, interrupted=1)
            print(f"\ninterrupted during generation {gen}; weights saved to "
                  f"{os.path.join(args.run, 'latest.pt')}", flush=True)
            break
        log.writerow(row)
        log_f.flush()

        if args.target_elo and reached_target(row, args, gen):
            break
    else:
        print(f"\n{args.generations} generations done in {(time.time() - t_run) / 3600:.1f} h", flush=True)

    log_f.close()
    print(f"curve: python -m az.dashboard {args.run}", flush=True)
    return 0


def run_generation(gen, args, net, opt, evaluator, buffer, openings, anchors, device, rng,
                   amp_dtype, scaler):
    """One generation. -> the log.csv row."""
    # --- self-play ---
    sp = play_generation(evaluator, args.games_per_gen, args.sims, n_parallel=args.n_parallel,
                         c_puct=args.c_puct, dirichlet_alpha=args.dirichlet_alpha,
                         dirichlet_eps=args.dirichlet_eps, temp_plies=args.temp_plies,
                         resign_threshold=args.resign_threshold,
                         seed=int(rng.integers(1, 2 ** 63)), progress=progress_printer(gen))
    stats = result_stats(sp["results"])
    chunk = examples_to_chunk(sp["examples"])
    cstats = chunk_stats(chunk)
    buffer.add(chunk)

    # --- train ---
    t0 = time.time()
    n_steps = steps_for(buffer, args)
    tr = train_steps(net, opt, buffer, n_steps, args.batch, device, rng,
                     lambda_value=args.lambda_value, amp_dtype=amp_dtype, scaler=scaler)
    train_seconds = time.time() - t0
    evaluator.refresh(net)

    save_checkpoint(net, os.path.join(args.run, f"gen_{gen:03d}.pt"), gen=gen, sims=args.sims)
    save_checkpoint(net, os.path.join(args.run, "latest.pt"), gen=gen, sims=args.sims)
    torch.save(opt.state_dict(), os.path.join(args.run, "opt.pt"))

    # --- eval ---
    t0 = time.time()
    rows = (run_eval(evaluator, anchors, args, openings, gen, seed=args.seed + gen)
            if anchors and args.eval_every and gen % args.eval_every == 0 else [])
    eval_seconds = time.time() - t0

    row = {"gen": gen, "games": stats["games"], "avg_plies": f"{stats['avg_plies']:.1f}",
           "draw_frac": f"{stats['draw_frac']:.4f}", "resign_frac": f"{stats['resign_frac']:.4f}",
           "white_score": f"{stats['white_score']:.4f}",
           "policy_loss": f"{tr['loss_p']:.4f}", "value_loss": f"{tr['loss_v']:.5f}",
           "buffer_size": len(buffer), "buffer_draw_frac": f"{buffer.draw_frac:.4f}",
           # empty, not nan, when no audit game triggered a resignation: the
           # column means "not measured this generation", and nan is not a rate
           "resign_fp_frac": (f"{stats['resign_fp_frac']:.4f}"
                              if stats["resign_fp_frac"] == stats["resign_fp_frac"] else ""),
           "visit_entropy": f"{cstats['visit_entropy']:.4f}",
           "policy_entropy": f"{tr['policy_entropy']:.4f}", "value_sign": f"{tr['value_sign']:.4f}",
           "steps": tr["steps"], "lr": f"{args.lr:.2e}", "sims": args.sims,
           "positions": cstats["positions"], "avg_batch": f"{sp['avg_batch']:.0f}",
           "games_per_h": f"{stats['games'] / max(sp['seconds'], 1e-9) * 3600:.0f}",
           "evals_per_s": f"{sp['evals_per_s']:.0f}",
           "sp_seconds": f"{sp['seconds']:.0f}", "train_seconds": f"{train_seconds:.0f}",
           "eval_seconds": f"{eval_seconds:.0f}", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for slot, r in zip(("a", "b"), rows):
        row[f"anchor_{slot}"] = r["name"]
        row[f"elo_vs_anchor_{slot}"] = f"{r['elo']:.1f}"
        row[f"elo_{slot}_lo"] = f"{r['elo_lo']:.1f}"
        row[f"elo_{slot}_hi"] = f"{r['elo_hi']:.1f}"

    print(f"gen {gen}: selfplay {stats['games']} games {stats['avg_plies']:.0f} plies "
          f"draw {stats['draw_frac']:.0%} resign {stats['resign_frac']:.0%} "
          f"white {stats['white_score']:.3f} ({sp['seconds']:.0f}s, "
          f"{stats['games'] / max(sp['seconds'], 1e-9) * 3600:,.0f} games/h, batch {sp['avg_batch']:.0f}) | "
          f"train {tr['steps']} steps p={tr['loss_p']:.4f} v={tr['loss_v']:.5f} "
          f"({train_seconds:.0f}s) | buffer {len(buffer):,} draws {buffer.draw_frac:.0%}", flush=True)
    for r in rows:
        absolute = absolute_elo(r)
        scale = f" ~{absolute[0]:.0f} abs" if absolute else ""
        print(f"  vs {r['name']}: +{r['match']['wins']} ={r['match']['draws']} -{r['match']['losses']} "
              f"({r['score']:.1%}) Elo {r['elo']:+.0f} [{r['elo_lo']:+.0f}, {r['elo_hi']:+.0f}]{scale} "
              f"({r['match']['seconds']:.0f}s)", flush=True)
    warn(stats, buffer, sp, args)
    return row


def progress_printer(gen):
    def report(done, total, elapsed, evals):
        print(f"  gen {gen}: {done}/{total} games, {elapsed:.0f}s, "
              f"{done / max(elapsed, 1e-9) * 3600:,.0f} games/h, {evals / max(elapsed, 1e-9):,.0f} evals/s",
              flush=True)
    return report


def warn(stats, buffer, sp, args):
    """docs/05's watch-list, checked every generation instead of remembered."""
    slots = min(args.n_parallel, args.games_per_gen)
    if args.games_per_gen <= args.n_parallel and sp["avg_batch"] < 0.6 * slots:
        # With one slot per game there is nothing to refill a slot whose game
        # ended, so the generation's tail searches a handful of positions per
        # GPU forward. More games than slots keeps the batch full.
        print(f"  ! self-play averaged {sp['avg_batch']:.0f} of {slots} slots per forward: the "
              f"generation's tail ran the fleet nearly empty. Raise --games-per-gen "
              f"(e.g. {4 * args.games_per_gen}) above --n-parallel so finished slots refill",
              flush=True)
    if buffer.draw_frac > 0.40:
        print(f"  ! buffer is {buffer.draw_frac:.0%} draws (>40%): the value head is starving. "
              "docs/05: raise --decisive-premium before touching lambda", flush=True)
    fp = stats["resign_fp_frac"]
    if stats["resign_triggers"] >= 10 and fp == fp and fp > 0.05:
        print(f"  ! resign false positives {fp:.0%} of {stats['resign_triggers']} audited triggers "
              "(>5%, docs/02): raise --resign-threshold toward -1", flush=True)


def reached_target(row, args, gen):
    """--target-elo, on the 95% lower bound against a rated anchor."""
    for slot in ("a", "b"):
        name, lo = row.get(f"anchor_{slot}"), row.get(f"elo_{slot}_lo")
        if not name or lo is None:
            continue
        nominal = azeval.parse_anchor(name).get("elo")
        if nominal is None:
            continue
        absolute = nominal + float(lo)
        if absolute >= args.target_elo:
            print(f"\nTARGET REACHED at generation {gen}: {absolute:.0f} Elo "
                  f"(95% lower bound vs {name}) >= {args.target_elo:.0f}", flush=True)
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
