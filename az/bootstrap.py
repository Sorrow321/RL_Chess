"""Supervised bootstrap training (docs/04): human moves -> a starting policy.

    python -m az.bootstrap --data data/shards --out runs/bootstrap --epochs 2

Hard-label cross-entropy on the move the human played, masked to the legal set
(docs/03), plus MSE on the game result from the mover's perspective. This is
the same `az.net.policy_loss` self-play will use — a one-hot target is just the
degenerate visit distribution — so there is no second loss implementation to
keep in sync.

The number to watch is **held-out top-1 agreement**. docs/04 expects 45-55%
after a full run on the 1400-2200 band; under 40% means a bug, most likely an
orientation or move-index mismatch, and the encoder round-trip tests in
tests/encoding_test.py are where to look first, not here.

Deliverable: `<out>/bootstrap.pt`, then the three sanity gates in az/gates.py.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from az.data import PositionShards, iter_batches, shard_stats
from az.net import PolicyValueNet, planes_to_tensor, policy_loss, save_checkpoint, load_checkpoint


def to_device(batch, device):
    """numpy batch -> tensors. Planes stay uint8 across the bus (az.net)."""
    x = planes_to_tensor(batch["planes"], device, torch.float32)
    return (x,
            torch.from_numpy(batch["legal"]).to(device, non_blocking=True),
            torch.from_numpy(batch["valid"]).to(device, non_blocking=True),
            torch.from_numpy(batch["label_pos"]).to(device, non_blocking=True),
            torch.from_numpy(batch["z"]).to(device, non_blocking=True))


def losses(net, x, legal, valid, label_pos, z, lambda_value):
    logits, v = net(x)
    logits = logits.float()
    target = torch.zeros(legal.shape, device=logits.device, dtype=torch.float32)
    target.scatter_(1, label_pos[:, None], 1.0)  # hard label = one-hot on the played move
    loss_p = policy_loss(logits, legal, target, valid)
    loss_v = F.mse_loss(v.float(), z)
    return loss_p + lambda_value * loss_v, loss_p, loss_v, logits, v


@torch.inference_mode()
def evaluate(net, shards, device, batch_size, max_positions, lambda_value, rng):
    """Held-out agreement and losses — the docs/04 45-55% gate."""
    net.eval()
    n = min(len(shards), max_positions)
    # With replacement on purpose: a no-replacement draw allocates a full
    # permutation of the corpus, which is hundreds of MB at doc scale, and a
    # few duplicate positions cannot move an agreement percentage.
    idx = rng.integers(0, len(shards), size=n)
    correct = seen = 0
    p_sum = v_sum = z_err = 0.0
    for start in range(0, n - batch_size + 1, batch_size):
        batch = shards.batch(np.sort(idx[start:start + batch_size]))
        x, legal, valid, label_pos, z = to_device(batch, device)
        _, loss_p, loss_v, logits, v = losses(net, x, legal, valid, label_pos, z, lambda_value)
        # argmax over the legal moves only, exactly as the greedy gate plays.
        gathered = logits.gather(1, legal).masked_fill(~valid, float("-inf"))
        correct += int((gathered.argmax(1) == label_pos).sum())
        seen += len(label_pos)
        p_sum += float(loss_p) * len(label_pos)
        v_sum += float(loss_v) * len(label_pos)
        z_err += float((torch.sign(v.float()) == torch.sign(z)).sum())
    net.train()
    if not seen:
        return {"agree": 0.0, "loss_p": 0.0, "loss_v": 0.0, "value_sign": 0.0, "n": 0}
    return {"agree": correct / seen, "loss_p": p_sum / seen, "loss_v": v_sum / seen,
            "value_sign": z_err / seen, "n": seen}


def cosine_lr(step, total, base_lr, warmup, floor=0.05):
    """Linear warmup, then cosine decay to `floor` x base (docs/04: Adam 1e-3)."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="training shard dir (az.pack)")
    ap.add_argument("--out", required=True, help="run dir: checkpoints + log.csv")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200, help="warmup steps")
    ap.add_argument("--lambda-value", type=float, default=1.0, help="weight on the value MSE")
    ap.add_argument("--val-shards", type=int, default=1)
    ap.add_argument("--val-positions", type=int, default=100_000)
    ap.add_argument("--val-every", type=int, default=2000, help="steps between validations")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (0 = full epochs)")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--seed", type=int, default=0)
    # No silent CPU fallback. CUDA init can fail transiently (it has here,
    # under page-cache pressure right after a large read), and a run that
    # quietly trains 100x slower instead of stopping is a whole night lost to
    # a warning nobody read. Ask for --device cpu if you mean it.
    ap.add_argument("--device", default=None, help="default: cuda, and it is an error if unavailable")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    args = ap.parse_args()

    if args.device is None:
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not available — refusing to train on CPU by accident.\n"
                "This is usually transient (driver context allocation losing a race with page-cache\n"
                "reclaim); check `nvidia-smi` and retry. Pass --device cpu to train on CPU on purpose.")
        args.device = "cuda"

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train, val = PositionShards.split(args.data, args.val_shards)
    # flush=True throughout: these runs are watched through a pipe or a log
    # tail, where an unflushed line is a line that does not exist yet.
    print(f"train {len(train):,} positions in {len(train.paths)} shard(s), "
          f"val {len(val) if val else 0:,}", flush=True)
    print("  corpus: " + ", ".join(f"{k}={v:.3g}" for k, v in shard_stats(train, rng=rng).items()),
          flush=True)

    net = (load_checkpoint(args.resume, device) if args.resume
           else PolicyValueNet(args.channels, args.blocks).to(device))
    net = net.to(memory_format=torch.channels_last).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    scaler = torch.amp.GradScaler(enabled=(args.amp == "fp16"))

    steps_per_epoch = len(train) // args.batch
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    print(f"{steps_per_epoch:,} steps/epoch, {total_steps:,} total")

    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "epoch", "lr", "loss_p", "loss_v", "train_agree",
                      "val_agree", "val_loss_p", "val_loss_v", "val_value_sign",
                      "positions_per_s", "seconds"])

    step, t_start = 0, time.time()
    p_sum = v_sum = agree_sum = 0.0
    window = 0
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in iter_batches(train, args.batch, rng=rng):
            lr = cosine_lr(step, total_steps, args.lr, args.warmup)
            for group in opt.param_groups:
                group["lr"] = lr

            x, legal, valid, label_pos, z = to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, loss_p, loss_v, logits, _ = losses(
                    net, x, legal, valid, label_pos, z, args.lambda_value)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                gathered = logits.gather(1, legal).masked_fill(~valid, float("-inf"))
                agree_sum += float((gathered.argmax(1) == label_pos).float().mean())
            p_sum += float(loss_p.detach())
            v_sum += float(loss_v.detach())
            window += 1
            step += 1

            if step % args.val_every == 0 or step == total_steps:
                elapsed = time.time() - t_start
                stats = (evaluate(net, val, device, args.batch, args.val_positions,
                                  args.lambda_value, rng) if val else
                         {"agree": 0.0, "loss_p": 0.0, "loss_v": 0.0, "value_sign": 0.0})
                pps = step * args.batch / elapsed
                log.writerow([step, epoch, f"{lr:.2e}", f"{p_sum / window:.4f}",
                              f"{v_sum / window:.5f}", f"{agree_sum / window:.4f}",
                              f"{stats['agree']:.4f}", f"{stats['loss_p']:.4f}",
                              f"{stats['loss_v']:.5f}", f"{stats['value_sign']:.4f}",
                              f"{pps:.0f}", f"{elapsed:.0f}"])
                log_f.flush()
                print(f"step {step:,}/{total_steps:,} (epoch {epoch}) lr {lr:.2e} | "
                      f"train p={p_sum / window:.4f} v={v_sum / window:.5f} "
                      f"agree={agree_sum / window:.1%} | "
                      f"val agree={stats['agree']:.1%} p={stats['loss_p']:.4f} "
                      f"v={stats['loss_v']:.5f} | {pps:,.0f} pos/s", flush=True)
                p_sum = v_sum = agree_sum = 0.0
                window = 0

            if step >= total_steps:
                done = True
                break

        save_checkpoint(net, os.path.join(args.out, f"epoch_{epoch:03d}.pt"), epoch=epoch, step=step)

    save_checkpoint(net, os.path.join(args.out, "bootstrap.pt"), step=step)
    log_f.close()
    print(f"\nsaved {os.path.join(args.out, 'bootstrap.pt')} after {step:,} steps "
          f"({(time.time() - t_start) / 60:.0f} min)")
    if val:
        final = evaluate(net, val, device, args.batch, args.val_positions, args.lambda_value, rng)
        print(f"held-out top-1 agreement: {final['agree']:.1%} over {final['n']:,} positions "
              f"(docs/04 expects 45-55%; under 40% is a bug hunt, start with the encoders)")
    print("next: python -m az.gates --ckpt " + os.path.join(args.out, "bootstrap.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
