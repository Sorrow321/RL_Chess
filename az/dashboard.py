"""Render the self-play curve from one or more run logs (docs/05).

    python -m az.dashboard runs/hunt1
    python -m az.dashboard runs/hunt1 runs/hunt2 runs/hunt3     # one lineage
    python -m az.dashboard runs/hunt* --tail 20 --width 100

Runs are concatenated in the order given and the generation axis keeps
counting across them, because a restart from `latest.pt` into a fresh run dir
is the same lineage with different hyperparameters — the boundaries are marked
on the axis so a step in the curve can be blamed on the right change.

lesson #7: plateau decisions need the curve, not memory of it. So this is
deliberately a *reading* tool — it draws the Elo curve, the metrics docs/05
says to watch, and the playbook hint the numbers imply. It never touches a
checkpoint.
"""
import argparse
import csv
import os
import sys

BLOCKS = " ▁▂▃▄▅▆▇█"


def read_log(path):
    """<run>/log.csv -> list of dicts, floats where they parse."""
    if os.path.isdir(path):
        path = os.path.join(path, "log.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no log.csv — has this run produced a generation yet?")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in list(row.items()):
            if value in (None, ""):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def load_runs(paths):
    """-> (rows, boundaries): rows renumbered end to end, boundaries per run."""
    rows, boundaries = [], []
    for path in paths:
        part = read_log(path)
        boundaries.append((len(rows), os.path.basename(os.path.normpath(path)), len(part)))
        for row in part:
            row["run"] = path
            rows.append(row)
    for i, row in enumerate(rows):
        row["x"] = float(i)
    return rows, boundaries


def column(rows, key):
    """(xs, ys) of the rows where `key` is a number — evals can be sparse."""
    xs = [r["x"] for r in rows if isinstance(r.get(key), float)]
    ys = [r[key] for r in rows if isinstance(r.get(key), float)]
    return xs, ys


def sparkline(values):
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return BLOCKS[4] * len(values)
    return "".join(BLOCKS[1 + int((v - lo) / (hi - lo) * (len(BLOCKS) - 2))] for v in values)


def chart(xs, ys, width=72, height=11, label="", fmt="{:+.0f}", band=None):
    """A line chart in text. `band` is an optional (lo, ys, hi) shaded interval.

    Points are placed by their x, so a series evaluated every 5th generation
    plots against the same axis as one evaluated every generation instead of
    being stretched to fill the frame.
    """
    if not ys:
        return [f"  {label}: no data"]
    x_max = max(max(xs), 1)
    grid = [[" "] * width for _ in range(height)]
    lo, hi = min(ys), max(ys)
    if band:
        lo, hi = min(lo, min(band[0])), max(hi, max(band[1]))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad

    def row_of(value):
        return height - 1 - int(round((value - lo) / (hi - lo) * (height - 1)))

    def col_of(x):
        return int(round(x / x_max * (width - 1)))

    if band:
        for x, blo, bhi in zip(xs, band[0], band[1]):
            c = col_of(x)
            for r in range(row_of(bhi), row_of(blo) + 1):
                grid[r][c] = "│"
    prev = None
    for x, y in zip(xs, ys):
        c, r = col_of(x), row_of(y)
        if prev is not None and prev[0] != c:
            # join to the previous point so a sparse series still reads as a line
            r0, c0 = prev[1], prev[0]
            for rr in range(min(r0, r), max(r0, r) + 1):
                if grid[rr][c] == " ":
                    grid[rr][c] = "│"
        grid[r][c] = "●"
        prev = (c, r)

    out = [f"  {label}"]
    for i, line in enumerate(grid):
        value = hi - (hi - lo) * i / (height - 1)
        gutter = fmt.format(value) if i in (0, height // 2, height - 1) else ""
        out.append(f"  {gutter:>8} │{''.join(line)}")
    out.append(f"  {'':>8} └" + "─" * width)
    return out


def axis_labels(boundaries, total, width=72):
    """The x axis: generation numbers plus a mark where each run started."""
    line = [" "] * (width + 1)
    marks = []
    for start, name, count in boundaries:
        col = int(round(start / max(total - 1, 1) * (width - 1)))
        if col < len(line):
            line[col] = "┴"
        marks.append(f"{name} (gen {start}-{start + count - 1})")
    return [f"  {'':>8}  {''.join(line)}",
            f"  {'':>8}  gen 0" + " " * max(0, width - 12) + f"gen {total - 1}",
            "  runs: " + "  ".join(marks)]


def nominal_elo(name):
    """`sf1320` -> 1320. Anchors without a rating (random, skill<N>) -> 0.

    It is Stockfish's own claim at *its* time management, not at the frozen
    node count the harness plays it at (docs/06 caveat 1) — a label for the
    axis, never a milestone claim on its own.
    """
    if name.startswith("sf"):
        try:
            return float(name[2:])
        except ValueError:
            pass
    return 0.0


def anchor_series(rows):
    """-> [(name, nominal, xs, elo, lo, hi)], one per anchor ever logged.

    Keyed by anchor *name*, not by column: climbing the ladder means swapping
    sf1320 out for sf2000 mid-lineage, and a reader keyed on the column would
    silently splice two different opponents into one curve.
    """
    series = {}
    for row in rows:
        for slot in ("a", "b"):
            name = row.get(f"anchor_{slot}")
            elo = row.get(f"elo_vs_anchor_{slot}")
            if not name or not isinstance(elo, float):
                continue
            nominal = nominal_elo(name)
            xs, ys, los, his = series.setdefault(name, ([], [], [], []))
            xs.append(row["x"])
            ys.append(elo + nominal)
            los.append((row.get(f"elo_{slot}_lo") or elo) + nominal)
            his.append((row.get(f"elo_{slot}_hi") or elo) + nominal)
    return [(name, nominal_elo(name), *cols) for name, cols in series.items()]


           # "gen" is the *lineage* index, not the run's own: two run dirs
           # both counting from 0 in one table is unreadable.
TABLE = [("gen", "x", "{:.0f}"), ("games", "games", "{:.0f}"), ("plies", "avg_plies", "{:.0f}"),
         ("draw", "draw_frac", "{:.0%}"), ("resign", "resign_frac", "{:.0%}"),
         ("white", "white_score", "{:.3f}"), ("loss_p", "policy_loss", "{:.3f}"),
         ("loss_v", "value_loss", "{:.4f}"), ("buffer", "buffer_size", "{:,.0f}"),
         ("steps", "steps", "{:.0f}"), ("sims", "sims", "{:.0f}"),
         ("elo_a", "elo_vs_anchor_a", "{:+.0f}"), ("elo_b", "elo_vs_anchor_b", "{:+.0f}"),
         ("sp_s", "sp_seconds", "{:.0f}"), ("tr_s", "train_seconds", "{:.0f}"),
         ("ev_s", "eval_seconds", "{:.0f}")]

SPARKS = [("draw_frac", "draw fraction (docs/05: 5-15% healthy, >40% starves the value head)", "{:.0%}"),
          ("avg_plies", "game length", "{:.0f}"),
          ("resign_frac", "resigned games", "{:.0%}"),
          ("policy_loss", "policy loss", "{:.3f}"),
          ("value_loss", "value loss", "{:.4f}"),
          ("policy_entropy", "search entropy (nats, target sharpness)", "{:.2f}"),
          ("value_sign", "value sign agreement on decisive positions", "{:.0%}"),
          ("games_per_h", "self-play throughput (games/h)", "{:,.0f}")]


def render(rows, boundaries, width=72, height=11, tail=12):
    total = len(rows)
    hours = sum((r.get(k) or 0.0) for r in rows
                for k in ("sp_seconds", "train_seconds", "eval_seconds")) / 3600
    games = sum(r.get("games") or 0.0 for r in rows)
    print(f"{total} generations, {games:,.0f} self-play games, {hours:.1f} h of compute")
    print()

    for name, nominal, xs, elo, lo, hi in anchor_series(rows):
        scale = f"absolute (anchor {name} = {nominal:.0f})" if nominal else f"relative to {name}"
        for line in chart(xs, elo, width, height, label=f"Elo vs {name} — {scale}",
                          fmt="{:+.0f}" if not nominal else "{:.0f}", band=(lo, hi)):
            print(line)
        print(f"  {'':>8}  latest {elo[-1]:.0f} [{lo[-1]:.0f}, {hi[-1]:.0f}] 95% CI"
              + (f", best {max(elo):.0f}" if len(elo) > 1 else ""))
        print()

    for line in axis_labels(boundaries, total, width):
        print(line)
    print()

    for key, label, fmt in SPARKS:
        _, ys = column(rows, key)
        if not ys:
            continue
        print(f"  {sparkline(ys[-width:])}  {fmt.format(ys[-1]):>8}  {label}")
    print()

    show = rows[-tail:]
    widths = [max(len(head), max(len(cell(r, key, fmt)) for r in show)) for head, key, fmt in TABLE]
    print("  " + "  ".join(f"{head:>{w}}" for (head, _k, _f), w in zip(TABLE, widths)))
    for r in show:
        print("  " + "  ".join(f"{cell(r, key, fmt):>{w}}" for (_h, key, fmt), w in zip(TABLE, widths)))
    print()
    advise(rows)


def cell(row, key, fmt):
    value = row.get(key)
    return fmt.format(value) if isinstance(value, float) else "-"


def advise(rows):
    """docs/05's watch-list and plateau playbook, read off the curve."""
    said = []
    last = rows[-1]
    draw = last.get("buffer_draw_frac") or last.get("draw_frac")
    if draw and draw > 0.40:
        said.append(f"draw fraction {draw:.0%} (>40%): the value head is starving. Playbook step 3 — "
                    "raise --decisive-premium, or upweight endgame plies.")
    fp = last.get("resign_fp_frac")
    if fp and fp > 0.05:
        said.append(f"resign false positives {fp:.0%} (>5%, docs/02): --resign-threshold is too hot, "
                    "move it toward -1.")

    plies = [r["avg_plies"] for r in rows[-6:] if isinstance(r.get("avg_plies"), float)]
    resign = [r["resign_frac"] for r in rows[-6:] if isinstance(r.get("resign_frac"), float)]
    if len(plies) >= 6 and plies[-1] < plies[0] * 0.85 and len(resign) >= 6 and resign[-1] > resign[0] + 0.1:
        said.append("game length falling while resignations rise — docs/05 calls that a resign "
                    "threshold that is too hot, not progress.")

    for name, _nominal, _xs, elo, _lo, _hi in anchor_series(rows):
        if len(elo) < 6:
            continue
        recent, before = sum(elo[-3:]) / 3, sum(elo[-6:-3]) / 3
        if recent - before < 10:
            sims = last.get("sims")
            said.append(f"Elo vs {name} flat over the last 6 evaluations ({before:+.0f} -> {recent:+.0f}). "
                        f"Playbook step 1: raise self-play sims"
                        + (f" ({sims:.0f} -> {sims * 1.6:.0f})" if sims else "")
                        + ", restarting from latest.pt into a fresh run dir.")
        break

    if said:
        print("notes")
        for line in said:
            print("  - " + line)
    else:
        print("notes: nothing on the watch-list is firing.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (or log.csv paths), oldest first")
    ap.add_argument("--width", type=int, default=72)
    ap.add_argument("--height", type=int, default=11)
    ap.add_argument("--tail", type=int, default=12, help="generations in the table")
    args = ap.parse_args(argv)

    rows, boundaries = load_runs(args.runs)
    if not rows:
        raise SystemExit("no generations logged yet")
    render(rows, boundaries, width=args.width, height=args.height, tail=args.tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
