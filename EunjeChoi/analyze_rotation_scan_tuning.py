"""
Analyze rotation scan tuning results.

Usage:
    python analyze_rotation_scan_tuning.py
    python analyze_rotation_scan_tuning.py ../logs/rotation_scan_tuning_20260602_143120.csv
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "matplotlib/numpy is required for the visual analyzer.\n"
        "Install it in your venv with:\n"
        "  pip install matplotlib numpy"
    ) from exc


LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


NUMERIC_COLUMNS = {
    "trial", "kp", "max_hold_speed", "angle_deg", "yaw_rate_deg_s",
    "start_x", "start_y", "end_x", "end_y",
    "start_yaw", "end_yaw", "yaw_delta",
    "final_drift", "max_drift", "mean_drift", "samples", "score",
}


def _as_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def _load_csv(path):

    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in NUMERIC_COLUMNS:
                if key in parsed:
                    parsed[key] = _as_float(parsed[key])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _latest_csv():
    files = sorted(LOG_DIR.glob("rotation_scan_tuning_*.csv"))
    if not files:
        raise FileNotFoundError(f"No rotation_scan_tuning_*.csv files in {LOG_DIR}")
    return files[-1]


def _pick_csv(limit=10):
    files = sorted(LOG_DIR.glob("rotation_scan_tuning_*.csv"))
    if not files:
        raise FileNotFoundError(f"No rotation_scan_tuning_*.csv files in {LOG_DIR}")

    shown = files[-limit:]
    print("Available rotation scan tuning logs:")
    for i, path in enumerate(shown):
        print(f"  [{i}] {path.name}")

    default_idx = len(shown) - 1
    raw = input(f"Select file number (Enter = latest [{default_idx}]): ").strip()
    idx = int(raw) if raw else default_idx
    if idx < 0 or idx >= len(shown):
        raise ValueError(f"Invalid file number: {idx}")
    return shown[idx]


def _mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def _group_stats(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)

    stats = []
    for value, items in grouped.items():
        stats.append({
            key: value,
            "n": len(items),
            "score": _mean([item["score"] for item in items]),
            "final_drift": _mean([item["final_drift"] for item in items]),
            "max_drift": _mean([item["max_drift"] for item in items]),
            "mean_drift": _mean([item["mean_drift"] for item in items]),
            "yaw_error": _mean([
                abs(abs(item["yaw_delta"]) - abs(item["angle_deg"]))
                for item in items
            ]),
        })
    return sorted(stats, key=lambda item: item["score"])


def _format_trial(row):
    return (
        f"trial={int(row['trial'])}, "
        f"kp={row['kp']:.3f}, "
        f"max_hold_speed={row['max_hold_speed']:.3f} m/s, "
        f"score={row['score']:.4f}, "
        f"final={row['final_drift']:.3f} m, "
        f"max={row['max_drift']:.3f} m, "
        f"mean={row['mean_drift']:.3f} m, "
        f"yaw_delta={row['yaw_delta']:.1f} deg"
    )


def _format_group(row, key):
    return (
        f"{key}={row[key]:.3f}, "
        f"n={row['n']}, "
        f"avg_score={row['score']:.4f}, "
        f"avg_final={row['final_drift']:.3f} m, "
        f"avg_max={row['max_drift']:.3f} m, "
        f"avg_yaw_error={row['yaw_error']:.2f} deg"
    )


def _build_report(path, rows, top_n):
    sorted_rows = sorted(rows, key=lambda row: row["score"])
    best = sorted_rows[0]
    kp_stats = _group_stats(rows, "kp")
    speed_stats = _group_stats(rows, "max_hold_speed")

    lines = []
    lines.append(f"Rotation scan tuning analysis: {path}")
    lines.append("")
    lines.append("Best trial")
    lines.append(f"  {_format_trial(best)}")
    lines.append("")
    lines.append(f"Top {min(top_n, len(sorted_rows))} trials")
    for row in sorted_rows[:top_n]:
        lines.append(f"  {_format_trial(row)}")
    lines.append("")
    lines.append("KP averages")
    for row in kp_stats:
        lines.append(f"  {_format_group(row, 'kp')}")
    lines.append("")
    lines.append("Max hold speed averages")
    for row in speed_stats:
        lines.append(f"  {_format_group(row, 'max_hold_speed')}")
    lines.append("")
    lines.append("Recommendation")
    lines.append(
        "  Use the best trial value first: "
        f"hold_kp={best['kp']:.3f}, "
        f"max_hold_speed={best['max_hold_speed']:.3f}."
    )
    lines.append(
        "  If the drone oscillates during rotation, lower kp or max_hold_speed; "
        "if it drifts smoothly away, raise kp or max_hold_speed slightly."
    )
    return "\n".join(lines)


def _plot_heatmap(ax, rows, value_key, title, cmap):
    kp_values = sorted({row["kp"] for row in rows})
    speed_values = sorted({row["max_hold_speed"] for row in rows})
    grid = np.full((len(kp_values), len(speed_values)), np.nan)

    for row in rows:
        r = kp_values.index(row["kp"])
        c = speed_values.index(row["max_hold_speed"])
        grid[r, c] = row[value_key]

    image = ax.imshow(grid, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("max_hold_speed (m/s)")
    ax.set_ylabel("kp")
    ax.set_xticks(range(len(speed_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in speed_values])
    ax.set_yticks(range(len(kp_values)))
    ax.set_yticklabels([f"{v:.1f}" for v in kp_values])

    for r in range(len(kp_values)):
        for c in range(len(speed_values)):
            if not math.isnan(grid[r, c]):
                ax.text(c, r, f"{grid[r, c]:.3f}",
                        ha="center", va="center", color="black", fontsize=9)

    return image


def _show_plots(path, rows, top_n):
    rows = sorted(rows, key=lambda row: row["score"])
    top_rows = rows[:top_n]
    best = rows[0]

    plt.style.use("default")
    fig = plt.figure(figsize=(13, 8))
    fig.canvas.manager.set_window_title("Rotation Scan Tuning Analysis")
    fig.suptitle(
        f"Rotation Scan Tuning: {path.name}\n"
        f"Best: kp={best['kp']:.3f}, max_hold_speed={best['max_hold_speed']:.3f} m/s, "
        f"final={best['final_drift']:.3f} m, max={best['max_drift']:.3f} m",
        fontsize=12,
    )

    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    ax_score = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[0, 2])
    ax_drift = fig.add_subplot(gs[1, 0:2])
    ax_text = fig.add_subplot(gs[1, 2])

    score_image = _plot_heatmap(ax_score, rows, "score", "Score heatmap", "YlGn_r")
    final_image = _plot_heatmap(ax_final, rows, "final_drift", "Final drift (m)", "Blues")
    fig.colorbar(score_image, ax=ax_score, fraction=0.046, pad=0.04)
    fig.colorbar(final_image, ax=ax_final, fraction=0.046, pad=0.04)

    labels = [
        f"T{int(row['trial'])}\nkp {row['kp']:.1f}\nspd {row['max_hold_speed']:.2f}"
        for row in top_rows
    ]
    scores = [row["score"] for row in top_rows]
    colors = ["#2e7d32"] + ["#90caf9"] * max(0, len(top_rows) - 1)
    ax_top.bar(range(len(top_rows)), scores, color=colors)
    ax_top.set_title(f"Top {len(top_rows)} trials")
    ax_top.set_ylabel("score")
    ax_top.set_xticks(range(len(top_rows)))
    ax_top.set_xticklabels(labels, fontsize=8)
    ax_top.grid(axis="y", alpha=0.25)

    trial_numbers = [int(row["trial"]) for row in rows]
    final_drifts = [row["final_drift"] for row in rows]
    max_drifts = [row["max_drift"] for row in rows]
    mean_drifts = [row["mean_drift"] for row in rows]
    ax_drift.plot(trial_numbers, final_drifts, marker="o", label="final drift")
    ax_drift.plot(trial_numbers, max_drifts, marker="s", label="max drift")
    ax_drift.plot(trial_numbers, mean_drifts, marker="^", label="mean drift")
    ax_drift.set_title("Drift by trial")
    ax_drift.set_xlabel("trial")
    ax_drift.set_ylabel("drift (m)")
    ax_drift.set_xticks(trial_numbers)
    ax_drift.grid(alpha=0.25)
    ax_drift.legend()

    kp_stats = _group_stats(rows, "kp")
    speed_stats = _group_stats(rows, "max_hold_speed")
    text_lines = [
        "Recommendation",
        f"hold_kp = {best['kp']:.3f}",
        f"max_hold_speed = {best['max_hold_speed']:.3f} m/s",
        "",
        "Best KP average",
        f"kp {kp_stats[0]['kp']:.3f}",
        f"avg score {kp_stats[0]['score']:.4f}",
        "",
        "Best speed average",
        f"{speed_stats[0]['max_hold_speed']:.3f} m/s",
        f"avg score {speed_stats[0]['score']:.4f}",
        "",
        "Tuning hint",
        "Oscillation: lower kp/speed",
        "Smooth drift: raise kp/speed",
    ]
    ax_text.axis("off")
    ax_text.text(
        0.02, 0.98, "\n".join(text_lines),
        va="top", ha="left", fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze rotation scan tuning CSV.")
    parser.add_argument("csv_path", nargs="?", help="CSV path. Defaults to latest tuning CSV.")
    parser.add_argument("--top", type=int, default=5, help="Number of top trials to print.")
    parser.add_argument("--list-count", type=int, default=10, help="Number of recent logs to show.")
    parser.add_argument("--save-report", action="store_true", help="Save a .txt report next to the CSV.")
    parser.add_argument("--print-report", action="store_true", help="Also print the text report.")
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser() if args.csv_path else _pick_csv(args.list_count)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    rows = _load_csv(path)
    report = _build_report(path, rows, max(1, args.top))
    if args.print_report:
        print(report)

    if args.save_report:
        report_path = path.with_suffix(".analysis.txt")
        report_path.write_text(report + "\n")
        print(f"\nSaved report: {report_path}")

    _show_plots(path, rows, max(1, args.top))


if __name__ == "__main__":
    main()
