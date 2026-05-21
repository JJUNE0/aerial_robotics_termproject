"""
Parse a Crazyflie square trajectory CSV log.

Example:
    .venv/bin/python src/aerial_robotics_termproject/parse_flight_log.py logs/flight_20260512_120000.csv
"""
import argparse
import csv
import math
from collections import defaultdict


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(row)
    return rows


def as_float(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def summarize(rows):
    if not rows:
        raise ValueError("log file has no rows")

    duration_s = as_float(rows[-1], "time_s") - as_float(rows[0], "time_s")
    xy_errors = [as_float(row, "error_xy_m") for row in rows]
    z_errors = [abs(as_float(row, "error_z_m")) for row in rows]
    errors_3d = [as_float(row, "error_3d_m") for row in rows]
    voltages = [as_float(row, "vbat_v") for row in rows if as_float(row, "vbat_v") is not None]

    return {
        "samples": len(rows),
        "duration_s": duration_s,
        "mean_xy_error_m": mean(xy_errors),
        "rmse_xy_error_m": rmse(xy_errors),
        "max_xy_error_m": max(xy_errors),
        "mean_z_error_m": mean(z_errors),
        "rmse_z_error_m": rmse(z_errors),
        "max_z_error_m": max(z_errors),
        "mean_3d_error_m": mean(errors_3d),
        "rmse_3d_error_m": rmse(errors_3d),
        "max_3d_error_m": max(errors_3d),
        "min_vbat_v": min(voltages) if voltages else None,
        "final_xy_error_m": xy_errors[-1],
        "final_z_error_m": z_errors[-1],
        "final_3d_error_m": errors_3d[-1],
    }


def summarize_by_phase(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["phase"]].append(row)

    summaries = {}
    for phase, phase_rows in grouped.items():
        xy_errors = [as_float(row, "error_xy_m") for row in phase_rows]
        z_errors = [abs(as_float(row, "error_z_m")) for row in phase_rows]
        summaries[phase] = {
            "samples": len(phase_rows),
            "mean_xy_error_m": mean(xy_errors),
            "max_xy_error_m": max(xy_errors),
            "mean_z_error_m": mean(z_errors),
            "max_z_error_m": max(z_errors),
        }
    return summaries


def mean(values):
    return sum(values) / len(values)


def rmse(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def cm(value_m):
    if value_m is None:
        return "n/a"
    return f"{value_m * 100.0:.1f} cm"


def print_summary(path, overall, phases):
    print("=" * 64)
    print(f"Log file: {path}")
    print("=" * 64)
    print(f"Samples: {overall['samples']}")
    print(f"Duration: {overall['duration_s']:.1f} s")
    if overall["min_vbat_v"] is None:
        print("Minimum battery: n/a")
    else:
        print(f"Minimum battery: {overall['min_vbat_v']:.2f} V")
    print()
    print("Overall tracking error")
    print(f"- XY mean / RMSE / max: {cm(overall['mean_xy_error_m'])} / {cm(overall['rmse_xy_error_m'])} / {cm(overall['max_xy_error_m'])}")
    print(f"- Z  mean / RMSE / max: {cm(overall['mean_z_error_m'])} / {cm(overall['rmse_z_error_m'])} / {cm(overall['max_z_error_m'])}")
    print(f"- 3D mean / RMSE / max: {cm(overall['mean_3d_error_m'])} / {cm(overall['rmse_3d_error_m'])} / {cm(overall['max_3d_error_m'])}")
    print()
    print("Final landing / stop error")
    print(f"- XY: {cm(overall['final_xy_error_m'])}")
    print(f"- Z:  {cm(overall['final_z_error_m'])}")
    print(f"- 3D: {cm(overall['final_3d_error_m'])}")
    print()
    print("Phase summary")
    for phase, stats in phases.items():
        print(
            f"- {phase:14s} samples={stats['samples']:4d} "
            f"XY mean/max={cm(stats['mean_xy_error_m'])}/{cm(stats['max_xy_error_m'])} "
            f"Z mean/max={cm(stats['mean_z_error_m'])}/{cm(stats['max_z_error_m'])}"
        )
    print()
    print("Tuning hints")
    if overall["max_xy_error_m"] > 0.15:
        print("- XY max error is over 15 cm: lower --speed-cm-s or increase segment duration.")
    if overall["mean_z_error_m"] > 0.08:
        print("- Z mean error is over 8 cm: check Flow deck surface and estimator reset stability.")
    if overall["min_vbat_v"] is not None and overall["min_vbat_v"] < 3.5:
        print("- Battery dropped below 3.5 V: recharge before tuning control behavior.")
    if overall["max_xy_error_m"] <= 0.15 and overall["mean_z_error_m"] <= 0.08:
        print("- Tracking looks reasonable for a first pass; compare repeated runs before tuning.")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a square trajectory flight log.")
    parser.add_argument("log_csv", help="Path to a CSV file saved by square_trajectory_flight.py")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.log_csv)
    overall = summarize(rows)
    phases = summarize_by_phase(rows)
    print_summary(args.log_csv, overall, phases)


if __name__ == "__main__":
    main()
