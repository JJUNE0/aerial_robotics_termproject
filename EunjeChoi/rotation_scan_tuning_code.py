"""
Rotation scan tuning code.

Takeoff -> hover -> run yaw-only rotation scans while tuning XY hold gains.

Usage:
    python rotation_scan_tuning_code.py

Optional:
    python rotation_scan_tuning_code.py --kp-values 1.0,1.5,2.0 --speed-values 0.04,0.06,0.08
"""

import argparse
import csv
import math
import time
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import config
import controller
from sensors import SensorHub


DEFAULT_KP_VALUES = (4.0,4.5)
DEFAULT_MAX_HOLD_SPEED_VALUES = (0.05,0.01)
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _clamp(value, low, high):
    return max(low, min(high, value))


def _yaw_delta_deg(start_yaw, end_yaw):
    return (end_yaw - start_yaw + 180.0) % 360.0 - 180.0


def _hover(cf, duration_s, z):
    end_t = time.time() + duration_s
    while time.time() < end_t:
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
        time.sleep(config.DT)


def _rotation_scan_trial(cf, hub, kp, max_hold_speed,
                         angle_deg, yaw_rate_deg_s, z):
    data0 = hub.read()
    hold_x, hold_y, _, start_yaw = data0.pose

    duration = abs(angle_deg) / yaw_rate_deg_s
    yaw_rate = math.copysign(yaw_rate_deg_s, angle_deg)
    end_t = time.time() + duration

    samples = []
    while time.time() < end_t:
        data = hub.read()
        cx, cy, cz, cyaw = data.pose
        ex, ey = hold_x - cx, hold_y - cy
        drift = math.hypot(ex, ey)

        vx_w = _clamp(ex * kp, -max_hold_speed, max_hold_speed)
        vy_w = _clamp(ey * kp, -max_hold_speed, max_hold_speed)
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)

        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, z)
        samples.append((time.time(), cx, cy, cz, cyaw, drift, vx_b, vy_b))
        time.sleep(config.DT)

    cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
    _hover(cf, 0.5, z)

    data1 = hub.read()
    end_x, end_y, _, end_yaw = data1.pose
    final_drift = math.hypot(end_x - hold_x, end_y - hold_y)
    max_drift = max((sample[5] for sample in samples), default=final_drift)
    mean_drift = (
        sum(sample[5] for sample in samples) / len(samples)
        if samples else final_drift
    )
    yaw_delta = _yaw_delta_deg(start_yaw, end_yaw)

    return {
        "kp": kp,
        "max_hold_speed": max_hold_speed,
        "start_x": hold_x,
        "start_y": hold_y,
        "end_x": end_x,
        "end_y": end_y,
        "start_yaw": start_yaw,
        "end_yaw": end_yaw,
        "yaw_delta": yaw_delta,
        "final_drift": final_drift,
        "max_drift": max_drift,
        "mean_drift": mean_drift,
        "samples": len(samples),
    }


def _score(result):
    yaw_error = abs(abs(result["yaw_delta"]) - abs(result["angle_deg"]))
    return result["max_drift"] * 2.0 + result["final_drift"] + yaw_error * 0.001


def _write_results(results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"rotation_scan_tuning_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "trial", "kp", "max_hold_speed", "angle_deg", "yaw_rate_deg_s",
        "start_x", "start_y", "end_x", "end_y",
        "start_yaw", "end_yaw", "yaw_delta",
        "final_drift", "max_drift", "mean_drift", "samples", "score",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, result in enumerate(results, start=1):
            row = {name: result.get(name) for name in fieldnames}
            row["trial"] = i
            writer.writerow(row)
    return path


def run(cf, args):
    if args.yaw_rate <= 0.0:
        raise ValueError("--yaw-rate must be positive")

    hub = SensorHub(cf)
    hub.start()

    try:
        controller.init_ekf(cf)
        controller.takeoff_vel(cf, args.height)
        _hover(cf, args.initial_hover, args.height)

        results = []
        kp_values = _parse_float_list(args.kp_values)
        speed_values = _parse_float_list(args.speed_values)

        for kp in kp_values:
            for max_hold_speed in speed_values:
                print(
                    f"[tune] trial kp={kp:.3f}, "
                    f"max_hold_speed={max_hold_speed:.3f} m/s"
                )
                _hover(cf, args.settle, args.height)
                result = _rotation_scan_trial(
                    cf, hub,
                    kp=kp,
                    max_hold_speed=max_hold_speed,
                    angle_deg=args.angle,
                    yaw_rate_deg_s=args.yaw_rate,
                    z=args.height,
                )
                result["angle_deg"] = args.angle
                result["yaw_rate_deg_s"] = args.yaw_rate
                result["score"] = _score(result)
                results.append(result)
                print(
                    "[tune] "
                    f"final={result['final_drift']:.3f} m, "
                    f"max={result['max_drift']:.3f} m, "
                    f"mean={result['mean_drift']:.3f} m, "
                    f"yaw_delta={result['yaw_delta']:.1f} deg, "
                    f"score={result['score']:.3f}"
                )

        if results:
            best = min(results, key=lambda item: item["score"])
            print(
                "[tune] best "
                f"kp={best['kp']:.3f}, "
                f"max_hold_speed={best['max_hold_speed']:.3f} m/s "
                f"(final={best['final_drift']:.3f} m, "
                f"max={best['max_drift']:.3f} m)"
            )
            path = _write_results(results)
            print(f"[tune] saved results: {path}")

        controller.land_vel(cf, hub)

    except KeyboardInterrupt:
        print("[tune] interrupted - landing")
        controller.land_vel(cf, hub)

    finally:
        hub.stop()
        controller.disarm(cf)
        print("[tune] done")


def main():
    parser = argparse.ArgumentParser(description="Tune rotation scan XY hold.")
    parser.add_argument("--kp-values", default=",".join(map(str, DEFAULT_KP_VALUES)))
    parser.add_argument(
        "--speed-values",
        default=",".join(map(str, DEFAULT_MAX_HOLD_SPEED_VALUES)),
        help="Comma-separated max XY hold speeds in m/s.",
    )
    parser.add_argument("--angle", type=float, default=config.SCAN_ROTATE_ANGLE)
    parser.add_argument("--yaw-rate", type=float, default=config.SCAN_ROTATE_RATE)
    parser.add_argument("--height", type=float, default=config.FLIGHT_Z)
    parser.add_argument("--initial-hover", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=1.0)
    args = parser.parse_args()

    cflib.crtp.init_drivers()
    print(f"[tune] connecting to {config.RADIO_URI}")
    with SyncCrazyflie(config.RADIO_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        run(scf.cf, args)


if __name__ == "__main__":
    main()
