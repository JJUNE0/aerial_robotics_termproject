"""
Rotation scan tuning code.

Takeoff -> hover -> run yaw-only rotation scans while tuning XY hold gains.

Usage:
    python rotation_scan_tuning_code.py

Optional:
    python rotation_scan_tuning_code.py --kp-values 1.0,1.5,2.0 --speed-values 0.04,0.06,0.08
    python rotation_scan_tuning_code.py --yaw-rate-values 8,10,12,15
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


DEFAULT_KP_VALUES = (4.7,4.8)
DEFAULT_MAX_HOLD_SPEED_VALUES = (0.05, 0.03)
DEFAULT_YAW_RATE_VALUES = (config.SCAN_ROTATE_RATE,)
DEFAULT_HOLD_DEADZONE = 0.015
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _clamp(value, low, high):
    return max(low, min(high, value))


def _hold_velocity_command(ex, ey, kp, max_hold_speed, deadzone_m):
    if math.hypot(ex, ey) < deadzone_m:
        return 0.0, 0.0
    vx_w = _clamp(ex * kp, -max_hold_speed, max_hold_speed)
    vy_w = _clamp(ey * kp, -max_hold_speed, max_hold_speed)
    return vx_w, vy_w


def _yaw_delta_deg(start_yaw, end_yaw):
    return (end_yaw - start_yaw + 180.0) % 360.0 - 180.0


def _hover(cf, duration_s, z):
    end_t = time.time() + duration_s
    while time.time() < end_t:
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
        time.sleep(config.DT)


def _position_hold(cf, hub, tx, ty, kp, max_hold_speed, deadzone_m, duration_s, z):
    end_t = time.time() + duration_s
    while time.time() < end_t:
        cx, cy, _, cyaw = hub.read().pose
        ex, ey = tx - cx, ty - cy
        vx_w, vy_w = _hold_velocity_command(
            ex, ey, kp, max_hold_speed, deadzone_m)
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0.0, z)
        time.sleep(config.DT)


def _return_to_home(cf, hub, home_x, home_y, kp, max_hold_speed,
                    deadzone_m, z, timeout_s):
    end_t = time.time() + timeout_s
    while time.time() < end_t:
        cx, cy, _, cyaw = hub.read().pose
        ex, ey = home_x - cx, home_y - cy
        dist = math.hypot(ex, ey)
        if dist < max(0.02, deadzone_m):
            _position_hold(
                cf, hub, home_x, home_y, kp, max_hold_speed,
                deadzone_m, 0.5, z)
            return True

        vx_w, vy_w = _hold_velocity_command(
            ex, ey, kp, max_hold_speed, deadzone_m)
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0.0, z)
        time.sleep(config.DT)

    return False


def _rotation_scan_trial(cf, hub, kp, max_hold_speed,
                         angle_deg, yaw_rate_deg_s, z,
                         deadzone_m=DEFAULT_HOLD_DEADZONE,
                         home_x=None, home_y=None,
                         hold_x=None, hold_y=None):
    data0 = hub.read()
    start_x, start_y, _, start_yaw = data0.pose
    if hold_x is None:
        hold_x = start_x
    if hold_y is None:
        hold_y = start_y
    if home_x is None:
        home_x = hold_x
    if home_y is None:
        home_y = hold_y

    duration = abs(angle_deg) / yaw_rate_deg_s
    yaw_rate = math.copysign(yaw_rate_deg_s, angle_deg)
    start_t = time.time()
    end_t = time.time() + duration

    samples = []
    while time.time() < end_t:
        data = hub.read()
        cx, cy, cz, cyaw = data.pose
        ex, ey = hold_x - cx, hold_y - cy
        drift = math.hypot(ex, ey)
        home_drift = math.hypot(home_x - cx, home_y - cy)

        vx_w, vy_w = _hold_velocity_command(
            ex, ey, kp, max_hold_speed, deadzone_m)
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)

        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, z)
        samples.append((time.time(), cx, cy, cz, cyaw,
                        drift, home_drift, vx_b, vy_b))
        time.sleep(config.DT)

    rotation_elapsed_s = time.time() - start_t
    cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
    _hover(cf, 0.5, z)

    elapsed_s = time.time() - start_t
    data1 = hub.read()
    end_x, end_y, _, end_yaw = data1.pose
    final_drift = math.hypot(end_x - hold_x, end_y - hold_y)
    home_final_drift = math.hypot(end_x - home_x, end_y - home_y)
    max_drift = max((sample[5] for sample in samples), default=final_drift)
    home_max_drift = max((sample[6] for sample in samples), default=home_final_drift)
    mean_drift = (
        sum(sample[5] for sample in samples) / len(samples)
        if samples else final_drift
    )
    home_mean_drift = (
        sum(sample[6] for sample in samples) / len(samples)
        if samples else home_final_drift
    )
    yaw_delta = _yaw_delta_deg(start_yaw, end_yaw)
    actual_yaw_rate = (
        yaw_delta / rotation_elapsed_s if rotation_elapsed_s > 0.0 else 0.0
    )

    return {
        "kp": kp,
        "max_hold_speed": max_hold_speed,
        "hold_deadzone": deadzone_m,
        "home_x": home_x,
        "home_y": home_y,
        "hold_x": hold_x,
        "hold_y": hold_y,
        "start_x": hold_x,
        "start_y": hold_y,
        "actual_start_x": start_x,
        "actual_start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "start_yaw": start_yaw,
        "end_yaw": end_yaw,
        "yaw_delta": yaw_delta,
        "actual_yaw_rate_deg_s": actual_yaw_rate,
        "elapsed_s": elapsed_s,
        "rotation_elapsed_s": rotation_elapsed_s,
        "final_drift": final_drift,
        "max_drift": max_drift,
        "mean_drift": mean_drift,
        "home_final_drift": home_final_drift,
        "home_max_drift": home_max_drift,
        "home_mean_drift": home_mean_drift,
        "samples": len(samples),
    }


def _score(result):
    yaw_error = abs(abs(result["yaw_delta"]) - abs(result["angle_deg"]))
    return (
        result["max_drift"] * 2.0
        + result["final_drift"]
        + result.get("home_final_drift", result["final_drift"])
        + yaw_error * 0.001
    )


def _write_results(results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"rotation_scan_tuning_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "trial", "kp", "max_hold_speed", "hold_deadzone",
        "angle_deg", "yaw_rate_deg_s",
        "actual_yaw_rate_deg_s", "elapsed_s", "rotation_elapsed_s",
        "home_x", "home_y", "hold_x", "hold_y",
        "start_x", "start_y", "actual_start_x", "actual_start_y",
        "end_x", "end_y", "start_yaw", "end_yaw", "yaw_delta",
        "final_drift", "max_drift", "mean_drift",
        "home_final_drift", "home_max_drift", "home_mean_drift",
        "samples", "score",
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

        # EKF is reset before takeoff, so the pre-takeoff pad is the origin.
        home_x, home_y = 0.0, 0.0
        takeoff_x, takeoff_y, _, _ = hub.read().pose
        takeoff_drift = math.hypot(takeoff_x - home_x, takeoff_y - home_y)
        print(
            f"[tune] takeoff drift before correction: {takeoff_drift:.3f} m "
            f"at ({takeoff_x:.3f}, {takeoff_y:.3f})"
        )

        ok = _return_to_home(
            cf, hub, home_x, home_y,
            kp=args.startup_kp,
            max_hold_speed=args.startup_speed,
            deadzone_m=args.deadzone,
            z=args.height,
            timeout_s=args.startup_timeout,
        )
        if not ok:
            print("[tune] warning: startup drift correction timed out")
        _position_hold(
            cf, hub, home_x, home_y,
            kp=args.startup_kp,
            max_hold_speed=args.startup_speed,
            deadzone_m=args.deadzone,
            duration_s=args.startup_hold,
            z=args.height,
        )
        corrected_x, corrected_y, _, _ = hub.read().pose
        corrected_drift = math.hypot(corrected_x - home_x, corrected_y - home_y)
        print(
            f"[tune] home ready: target=({home_x:.3f}, {home_y:.3f}), "
            f"actual=({corrected_x:.3f}, {corrected_y:.3f}), "
            f"drift={corrected_drift:.3f} m"
        )

        results = []
        kp_values = _parse_float_list(args.kp_values)
        speed_values = _parse_float_list(args.speed_values)
        yaw_rate_values = _parse_float_list(args.yaw_rate_values)

        for kp in kp_values:
            for max_hold_speed in speed_values:
                for yaw_rate in yaw_rate_values:
                    print(
                        f"[tune] trial kp={kp:.3f}, "
                        f"max_hold_speed={max_hold_speed:.3f} m/s, "
                        f"yaw_rate={yaw_rate:.1f} deg/s"
                    )
                    if args.return_home:
                        ok = _return_to_home(
                            cf, hub, home_x, home_y,
                            kp=kp,
                            max_hold_speed=max_hold_speed,
                            deadzone_m=args.deadzone,
                            z=args.height,
                            timeout_s=args.return_timeout,
                        )
                        if not ok:
                            print("[tune] warning: return-to-home timed out")
                    else:
                        _position_hold(
                            cf, hub, home_x, home_y,
                            kp=kp,
                            max_hold_speed=max_hold_speed,
                            deadzone_m=args.deadzone,
                            duration_s=args.settle,
                            z=args.height,
                        )

                    result = _rotation_scan_trial(
                        cf, hub,
                        kp=kp,
                        max_hold_speed=max_hold_speed,
                        angle_deg=args.angle,
                        yaw_rate_deg_s=yaw_rate,
                        z=args.height,
                        deadzone_m=args.deadzone,
                        home_x=home_x,
                        home_y=home_y,
                        hold_x=home_x if args.global_hold else None,
                        hold_y=home_y if args.global_hold else None,
                    )
                    result["angle_deg"] = args.angle
                    result["yaw_rate_deg_s"] = yaw_rate
                    result["score"] = _score(result)
                    results.append(result)
                    print(
                        "[tune] "
                        f"local_final={result['final_drift']:.3f} m, "
                        f"home_final={result['home_final_drift']:.3f} m, "
                        f"max={result['max_drift']:.3f} m, "
                        f"home_max={result['home_max_drift']:.3f} m, "
                        f"yaw_delta={result['yaw_delta']:.1f} deg, "
                        f"actual_rate={result['actual_yaw_rate_deg_s']:.1f} deg/s, "
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
    parser.add_argument(
        "--yaw-rate-values",
        default=",".join(map(str, DEFAULT_YAW_RATE_VALUES)),
        help="Comma-separated yaw rates in deg/s. Overrides --yaw-rate for trials.",
    )
    parser.add_argument("--height", type=float, default=config.FLIGHT_Z)
    parser.add_argument("--initial-hover", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument(
        "--deadzone",
        type=float,
        default=DEFAULT_HOLD_DEADZONE,
        help="XY hold dead zone in meters. Velocity command is zero inside it.",
    )
    parser.add_argument("--startup-kp", type=float, default=4.0)
    parser.add_argument("--startup-speed", type=float, default=0.3)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--startup-hold", type=float, default=1.0)
    parser.add_argument("--return-timeout", type=float, default=4.0)
    parser.add_argument(
        "--return-home",
        action="store_true",
        help="Move back to the takeoff hover point before every trial.",
    )
    parser.add_argument(
        "--global-hold",
        action="store_true",
        help="Use the takeoff hover point as the hold point for every scan.",
    )
    args = parser.parse_args()

    if args.yaw_rate_values == ",".join(map(str, DEFAULT_YAW_RATE_VALUES)):
        args.yaw_rate_values = str(args.yaw_rate)

    cflib.crtp.init_drivers()
    print(f"[tune] connecting to {config.RADIO_URI}")
    with SyncCrazyflie(config.RADIO_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        run(scf.cf, args)


if __name__ == "__main__":
    main()
