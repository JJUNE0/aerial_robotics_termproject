"""
Position drift tuning code.

Takeoff -> correct startup drift to EKF origin -> move +X -> return to origin.
No yaw rotation is commanded; yaw_rate is always 0.

Usage:
    python position_drift_tuning_code.py

Optional:
    python position_drift_tuning_code.py --kp-values 2,3,4,5 --speed-values 0.05,0.08,0.10
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


DEFAULT_KP_VALUES = (2.0, 3.0)
DEFAULT_MAX_SPEED_VALUES = (0.3, 0.4)
DEFAULT_DEADZONE = 0.01
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _parse_float_list(text):
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _clamp(value, low, high):
    return max(low, min(high, value))


def _hold_velocity_command(ex, ey, kp, max_speed, deadzone_m):
    if math.hypot(ex, ey) < deadzone_m:
        return 0.0, 0.0
    vx_w = _clamp(ex * kp, -max_speed, max_speed)
    vy_w = _clamp(ey * kp, -max_speed, max_speed)
    return vx_w, vy_w


def _hover(cf, duration_s, z):
    end_t = time.time() + duration_s
    while time.time() < end_t:
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
        time.sleep(config.DT)


def _position_hold(cf, hub, tx, ty, kp, max_speed, deadzone_m, duration_s, z):
    samples = []
    end_t = time.time() + duration_s
    while time.time() < end_t:
        cx, cy, cz, cyaw = hub.read().pose
        ex, ey = tx - cx, ty - cy
        vx_w, vy_w = _hold_velocity_command(ex, ey, kp, max_speed, deadzone_m)
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0.0, z)
        samples.append((time.time(), cx, cy, cz, cyaw, ex, ey, vx_b, vy_b))
        time.sleep(config.DT)
    return samples


def _move_to(cf, hub, tx, ty, kp, max_speed, deadzone_m,
             z, arrive_threshold, timeout_s, settle_s):
    samples = []
    start_t = time.time()
    settled_since = None

    while time.time() - start_t < timeout_s:
        cx, cy, cz, cyaw = hub.read().pose
        ex, ey = tx - cx, ty - cy
        dist = math.hypot(ex, ey)

        if dist < arrive_threshold:
            if settled_since is None:
                settled_since = time.time()
            if time.time() - settled_since >= settle_s:
                cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
                return True, samples
        else:
            settled_since = None

        vx_w, vy_w = _hold_velocity_command(ex, ey, kp, max_speed, deadzone_m)
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0.0, z)
        samples.append((time.time(), cx, cy, cz, cyaw, ex, ey, dist, vx_b, vy_b))
        time.sleep(config.DT)

    cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, z)
    return False, samples


def _summarize_segment(samples, tx, ty):
    if not samples:
        return {
            "end_x": math.nan,
            "end_y": math.nan,
            "final_error": math.nan,
            "max_error": math.nan,
            "mean_error": math.nan,
            "max_abs_y": math.nan,
        }

    errors = []
    abs_y = []
    for sample in samples:
        cx, cy = sample[1], sample[2]
        errors.append(math.hypot(tx - cx, ty - cy))
        abs_y.append(abs(cy - ty))

    end_x, end_y = samples[-1][1], samples[-1][2]
    return {
        "end_x": end_x,
        "end_y": end_y,
        "final_error": math.hypot(tx - end_x, ty - end_y),
        "max_error": max(errors),
        "mean_error": sum(errors) / len(errors),
        "max_abs_y": max(abs_y),
    }


def _trial(cf, hub, kp, max_speed, deadzone_m, distance_m,
           z, arrive_threshold, timeout_s, settle_s):
    home_x, home_y = 0.0, 0.0
    start_x, start_y, _, _ = hub.read().pose

    ok_out, out_samples = _move_to(
        cf, hub,
        tx=home_x + distance_m,
        ty=home_y,
        kp=kp,
        max_speed=max_speed,
        deadzone_m=deadzone_m,
        z=z,
        arrive_threshold=arrive_threshold,
        timeout_s=timeout_s,
        settle_s=settle_s,
    )
    _position_hold(cf, hub, home_x + distance_m, home_y,
                   kp, max_speed, deadzone_m, 0.5, z)

    ok_back, back_samples = _move_to(
        cf, hub,
        tx=home_x,
        ty=home_y,
        kp=kp,
        max_speed=max_speed,
        deadzone_m=deadzone_m,
        z=z,
        arrive_threshold=arrive_threshold,
        timeout_s=timeout_s,
        settle_s=settle_s,
    )
    _position_hold(cf, hub, home_x, home_y,
                   kp, max_speed, deadzone_m, 0.5, z)

    out = _summarize_segment(out_samples, home_x + distance_m, home_y)
    back = _summarize_segment(back_samples, home_x, home_y)
    end_x, end_y, _, _ = hub.read().pose

    return {
        "kp": kp,
        "max_speed": max_speed,
        "deadzone": deadzone_m,
        "distance_m": distance_m,
        "start_x": start_x,
        "start_y": start_y,
        "out_ok": int(ok_out),
        "out_end_x": out["end_x"],
        "out_end_y": out["end_y"],
        "out_final_error": out["final_error"],
        "out_max_error": out["max_error"],
        "out_mean_error": out["mean_error"],
        "out_max_abs_y": out["max_abs_y"],
        "back_ok": int(ok_back),
        "back_end_x": back["end_x"],
        "back_end_y": back["end_y"],
        "back_final_error": back["final_error"],
        "back_max_error": back["max_error"],
        "back_mean_error": back["mean_error"],
        "back_max_abs_y": back["max_abs_y"],
        "home_final_x": end_x,
        "home_final_y": end_y,
        "home_final_error": math.hypot(end_x - home_x, end_y - home_y),
        "out_samples": len(out_samples),
        "back_samples": len(back_samples),
    }


def _score(result):
    return (
        result["home_final_error"] * 3.0
        + result["back_final_error"] * 2.0
        + result["out_final_error"]
        + result["out_max_abs_y"]
        + result["back_max_abs_y"]
    )


def _write_results(results):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"position_drift_tuning_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "trial", "kp", "max_speed", "deadzone", "distance_m",
        "start_x", "start_y",
        "out_ok", "out_end_x", "out_end_y",
        "out_final_error", "out_max_error", "out_mean_error", "out_max_abs_y",
        "back_ok", "back_end_x", "back_end_y",
        "back_final_error", "back_max_error", "back_mean_error", "back_max_abs_y",
        "home_final_x", "home_final_y", "home_final_error",
        "out_samples", "back_samples", "score",
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
    hub = SensorHub(cf)
    hub.start()

    try:
        controller.init_ekf(cf)
        controller.takeoff_vel(cf, args.height)
        _hover(cf, args.initial_hover, args.height)

        print("[pos-tune] correcting startup drift to EKF origin")
        _move_to(
            cf, hub,
            tx=0.0,
            ty=0.0,
            kp=args.startup_kp,
            max_speed=args.startup_speed,
            deadzone_m=args.deadzone,
            z=args.height,
            arrive_threshold=args.arrive_threshold,
            timeout_s=args.startup_timeout,
            settle_s=args.settle,
        )
        _position_hold(cf, hub, 0.0, 0.0,
                       args.startup_kp, args.startup_speed,
                       args.deadzone, args.startup_hold, args.height)
        x0, y0, _, _ = hub.read().pose
        print(f"[pos-tune] start ready: ({x0:.3f}, {y0:.3f})")

        results = []
        kp_values = _parse_float_list(args.kp_values)
        speed_values = _parse_float_list(args.speed_values)

        for kp in kp_values:
            for max_speed in speed_values:
                print(
                    f"[pos-tune] trial kp={kp:.3f}, "
                    f"max_speed={max_speed:.3f} m/s"
                )
                _move_to(
                    cf, hub,
                    tx=0.0,
                    ty=0.0,
                    kp=args.startup_kp,
                    max_speed=args.startup_speed,
                    deadzone_m=args.deadzone,
                    z=args.height,
                    arrive_threshold=args.arrive_threshold,
                    timeout_s=args.startup_timeout,
                    settle_s=args.settle,
                )
                _position_hold(cf, hub, 0.0, 0.0,
                               args.startup_kp, args.startup_speed,
                               args.deadzone, args.startup_hold, args.height)

                result = _trial(
                    cf, hub,
                    kp=kp,
                    max_speed=max_speed,
                    deadzone_m=args.deadzone,
                    distance_m=args.distance,
                    z=args.height,
                    arrive_threshold=args.arrive_threshold,
                    timeout_s=args.move_timeout,
                    settle_s=args.settle,
                )
                result["score"] = _score(result)
                results.append(result)
                print(
                    "[pos-tune] "
                    f"out_err={result['out_final_error']:.3f} m, "
                    f"back_err={result['back_final_error']:.3f} m, "
                    f"home_err={result['home_final_error']:.3f} m, "
                    f"y_swing=max({result['out_max_abs_y']:.3f},"
                    f"{result['back_max_abs_y']:.3f}) m, "
                    f"score={result['score']:.3f}"
                )

        if results:
            best = min(results, key=lambda item: item["score"])
            print(
                "[pos-tune] best "
                f"kp={best['kp']:.3f}, max_speed={best['max_speed']:.3f} m/s, "
                f"home_err={best['home_final_error']:.3f} m"
            )
            path = _write_results(results)
            print(f"[pos-tune] saved results: {path}")

        controller.land_vel(cf, hub)

    except KeyboardInterrupt:
        print("[pos-tune] interrupted - landing")
        controller.land_vel(cf, hub)

    finally:
        hub.stop()
        controller.disarm(cf)
        print("[pos-tune] done")


def main():
    parser = argparse.ArgumentParser(description="Tune straight-line XY drift.")
    parser.add_argument("--kp-values", default=",".join(map(str, DEFAULT_KP_VALUES)))
    parser.add_argument(
        "--speed-values",
        default=",".join(map(str, DEFAULT_MAX_SPEED_VALUES)),
        help="Comma-separated max XY speeds in m/s.",
    )
    parser.add_argument("--deadzone", type=float, default=DEFAULT_DEADZONE)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=config.FLIGHT_Z)
    parser.add_argument("--initial-hover", type=float, default=1.0)
    parser.add_argument("--settle", type=float, default=0.4)
    parser.add_argument("--arrive-threshold", type=float, default=0.03)
    parser.add_argument("--move-timeout", type=float, default=20.0)
    parser.add_argument("--startup-kp", type=float, default=3.0)
    parser.add_argument("--startup-speed", type=float, default=0.08)
    parser.add_argument("--startup-timeout", type=float, default=5.0)
    parser.add_argument("--startup-hold", type=float, default=1.0)
    args = parser.parse_args()

    cflib.crtp.init_drivers()
    print(f"[pos-tune] connecting to {config.RADIO_URI}")
    with SyncCrazyflie(config.RADIO_URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        run(scf.cf, args)


if __name__ == "__main__":
    main()
