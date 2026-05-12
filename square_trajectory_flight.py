"""
Crazyflie square trajectory flight.

Scenario:
1. Reset the estimator while the Crazyflie is sitting on a 15 cm box.
2. Take off vertically from the box top.
3. Fly a rectangle in the horizontal plane.
4. Return to the takeoff point and land on the same box.
5. Send motor-stop commands whenever the program exits or is interrupted.

The 15 cm box is not added to the target height. After estimator reset, the
box top is treated as z=0, so flight_height_cm means height above the box.
"""
import argparse
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crazyflie_matplotlib")

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
import matplotlib.pyplot as plt


URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E5")

BOX_HEIGHT_CM = 15.0
DEFAULT_WIDTH_CM = 50.0
DEFAULT_HEIGHT_CM = 50.0
DEFAULT_FLIGHT_HEIGHT_CM = 40.0
DEFAULT_SPEED_CM_S = 10.0
TAKEOFF_DURATION_S = 3.0
LANDING_DURATION_S = 3.0
HOVER_TIME_S = 1.0
LOG_PERIOD_MS = 100

logging.basicConfig(level=logging.ERROR)

_active_cf = None
_active_monitor = None


def cm_to_m(value_cm):
    return value_cm / 100.0


def clamp_duration(distance_m, speed_m_s, minimum_s=2.0):
    return max(minimum_s, distance_m / speed_m_s)


def ned_to_crazyflie(north_m, east_m, down_m):
    # Crazyflie high-level commander uses x=forward, y=left, z=up.
    # NED uses x=north/forward, y=east/right, z=down.
    return north_m, -east_m, -down_m


def crazyflie_to_ned(x_m, y_m, z_m):
    return x_m, -y_m, -z_m


class FlightMonitor:
    def __init__(self, scf, log_dir, plot_enabled, width_m, height_m, frame):
        self._cf = scf.cf
        self._log_dir = log_dir
        self._plot_enabled = plot_enabled
        self._width_m = width_m
        self._height_m = height_m
        self._frame = frame
        self._lock = None
        self._running = False
        self._log_configs = []
        self._csv_file = None
        self._writer = None
        self._start_time = None
        self._latest = None
        self._path_x = []
        self._path_y = []
        self._figure = None
        self._axis = None
        self._actual_line = None
        self._target_point = None
        self._drone_point = None
        self._status_text = None
        self._attitude = {
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        }
        self._target = {
            "start_x": 0.0,
            "start_y": 0.0,
            "start_z": 0.0,
            "end_x": 0.0,
            "end_y": 0.0,
            "end_z": 0.0,
            "start_time": None,
            "duration_s": 0.0,
            "phase": "init",
        }
        self.path = None

    def __enter__(self):
        import threading

        self._lock = threading.Lock()
        os.makedirs(self._log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self._log_dir, f"flight_{timestamp}.csv")
        self._csv_file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "time_s",
                "phase",
                "target_x_m",
                "target_y_m",
                "target_z_m",
                "target_n_m",
                "target_e_m",
                "target_d_m",
                "actual_x_m",
                "actual_y_m",
                "actual_z_m",
                "actual_n_m",
                "actual_e_m",
                "actual_d_m",
                "error_x_m",
                "error_y_m",
                "error_z_m",
                "error_xy_m",
                "error_3d_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "vbat_v",
            ],
        )
        self._writer.writeheader()
        self._start_time = time.time()
        self._running = True

        position_log = LogConfig(name="FlightPosition", period_in_ms=LOG_PERIOD_MS)
        position_log.add_variable("stateEstimate.x", "float")
        position_log.add_variable("stateEstimate.y", "float")
        position_log.add_variable("stateEstimate.z", "float")
        position_log.add_variable("pm.vbat", "float")
        self._cf.log.add_config(position_log)
        position_log.data_received_cb.add_callback(self._position_log_callback)
        position_log.start()

        attitude_log = LogConfig(name="FlightAttitude", period_in_ms=LOG_PERIOD_MS)
        attitude_log.add_variable("stabilizer.roll", "float")
        attitude_log.add_variable("stabilizer.pitch", "float")
        attitude_log.add_variable("stabilizer.yaw", "float")
        self._cf.log.add_config(attitude_log)
        attitude_log.data_received_cb.add_callback(self._attitude_log_callback)
        attitude_log.start()
        self._log_configs = [position_log, attitude_log]

        if self._plot_enabled:
            self._setup_plot()

        print(f"Flight log: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def set_target(self, x, y, z, phase):
        with self._lock:
            self._target = {
                "start_x": x,
                "start_y": y,
                "start_z": z,
                "end_x": x,
                "end_y": y,
                "end_z": z,
                "start_time": None,
                "duration_s": 0.0,
                "phase": phase,
            }

    def start_segment(self, start_xyz, end_xyz, duration_s, phase):
        with self._lock:
            self._target = {
                "start_x": start_xyz[0],
                "start_y": start_xyz[1],
                "start_z": start_xyz[2],
                "end_x": end_xyz[0],
                "end_y": end_xyz[1],
                "end_z": end_xyz[2],
                "start_time": time.time(),
                "duration_s": duration_s,
                "phase": phase,
            }

    def stop(self):
        self._running = False
        for log_config in self._log_configs:
            try:
                log_config.stop()
            except Exception:
                pass
        self._log_configs = []

        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None

    def update_plot(self):
        if not self._plot_enabled or self._figure is None:
            return
        if not plt.fignum_exists(self._figure.number):
            self._figure = None
            return

        with self._lock:
            latest = dict(self._latest) if self._latest else None
            target = self._current_target_locked()
            target_n, target_e, target_d = crazyflie_to_ned(
                target["x"], target["y"], target["z"]
            )

        if latest:
            actual_x = float(latest["actual_x_m"])
            actual_y = float(latest["actual_y_m"])
            actual_z = float(latest["actual_z_m"])
            actual_n, actual_e, actual_d = crazyflie_to_ned(
                actual_x, actual_y, actual_z
            )
            error_xy_cm = float(latest["error_xy_m"]) * 100.0
            error_z_cm = float(latest["error_z_m"]) * 100.0
            vbat = float(latest["vbat_v"])
            self._path_x.append(actual_n)
            self._path_y.append(actual_e)
        else:
            actual_n = actual_e = actual_d = 0.0
            error_xy_cm = error_z_cm = 0.0
            vbat = 0.0

        self._actual_line.set_data(self._path_y, self._path_x)
        self._target_point.set_data([target_e], [target_n])
        self._drone_point.set_data([actual_e], [actual_n])
        self._status_text.set_text(
            f"phase: {target['phase']}\n"
            f"target N,E,D: {target_n:.2f}, {target_e:.2f}, {target_d:.2f} m\n"
            f"actual N,E,D: {actual_n:.2f}, {actual_e:.2f}, {actual_d:.2f} m\n"
            f"error XY: {error_xy_cm:.1f} cm   Z: {error_z_cm:.1f} cm\n"
            f"battery: {vbat:.2f} V"
        )
        self._figure.canvas.draw_idle()
        plt.pause(0.001)

    def _setup_plot(self):
        plt.ion()
        self._figure, self._axis = plt.subplots(figsize=(7.5, 7.0))
        self._figure.canvas.manager.set_window_title("Crazyflie square tracking")

        # Plot in NED view: matplotlib x-axis is East, y-axis is North.
        square_e = [0.0, 0.0, self._height_m, self._height_m, 0.0]
        square_n = [0.0, self._width_m, self._width_m, 0.0, 0.0]
        self._axis.plot(square_e, square_n, "k--", linewidth=1.4, label="planned square")
        self._actual_line, = self._axis.plot([], [], color="#1f77b4", linewidth=2.0, label="actual path")
        self._target_point, = self._axis.plot([], [], "o", color="#ff7f0e", markersize=9, label="target")
        self._drone_point, = self._axis.plot([], [], "o", color="#2ca02c", markersize=9, label="drone")
        self._axis.plot([0.0], [0.0], "s", color="#444444", markersize=7, label="takeoff / landing")

        margin_m = 0.15
        self._axis.set_xlim(-margin_m, self._height_m + margin_m)
        self._axis.set_ylim(-margin_m, self._width_m + margin_m)
        self._axis.set_aspect("equal", adjustable="box")
        self._axis.grid(True, linestyle=":", linewidth=0.8)
        self._axis.set_xlabel("East / right (m)")
        self._axis.set_ylabel("North / forward (m)")
        self._axis.set_title("Crazyflie live 2D tracking - NED view")
        self._axis.legend(loc="upper right")
        self._status_text = self._axis.text(
            0.02,
            0.98,
            "waiting for log data...",
            transform=self._axis.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
        )
        self._figure.tight_layout()
        self._figure.show()

    def _attitude_log_callback(self, _timestamp, data, _logconf):
        with self._lock:
            self._attitude = {
                "roll_deg": data["stabilizer.roll"],
                "pitch_deg": data["stabilizer.pitch"],
                "yaw_deg": data["stabilizer.yaw"],
            }

    def _position_log_callback(self, _timestamp, data, _logconf):
        actual_x = data["stateEstimate.x"]
        actual_y = data["stateEstimate.y"]
        actual_z = data["stateEstimate.z"]

        with self._lock:
            target = self._current_target_locked()
            attitude = dict(self._attitude)

        target_n, target_e, target_d = crazyflie_to_ned(
            target["x"], target["y"], target["z"]
        )
        actual_n, actual_e, actual_d = crazyflie_to_ned(actual_x, actual_y, actual_z)
        error_x = target["x"] - actual_x
        error_y = target["y"] - actual_y
        error_z = target["z"] - actual_z
        error_xy = (error_x ** 2 + error_y ** 2) ** 0.5
        error_3d = (error_xy ** 2 + error_z ** 2) ** 0.5
        row = {
            "time_s": f"{time.time() - self._start_time:.3f}",
            "phase": target["phase"],
            "target_x_m": f"{target['x']:.4f}",
            "target_y_m": f"{target['y']:.4f}",
            "target_z_m": f"{target['z']:.4f}",
            "target_n_m": f"{target_n:.4f}",
            "target_e_m": f"{target_e:.4f}",
            "target_d_m": f"{target_d:.4f}",
            "actual_x_m": f"{actual_x:.4f}",
            "actual_y_m": f"{actual_y:.4f}",
            "actual_z_m": f"{actual_z:.4f}",
            "actual_n_m": f"{actual_n:.4f}",
            "actual_e_m": f"{actual_e:.4f}",
            "actual_d_m": f"{actual_d:.4f}",
            "error_x_m": f"{error_x:.4f}",
            "error_y_m": f"{error_y:.4f}",
            "error_z_m": f"{error_z:.4f}",
            "error_xy_m": f"{error_xy:.4f}",
            "error_3d_m": f"{error_3d:.4f}",
            "roll_deg": f"{attitude['roll_deg']:.3f}",
            "pitch_deg": f"{attitude['pitch_deg']:.3f}",
            "yaw_deg": f"{attitude['yaw_deg']:.3f}",
            "vbat_v": f"{data['pm.vbat']:.3f}",
        }

        with self._lock:
            self._latest = row
            writer = self._writer

        if writer is not None:
            writer.writerow(row)

    def _current_target_locked(self):
        target = self._target
        start_time = target["start_time"]
        duration_s = target["duration_s"]
        if start_time is None or duration_s <= 0.0:
            ratio = 1.0
        else:
            ratio = (time.time() - start_time) / duration_s
            ratio = min(1.0, max(0.0, ratio))

        return {
            "x": target["start_x"] + (target["end_x"] - target["start_x"]) * ratio,
            "y": target["start_y"] + (target["end_y"] - target["start_y"]) * ratio,
            "z": target["start_z"] + (target["end_z"] - target["start_z"]) * ratio,
            "phase": target["phase"],
        }


def sleep_with_plot(monitor, duration_s):
    end_time = time.time() + duration_s
    while time.time() < end_time:
        monitor.update_plot()
        time.sleep(0.05)
    monitor.update_plot()


def emergency_stop(cf):
    """Stop the current trajectory and force motor stop."""
    if cf is None:
        return

    for _ in range(3):
        try:
            cf.high_level_commander.stop()
        except Exception:
            pass

        try:
            cf.commander.send_stop_setpoint()
        except Exception:
            pass

        try:
            cf.platform.send_arming_request(False)
        except Exception:
            pass

        time.sleep(0.05)


def signal_handler(signum, _frame):
    print(f"\nSignal {signum} received. Emergency stop.")
    emergency_stop(_active_cf)
    sys.exit(128 + signum)


def reset_estimator(cf):
    print("Estimator reset: current box top is z=0.")
    cf.param.set_value("stabilizer.estimator", "2")
    time.sleep(0.2)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)


def setup_high_level_commander(cf):
    cf.param.set_value("commander.enHighLevel", "1")
    time.sleep(0.2)

    try:
        cf.platform.send_arming_request(True)
        time.sleep(0.5)
    except Exception:
        print("Arming request is not supported by this firmware; continuing.")


def fly_square(
    scf,
    width_cm,
    height_cm,
    flight_height_cm,
    speed_cm_s,
    log_dir,
    plot_enabled,
    frame,
):
    global _active_monitor

    cf = scf.cf
    commander = cf.high_level_commander

    width_m = cm_to_m(width_cm)
    height_m = cm_to_m(height_cm)
    flight_height_m = cm_to_m(flight_height_cm)
    speed_m_s = cm_to_m(speed_cm_s)

    x_duration = clamp_duration(width_m, speed_m_s)
    y_duration = clamp_duration(height_m, speed_m_s)

    print("=" * 60)
    print("Flight plan")
    print(f"- Takeoff/landing surface: {BOX_HEIGHT_CM:.0f} cm box top")
    print(f"- Rectangle: {width_cm:.1f} cm x {height_cm:.1f} cm")
    print(f"- Flight height: {flight_height_cm:.1f} cm above the box")
    print(f"- Horizontal speed: {speed_cm_s:.1f} cm/s")
    print(f"- Command frame: {frame.upper()}")
    print("=" * 60)

    reset_estimator(cf)
    setup_high_level_commander(cf)

    with FlightMonitor(scf, log_dir, plot_enabled, width_m, height_m, frame) as monitor:
        _active_monitor = monitor

        print("Takeoff.")
        current_position = ned_to_crazyflie(0.0, 0.0, 0.0)
        hover_position = ned_to_crazyflie(0.0, 0.0, -flight_height_m)
        monitor.start_segment(
            current_position,
            hover_position,
            TAKEOFF_DURATION_S,
            "takeoff",
        )
        commander.takeoff(flight_height_m, TAKEOFF_DURATION_S, yaw=0.0)
        sleep_with_plot(monitor, TAKEOFF_DURATION_S)
        monitor.set_target(*hover_position, phase="hover")
        sleep_with_plot(monitor, HOVER_TIME_S)
        current_position = hover_position

        # Commands below are expressed in NED:
        # N is forward, E is right, D is down. They are converted before
        # sending to the Crazyflie high-level commander.
        ned_corners = [
            (width_m, 0.0, -flight_height_m, x_duration, "north / forward"),
            (width_m, height_m, -flight_height_m, y_duration, "east / right"),
            (0.0, height_m, -flight_height_m, x_duration, "south / back"),
            (0.0, 0.0, -flight_height_m, y_duration, "west / return"),
        ]

        for north, east, down, duration_s, label in ned_corners:
            next_position = ned_to_crazyflie(north, east, down)
            monitor.start_segment(current_position, next_position, duration_s, label)
            x, y, z = next_position
            print(
                f"Move {label}: N={north:.2f} m, E={east:.2f} m, D={down:.2f} m "
                f"-> CF x={x:.2f}, y={y:.2f}, z={z:.2f}"
            )
            commander.go_to(x, y, z, 0.0, duration_s, relative=False, linear=True)
            sleep_with_plot(monitor, duration_s)
            monitor.set_target(*next_position, phase=f"{label} hover")
            sleep_with_plot(monitor, HOVER_TIME_S)
            current_position = next_position

        landing_position = ned_to_crazyflie(0.0, 0.0, 0.0)
        monitor.start_segment(
            current_position,
            landing_position,
            LANDING_DURATION_S,
            "landing",
        )
        print("Land on the original box.")
        commander.land(0.0, LANDING_DURATION_S, yaw=0.0)
        sleep_with_plot(monitor, LANDING_DURATION_S)
        monitor.set_target(*landing_position, phase="landed")
        sleep_with_plot(monitor, 1.0)
        print(f"Saved flight log: {monitor.path}")
        _active_monitor = None


def positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fly a rectangular trajectory and land on the takeoff box."
    )
    parser.add_argument("--uri", default=URI, help="Crazyflie radio URI")
    parser.add_argument("--width-cm", type=positive_float, default=DEFAULT_WIDTH_CM)
    parser.add_argument("--height-cm", type=positive_float, default=DEFAULT_HEIGHT_CM)
    parser.add_argument(
        "--flight-height-cm",
        type=positive_float,
        default=DEFAULT_FLIGHT_HEIGHT_CM,
        help="Height above the 15 cm box top",
    )
    parser.add_argument("--speed-cm-s", type=positive_float, default=DEFAULT_SPEED_CM_S)
    parser.add_argument(
        "--frame",
        choices=["ned"],
        default="ned",
        help="Command frame. NED means x=north/forward, y=east/right, z=down",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory where CSV flight logs are saved",
    )
    parser.add_argument(
        "--no-grid",
        "--no-plot",
        action="store_true",
        dest="no_plot",
        help="Disable the live matplotlib tracking window",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the safety confirmation prompt",
    )
    return parser.parse_args()


def confirm_safety(args):
    print("=" * 60)
    print("Safety check")
    print("- Propellers are attached correctly and the flight area is clear.")
    print("- The Crazyflie starts on the 15 cm box and the box is stable.")
    print("- The original box position is clear for landing.")
    print("- Press Ctrl+C at any time to send an emergency motor stop.")
    print("=" * 60)
    print(
        f"Plan: {args.width_cm:.1f} cm north x {args.height_cm:.1f} cm east rectangle, "
        f"{args.flight_height_cm:.1f} cm above the box, frame={args.frame.upper()}."
    )

    if args.yes:
        return True

    answer = input("Type 'yes' to start flight: ")
    return answer.strip().lower() == "yes"


def main():
    global _active_cf, _active_monitor

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()
    if not confirm_safety(args):
        print("Flight cancelled.")
        return

    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache="./cache")

    try:
        print(f"Connecting to {args.uri}...")
        with SyncCrazyflie(args.uri, cf=cf) as scf:
            _active_cf = scf.cf
            print("Connected.")
            fly_square(
                scf,
                args.width_cm,
                args.height_cm,
                args.flight_height_cm,
                args.speed_cm_s,
                args.log_dir,
                not args.no_plot,
                args.frame,
            )
            print("Flight completed.")
    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Emergency stop.")
    finally:
        print("Final motor stop / disarm.")
        if _active_monitor is not None:
            _active_monitor.stop()
            _active_monitor = None
        emergency_stop(_active_cf or cf)
        _active_cf = None


if __name__ == "__main__":
    main()
