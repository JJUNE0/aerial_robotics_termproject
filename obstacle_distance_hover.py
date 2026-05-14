"""
Crazyflie obstacle-distance hover GUI.

The Crazyflie hovers by default. When a Multi-ranger reading approaches the
target distance, the controller sends a stronger velocity command away from the
nearest obstacle while keeping the same hover height.
"""
import argparse
import logging
import queue
import signal
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import warnings

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E5")

DEFAULT_HOVER_HEIGHT_M = 0.40
DEFAULT_TARGET_DISTANCE_CM = 10.0
DEFAULT_REACTION_MARGIN_CM = 7.0
DEFAULT_MAX_SPEED_M_S = 0.35
DEFAULT_MIN_SPEED_M_S = 0.10
DEFAULT_GAIN = 8.0
LOG_PERIOD_MS = 50
CONTROL_PERIOD_S = 0.05
TAKEOFF_STEP_M = 0.02
TAKEOFF_STEP_S = 0.08
LANDING_STEP_M = 0.02
LANDING_STEP_S = 0.08
OUT_OF_RANGE_MM = 4000

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="cflib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The supervisor subsystem requires CRTP protocol version.*",
    category=UserWarning,
)

_controller = None


class RangeReader:
    """Thread-safe storage for the latest Multi-ranger measurements."""

    def __init__(self, cf):
        self._cf = cf
        self._lock = threading.Lock()
        self._latest = {
            "front": OUT_OF_RANGE_MM,
            "back": OUT_OF_RANGE_MM,
            "left": OUT_OF_RANGE_MM,
            "right": OUT_OF_RANGE_MM,
            "up": OUT_OF_RANGE_MM,
            "zrange": OUT_OF_RANGE_MM,
        }
        self._log_config = None

    def __enter__(self):
        log_config = LogConfig(name="ObstacleRanges", period_in_ms=LOG_PERIOD_MS)
        log_config.add_variable("range.front", "uint16_t")
        log_config.add_variable("range.back", "uint16_t")
        log_config.add_variable("range.left", "uint16_t")
        log_config.add_variable("range.right", "uint16_t")
        log_config.add_variable("range.up", "uint16_t")
        log_config.add_variable("range.zrange", "uint16_t")

        self._cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(self._log_callback)
        log_config.start()
        self._log_config = log_config
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        if self._log_config is None:
            return
        try:
            self._log_config.stop()
        except Exception:
            pass
        self._log_config = None

    def snapshot(self):
        with self._lock:
            return dict(self._latest)

    def _log_callback(self, _timestamp, data, _logconf):
        with self._lock:
            self._latest = {
                "front": data["range.front"],
                "back": data["range.back"],
                "left": data["range.left"],
                "right": data["range.right"],
                "up": data["range.up"],
                "zrange": data["range.zrange"],
            }


def cm_to_mm(value_cm):
    return value_cm * 10.0


def clamp(value, low, high):
    return max(low, min(high, value))


def valid_range_mm(value):
    return 0 < value < OUT_OF_RANGE_MM


def fmt_range(value_mm):
    if not valid_range_mm(value_mm):
        return "OUT"
    return f"{value_mm / 10.0:.1f} cm"


def emergency_stop(cf):
    if cf is None:
        return

    for _ in range(3):
        try:
            cf.commander.send_stop_setpoint()
        except Exception:
            pass
        send_arming_request(cf, False)
        time.sleep(0.05)


def send_arming_request(cf, do_arm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        try:
            cf.supervisor.send_arming_request(do_arm)
            return
        except Exception:
            pass

        try:
            cf.platform.send_arming_request(do_arm)
        except Exception:
            pass


def setup_controller(cf):
    cf.param.set_value("stabilizer.estimator", "2")
    time.sleep(0.2)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)

    send_arming_request(cf, True)
    time.sleep(0.5)


def send_hover(cf, vx_m_s, vy_m_s, height_m):
    # vx: +forward/-back, vy: +left/-right in the Crazyflie body frame.
    cf.commander.send_hover_setpoint(vx_m_s, vy_m_s, 0.0, height_m)


def takeoff(cf, target_height_m, stop_event):
    height_m = 0.0
    while height_m < target_height_m and not stop_event.is_set():
        height_m = min(target_height_m, height_m + TAKEOFF_STEP_M)
        send_hover(cf, 0.0, 0.0, height_m)
        time.sleep(TAKEOFF_STEP_S)
    time.sleep(0.5)


def land(cf, start_height_m):
    height_m = start_height_m
    while height_m > 0.05:
        height_m = max(0.05, height_m - LANDING_STEP_M)
        send_hover(cf, 0.0, 0.0, height_m)
        time.sleep(LANDING_STEP_S)
    emergency_stop(cf)


def avoidance_velocity(
    ranges,
    target_distance_cm,
    reaction_margin_cm,
    max_speed_m_s,
    min_speed_m_s,
    gain,
):
    target_mm = cm_to_mm(target_distance_cm)
    active_mm = cm_to_mm(target_distance_cm + reaction_margin_cm)
    vx = 0.0
    vy = 0.0
    active_directions = []

    direction_signs = {
        "front": (-1.0, 0.0),
        "back": (1.0, 0.0),
        "left": (0.0, -1.0),
        "right": (0.0, 1.0),
    }

    for direction, (x_sign, y_sign) in direction_signs.items():
        distance_mm = ranges[direction]
        if not valid_range_mm(distance_mm) or distance_mm >= active_mm:
            continue

        error_m = max((target_mm - distance_mm) / 1000.0, 0.0)
        prepush_m = max((active_mm - distance_mm) / 1000.0, 0.0) * 0.35
        speed_m_s = clamp(gain * (error_m + prepush_m), min_speed_m_s, max_speed_m_s)
        vx += x_sign * speed_m_s
        vy += y_sign * speed_m_s
        active_directions.append(direction)

    vx = clamp(vx, -max_speed_m_s, max_speed_m_s)
    vy = clamp(vy, -max_speed_m_s, max_speed_m_s)
    return vx, vy, active_directions


class ObstacleHoverController:
    def __init__(self, status_queue):
        self._queue = status_queue
        self._thread = None
        self._stop_event = threading.Event()
        self._emergency_event = threading.Event()
        self._cf = None
        self._hover_height_m = DEFAULT_HOVER_HEIGHT_M

    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, settings):
        if self.running():
            self._publish(status="Already flying.")
            return

        self._stop_event.clear()
        self._emergency_event.clear()
        self._hover_height_m = settings["hover_height_m"]
        self._thread = threading.Thread(
            target=self._flight_worker,
            args=(settings,),
            daemon=True,
        )
        self._thread.start()

    def land(self):
        self._publish(status="Landing requested.")
        self._stop_event.set()

    def emergency(self):
        self._publish(status="Emergency stop requested.")
        self._emergency_event.set()
        self._stop_event.set()
        emergency_stop(self._cf)

    def _publish(self, **data):
        self._queue.put(data)

    def _flight_worker(self, settings):
        cf = Crazyflie(rw_cache="./cache")

        try:
            self._publish(status="Initializing radio drivers...", running=True)
            cflib.crtp.init_drivers()
            self._publish(status=f"Connecting to {settings['uri']}...")

            with SyncCrazyflie(settings["uri"], cf=cf) as scf:
                self._cf = scf.cf
                self._publish(status="Connected. Resetting estimator...")
                setup_controller(scf.cf)

                with RangeReader(scf.cf) as ranges:
                    self._publish(status="Taking off...")
                    takeoff(scf.cf, settings["hover_height_m"], self._stop_event)

                    while not self._stop_event.is_set():
                        latest = ranges.snapshot()
                        vx, vy, active = avoidance_velocity(
                            latest,
                            settings["target_distance_cm"],
                            settings["reaction_margin_cm"],
                            settings["max_speed_m_s"],
                            settings["min_speed_m_s"],
                            settings["gain"],
                        )
                        send_hover(scf.cf, vx, vy, settings["hover_height_m"])
                        mode = "avoid: " + ",".join(active) if active else "hover"
                        self._publish(
                            status=mode,
                            ranges=latest,
                            vx=vx,
                            vy=vy,
                            running=True,
                        )
                        time.sleep(CONTROL_PERIOD_S)

                    if self._emergency_event.is_set():
                        emergency_stop(scf.cf)
                        self._publish(status="Emergency stopped.")
                    else:
                        self._publish(status="Landing...")
                        land(scf.cf, settings["hover_height_m"])
                        self._publish(status="Landed.")

        except Exception as exc:
            self._publish(status=f"Error: {exc}")
            emergency_stop(self._cf or cf)
        finally:
            self._cf = None
            self._publish(running=False)


class ObstacleHoverApp:
    def __init__(self, root, args):
        self.root = root
        self.root.title("Crazyflie Distance Hover")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_queue = queue.Queue()
        self.controller = ObstacleHoverController(self.status_queue)
        global _controller
        _controller = self.controller

        self.uri_var = tk.StringVar(value=args.uri)
        self.height_var = tk.StringVar(value=f"{args.hover_height_m:.2f}")
        self.target_var = tk.StringVar(value=f"{args.target_distance_cm:.1f}")
        self.margin_var = tk.StringVar(value=f"{args.reaction_margin_cm:.1f}")
        self.max_speed_var = tk.StringVar(value=f"{args.max_speed_m_s:.2f}")
        self.min_speed_var = tk.StringVar(value=f"{args.min_speed_m_s:.2f}")
        self.gain_var = tk.StringVar(value=f"{args.gain:.1f}")
        self.status_var = tk.StringVar(value="Ready.")
        self.velocity_var = tk.StringVar(value="vx +0.00 m/s   vy +0.00 m/s")
        self.range_vars = {
            "front": tk.StringVar(value="OUT"),
            "back": tk.StringVar(value="OUT"),
            "left": tk.StringVar(value="OUT"),
            "right": tk.StringVar(value="OUT"),
            "up": tk.StringVar(value="OUT"),
            "zrange": tk.StringVar(value="OUT"),
        }

        self._build()
        self._poll_queue()

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(main, text="Flight settings", padding=10)
        settings.grid(row=0, column=0, columnspan=2, sticky="ew")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        self._entry(settings, "URI", self.uri_var, 0, 0, width=34)
        self._entry(settings, "Hover height m", self.height_var, 1, 0)
        self._entry(settings, "Target distance cm", self.target_var, 1, 2)
        self._entry(settings, "Reaction margin cm", self.margin_var, 2, 0)
        self._entry(settings, "Max speed m/s", self.max_speed_var, 2, 2)
        self._entry(settings, "Min speed m/s", self.min_speed_var, 3, 0)
        self._entry(settings, "Gain", self.gain_var, 3, 2)

        ranges = ttk.LabelFrame(main, text="Multi-ranger", padding=10)
        ranges.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        for index, (label, key) in enumerate(
            [
                ("Front", "front"),
                ("Back", "back"),
                ("Left", "left"),
                ("Right", "right"),
                ("Up", "up"),
                ("Down", "zrange"),
            ]
        ):
            ttk.Label(ranges, text=label, width=8).grid(row=index, column=0, sticky="w")
            ttk.Label(ranges, textvariable=self.range_vars[key], width=12).grid(
                row=index, column=1, sticky="e", padx=(16, 0)
            )

        controls = ttk.LabelFrame(main, text="Control", padding=10)
        controls.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(controls, text="Start Hover", command=self._start)
        self.start_button.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.land_button = ttk.Button(controls, text="Land", command=self._land, state="disabled")
        self.land_button.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.stop_button = ttk.Button(
            controls,
            text="Emergency Stop",
            command=self._emergency,
            state="disabled",
        )
        self.stop_button.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Separator(controls).grid(row=2, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(controls, textvariable=self.velocity_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

    def _entry(self, parent, label, variable, row, column, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row,
            column=column + 1,
            sticky="ew",
            pady=4,
        )

    def _settings(self):
        settings = {
            "uri": self.uri_var.get().strip(),
            "hover_height_m": float(self.height_var.get()),
            "target_distance_cm": float(self.target_var.get()),
            "reaction_margin_cm": float(self.margin_var.get()),
            "max_speed_m_s": float(self.max_speed_var.get()),
            "min_speed_m_s": float(self.min_speed_var.get()),
            "gain": float(self.gain_var.get()),
        }

        if settings["hover_height_m"] <= 0.0:
            raise ValueError("Hover height must be positive.")
        if settings["target_distance_cm"] <= 0.0:
            raise ValueError("Target distance must be positive.")
        if settings["reaction_margin_cm"] < 0.0:
            raise ValueError("Reaction margin must be zero or positive.")
        if settings["max_speed_m_s"] <= 0.0 or settings["min_speed_m_s"] <= 0.0:
            raise ValueError("Speeds must be positive.")
        if settings["min_speed_m_s"] > settings["max_speed_m_s"]:
            raise ValueError("Min speed must be smaller than max speed.")
        if settings["gain"] <= 0.0:
            raise ValueError("Gain must be positive.")
        return settings

    def _start(self):
        try:
            settings = self._settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.start_button.configure(state="disabled")
        self.land_button.configure(state="normal")
        self.stop_button.configure(state="normal")
        self.controller.start(settings)

    def _land(self):
        self.land_button.configure(state="disabled")
        self.controller.land()

    def _emergency(self):
        self.land_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.controller.emergency()

    def _on_close(self):
        if self.controller.running():
            self.controller.emergency()
            self.root.after(300, self.root.destroy)
        else:
            self.root.destroy()

    def _poll_queue(self):
        try:
            while True:
                data = self.status_queue.get_nowait()
                self._apply_status(data)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _apply_status(self, data):
        if "status" in data:
            self.status_var.set(data["status"])
        if "ranges" in data:
            for key, value in data["ranges"].items():
                if key in self.range_vars:
                    self.range_vars[key].set(fmt_range(value))
        if "vx" in data and "vy" in data:
            self.velocity_var.set(f"vx {data['vx']:+.2f} m/s   vy {data['vy']:+.2f} m/s")
        if data.get("running") is False:
            self.start_button.configure(state="normal")
            self.land_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")


def signal_handler(_signum, _frame):
    if _controller is not None:
        _controller.emergency()


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value):
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a GUI for Crazyflie hover and distance keeping."
    )
    parser.add_argument("--uri", default=URI, help="Crazyflie radio URI")
    parser.add_argument("--hover-height-m", type=positive_float, default=DEFAULT_HOVER_HEIGHT_M)
    parser.add_argument(
        "--target-distance-cm",
        type=positive_float,
        default=DEFAULT_TARGET_DISTANCE_CM,
    )
    parser.add_argument(
        "--reaction-margin-cm",
        type=non_negative_float,
        default=DEFAULT_REACTION_MARGIN_CM,
    )
    parser.add_argument("--max-speed-m-s", type=positive_float, default=DEFAULT_MAX_SPEED_M_S)
    parser.add_argument("--min-speed-m-s", type=positive_float, default=DEFAULT_MIN_SPEED_M_S)
    parser.add_argument("--gain", type=positive_float, default=DEFAULT_GAIN)
    return parser.parse_args()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()
    root = tk.Tk()
    ObstacleHoverApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
