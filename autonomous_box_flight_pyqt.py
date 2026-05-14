"""
Crazyflie autonomous box-to-box flight with a PyQt GUI.

Assumptions and limits:
- The start and goal boxes are about 12 cm above the floor.
- The goal position is unknown.
- The current goal detector is a placeholder. Replace is_goal_detected() with
  camera, AprilTag, color, or floor/box detection logic later.
- Range variables follow the Multi-ranger/Flow deck names used by Bitcraze
  firmware: range.front/back/left/right/up/zrange, in millimeters.
"""
import argparse
import logging
import math
import signal
import sys
import threading
import time
import warnings

try:
    from PyQt5 import QtCore, QtWidgets

    PYQT_AVAILABLE = True
except ImportError:
    QtCore = None
    QtWidgets = None
    PYQT_AVAILABLE = False

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E5")

# Start/goal boxes are 12 cm high. This target is height above the estimator's
# takeoff reference. Put the Crazyflie on the start box before pressing Start.
TARGET_HEIGHT = 0.40
BOX_HEIGHT = 0.12

FORWARD_SPEED = 0.15
SIDE_SPEED = 0.18
BACK_SPEED = 0.12

FRONT_THRESHOLD = 0.45
SIDE_CLEAR_THRESHOLD = 0.50
TOO_CLOSE_FRONT_THRESHOLD = 0.18
TOO_CLOSE_SIDE_THRESHOLD = 0.12

AVOIDANCE_TIME = 1.0
MAX_FLIGHT_TIME = 45.0
CONTROL_DT = 0.05
LOG_PERIOD_MS = 50
RANGE_FILTER_ALPHA = 0.35
MAX_VELOCITY_STEP = 0.025

GOAL_DETECTION_MIN_TIME = 5.0
GOAL_ZRANGE_TOLERANCE = 0.06
GOAL_STABLE_COUNT = 8
GOAL_APPROACH_HEIGHT_REDUCTION = 0.08

# Hover commands do not expose a direct motor-thrust cap. This software guard
# limits sudden upward height commands, which is the safer way to keep height
# hold while preventing a jump when zrange changes over a box.
MAX_HEIGHT_COMMAND = TARGET_HEIGHT
MAX_HEIGHT_STEP_UP = 0.01

TAKEOFF_STEP_M = 0.02
TAKEOFF_STEP_S = 0.08
LANDING_STEP_M = 0.02
LANDING_STEP_S = 0.10
GOAL_LANDING_STEP_M = 0.005
GOAL_LANDING_STEP_S = 0.12
GOAL_LANDING_FORWARD_SPEED = 0.03
GOAL_LANDING_SIDE_SPEED_LIMIT = 0.04
OUT_OF_RANGE_MM = 4000

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The supervisor subsystem requires CRTP protocol version.*",
    category=UserWarning,
)

_active_controller = None
_last_height_command = 0.0
_last_vx_command = 0.0
_last_vy_command = 0.0


class RangeReader:
    """Reads Multi-ranger and Flow deck range data into a thread-safe snapshot."""

    def __init__(self, cf):
        self._cf = cf
        self._lock = threading.Lock()
        self._latest = {
            "front": None,
            "left": None,
            "right": None,
            "back": None,
            "up": None,
            "zrange": None,
            "height": None,
            "vbat": None,
        }
        self._log_config = None

    def __enter__(self):
        log_config = LogConfig(name="BoxFlightRanges", period_in_ms=LOG_PERIOD_MS)
        log_config.add_variable("range.front", "uint16_t")
        log_config.add_variable("range.left", "uint16_t")
        log_config.add_variable("range.right", "uint16_t")
        log_config.add_variable("range.back", "uint16_t")
        log_config.add_variable("range.up", "uint16_t")
        log_config.add_variable("range.zrange", "uint16_t")
        log_config.add_variable("pm.vbat", "float")

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
        raw_values = {
            "front": mm_to_m(data.get("range.front")),
            "left": mm_to_m(data.get("range.left")),
            "right": mm_to_m(data.get("range.right")),
            "back": mm_to_m(data.get("range.back")),
            "up": mm_to_m(data.get("range.up")),
            "zrange": mm_to_m(data.get("range.zrange")),
            "vbat": data.get("pm.vbat"),
        }
        with self._lock:
            filtered = dict(self._latest)
            for key, raw_value in raw_values.items():
                if key == "vbat":
                    filtered[key] = raw_value
                    continue

                previous = filtered.get(key)
                if raw_value is None:
                    filtered[key] = previous
                elif previous is None:
                    filtered[key] = raw_value
                else:
                    filtered[key] = (
                        RANGE_FILTER_ALPHA * raw_value
                        + (1.0 - RANGE_FILTER_ALPHA) * previous
                    )
            filtered["height"] = filtered["zrange"]
            self._latest = filtered


def mm_to_m(value_mm):
    """Convert firmware millimeters to meters, treating invalid/out values as None."""
    if value_mm is None:
        return None
    try:
        value_mm = float(value_mm)
    except (TypeError, ValueError):
        return None
    if value_mm <= 0 or value_mm >= OUT_OF_RANGE_MM or math.isinf(value_mm):
        return None
    return value_mm / 1000.0


def fmt_distance(value_m):
    if value_m is None or math.isinf(value_m):
        return "OUT"
    return f"{value_m:.2f} m"


def battery_percent(vbat):
    if vbat is None:
        return 0
    return int(max(0.0, min(100.0, (float(vbat) - 3.2) / (4.2 - 3.2) * 100.0)))


def battery_level_name(vbat):
    if vbat is None:
        return "unknown"
    if vbat < 3.4:
        return "critical"
    if vbat < 3.6:
        return "low"
    return "good"


def fmt_battery(vbat):
    if vbat is None:
        return "Battery --"
    return f"Battery {vbat:.2f} V  ·  {battery_percent(vbat)}%"


def finite_or(value, fallback):
    if value is None or math.isinf(value):
        return fallback
    return value


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


def setup_estimator_and_arm(cf):
    """Use the Kalman estimator and reset the origin on the start box."""
    cf.param.set_value("stabilizer.estimator", "2")
    time.sleep(0.2)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)
    send_arming_request(cf, True)
    time.sleep(0.5)


def send_velocity(cf, vx, vy, target_height):
    """Velocity command in Crazyflie body frame with height hold."""
    cf.commander.send_hover_setpoint(vx, vy, 0.0, target_height)


def reset_height_limiter(initial_height=0.0):
    global _last_height_command
    _last_height_command = initial_height


def reset_velocity_limiter(initial_vx=0.0, initial_vy=0.0):
    global _last_vx_command, _last_vy_command
    _last_vx_command = initial_vx
    _last_vy_command = initial_vy


def limited_velocity_command(requested_vx, requested_vy):
    """Limit velocity step changes to reduce jerky lateral/forward motion."""
    global _last_vx_command, _last_vy_command
    vx_delta = max(-MAX_VELOCITY_STEP, min(MAX_VELOCITY_STEP, requested_vx - _last_vx_command))
    vy_delta = max(-MAX_VELOCITY_STEP, min(MAX_VELOCITY_STEP, requested_vy - _last_vy_command))
    _last_vx_command += vx_delta
    _last_vy_command += vy_delta
    return _last_vx_command, _last_vy_command


def limited_height_command(requested_height, max_height=MAX_HEIGHT_COMMAND):
    """
    Clamp upward height changes to reduce sudden thrust increases.

    This is not a raw motor-thrust limit. It is a height-command limiter for
    send_hover_setpoint(), which preserves Crazyflie's internal height hold.
    """
    global _last_height_command
    capped_height = min(requested_height, max_height)
    if capped_height > _last_height_command:
        capped_height = min(capped_height, _last_height_command + MAX_HEIGHT_STEP_UP)
    _last_height_command = capped_height
    return capped_height


def send_velocity_limited(cf, vx, vy, requested_height, max_height=MAX_HEIGHT_COMMAND):
    height = limited_height_command(requested_height, max_height)
    limited_vx, limited_vy = limited_velocity_command(vx, vy)
    send_velocity(cf, limited_vx, limited_vy, height)
    return height, limited_vx, limited_vy


def takeoff(cf, target_height=TARGET_HEIGHT, stop_event=None, status_cb=None):
    """Smoothly climb from the start box to target_height."""
    reset_height_limiter(0.0)
    reset_velocity_limiter(0.0, 0.0)
    height = 0.0
    while height < target_height:
        if stop_event is not None and stop_event.is_set():
            break
        height = min(target_height, height + TAKEOFF_STEP_M)
        send_velocity(cf, 0.0, 0.0, height)
        if status_cb:
            status_cb("takeoff", {"height_command": height})
        time.sleep(TAKEOFF_STEP_S)
    reset_height_limiter(height)
    reset_velocity_limiter(0.0, 0.0)
    time.sleep(0.5)


def land(cf, start_height=TARGET_HEIGHT, status_cb=None):
    """Slow landing for a 12 cm goal box; final motor stop happens near 5 cm."""
    reset_velocity_limiter(0.0, 0.0)
    height = start_height
    while height > 0.05:
        height = max(0.05, height - LANDING_STEP_M)
        send_velocity(cf, 0.0, 0.0, height)
        if status_cb:
            status_cb("landing", {"height_command": height})
        time.sleep(LANDING_STEP_S)
    emergency_stop(cf)


def land_on_goal_box(cf, start_height=TARGET_HEIGHT, status_cb=None):
    """
    Gentle goal-box landing.

    After detecting the box, keep a tiny forward crawl while lowering the height
    command in small steps. This avoids the abrupt "detected -> vertical drop"
    behavior and gives the drone time to settle over the box.
    """
    reset_height_limiter(start_height)
    reset_velocity_limiter(0.0, 0.0)
    height = start_height
    while height > 0.04:
        height = max(0.04, height - GOAL_LANDING_STEP_M)
        commanded_height, commanded_vx, commanded_vy = send_velocity_limited(
            cf,
            GOAL_LANDING_FORWARD_SPEED,
            0.0,
            height,
            start_height,
        )
        if status_cb:
            status_cb(
                    "goal landing: forward/slow descent",
                {
                    "height_command": commanded_height,
                    "vx": commanded_vx,
                    "vy": commanded_vy,
                },
            )
        time.sleep(GOAL_LANDING_STEP_S)

    for _ in range(8):
        send_velocity(cf, 0.0, 0.0, 0.04)
        time.sleep(CONTROL_DT)
    emergency_stop(cf)


def get_sensor_data(range_reader):
    """Return front/left/right/back/up/zrange/height in meters."""
    data = range_reader.snapshot()
    for key in ("front", "left", "right", "back", "up", "zrange", "height", "vbat"):
        data.setdefault(key, None)
    return data


def is_obstacle_ahead(sensor_data, threshold=FRONT_THRESHOLD):
    front = sensor_data.get("front")
    return front is not None and front < threshold


def choose_avoidance_direction(
    sensor_data,
    side_clear_threshold=SIDE_CLEAR_THRESHOLD,
):
    """
    Choose a side according to the project rule.

    If both sides are clear, move right. Otherwise move toward the side with the
    smaller measured distance, exactly as requested. If your physical setup
    should move toward the more open side, swap the comparison here.
    """
    left = finite_or(sensor_data.get("left"), float("inf"))
    right = finite_or(sensor_data.get("right"), float("inf"))

    if left > side_clear_threshold and right > side_clear_threshold:
        return "right"
    if left < right:
        return "left"
    return "right"


def reset_goal_detector():
    is_goal_detected.seen_floor = False
    is_goal_detected.stable_count = 0


def is_goal_box_candidate(sensor_data, target_height=TARGET_HEIGHT):
    """
    Detect the 12 cm box by the bottom range pattern.

    At target_height above the start box, range.zrange should be around
    target_height while flying over another same-height box. Over the floor it
    should be roughly target_height + BOX_HEIGHT. We require seeing floor first
    to avoid detecting the start box immediately after takeoff.
    """
    zrange = sensor_data.get("zrange")
    if zrange is None:
        return False
    return abs(zrange - target_height) <= GOAL_ZRANGE_TOLERANCE


def is_goal_detected(sensor_data, elapsed_s):
    """
    Goal-box detector using downward range as a first practical placeholder.

    Replace this with a real detector later:
    - camera / AprilTag
    - color marker
    - optical-flow map
    - a better bottom-distance classifier
    """
    if elapsed_s < GOAL_DETECTION_MIN_TIME:
        return False

    zrange = sensor_data.get("zrange")
    if zrange is None:
        is_goal_detected.stable_count = 0
        return False

    floor_threshold = TARGET_HEIGHT + BOX_HEIGHT * 0.45
    if zrange > floor_threshold:
        is_goal_detected.seen_floor = True

    if not is_goal_detected.seen_floor:
        return False

    if is_goal_box_candidate(sensor_data, TARGET_HEIGHT):
        is_goal_detected.stable_count += 1
    else:
        is_goal_detected.stable_count = 0

    return is_goal_detected.stable_count >= GOAL_STABLE_COUNT


reset_goal_detector()


def compute_velocity_command(sensor_data, state):
    """
    Decide vx/vy from current sensor data and avoidance state.

    vx > 0: forward
    vx < 0: backward
    vy > 0: left
    vy < 0: right
    """
    now = time.time()
    front = sensor_data.get("front")
    left = sensor_data.get("left")
    right = sensor_data.get("right")

    if front is not None and front < TOO_CLOSE_FRONT_THRESHOLD:
        state["avoid_until"] = now + 0.4
        state["avoid_direction"] = "back"
        return -BACK_SPEED, 0.0, "too close: backing"

    if left is not None and left < TOO_CLOSE_SIDE_THRESHOLD:
        state["avoid_until"] = now + 0.4
        state["avoid_direction"] = "right"
        return 0.0, -SIDE_SPEED, "left too close: right"

    if right is not None and right < TOO_CLOSE_SIDE_THRESHOLD:
        state["avoid_until"] = now + 0.4
        state["avoid_direction"] = "left"
        return 0.0, SIDE_SPEED, "right too close: left"

    if is_obstacle_ahead(sensor_data, FRONT_THRESHOLD):
        direction = choose_avoidance_direction(sensor_data, SIDE_CLEAR_THRESHOLD)
        state["avoid_until"] = now + AVOIDANCE_TIME
        state["avoid_direction"] = direction

    if now < state.get("avoid_until", 0.0):
        direction = state.get("avoid_direction")
        if direction == "left":
            return 0.02, SIDE_SPEED, "avoid left"
        if direction == "right":
            return 0.02, -SIDE_SPEED, "avoid right"
        if direction == "back":
            return -BACK_SPEED, 0.0, "avoid back"

    state["avoid_direction"] = None
    return FORWARD_SPEED, 0.0, "forward"


def main_control_loop(
    cf,
    range_reader,
    stop_event,
    settings,
    status_cb=None,
):
    """
    Fly forward, avoid random obstacles, and land on goal detection or timeout.
    """
    start_time = time.time()
    avoidance_state = {"avoid_until": 0.0, "avoid_direction": None}
    target_height = settings["target_height"]
    max_flight_time = settings["max_flight_time"]
    max_height_command = settings["max_height_command"]
    reset_goal_detector()
    reset_height_limiter(target_height)
    reset_velocity_limiter(0.0, 0.0)

    while not stop_event.is_set():
        elapsed = time.time() - start_time
        sensor_data = get_sensor_data(range_reader)
        goal_candidate = is_goal_box_candidate(sensor_data, target_height)

        if is_goal_detected(sensor_data, elapsed):
            if status_cb:
                status_cb("goal detected", sensor_data)
            return "goal"

        if elapsed >= max_flight_time:
            if status_cb:
                status_cb("max flight time reached", sensor_data)
            return "timeout"

        vx, vy, mode = compute_velocity_command(sensor_data, avoidance_state)
        height_request = target_height
        if goal_candidate:
            height_request = max(0.20, target_height - GOAL_APPROACH_HEIGHT_REDUCTION)
            vx *= 0.35
            vy *= 0.35
            mode = "goal candidate: slow/low"

        height_command, commanded_vx, commanded_vy = send_velocity_limited(
            cf,
            vx,
            vy,
            height_request,
            max_height_command,
        )

        if status_cb:
            payload = dict(sensor_data)
            payload.update(
                {
                    "elapsed": elapsed,
                    "vx": commanded_vx,
                    "vy": commanded_vy,
                    "requested_vx": vx,
                    "requested_vy": vy,
                    "height_command": height_command,
                    "goal_candidate": goal_candidate,
                    "goal_stable_count": is_goal_detected.stable_count,
                }
            )
            status_cb(mode, payload)

        time.sleep(CONTROL_DT)

    return "stopped"


class FlightController(QtCore.QThread if PYQT_AVAILABLE else object):
    if PYQT_AVAILABLE:
        status_changed = QtCore.pyqtSignal(str, dict)
        running_changed = QtCore.pyqtSignal(bool)

    def __init__(self, settings):
        if PYQT_AVAILABLE:
            super().__init__()
        self.settings = settings
        self.stop_event = threading.Event()
        self.emergency_event = threading.Event()
        self.cf = None

    def request_land(self):
        self.stop_event.set()

    def request_emergency_stop(self):
        self.emergency_event.set()
        self.stop_event.set()
        emergency_stop(self.cf)

    def publish(self, status, payload=None):
        if payload is None:
            payload = {}
        if PYQT_AVAILABLE:
            self.status_changed.emit(status, payload)

    def run(self):
        if PYQT_AVAILABLE:
            self.running_changed.emit(True)

        cf = Crazyflie(rw_cache="./cache")
        self.cf = cf

        try:
            self.publish("initializing radio")
            cflib.crtp.init_drivers()
            self.publish(f"connecting: {self.settings['uri']}")

            with SyncCrazyflie(self.settings["uri"], cf=cf) as scf:
                self.cf = scf.cf
                self.publish("connected: resetting estimator")
                setup_estimator_and_arm(scf.cf)

                with RangeReader(scf.cf) as range_reader:
                    self.publish("takeoff")
                    takeoff(
                        scf.cf,
                        self.settings["target_height"],
                        self.stop_event,
                        self.publish,
                    )

                    if self.stop_event.is_set():
                        result = "stopped before control loop"
                    else:
                        result = main_control_loop(
                            scf.cf,
                            range_reader,
                            self.stop_event,
                            self.settings,
                            self.publish,
                        )

                    if self.emergency_event.is_set():
                        emergency_stop(scf.cf)
                        self.publish("emergency stopped")
                    elif result == "goal":
                        self.publish("goal landing")
                        land_on_goal_box(
                            scf.cf,
                            self.settings["target_height"],
                            self.publish,
                        )
                        self.publish("landed on goal")
                    else:
                        self.publish(f"landing: {result}")
                        land(scf.cf, self.settings["target_height"], self.publish)
                        self.publish("landed")

        except Exception as exc:
            self.publish(f"error: {exc}")
            emergency_stop(self.cf or cf)
        finally:
            self.cf = None
            if PYQT_AVAILABLE:
                self.running_changed.emit(False)


if PYQT_AVAILABLE:

    class MainWindow(QtWidgets.QWidget):
        def __init__(self, args):
            super().__init__()
            self.setWindowTitle("Crazyflie Autonomous Box Flight")
            self.setStyleSheet(
                """
                QWidget {
                    background: #f6f7fb;
                    color: #1f2937;
                    font-size: 13px;
                }
                QGroupBox {
                    background: #ffffff;
                    border: 1px solid #d9dee8;
                    border-radius: 8px;
                    margin-top: 12px;
                    padding: 12px;
                    font-weight: 600;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 5px;
                }
                QLineEdit, QDoubleSpinBox {
                    background: #ffffff;
                    border: 1px solid #cfd6e3;
                    border-radius: 6px;
                    padding: 5px 7px;
                }
                QPushButton {
                    background: #2563eb;
                    color: #ffffff;
                    border: none;
                    border-radius: 7px;
                    padding: 9px 12px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #1d4ed8;
                }
                QPushButton:disabled {
                    background: #aeb7c7;
                }
                QPushButton#dangerButton {
                    background: #dc2626;
                }
                QPushButton#dangerButton:hover {
                    background: #b91c1c;
                }
                QLabel#statusLabel {
                    background: #111827;
                    color: #ffffff;
                    border-radius: 8px;
                    padding: 10px 12px;
                    font-weight: 600;
                }
                QLabel#metricLabel {
                    color: #4b5563;
                    padding: 2px 0;
                }
                """
            )
            self.controller = None

            self.uri_edit = QtWidgets.QLineEdit(args.uri)
            self.height_spin = self._spin(0.20, 0.80, args.target_height, 0.01)
            self.max_height_spin = self._spin(0.20, 0.80, args.target_height, 0.01)
            self.forward_spin = self._spin(0.05, 0.30, FORWARD_SPEED, 0.01)
            self.side_spin = self._spin(0.05, 0.35, SIDE_SPEED, 0.01)
            self.front_spin = self._spin(0.20, 1.00, FRONT_THRESHOLD, 0.01)
            self.side_clear_spin = self._spin(0.20, 1.20, SIDE_CLEAR_THRESHOLD, 0.01)
            self.max_time_spin = self._spin(5.0, 180.0, args.max_flight_time, 1.0)

            self.status_label = QtWidgets.QLabel("Ready")
            self.status_label.setObjectName("statusLabel")
            self.velocity_label = QtWidgets.QLabel("vx +0.00 m/s, vy +0.00 m/s")
            self.velocity_label.setObjectName("metricLabel")
            self.height_label = QtWidgets.QLabel("height command 0.00 m")
            self.height_label.setObjectName("metricLabel")
            self.goal_label = QtWidgets.QLabel("goal stable 0")
            self.goal_label.setObjectName("metricLabel")
            self.battery_label = QtWidgets.QLabel("Battery --")
            self.battery_label.setObjectName("metricLabel")
            self.battery_bar = QtWidgets.QProgressBar()
            self.battery_bar.setRange(0, 100)
            self.battery_bar.setValue(0)
            self.battery_bar.setTextVisible(False)
            self._set_battery_style(None)
            self.range_labels = {
                key: QtWidgets.QLabel("OUT")
                for key in ("front", "left", "right", "back", "up", "zrange")
            }

            self.start_button = QtWidgets.QPushButton("Start Autonomous Flight")
            self.land_button = QtWidgets.QPushButton("Land")
            self.stop_button = QtWidgets.QPushButton("Emergency Stop")
            self.stop_button.setObjectName("dangerButton")
            self.land_button.setEnabled(False)
            self.stop_button.setEnabled(False)

            self.start_button.clicked.connect(self.start_flight)
            self.land_button.clicked.connect(self.land)
            self.stop_button.clicked.connect(self.emergency_stop)

            self._build_layout()

        def _spin(self, low, high, value, step):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setValue(value)
            spin.setSingleStep(step)
            spin.setDecimals(2)
            return spin

        def _set_battery_style(self, vbat):
            level = battery_level_name(vbat)
            color = {
                "good": "#16a34a",
                "low": "#f59e0b",
                "critical": "#dc2626",
                "unknown": "#9ca3af",
            }[level]
            self.battery_bar.setStyleSheet(
                f"""
                QProgressBar {{
                    background: #e5e7eb;
                    border: none;
                    border-radius: 5px;
                    height: 10px;
                    min-width: 150px;
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 5px;
                }}
                """
            )

        def _build_layout(self):
            root = QtWidgets.QVBoxLayout(self)
            root.setSpacing(10)

            header = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel("Autonomous Box Flight")
            title.setStyleSheet("font-size: 20px; font-weight: 700; color: #111827;")
            header.addWidget(title)
            header.addStretch(1)

            battery_panel = QtWidgets.QVBoxLayout()
            battery_panel.addWidget(self.battery_label)
            battery_panel.addWidget(self.battery_bar)
            header.addLayout(battery_panel)
            root.addLayout(header)

            settings = QtWidgets.QFormLayout()
            settings.addRow("URI", self.uri_edit)
            settings.addRow("Target height m", self.height_spin)
            settings.addRow("Max height command m", self.max_height_spin)
            settings.addRow("Forward speed m/s", self.forward_spin)
            settings.addRow("Side speed m/s", self.side_spin)
            settings.addRow("Front threshold m", self.front_spin)
            settings.addRow("Side clear threshold m", self.side_clear_spin)
            settings.addRow("Max flight time s", self.max_time_spin)

            settings_box = QtWidgets.QGroupBox("Settings")
            settings_box.setLayout(settings)
            root.addWidget(settings_box)

            ranges = QtWidgets.QGridLayout()
            for row, key in enumerate(("front", "left", "right", "back", "up", "zrange")):
                ranges.addWidget(QtWidgets.QLabel(key), row, 0)
                ranges.addWidget(self.range_labels[key], row, 1)

            ranges_box = QtWidgets.QGroupBox("Sensors")
            ranges_box.setLayout(ranges)
            root.addWidget(ranges_box)

            buttons = QtWidgets.QHBoxLayout()
            buttons.addWidget(self.start_button)
            buttons.addWidget(self.land_button)
            buttons.addWidget(self.stop_button)
            root.addLayout(buttons)

            root.addWidget(self.status_label)
            root.addWidget(self.velocity_label)
            root.addWidget(self.height_label)
            root.addWidget(self.goal_label)

        def current_settings(self):
            # Update module-level tuning values so the required helper functions
            # stay simple and easy to edit for experiments.
            global FORWARD_SPEED, SIDE_SPEED, FRONT_THRESHOLD, SIDE_CLEAR_THRESHOLD
            global TARGET_HEIGHT, MAX_HEIGHT_COMMAND
            TARGET_HEIGHT = self.height_spin.value()
            FORWARD_SPEED = self.forward_spin.value()
            SIDE_SPEED = self.side_spin.value()
            FRONT_THRESHOLD = self.front_spin.value()
            SIDE_CLEAR_THRESHOLD = self.side_clear_spin.value()
            MAX_HEIGHT_COMMAND = self.max_height_spin.value()

            return {
                "uri": self.uri_edit.text().strip(),
                "target_height": self.height_spin.value(),
                "max_height_command": self.max_height_spin.value(),
                "max_flight_time": self.max_time_spin.value(),
            }

        def start_flight(self):
            if self.controller is not None and self.controller.isRunning():
                return

            self.controller = FlightController(self.current_settings())
            global _active_controller
            _active_controller = self.controller
            self.controller.status_changed.connect(self.update_status)
            self.controller.running_changed.connect(self.update_running)
            self.controller.start()

        def land(self):
            if self.controller is not None:
                self.controller.request_land()
                self.status_label.setText("Landing requested")

        def emergency_stop(self):
            if self.controller is not None:
                self.controller.request_emergency_stop()
            self.status_label.setText("Emergency stop requested")

        def update_running(self, running):
            self.start_button.setEnabled(not running)
            self.land_button.setEnabled(running)
            self.stop_button.setEnabled(running)

        def update_status(self, status, payload):
            self.status_label.setText(status)
            for key, label in self.range_labels.items():
                if key in payload:
                    label.setText(fmt_distance(payload[key]))
            if "vx" in payload and "vy" in payload:
                self.velocity_label.setText(
                    f"vx {payload['vx']:+.2f} m/s, vy {payload['vy']:+.2f} m/s"
                )
            if "height_command" in payload:
                self.height_label.setText(
                    f"height command {payload['height_command']:.2f} m"
                )
            if "goal_stable_count" in payload:
                candidate = "candidate" if payload.get("goal_candidate") else "searching"
                self.goal_label.setText(
                    f"goal {candidate}, stable {payload['goal_stable_count']}/{GOAL_STABLE_COUNT}"
                )
            if "vbat" in payload:
                self.battery_label.setText(fmt_battery(payload["vbat"]))
                self.battery_bar.setValue(battery_percent(payload["vbat"]))
                self._set_battery_style(payload["vbat"])

        def closeEvent(self, event):
            if self.controller is not None and self.controller.isRunning():
                self.controller.request_emergency_stop()
                self.controller.wait(1000)
            event.accept()


def signal_handler(_signum, _frame):
    if _active_controller is not None:
        _active_controller.request_emergency_stop()


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyQt GUI for Crazyflie autonomous box-to-box obstacle flight."
    )
    parser.add_argument("--uri", default=URI, help="Crazyflie radio URI")
    parser.add_argument("--target-height", type=positive_float, default=TARGET_HEIGHT)
    parser.add_argument("--max-flight-time", type=positive_float, default=MAX_FLIGHT_TIME)
    return parser.parse_args()


def main():
    args = parse_args()

    if not PYQT_AVAILABLE:
        print("PyQt5 is not installed in this virtual environment.")
        print("Install it first, for example: pip install PyQt5")
        return 1

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args)
    window.resize(520, 520)
    window.show()

    global _active_controller
    _active_controller = window.controller
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
