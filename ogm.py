"""
Crazyflie Gaussian-goal landing with concurrent Occupancy Grid Mapping (OGM).

Purpose
-------
Sequence per session:
  1. User presses Connect: SyncCrazyflie opens, RangePoseReader streams
     sensor + state data into the GUI. No motors run.
  2. User sets map W x H, start (sx, sy), goal (gx, gy) in *world* meters,
     and target flight height.
  3. User presses Start Goal Landing:
       takeoff -> choose a clear landing cell inside the goal Gaussian ->
       seek it while avoiding obstacles -> land.
  4. After landing, the GUI shows goal mean vs Kalman final pose error.
  5. While flying, Multi-ranger F/L/R/B beams are ray-cast into an
     occupancy grid and visualised in the GUI.

Assumptions
-----------
- Crazyflie has Flow deck (for stateEstimate.x/y) and Multi-ranger.
- Takeoff happens at world coordinate (sx, sy). i.e. user is expected
  to put the drone there before connecting.
- No yaw rotation during flight (holonomic motion).
"""

import argparse
import csv
import logging
import math
from pathlib import Path
import signal
import sys
import threading
import time
import warnings

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    PYQT_AVAILABLE = True
except ImportError:
    QtCore = None
    QtGui = None
    QtWidgets = None
    PYQT_AVAILABLE = False

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper


# ===========================================================================
# Configuration
# ===========================================================================
URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E5")

TARGET_HEIGHT = 0.40
SPEED_X = 0.3
SPEED_Y = 0.2
ARRIVAL_RADIUS = 0.10            # axis 도착 판정 반경

# Goal distribution / landing search
GOAL_SIGMA = 0.30                # goal point 주변 Gaussian 표준편차
GOAL_SEARCH_RADIUS = 0.75        # landing 후보를 찾을 최대 반경
LANDING_CLEAR_RADIUS = 0.18      # 착륙 후보 주변 장애물 여유 반경
LANDING_OCCUPIED_LIMIT = 0.75    # 이 log-odds 이상이면 landing 후보 제외
LANDING_Z_DELTA = 0.045          # 주변 바닥 대비 zrange 변화량
LANDING_Z_STEP_DELTA = 0.025     # 이전 스텝 대비 zrange 급변 감지 기준
LANDING_Z_STABLE_COUNT = 3       # landing 높이 후보 연속 검출 횟수
LANDING_START_RADIUS = 0.15      # goal 평균점에 이만큼 접근한 뒤 착륙 탐색 시작
GOAL_SCAN_RADIUS = 0.35          # 분포 내부 z-search 순회 반경
GOAL_SCAN_SPEED = 0.08

# Safety: 4 ranger 10 cm 유지
SAFETY_DIST = 0.10
LANDING_DESCENT_MIN_DIST = 0.12  # goal 내부 착륙 하강을 허용할 최소 수평 거리
SAFETY_BRAKE_DIST = 0.25         # 25cm 안으로 들어오면 비례 감속 시작
FRONT_AVOID_DIST = 0.35          # front가 이보다 가까우면 좌/우 회피
SIDE_RETURN_DIST = 0.18          # 선택한 side가 가까워지면 goal 분포로 복귀
SIDE_STEP_SPEED = 0.12
GOAL_SEEK_SPEED = 0.16
SIDE_AVOID_MAX_S = 2.0

# OGM
MAP_RESOLUTION = 0.10
DEFAULT_MAP_W = 3.0
DEFAULT_MAP_H = 5.0
DEFAULT_START = (0.5, 0.5)
DEFAULT_GOAL = (2.5, 4.5)

LOG_ODDS_OCC = 0.85
LOG_ODDS_FREE = -0.4
LOG_ODDS_MAX = 5.0
LOG_ODDS_MIN = -5.0
RAY_MAX_RANGE = 3.5              # ranger OUT 처리 시 free로 그릴 거리

# 기존 보존
CONTROL_DT = 0.05
LOG_PERIOD_MS = 50
RANGE_FILTER_ALPHA = 0.35
MAX_VELOCITY_STEP = 0.025
MAX_HEIGHT_COMMAND = TARGET_HEIGHT
MAX_HEIGHT_STEP_UP = 0.01
TAKEOFF_STEP_M = 0.02
TAKEOFF_STEP_S = 0.08
LANDING_STEP_M = 0.02
LANDING_STEP_S = 0.10
OUT_OF_RANGE_MM = 4000
SENSOR_LOG_DIR = Path("logs")

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*supervisor.*", category=UserWarning)

_active_worker = None
_last_height_command = 0.0
_last_vx_command = 0.0
_last_vy_command = 0.0


# ===========================================================================
# Helpers
# ===========================================================================
def mm_to_m(value_mm):
    if value_mm is None:
        return None
    try:
        v = float(value_mm)
    except (TypeError, ValueError):
        return None
    if v <= 0 or v >= OUT_OF_RANGE_MM or math.isinf(v):
        return None
    return v / 1000.0


def fmt_distance(v):
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return "OUT"
    return f"{v:.2f} m"


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


def send_arming_request(cf, do_arm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


# ---------------------------------------------------------------------------
# Velocity / height limiters (기존 코드 유지)
# ---------------------------------------------------------------------------
def reset_height_limiter(h=0.0):
    global _last_height_command
    _last_height_command = h


def reset_velocity_limiter(vx=0.0, vy=0.0):
    global _last_vx_command, _last_vy_command
    _last_vx_command = vx
    _last_vy_command = vy


def limited_velocity(rvx, rvy):
    global _last_vx_command, _last_vy_command
    dvx = max(-MAX_VELOCITY_STEP, min(MAX_VELOCITY_STEP, rvx - _last_vx_command))
    dvy = max(-MAX_VELOCITY_STEP, min(MAX_VELOCITY_STEP, rvy - _last_vy_command))
    _last_vx_command += dvx
    _last_vy_command += dvy
    return _last_vx_command, _last_vy_command


def limited_height(rh, max_h=MAX_HEIGHT_COMMAND):
    global _last_height_command
    h = min(rh, max_h)
    if h > _last_height_command:
        h = min(h, _last_height_command + MAX_HEIGHT_STEP_UP)
    _last_height_command = h
    return h


def send_velocity_limited(cf, vx, vy, rh, max_h=MAX_HEIGHT_COMMAND):
    h = limited_height(rh, max_h)
    vx_l, vy_l = limited_velocity(vx, vy)
    cf.commander.send_hover_setpoint(vx_l, vy_l, 0.0, h)
    return h, vx_l, vy_l


def clamp(value, low, high):
    return max(low, min(high, value))


def gaussian_weight(x, y, mean_xy, sigma=GOAL_SIGMA):
    if sigma <= 0:
        return 1.0 if (x, y) == tuple(mean_xy) else 0.0
    dx = x - mean_xy[0]
    dy = y - mean_xy[1]
    return math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def goal_distribution_threshold():
    return gaussian_weight(GOAL_SIGMA, 0.0, (0.0, 0.0))


def in_goal_distribution(x, y, goal_xy):
    return gaussian_weight(x, y, goal_xy) >= goal_distribution_threshold()


# ===========================================================================
# Occupancy Grid Map (log-odds, ray-cast)
# ===========================================================================
class OccupancyGrid:
    """
    Log-odds occupancy grid.

    World coordinate system:
      x : 0 .. width_m
      y : 0 .. height_m
      origin is map corner (0, 0). y-axis points "up" in plotting.

    Cell indexing:
      col = floor(x / res)
      row = floor(y / res)
    """

    def __init__(self, width_m, height_m, resolution_m=MAP_RESOLUTION):
        self.width_m = width_m
        self.height_m = height_m
        self.res = resolution_m
        self.cols = max(1, int(round(width_m / resolution_m)))
        self.rows = max(1, int(round(height_m / resolution_m)))
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.lock = threading.Lock()

    def world_to_cell(self, x, y):
        col = int(x / self.res)
        row = int(y / self.res)
        return row, col

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    @staticmethod
    def _bresenham(r0, c0, r1, c1):
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr
        r, c = r0, c0
        # safety cap to avoid runaway loops
        max_cells = (dr + dc) * 2 + 4
        while max_cells > 0:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
            max_cells -= 1
        return cells

    def _update_ray(self, x0, y0, x1, y1, hit_endpoint):
        r0, c0 = self.world_to_cell(x0, y0)
        r1, c1 = self.world_to_cell(x1, y1)
        cells = self._bresenham(r0, c0, r1, c1)
        if not cells:
            return
        with self.lock:
            for i, (r, c) in enumerate(cells):
                if not self.in_bounds(r, c):
                    continue
                last = (i == len(cells) - 1)
                if last and hit_endpoint:
                    self.log_odds[r, c] = min(
                        LOG_ODDS_MAX, self.log_odds[r, c] + LOG_ODDS_OCC
                    )
                else:
                    self.log_odds[r, c] = max(
                        LOG_ODDS_MIN, self.log_odds[r, c] + LOG_ODDS_FREE
                    )

    # body-frame ray offsets (CCW from front)
    _RAY_ANGLES = {
        "front": 0.0,
        "left":  math.pi / 2,
        "back":  math.pi,
        "right": -math.pi / 2,
    }

    def update_from_ranges(self, pose_x, pose_y, pose_yaw_rad, ranges_dict):
        """
        Multi-ranger F/L/R/B 빔을 ray-cast해 log-odds 업데이트.
        pose_x, pose_y: world frame meters.
        pose_yaw_rad: yaw in radians (CCW positive).
        ranges_dict: {"front": m, "left": m, "right": m, "back": m}, None=OUT.
        """
        for name, dist in ranges_dict.items():
            if name not in self._RAY_ANGLES:
                continue
            angle = pose_yaw_rad + self._RAY_ANGLES[name]
            if dist is None or dist >= RAY_MAX_RANGE:
                end_x = pose_x + RAY_MAX_RANGE * math.cos(angle)
                end_y = pose_y + RAY_MAX_RANGE * math.sin(angle)
                self._update_ray(pose_x, pose_y, end_x, end_y, hit_endpoint=False)
            else:
                end_x = pose_x + dist * math.cos(angle)
                end_y = pose_y + dist * math.sin(angle)
                self._update_ray(pose_x, pose_y, end_x, end_y, hit_endpoint=True)

    def snapshot(self):
        with self.lock:
            return self.log_odds.copy()

    def cell_center(self, row, col):
        return (col + 0.5) * self.res, (row + 0.5) * self.res

    def is_landing_clear(self, x, y, clear_radius=LANDING_CLEAR_RADIUS):
        """Return True when the local OGM patch has no likely occupied cells."""
        row, col = self.world_to_cell(x, y)
        radius_cells = max(1, int(math.ceil(clear_radius / self.res)))
        with self.lock:
            for rr in range(row - radius_cells, row + radius_cells + 1):
                for cc in range(col - radius_cells, col + radius_cells + 1):
                    if not self.in_bounds(rr, cc):
                        return False
                    cx, cy = self.cell_center(rr, cc)
                    if math.hypot(cx - x, cy - y) > clear_radius:
                        continue
                    if self.log_odds[rr, cc] >= LANDING_OCCUPIED_LIMIT:
                        return False
        return True

    def best_landing_target(self, goal_xy):
        """
        Pick a landing cell inside the Gaussian goal region.

        The score favors cells near the goal mean and cells observed as free.
        Occupied cells, map edges, and cells without enough clearance are rejected.
        """
        best = None
        best_score = -float("inf")
        search_cells = max(1, int(math.ceil(GOAL_SEARCH_RADIUS / self.res)))
        grow, gcol = self.world_to_cell(*goal_xy)
        with self.lock:
            log_odds = self.log_odds.copy()

        for row in range(grow - search_cells, grow + search_cells + 1):
            for col in range(gcol - search_cells, gcol + search_cells + 1):
                if not self.in_bounds(row, col):
                    continue
                x, y = self.cell_center(row, col)
                dist = math.hypot(x - goal_xy[0], y - goal_xy[1])
                if dist > GOAL_SEARCH_RADIUS:
                    continue
                lo = float(log_odds[row, col])
                if lo >= LANDING_OCCUPIED_LIMIT:
                    continue
                if not self.is_landing_clear(x, y):
                    continue

                goal_score = gaussian_weight(x, y, goal_xy)
                free_bonus = clamp(-lo / abs(LOG_ODDS_MIN), 0.0, 1.0)
                unknown_penalty = 0.10 if abs(lo) < 0.05 else 0.0
                score = goal_score + 0.35 * free_bonus - unknown_penalty
                if score > best_score:
                    best_score = score
                    best = (x, y)

        return best if best is not None else tuple(goal_xy)


# ===========================================================================
# Range + Pose Reader  (Multi-ranger + Kalman state)
# ===========================================================================
class RangePoseReader:
    """
    두 LogConfig를 동시에 운용:
      - range.front/left/right/back/up/zrange + pm.vbat  (50ms)
      - stateEstimate.x/y/yaw                            (50ms)
    snapshot() 으로 thread-safe하게 최신값 dict 반환.
    """

    def __init__(self, cf):
        self._cf = cf
        self._lock = threading.Lock()
        self._latest = {
            "front": None, "left": None, "right": None, "back": None,
            "up": None, "zrange": None, "zrange_raw": None, "vbat": None,
            "x": None, "y": None, "z": None, "yaw": None,
        }
        self._range_log = None
        self._state_log = None

    def __enter__(self):
        rl = LogConfig(name="Ranges", period_in_ms=LOG_PERIOD_MS)
        rl.add_variable("range.front", "uint16_t")
        rl.add_variable("range.left", "uint16_t")
        rl.add_variable("range.right", "uint16_t")
        rl.add_variable("range.back", "uint16_t")
        rl.add_variable("range.up", "uint16_t")
        rl.add_variable("range.zrange", "uint16_t")
        rl.add_variable("pm.vbat", "float")
        self._cf.log.add_config(rl)
        rl.data_received_cb.add_callback(self._range_cb)
        rl.start()
        self._range_log = rl

        sl = LogConfig(name="Pose", period_in_ms=LOG_PERIOD_MS)
        sl.add_variable("stateEstimate.x", "float")
        sl.add_variable("stateEstimate.y", "float")
        sl.add_variable("stateEstimate.z", "float")
        sl.add_variable("stateEstimate.yaw", "float")
        self._cf.log.add_config(sl)
        sl.data_received_cb.add_callback(self._state_cb)
        sl.start()
        self._state_log = sl
        return self

    def __exit__(self, *_a):
        self.stop()

    def stop(self):
        for log in (self._range_log, self._state_log):
            if log is not None:
                try:
                    log.stop()
                except Exception:
                    pass
        self._range_log = None
        self._state_log = None

    def _range_cb(self, _ts, data, _conf):
        zrange_raw = mm_to_m(data.get("range.zrange"))
        raw = {
            "front":  mm_to_m(data.get("range.front")),
            "left":   mm_to_m(data.get("range.left")),
            "right":  mm_to_m(data.get("range.right")),
            "back":   mm_to_m(data.get("range.back")),
            "up":     mm_to_m(data.get("range.up")),
            "zrange": zrange_raw,
            "zrange_raw": zrange_raw,
            "vbat":   data.get("pm.vbat"),
        }
        with self._lock:
            for k, v in raw.items():
                if k in ("vbat", "zrange_raw"):
                    self._latest[k] = v
                    continue
                prev = self._latest.get(k)
                if v is None:
                    pass  # keep previous
                elif prev is None:
                    self._latest[k] = v
                else:
                    self._latest[k] = (
                        RANGE_FILTER_ALPHA * v + (1.0 - RANGE_FILTER_ALPHA) * prev
                    )

    def _state_cb(self, _ts, data, _conf):
        with self._lock:
            self._latest["x"] = data.get("stateEstimate.x")
            self._latest["y"] = data.get("stateEstimate.y")
            self._latest["z"] = data.get("stateEstimate.z")
            self._latest["yaw"] = data.get("stateEstimate.yaw")  # degrees

    def snapshot(self):
        with self._lock:
            return dict(self._latest)


# ===========================================================================
# Crazyflie Worker Thread
# ===========================================================================
if PYQT_AVAILABLE:

    class CrazyflieWorker(QtCore.QThread):
        """
        하나의 thread가 SyncCrazyflie context를 보유.
        States: disconnected -> connecting -> connected -> flying -> connected ...
        """

        sensor_updated = QtCore.pyqtSignal(dict)
        map_changed = QtCore.pyqtSignal()
        status_text = QtCore.pyqtSignal(str)
        connection_state = QtCore.pyqtSignal(str)
        flight_result = QtCore.pyqtSignal(dict)
        landing_found = QtCore.pyqtSignal(float, float)

        def __init__(self):
            super().__init__()
            self.uri = URI
            self.ogm = None
            self.start_xy = DEFAULT_START
            self.goal_xy = DEFAULT_GOAL
            self.target_height = TARGET_HEIGHT
            self._landing_found_world = None
            self._sensor_log_path = None

            self._quit = False
            self._disconnect_request = False
            self._fly_request = False
            self._land_request = False
            self._estop_request = False
            self._mapping_active = False     # idle_loop에서 매핑 ON/OFF
            self._kalman_reset_pending = False  # 다음 idle tick에서 Kalman 재리셋

            self._cf = None
            self._reader = None
            self._command_event = threading.Event()
            self._map_lock = threading.Lock()

        # ---- public requests ----
        def request_connect(self, uri):
            self.uri = uri
            if not self.isRunning():
                self.start()

        def request_disconnect(self):
            self._disconnect_request = True
            self._command_event.set()

        def request_fly(self, ogm, start_xy, goal_xy, target_height):
            self.ogm = ogm
            self.start_xy = start_xy
            self.goal_xy = goal_xy
            self.target_height = target_height
            self._landing_found_world = None
            self._land_request = False
            self._estop_request = False
            # 매핑 중이 아닐 때만 Kalman 재리셋 (매핑 중이면 이미 정렬된 좌표계 유지)
            if not self._mapping_active:
                self._kalman_reset_pending = True
            self._fly_request = True
            self._command_event.set()

        def request_build_map(self, ogm, start_xy, goal_xy):
            """매핑 시작: ogm을 worker에 설정하고 idle_loop에서 ray-cast 활성."""
            with self._map_lock:
                self.ogm = ogm
                self.start_xy = start_xy
                self.goal_xy = goal_xy
                self._mapping_active = True
            # 시작 시점의 드론 위치 = 시작점이 되도록 Kalman 재리셋
            self._kalman_reset_pending = True
            self._command_event.set()

        def request_stop_mapping(self):
            with self._map_lock:
                self._mapping_active = False

        def request_reset_map(self):
            """ogm은 유지하되 log_odds를 0으로."""
            with self._map_lock:
                if self.ogm is not None:
                    with self.ogm.lock:
                        self.ogm.log_odds.fill(0.0)
                    self.map_changed.emit()

        def request_land(self):
            self._land_request = True

        def request_emergency_stop(self):
            self._estop_request = True
            self._land_request = True
            emergency_stop(self._cf)

        def request_quit(self):
            self._quit = True
            self._disconnect_request = True
            self._estop_request = True
            self._command_event.set()

        # ---- main loop ----
        def run(self):
            try:
                cflib.crtp.init_drivers()
                self.status_text.emit("Initializing radio")
                self.connection_state.emit("connecting")

                cf = Crazyflie(rw_cache="./cache")
                self._cf = cf

                with SyncCrazyflie(self.uri, cf=cf) as scf:
                    self._cf = scf.cf
                    self.status_text.emit("Connected. Resetting estimator…")
                    self._setup_estimator(scf.cf)

                    with RangePoseReader(scf.cf) as reader:
                        self._reader = reader
                        self.connection_state.emit("connected")
                        self.status_text.emit("Connected. Ready to fly.")
                        self._idle_loop(scf.cf)

                self._cf = None
                self._reader = None
                self.connection_state.emit("disconnected")
                self.status_text.emit("Disconnected")
            except Exception as exc:
                self.status_text.emit(f"Error: {exc}")
                emergency_stop(self._cf)
                self.connection_state.emit("disconnected")

        def _setup_estimator(self, cf):
            cf.param.set_value("stabilizer.estimator", "2")
            time.sleep(0.2)
            self._reset_kalman_only(cf)
            send_arming_request(cf, True)
            time.sleep(0.5)

        def _reset_kalman_only(self, cf):
            """Kalman EKF state reset (현재 드론 위치/자세를 origin으로 만듦)."""
            cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.1)
            cf.param.set_value("kalman.resetEstimation", "0")
            time.sleep(2.0)

        def _idle_loop(self, cf):
            """Connected but not flying. Stream sensors, do mapping if active."""
            last_emit = 0.0
            last_map_emit = 0.0
            while True:
                if self._quit or self._disconnect_request:
                    self._disconnect_request = False
                    return

                # Kalman 재리셋 요청 처리 (Build Map / Start Flight 직전)
                if self._kalman_reset_pending:
                    self._kalman_reset_pending = False
                    self.status_text.emit(
                        "Aligning origin: hold drone still ~2s…"
                    )
                    # 매핑이 켜져있다면 reset 동안 누적된 잘못된 좌표를 지움
                    if self._mapping_active and self.ogm is not None:
                        with self.ogm.lock:
                            self.ogm.log_odds.fill(0.0)
                        self.map_changed.emit()
                    self._reset_kalman_only(cf)
                    self.status_text.emit("Origin aligned. Drone is at start point.")

                if self._fly_request:
                    self._fly_request = False
                    self.connection_state.emit("flying")
                    try:
                        result = self._execute_flight(cf)
                    except Exception as exc:
                        self.status_text.emit(f"Flight error: {exc}")
                        emergency_stop(cf)
                        result = {"status": "error", "error": str(exc)}
                    self.flight_result.emit(result)
                    self.connection_state.emit("connected")
                    self.status_text.emit(
                        f"Flight done ({result.get('status', '?')}). Idle."
                    )
                    self._land_request = False
                    self._estop_request = False

                # sensor emit (~20Hz) + mapping
                now = time.time()
                if now - last_emit >= 0.05 and self._reader is not None:
                    snap = self._reader.snapshot()
                    self.sensor_updated.emit(snap)
                    last_emit = now

                    # idle 상태에서도 매핑 active이면 OGM 업데이트
                    if self._mapping_active:
                        with self._map_lock:
                            ogm = self.ogm
                            origin = self.start_xy
                        if ogm is not None and snap.get("x") is not None:
                            world_x = origin[0] + snap["x"]
                            world_y = origin[1] + snap["y"]
                            yaw_rad = math.radians(snap.get("yaw") or 0.0)
                            ogm.update_from_ranges(world_x, world_y, yaw_rad, {
                                "front": snap.get("front"),
                                "left":  snap.get("left"),
                                "right": snap.get("right"),
                                "back":  snap.get("back"),
                            })
                            if now - last_map_emit > 0.1:
                                self.map_changed.emit()
                                last_map_emit = now

                self._command_event.wait(timeout=0.05)
                self._command_event.clear()

        # ---- flight execution ----
        def _execute_flight(self, cf):
            """Takeoff -> Gaussian goal-region seek with reactive side avoidance."""
            log_file, log_writer = self._open_sensor_log()
            self.status_text.emit("Takeoff")
            reset_height_limiter(0.0)
            reset_velocity_limiter(0.0, 0.0)
            h = 0.0
            try:
                while h < self.target_height:
                    if self._land_request or self._estop_request:
                        break
                    h = min(self.target_height, h + TAKEOFF_STEP_M)
                    cf.commander.send_hover_setpoint(0, 0, 0, h)
                    if self._reader is not None:
                        self._write_sensor_log(
                            log_writer,
                            phase="takeoff",
                            mode="takeoff",
                            snap=self._reader.snapshot(),
                            height_cmd=h,
                        )
                    time.sleep(TAKEOFF_STEP_S)
                reset_height_limiter(h)
                time.sleep(0.5)

                path_log = []
                completed = False
                if not (self._land_request or self._estop_request):
                    completed = self._seek_goal_distribution(cf, path_log, log_writer)

                if self._land_request or self._estop_request:
                    status = "aborted"
                else:
                    status = "completed" if completed else "timeout"
                return self._finalize(cf, path_log, status)
            finally:
                if log_file is not None:
                    try:
                        log_file.flush()
                        log_file.close()
                    except Exception:
                        pass

        def _seek_goal_distribution(self, cf, path_log, log_writer=None):
            start_world = tuple(self.start_xy)
            goal_world = tuple(self.goal_xy)
            last_map_emit = 0.0
            last_target_update = 0.0
            landing_target = goal_world
            mode = "seek_goal"
            return_mode = "seek_goal"
            side_sign = 0.0
            side_started = 0.0
            z_floor_ref = None
            prev_zrange_raw = None
            z_edge_count = 0
            z_candidate_count = 0
            clear_landing_count = 0
            scan_points = self._goal_scan_points(goal_world)
            scan_index = 0
            flight_started = time.time()
            max_flight_s = 180.0

            self.status_text.emit(
                f"Seeking Gaussian goal region: mean=({goal_world[0]:.2f}, {goal_world[1]:.2f}), "
                f"sigma={GOAL_SIGMA:.2f} m"
            )

            while True:
                if self._land_request or self._estop_request:
                    return False
                if time.time() - flight_started > max_flight_s:
                    self.status_text.emit("Goal seek timeout")
                    return False

                snap = self._reader.snapshot()
                cur_x = snap.get("x")
                cur_y = snap.get("y")
                cur_yaw = snap.get("yaw") or 0.0
                yaw_rad = math.radians(cur_yaw)
                if cur_x is None or cur_y is None:
                    cf.commander.send_hover_setpoint(0, 0, 0, self.target_height)
                    time.sleep(CONTROL_DT)
                    continue

                world_x = start_world[0] + cur_x
                world_y = start_world[1] + cur_y
                dist_to_goal_mean = math.hypot(
                    goal_world[0] - world_x,
                    goal_world[1] - world_y,
                )

                if self.ogm is not None:
                    self.ogm.update_from_ranges(world_x, world_y, yaw_rad, {
                        "front": snap.get("front"),
                        "left":  snap.get("left"),
                        "right": snap.get("right"),
                        "back":  snap.get("back"),
                    })
                    now = time.time()
                    if now - last_map_emit > 0.1:
                        self.map_changed.emit()
                        last_map_emit = now
                    if mode != "seek_goal" and now - last_target_update > 0.4:
                        landing_target = self.ogm.best_landing_target(goal_world)
                        last_target_update = now

                inside_goal_distribution = in_goal_distribution(
                    world_x, world_y, goal_world
                )
                zrange = snap.get("zrange")
                zrange_raw = snap.get("zrange_raw") or zrange
                if zrange_raw is not None and not inside_goal_distribution:
                    if z_floor_ref is None:
                        z_floor_ref = zrange_raw
                    else:
                        z_floor_ref = 0.98 * z_floor_ref + 0.02 * zrange_raw

                if mode == "seek_goal":
                    landing_target = goal_world

                if (
                    inside_goal_distribution and
                    dist_to_goal_mean <= LANDING_START_RADIUS and
                    mode != "search_landing"
                ):
                    mode = "search_landing"
                    side_sign = 0.0
                    z_candidate_count = 0
                    self.status_text.emit("Inside goal distribution: landing has priority")

                if mode == "search_landing":
                    landing_target = scan_points[scan_index]
                    if math.hypot(landing_target[0] - world_x, landing_target[1] - world_y) <= ARRIVAL_RADIUS:
                        scan_index = (scan_index + 1) % len(scan_points)
                        landing_target = scan_points[scan_index]

                dx = landing_target[0] - world_x
                dy = landing_target[1] - world_y
                dist = math.hypot(dx, dy)
                landing_clear = (
                    self.ogm is None or self.ogm.is_landing_clear(world_x, world_y)
                )
                descent_clear = self._is_descent_clear(snap)
                if mode == "search_landing" and (landing_clear or descent_clear):
                    clear_landing_count += 1
                else:
                    clear_landing_count = 0

                z_step = None
                if zrange_raw is not None and prev_zrange_raw is not None:
                    z_step = zrange_raw - prev_zrange_raw
                z_edge = (
                    mode == "search_landing" and
                    z_step is not None and
                    abs(z_step) >= LANDING_Z_STEP_DELTA
                )
                if (
                    mode == "search_landing" and
                    self._is_landing_height_candidate(zrange_raw, z_floor_ref, z_edge)
                ):
                    z_candidate_count += 1
                    if z_edge:
                        z_edge_count += 1
                else:
                    z_candidate_count = max(0, z_candidate_count - 1)

                if (
                    mode == "search_landing" and
                    z_candidate_count >= LANDING_Z_STABLE_COUNT and
                    z_edge_count >= 1 and
                    (landing_clear or descent_clear)
                ):
                    self._landing_found_world = (world_x, world_y)
                    self.landing_found.emit(world_x, world_y)
                    cf.commander.send_hover_setpoint(0, 0, 0, self.target_height)
                    self.status_text.emit(
                        f"Landing z-height found: ({world_x:.2f}, {world_y:.2f})"
                    )
                    return True

                front = snap.get("front")
                if mode == "seek_goal" and front is not None and front < FRONT_AVOID_DIST:
                    side_sign = self._choose_side_direction(snap)
                    if side_sign != 0.0:
                        return_mode = mode
                        mode = "side_avoid"
                        side_started = time.time()
                        label = "left" if side_sign > 0 else "right"
                        self.status_text.emit(f"Front obstacle: sidestep {label}")

                if mode == "side_avoid":
                    selected_side = snap.get("left") if side_sign > 0 else snap.get("right")
                    timed_out = time.time() - side_started >= SIDE_AVOID_MAX_S
                    side_close = selected_side is not None and selected_side <= SIDE_RETURN_DIST
                    front_clear = front is None or front > FRONT_AVOID_DIST + 0.10
                    if side_close or (timed_out and front_clear):
                        mode = return_mode
                        side_sign = 0.0
                        self.status_text.emit("Returning to landing search")

                if mode == "side_avoid":
                    vx = 0.0
                    vy = side_sign * SIDE_STEP_SPEED
                else:
                    max_speed = GOAL_SCAN_SPEED if mode == "search_landing" else GOAL_SEEK_SPEED
                    speed = min(max_speed, max(0.04, dist))
                    if dist > 1e-6:
                        vx = clamp(dx / dist * speed, -max_speed, max_speed)
                        vy = clamp(dy / dist * speed, -max_speed, max_speed)
                    else:
                        vx = 0.0
                        vy = 0.0

                vx, vy, throttled = self._apply_safety(snap, vx, vy)
                _hc, c_vx, c_vy = send_velocity_limited(
                    cf, vx, vy, self.target_height
                )
                path_log.append({
                    "t": time.time(),
                    "mode": mode,
                    "cur_x": cur_x, "cur_y": cur_y, "yaw": cur_yaw,
                    "world_x": world_x, "world_y": world_y,
                    "target_x": landing_target[0], "target_y": landing_target[1],
                    "zrange": zrange, "zrange_raw": zrange_raw,
                    "z_step": z_step, "z_floor_ref": z_floor_ref,
                    "z_candidate_count": z_candidate_count,
                    "z_edge_count": z_edge_count,
                    "clear_landing_count": clear_landing_count,
                    "descent_clear": descent_clear,
                    "vx_cmd": c_vx, "vy_cmd": c_vy,
                    "throttled": throttled,
                })
                self._write_sensor_log(
                    log_writer,
                    phase="flight",
                    mode=mode,
                    snap=snap,
                    world_x=world_x,
                    world_y=world_y,
                    target_x=landing_target[0],
                    target_y=landing_target[1],
                    height_cmd=self.target_height,
                    vx_cmd=c_vx,
                    vy_cmd=c_vy,
                    landing_clear=landing_clear,
                    descent_clear=descent_clear,
                    clear_landing_count=clear_landing_count,
                    inside_goal=inside_goal_distribution,
                    z_floor_ref=z_floor_ref,
                    z_step=z_step,
                    z_edge=z_edge,
                    z_candidate_count=z_candidate_count,
                    z_edge_count=z_edge_count,
                    scan_index=scan_index,
                    throttled=throttled,
                )
                self.sensor_updated.emit(snap)
                if zrange_raw is not None:
                    prev_zrange_raw = zrange_raw
                time.sleep(CONTROL_DT)

        def _open_sensor_log(self):
            try:
                SENSOR_LOG_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = SENSOR_LOG_DIR / f"ogm_sensor_{stamp}.csv"
                log_file = path.open("w", newline="", encoding="utf-8", buffering=1)
                fields = [
                    "time_s", "phase", "mode",
                    "x", "y", "z", "yaw",
                    "world_x", "world_y",
                    "target_x", "target_y",
                    "front", "left", "right", "back", "up",
                    "zrange", "zrange_raw", "z_step", "z_floor_ref",
                    "vbat", "height_cmd", "vx_cmd", "vy_cmd",
                    "inside_goal", "landing_clear", "z_edge",
                    "descent_clear", "clear_landing_count",
                    "z_candidate_count", "z_edge_count", "scan_index",
                    "throttled",
                ]
                writer = csv.DictWriter(log_file, fieldnames=fields)
                writer.writeheader()
                self._sensor_log_path = str(path)
                self.status_text.emit(f"Sensor logging: {path}")
                return log_file, writer
            except Exception as exc:
                self._sensor_log_path = None
                self.status_text.emit(f"Sensor log disabled: {exc}")
                return None, None

        def _write_sensor_log(self, writer, phase, mode, snap, **extra):
            if writer is None:
                return
            row = {
                "time_s": time.time(),
                "phase": phase,
                "mode": mode,
                "x": snap.get("x"),
                "y": snap.get("y"),
                "z": snap.get("z"),
                "yaw": snap.get("yaw"),
                "world_x": extra.get("world_x"),
                "world_y": extra.get("world_y"),
                "target_x": extra.get("target_x"),
                "target_y": extra.get("target_y"),
                "front": snap.get("front"),
                "left": snap.get("left"),
                "right": snap.get("right"),
                "back": snap.get("back"),
                "up": snap.get("up"),
                "zrange": snap.get("zrange"),
                "zrange_raw": snap.get("zrange_raw"),
                "z_step": extra.get("z_step"),
                "z_floor_ref": extra.get("z_floor_ref"),
                "vbat": snap.get("vbat"),
                "height_cmd": extra.get("height_cmd"),
                "vx_cmd": extra.get("vx_cmd"),
                "vy_cmd": extra.get("vy_cmd"),
                "inside_goal": extra.get("inside_goal"),
                "landing_clear": extra.get("landing_clear"),
                "z_edge": extra.get("z_edge"),
                "descent_clear": extra.get("descent_clear"),
                "clear_landing_count": extra.get("clear_landing_count"),
                "z_candidate_count": extra.get("z_candidate_count"),
                "z_edge_count": extra.get("z_edge_count"),
                "scan_index": extra.get("scan_index"),
                "throttled": extra.get("throttled"),
            }
            try:
                writer.writerow(row)
            except Exception:
                pass

        def _goal_scan_points(self, goal_world):
            radius = min(GOAL_SCAN_RADIUS, GOAL_SEARCH_RADIUS, GOAL_SIGMA * 1.5)
            offsets = [
                (0.0, 0.0),
                (radius, 0.0),
                (0.0, radius),
                (-radius, 0.0),
                (0.0, -radius),
                (radius * 0.7, radius * 0.7),
                (-radius * 0.7, radius * 0.7),
                (-radius * 0.7, -radius * 0.7),
                (radius * 0.7, -radius * 0.7),
            ]
            points = []
            for dx, dy in offsets:
                x = clamp(goal_world[0] + dx, 0.0, self.ogm.width_m if self.ogm else goal_world[0] + dx)
                y = clamp(goal_world[1] + dy, 0.0, self.ogm.height_m if self.ogm else goal_world[1] + dy)
                if in_goal_distribution(x, y, goal_world):
                    points.append((x, y))
            return points or [tuple(goal_world)]

        def _is_landing_height_candidate(self, zrange, z_floor_ref, z_edge=False):
            if zrange is None:
                return False
            if z_edge:
                return True
            ref = z_floor_ref if z_floor_ref is not None else self.target_height
            if ref is None or ref <= 0:
                return False
            return abs(zrange - ref) >= LANDING_Z_DELTA

        def _choose_side_direction(self, snap):
            """
            Pick the shorter left/right ranger direction when it is still safe.
            If the shorter side is already too close, fall back to the other side.
            """
            left = snap.get("left")
            right = snap.get("right")
            left_ok = left is None or left > SAFETY_BRAKE_DIST
            right_ok = right is None or right > SAFETY_BRAKE_DIST

            if left is None and right is None:
                return 1.0
            if left is None:
                return -1.0 if right_ok else 0.0
            if right is None:
                return 1.0 if left_ok else 0.0

            preferred = 1.0 if left <= right else -1.0
            if preferred > 0 and left_ok:
                return preferred
            if preferred < 0 and right_ok:
                return preferred
            if left_ok:
                return 1.0
            if right_ok:
                return -1.0
            return 0.0

        def _is_descent_clear(self, snap):
            """Allow landing in the goal when no ranger reports immediate collision risk."""
            for name in ("front", "left", "right", "back"):
                dist = snap.get(name)
                if dist is not None and dist < LANDING_DESCENT_MIN_DIST:
                    return False
            return True

        def _move_axis(self, cf, axis, delta, path_log, world_origin):
            if abs(delta) < 1e-3:
                return
            direction = 1.0 if delta > 0 else -1.0

            snap0 = self._reader.snapshot()
            x0 = snap0.get("x") or 0.0
            y0 = snap0.get("y") or 0.0

            last_map_emit = 0.0

            while True:
                if self._land_request or self._estop_request:
                    break

                snap = self._reader.snapshot()
                cur_x = snap.get("x")
                cur_y = snap.get("y")
                cur_yaw = snap.get("yaw") or 0.0
                yaw_rad = math.radians(cur_yaw)

                if cur_x is None or cur_y is None:
                    cf.commander.send_hover_setpoint(0, 0, 0, self.target_height)
                    time.sleep(CONTROL_DT)
                    continue

                progress = (cur_x - x0) if axis == "x" else (cur_y - y0)
                remaining = delta - progress
                if abs(remaining) < ARRIVAL_RADIUS:
                    cf.commander.send_hover_setpoint(0, 0, 0, self.target_height)
                    break

                # Command velocity (single axis)
                speed = SPEED_X if axis == "x" else SPEED_Y
                vx = direction * speed if axis == "x" else 0.0
                vy = direction * speed if axis == "y" else 0.0

                # Safety: 4-ranger 10 cm rule
                vx, vy, throttled = self._apply_safety(snap, vx, vy)

                # OGM update (world frame)
                if self.ogm is not None:
                    world_x = world_origin[0] + cur_x
                    world_y = world_origin[1] + cur_y
                    self.ogm.update_from_ranges(world_x, world_y, yaw_rad, {
                        "front": snap.get("front"),
                        "left":  snap.get("left"),
                        "right": snap.get("right"),
                        "back":  snap.get("back"),
                    })
                    now = time.time()
                    if now - last_map_emit > 0.1:
                        self.map_changed.emit()
                        last_map_emit = now

                hc, c_vx, c_vy = send_velocity_limited(cf, vx, vy, self.target_height)
                path_log.append({
                    "t": time.time(),
                    "cur_x": cur_x, "cur_y": cur_y, "yaw": cur_yaw,
                    "vx_cmd": c_vx, "vy_cmd": c_vy,
                    "throttled": throttled,
                })

                # Push a sensor update for the GUI as well
                self.sensor_updated.emit(snap)

                time.sleep(CONTROL_DT)

        def _apply_safety(self, snap, vx, vy):
            """
            4방향 ranger 10cm 유지.
            - front < SAFETY_DIST 이고 vx > 0 → vx = 0
            - SAFETY_DIST < d < SAFETY_BRAKE_DIST 이면 비례 감속
            - 반대 방향은 그대로 (멀어지는 건 허용)
            """
            def brake(d, v_into_wall):
                if d is None:
                    return v_into_wall
                if v_into_wall <= 0:
                    return v_into_wall
                if d < SAFETY_DIST:
                    return 0.0
                if d < SAFETY_BRAKE_DIST:
                    ratio = (d - SAFETY_DIST) / (SAFETY_BRAKE_DIST - SAFETY_DIST)
                    return v_into_wall * max(0.0, min(1.0, ratio))
                return v_into_wall

            f = snap.get("front")
            b = snap.get("back")
            l = snap.get("left")
            r = snap.get("right")

            orig_vx, orig_vy = vx, vy
            if vx > 0:
                vx = brake(f, vx)
            elif vx < 0:
                vx = -brake(b, -vx)
            if vy > 0:
                vy = brake(l, vy)
            elif vy < 0:
                vy = -brake(r, -vy)

            throttled = (vx != orig_vx) or (vy != orig_vy)
            return vx, vy, throttled

        def _finalize(self, cf, path_log, status):
            """Hover-capture final pose → land → report drift."""
            # 착륙 직전 pose 캡처 (commanded vs measured)
            snap = self._reader.snapshot()
            final_x = snap.get("x") or 0.0
            final_y = snap.get("y") or 0.0
            commanded_dx = self.goal_xy[0] - self.start_xy[0]
            commanded_dy = self.goal_xy[1] - self.start_xy[1]

            error_x = final_x - commanded_dx
            error_y = final_y - commanded_dy
            error_norm = math.hypot(error_x, error_y)

            self.status_text.emit("Landing")
            reset_velocity_limiter(0.0, 0.0)
            h = self.target_height
            while h > 0.05:
                if self._estop_request:
                    break
                h = max(0.05, h - LANDING_STEP_M)
                cf.commander.send_hover_setpoint(0, 0, 0, h)
                time.sleep(LANDING_STEP_S)
            emergency_stop(cf)

            return {
                "status": status,
                "commanded_world": list(self.goal_xy),
                "commanded_drone": [commanded_dx, commanded_dy],
                "measured_drone": [final_x, final_y],
                "error_x": error_x,
                "error_y": error_y,
                "error_norm": error_norm,
                "landing_world": list(self._landing_found_world) if self._landing_found_world else None,
                "sensor_log": self._sensor_log_path,
                "samples": len(path_log),
            }


# ===========================================================================
# Map view (Qt widget)
# ===========================================================================
if PYQT_AVAILABLE:

    class MapView(QtWidgets.QWidget):
        """Render OGM + start/goal markers + drone trace."""

        def __init__(self):
            super().__init__()
            self.ogm = None
            self.drone_world = None
            self.start_world = None
            self.goal_world = None
            self.landing_world = None
            self.path_world = []
            self.setMinimumSize(360, 480)
            self.setStyleSheet(
                "background: white; border: 1px solid #d0d6e0; border-radius: 8px;"
            )

        def set_map(self, ogm, start_xy, goal_xy):
            self.ogm = ogm
            self.start_world = start_xy
            self.goal_world = goal_xy
            self.landing_world = None
            self.path_world = []
            self.update()

        def set_landing_marker(self, world_x, world_y):
            self.landing_world = (world_x, world_y)
            self.update()

        def update_drone(self, world_x, world_y):
            self.drone_world = (world_x, world_y)
            self.path_world.append((world_x, world_y))
            if len(self.path_world) > 4000:
                self.path_world = self.path_world[-4000:]
            self.update()

        def map_changed(self):
            self.update()

        def paintEvent(self, _ev):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)

            w = self.width()
            h = self.height()

            if self.ogm is None:
                p.setPen(QtGui.QColor("#9ca3af"))
                p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                           "Press 'Build Map' (after Connect) to start mapping")
                return

            # 비대칭 margin: 화면 좌측에 x 라벨, 화면 하단에 y 라벨
            # (drone forward = 화면 위, drone left = 화면 왼쪽 으로 회전 표시)
            margin_l = 36
            margin_b = 24
            margin_t = 14
            margin_r = 14
            avail_w = w - margin_l - margin_r
            avail_h = h - margin_t - margin_b
            # world x(=forward)는 화면 세로, world y(=left)는 화면 가로에 매핑
            scale = min(avail_w / self.ogm.height_m, avail_h / self.ogm.width_m)

            def w2s(wx, wy):
                # world +x (drone forward) → screen up  (sy 감소)
                # world +y (drone left)    → screen left (sx 감소)
                sx = (w - margin_r) - wy * scale
                sy = (h - margin_b) - wx * scale
                return sx, sy

            # ---- 격자 (0.5m 점선) ----
            grid_step = 0.5
            p.setPen(QtGui.QPen(QtGui.QColor(220, 224, 232), 1, QtCore.Qt.DotLine))
            gx = 0.0
            while gx <= self.ogm.width_m + 1e-6:
                sx_a, sy_a = w2s(gx, 0)
                sx_b, sy_b = w2s(gx, self.ogm.height_m)
                p.drawLine(QtCore.QPointF(sx_a, sy_a), QtCore.QPointF(sx_b, sy_b))
                gx += grid_step
            gy = 0.0
            while gy <= self.ogm.height_m + 1e-6:
                sx_a, sy_a = w2s(0, gy)
                sx_b, sy_b = w2s(self.ogm.width_m, gy)
                p.drawLine(QtCore.QPointF(sx_a, sy_a), QtCore.QPointF(sx_b, sy_b))
                gy += grid_step

            # ---- 눈금 라벨 ----
            p.setPen(QtGui.QColor("#6b7280"))
            font = p.font()
            font.setPointSize(8)
            p.setFont(font)
            # x 라벨 → 화면 좌측 (각 wx 격자선의 왼쪽 끝)
            gx = 0.0
            while gx <= self.ogm.width_m + 1e-6:
                sx, sy = w2s(gx, self.ogm.height_m)
                p.drawText(QtCore.QPointF(sx - 30, sy + 4), f"{gx:.1f}")
                gx += grid_step
            # y 라벨 → 화면 하단 (각 wy 격자선의 아래쪽 끝)
            gy = 0.0
            while gy <= self.ogm.height_m + 1e-6:
                sx, sy = w2s(0, gy)
                p.drawText(QtCore.QPointF(sx - 8, sy + 16), f"{gy:.1f}")
                gy += grid_step

            # boundary
            sx0, sy0 = w2s(0, 0)
            sx1, sy1 = w2s(self.ogm.width_m, self.ogm.height_m)
            p.setPen(QtGui.QPen(QtGui.QColor("#9ca3af"), 1))
            rect = QtCore.QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0).normalized()
            p.drawRect(rect)

            # ---- 원점 좌표축 화살표 (x=빨강 위로, y=초록 왼쪽으로) ----
            arrow_len_world = 0.35  # meters
            ox, oy = w2s(0, 0)
            exs, eys = w2s(arrow_len_world, 0)   # x축 끝점 (화면 위)
            yxs, yys = w2s(0, arrow_len_world)   # y축 끝점 (화면 왼쪽)
            # x축 (빨강, 위로)
            p.setPen(QtGui.QPen(QtGui.QColor("#ef4444"), 2))
            p.drawLine(QtCore.QPointF(ox, oy), QtCore.QPointF(exs, eys))
            p.drawLine(QtCore.QPointF(exs, eys), QtCore.QPointF(exs - 4, eys + 6))
            p.drawLine(QtCore.QPointF(exs, eys), QtCore.QPointF(exs + 4, eys + 6))
            font_b = p.font()
            font_b.setBold(True)
            font_b.setPointSize(10)
            p.setFont(font_b)
            p.setPen(QtGui.QColor("#ef4444"))
            p.drawText(QtCore.QPointF(exs + 6, eys + 4), "x (fwd)")
            # y축 (초록, 왼쪽으로)
            p.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 2))
            p.drawLine(QtCore.QPointF(ox, oy), QtCore.QPointF(yxs, yys))
            p.drawLine(QtCore.QPointF(yxs, yys), QtCore.QPointF(yxs + 6, yys - 4))
            p.drawLine(QtCore.QPointF(yxs, yys), QtCore.QPointF(yxs + 6, yys + 4))
            p.setPen(QtGui.QColor("#10b981"))
            p.drawText(QtCore.QPointF(yxs - 32, yys - 4), "y (left)")
            p.setFont(font)

            # cells (vectorised lookup but simple iteration for clarity)
            log_odds = self.ogm.snapshot()
            cell_w = self.ogm.res * scale
            for row in range(self.ogm.rows):
                for col in range(self.ogm.cols):
                    lo = log_odds[row, col]
                    if abs(lo) < 0.05:
                        continue
                    cx = col * self.ogm.res
                    cy = row * self.ogm.res
                    px, py = w2s(cx, cy + self.ogm.res)
                    if lo > 0:
                        a = int(min(220, lo / LOG_ODDS_MAX * 220 + 35))
                        color = QtGui.QColor(220, 60, 60, a)
                    else:
                        a = int(min(140, -lo / LOG_ODDS_MAX * 120 + 20))
                        color = QtGui.QColor(100, 180, 100, a)
                    p.fillRect(QtCore.QRectF(px, py, cell_w, cell_w), color)

            # commanded reference line (start -> goal mean)
            if self.start_world and self.goal_world:
                p.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180, 160), 1, QtCore.Qt.DashLine))
                sxs, sys_ = w2s(*self.start_world)
                gxs, gys = w2s(*self.goal_world)
                p.drawLine(QtCore.QPointF(sxs, sys_), QtCore.QPointF(gxs, gys))

            # Gaussian goal distribution: 1-sigma filled area and search boundary.
            if self.goal_world:
                gxs, gys = w2s(*self.goal_world)
                sigma_px = GOAL_SIGMA * scale
                search_px = GOAL_SEARCH_RADIUS * scale
                p.setBrush(QtGui.QColor(245, 158, 11, 42))
                p.setPen(QtGui.QPen(QtGui.QColor(245, 158, 11, 150), 1.5))
                p.drawEllipse(QtCore.QPointF(gxs, gys), sigma_px, sigma_px)
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(QtGui.QColor(245, 158, 11, 95), 1, QtCore.Qt.DashLine))
                p.drawEllipse(QtCore.QPointF(gxs, gys), search_px, search_px)

            # measured trajectory
            if len(self.path_world) >= 2:
                p.setPen(QtGui.QPen(QtGui.QColor("#3b82f6"), 1.8))
                pts = [QtCore.QPointF(*w2s(x, y)) for x, y in self.path_world]
                path = QtGui.QPainterPath()
                path.moveTo(pts[0])
                for q in pts[1:]:
                    path.lineTo(q)
                p.drawPath(path)

            def marker(wx, wy, color, label):
                sx, sy = w2s(wx, wy)
                p.setBrush(QtGui.QColor(color))
                p.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
                p.drawEllipse(QtCore.QPointF(sx, sy), 6, 6)
                p.drawText(QtCore.QPointF(sx + 9, sy + 4), label)

            if self.start_world:
                marker(*self.start_world, color="#10b981", label="S")
            if self.goal_world:
                marker(*self.goal_world, color="#f59e0b", label="Gμ")

            if self.landing_world:
                sx, sy = w2s(*self.landing_world)
                p.setPen(QtGui.QPen(QtGui.QColor("#dc2626"), 3))
                p.drawLine(QtCore.QPointF(sx - 8, sy - 8), QtCore.QPointF(sx + 8, sy + 8))
                p.drawLine(QtCore.QPointF(sx - 8, sy + 8), QtCore.QPointF(sx + 8, sy - 8))
                p.drawText(QtCore.QPointF(sx + 10, sy - 8), "L")

            if self.drone_world:
                sx, sy = w2s(*self.drone_world)
                p.setBrush(QtGui.QColor("#1e3a8a"))
                p.setPen(QtGui.QPen(QtGui.QColor("#1e3a8a"), 2))
                p.drawEllipse(QtCore.QPointF(sx, sy), 5, 5)


# ===========================================================================
# Main window
# ===========================================================================
if PYQT_AVAILABLE:

    class MainWindow(QtWidgets.QWidget):
        def __init__(self, args):
            super().__init__()
            self.setWindowTitle("Crazyflie OGM + Gaussian Goal Landing")
            self.setStyleSheet(self._stylesheet())

            # widgets
            self.uri_edit = QtWidgets.QLineEdit(args.uri)
            self.connect_btn = QtWidgets.QPushButton("Connect")
            self.connect_btn.setObjectName("connectButton")
            self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
            self.disconnect_btn.setObjectName("dangerButton")
            self.disconnect_btn.setEnabled(False)

            self.map_w_spin = self._spin(0.5, 10.0, DEFAULT_MAP_W, 0.5)
            self.map_h_spin = self._spin(0.5, 10.0, DEFAULT_MAP_H, 0.5)
            self.start_x_spin = self._spin(0.0, 10.0, DEFAULT_START[0], 0.1)
            self.start_y_spin = self._spin(0.0, 10.0, DEFAULT_START[1], 0.1)
            self.goal_x_spin = self._spin(0.0, 10.0, DEFAULT_GOAL[0], 0.1)
            self.goal_y_spin = self._spin(0.0, 10.0, DEFAULT_GOAL[1], 0.1)
            self.height_spin = self._spin(0.20, 0.80, args.target_height, 0.05)

            self.start_btn = QtWidgets.QPushButton("Start Goal Landing")
            self.land_btn = QtWidgets.QPushButton("Land")
            self.estop_btn = QtWidgets.QPushButton("Emergency Stop")
            self.estop_btn.setObjectName("dangerButton")
            self.start_btn.setEnabled(False)
            self.land_btn.setEnabled(False)
            self.estop_btn.setEnabled(False)

            # Mapping buttons (Build = toggle)
            self.build_btn = QtWidgets.QPushButton("Build Map")
            self.build_btn.setObjectName("mapButton")
            self.build_btn.setCheckable(True)
            self.reset_btn = QtWidgets.QPushButton("Reset Map")
            self.build_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)

            self.status_label = QtWidgets.QLabel("Disconnected")
            self.status_label.setObjectName("statusLabel")
            self.battery_label = QtWidgets.QLabel("Battery --")
            self.battery_label.setObjectName("metric")
            self.battery_bar = QtWidgets.QProgressBar()
            self.battery_bar.setRange(0, 100)
            self.battery_bar.setTextVisible(False)
            self._set_battery_style(None)

            self.range_labels = {k: QtWidgets.QLabel("--") for k in
                                 ("front", "left", "right", "back", "up", "zrange")}
            self.pose_label = QtWidgets.QLabel("x: --  y: --  yaw: --")
            self.pose_label.setObjectName("metric")

            self.error_label = QtWidgets.QLabel("Press Start to fly. Drift report will appear here.")
            self.error_label.setObjectName("errorLabel")
            self.error_label.setWordWrap(True)

            self.map_view = MapView()

            # worker wiring
            self.worker = CrazyflieWorker()
            self.worker.sensor_updated.connect(self.on_sensor)
            self.worker.connection_state.connect(self.on_connection_state)
            self.worker.status_text.connect(self.status_label.setText)
            self.worker.map_changed.connect(self.map_view.map_changed)
            self.worker.flight_result.connect(self.on_flight_result)
            self.worker.landing_found.connect(self.map_view.set_landing_marker)

            self.connect_btn.clicked.connect(self.on_connect)
            self.disconnect_btn.clicked.connect(self.on_disconnect)
            self.start_btn.clicked.connect(self.on_start_flight)
            self.land_btn.clicked.connect(self.on_land)
            self.estop_btn.clicked.connect(self.on_estop)
            self.build_btn.toggled.connect(self.on_build_toggled)
            self.reset_btn.clicked.connect(self.on_reset_map)

            self.ogm = None
            self._build_layout()

        def _stylesheet(self):
            return """
            QWidget { background: #f6f7fb; color: #1f2937; font-size: 13px; }
            QGroupBox { background: #ffffff; border: 1px solid #d9dee8;
                        border-radius: 8px; margin-top: 12px; padding: 12px;
                        font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
            QLineEdit, QDoubleSpinBox { background: #ffffff; border: 1px solid #cfd6e3;
                                        border-radius: 6px; padding: 4px 6px; }
            QPushButton { background: #2563eb; color: white; border: none;
                          border-radius: 7px; padding: 8px 12px; font-weight: 600; }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton:disabled { background: #aeb7c7; }
            QPushButton#dangerButton { background: #dc2626; }
            QPushButton#dangerButton:hover { background: #b91c1c; }
            QPushButton#connectButton { background: #16a34a; }
            QPushButton#connectButton:hover { background: #15803d; }
            QPushButton#mapButton { background: #7c3aed; }
            QPushButton#mapButton:hover { background: #6d28d9; }
            QPushButton#mapButton:checked { background: #5b21b6; }
            QLabel#statusLabel { background: #111827; color: white;
                                 border-radius: 8px; padding: 10px 12px; font-weight: 600; }
            QLabel#metric { color: #4b5563; padding: 2px 0; }
            QLabel#errorLabel { background: #fef3c7; color: #92400e;
                                border: 1px solid #fbbf24; border-radius: 6px;
                                padding: 8px; font-family: monospace; }
            """

        def _spin(self, lo, hi, val, step):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(2)
            return s

        def _set_battery_style(self, vbat):
            level = battery_level_name(vbat)
            color = {"good": "#16a34a", "low": "#f59e0b",
                     "critical": "#dc2626", "unknown": "#9ca3af"}[level]
            self.battery_bar.setStyleSheet(
                f"QProgressBar {{ background: #e5e7eb; border: none; "
                f"  border-radius: 5px; height: 10px; min-width: 150px; }}"
                f"QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}"
            )

        def _build_layout(self):
            root = QtWidgets.QHBoxLayout(self)

            left = QtWidgets.QVBoxLayout()

            # Connection
            conn_box = QtWidgets.QGroupBox("Connection")
            conn = QtWidgets.QFormLayout()
            conn.addRow("URI", self.uri_edit)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.connect_btn)
            row.addWidget(self.disconnect_btn)
            conn.addRow(row)
            conn_box.setLayout(conn)
            left.addWidget(conn_box)

            # Map setup
            map_box = QtWidgets.QGroupBox("Map setup (world coords, meters)")
            mform = QtWidgets.QFormLayout()
            wh = QtWidgets.QHBoxLayout()
            wh.addWidget(QtWidgets.QLabel("W"))
            wh.addWidget(self.map_w_spin)
            wh.addWidget(QtWidgets.QLabel("H"))
            wh.addWidget(self.map_h_spin)
            mform.addRow("Size", wh)
            s = QtWidgets.QHBoxLayout()
            s.addWidget(QtWidgets.QLabel("x"))
            s.addWidget(self.start_x_spin)
            s.addWidget(QtWidgets.QLabel("y"))
            s.addWidget(self.start_y_spin)
            mform.addRow("Start", s)
            g = QtWidgets.QHBoxLayout()
            g.addWidget(QtWidgets.QLabel("x"))
            g.addWidget(self.goal_x_spin)
            g.addWidget(QtWidgets.QLabel("y"))
            g.addWidget(self.goal_y_spin)
            mform.addRow("Goal", g)
            mform.addRow("Height", self.height_spin)
            map_btn_row = QtWidgets.QHBoxLayout()
            map_btn_row.addWidget(self.build_btn)
            map_btn_row.addWidget(self.reset_btn)
            mform.addRow(map_btn_row)
            map_box.setLayout(mform)
            left.addWidget(map_box)

            # Flight buttons
            fl_box = QtWidgets.QGroupBox("Flight")
            fl = QtWidgets.QVBoxLayout()
            fl.addWidget(self.start_btn)
            fl.addWidget(self.land_btn)
            fl.addWidget(self.estop_btn)
            fl_box.setLayout(fl)
            left.addWidget(fl_box)

            # Sensors
            sens_box = QtWidgets.QGroupBox("Sensors")
            sens = QtWidgets.QGridLayout()
            for row_i, k in enumerate(("front", "left", "right", "back", "up", "zrange")):
                sens.addWidget(QtWidgets.QLabel(k), row_i, 0)
                sens.addWidget(self.range_labels[k], row_i, 1)
            sens_box.setLayout(sens)
            left.addWidget(sens_box)

            left.addWidget(self.pose_label)

            # Battery
            bat = QtWidgets.QHBoxLayout()
            bat.addWidget(self.battery_label)
            bat.addWidget(self.battery_bar)
            left.addLayout(bat)

            left.addWidget(self.status_label)
            left.addWidget(self.error_label)
            left.addStretch(1)

            # Right: map
            right = QtWidgets.QVBoxLayout()
            map_title = QtWidgets.QLabel(
                "Occupancy Grid (top-down, drone forward = ↑)   "
                "red=occupied · green=free · blue=path · orange=goal distribution · x↑ red, y← green"
            )
            map_title.setStyleSheet("font-weight: 600; padding: 4px;")
            right.addWidget(map_title)
            right.addWidget(self.map_view, 1)

            root.addLayout(left, 0)
            root.addLayout(right, 1)

        # ---- handlers ----
        def on_connect(self):
            self.status_label.setText("Connecting…")
            self.connect_btn.setEnabled(False)
            self.worker.request_connect(self.uri_edit.text().strip())

        def on_disconnect(self):
            self.status_label.setText("Disconnecting…")
            self.worker.request_disconnect()

        def on_start_flight(self):
            # OGM이 이미 있으면 (매핑 중이었으면) 재사용, 없으면 새로 생성
            if self.ogm is None:
                self.ogm = OccupancyGrid(
                    self.map_w_spin.value(), self.map_h_spin.value()
                )
            start_xy = (self.start_x_spin.value(), self.start_y_spin.value())
            goal_xy = (self.goal_x_spin.value(), self.goal_y_spin.value())
            self.map_view.set_map(self.ogm, start_xy, goal_xy)
            self.error_label.setText("Flight in progress…")
            self.worker.request_fly(
                self.ogm, start_xy, goal_xy, self.height_spin.value()
            )

        def on_build_toggled(self, checked):
            if checked:
                # 매핑 시작
                if self.ogm is None:
                    self.ogm = OccupancyGrid(
                        self.map_w_spin.value(), self.map_h_spin.value()
                    )
                start_xy = (self.start_x_spin.value(), self.start_y_spin.value())
                goal_xy = (self.goal_x_spin.value(), self.goal_y_spin.value())
                self.map_view.set_map(self.ogm, start_xy, goal_xy)
                self.worker.request_build_map(self.ogm, start_xy, goal_xy)
                self.build_btn.setText("Stop Mapping")
                self.reset_btn.setEnabled(True)
                # 매핑 중에는 사이즈 변경 잠금 (논리 일관성)
                for w in (self.map_w_spin, self.map_h_spin,
                          self.start_x_spin, self.start_y_spin):
                    w.setEnabled(False)
            else:
                self.worker.request_stop_mapping()
                self.build_btn.setText("Build Map")
                for w in (self.map_w_spin, self.map_h_spin,
                          self.start_x_spin, self.start_y_spin):
                    w.setEnabled(True)

        def on_reset_map(self):
            self.worker.request_reset_map()
            self.map_view.path_world = []
            self.map_view.landing_world = None
            self.map_view.update()

        def on_land(self):
            self.worker.request_land()
            self.status_label.setText("Landing requested")

        def on_estop(self):
            self.worker.request_emergency_stop()
            self.status_label.setText("E-STOP")

        def on_connection_state(self, state):
            connected = state in ("connected", "flying")
            self.connect_btn.setEnabled(state == "disconnected")
            self.disconnect_btn.setEnabled(connected)
            self.start_btn.setEnabled(state == "connected")
            self.land_btn.setEnabled(state == "flying")
            self.estop_btn.setEnabled(connected)
            # Build Map은 connected에서만 (flying 중에는 자동 매핑이라 토글 무의미)
            self.build_btn.setEnabled(state == "connected")
            # Disconnect 시 매핑 토글도 풀기
            if state == "disconnected" and self.build_btn.isChecked():
                self.build_btn.blockSignals(True)
                self.build_btn.setChecked(False)
                self.build_btn.setText("Build Map")
                self.build_btn.blockSignals(False)
                self.reset_btn.setEnabled(False)
                for w in (self.map_w_spin, self.map_h_spin,
                          self.start_x_spin, self.start_y_spin):
                    w.setEnabled(True)

        def on_sensor(self, snap):
            for k in self.range_labels:
                self.range_labels[k].setText(fmt_distance(snap.get(k)))
            x = snap.get("x"); y = snap.get("y"); yaw = snap.get("yaw")
            if x is not None and y is not None:
                self.pose_label.setText(
                    f"x: {x:+.3f}  y: {y:+.3f}  yaw: {yaw if yaw is not None else 0.0:+.1f}°"
                )
                if self.ogm is not None:
                    start = (self.start_x_spin.value(), self.start_y_spin.value())
                    self.map_view.update_drone(start[0] + x, start[1] + y)
            vbat = snap.get("vbat")
            self.battery_label.setText(fmt_battery(vbat))
            self.battery_bar.setValue(battery_percent(vbat))
            self._set_battery_style(vbat)

        def on_flight_result(self, result):
            ex = result.get("error_x", 0.0)
            ey = result.get("error_y", 0.0)
            en = result.get("error_norm", 0.0)
            cd = result.get("commanded_drone", [0, 0])
            md = result.get("measured_drone", [0, 0])
            lw = result.get("landing_world")
            sensor_log = result.get("sensor_log")
            landing_text = (
                f"\nLanding world:      ({lw[0]:+.3f}, {lw[1]:+.3f}) m"
                if lw else ""
            )
            log_text = f"\nSensor log: {sensor_log}" if sensor_log else ""
            text = (
                f"Status:   {result.get('status', '?')}\n"
                f"Cmd (drone-frame):  ({cd[0]:+.3f}, {cd[1]:+.3f}) m\n"
                f"Meas (drone-frame): ({md[0]:+.3f}, {md[1]:+.3f}) m\n"
                f"Error:    dx={ex:+.3f}  dy={ey:+.3f}   ||e||={en:.3f} m"
                f"{landing_text}"
                f"{log_text}"
            )
            self.error_label.setText(text)

        def closeEvent(self, ev):
            self.worker.request_emergency_stop()
            self.worker.request_quit()
            self.worker.wait(2000)
            ev.accept()


# ===========================================================================
# Entrypoint
# ===========================================================================
def signal_handler(_sn, _fr):
    if _active_worker is not None:
        _active_worker.request_emergency_stop()


def parse_args():
    p = argparse.ArgumentParser(
        description="Crazyflie OGM Gaussian-goal landing"
    )
    p.add_argument("--uri", default=URI)
    p.add_argument("--target-height", type=float, default=TARGET_HEIGHT)
    return p.parse_args()


def main():
    args = parse_args()
    if not PYQT_AVAILABLE:
        print("PyQt5 not installed. Try: pip install PyQt5")
        return 1
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args)
    win.resize(1080, 740)
    win.show()
    global _active_worker
    _active_worker = win.worker
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
