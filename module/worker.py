"""Crazyflie worker thread: connection, idle mapping, and flight execution."""
import csv
import math
import threading
import time

from PyQt5 import QtCore

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from utils.config import (
    ARRIVAL_RADIUS,
    CONTROL_DT,
    DEFAULT_GOAL,
    DEFAULT_START,
    FRONT_AVOID_DIST,
    GOAL_SCAN_RADIUS,
    GOAL_SCAN_SPEED,
    GOAL_SEARCH_RADIUS,
    GOAL_SEEK_SPEED,
    GOAL_SIGMA,
    LANDING_DESCENT_MIN_DIST,
    LANDING_START_RADIUS,
    LANDING_STEP_M,
    LANDING_STEP_S,
    LANDING_Z_DELTA,
    LANDING_Z_STABLE_COUNT,
    LANDING_Z_STEP_DELTA,
    SAFETY_BRAKE_DIST,
    SAFETY_DIST,
    SENSOR_LOG_DIR,
    SIDE_AVOID_MAX_S,
    SIDE_RETURN_DIST,
    SIDE_STEP_SPEED,
    SPEED_X,
    SPEED_Y,
    TAKEOFF_STEP_M,
    TAKEOFF_STEP_S,
    TARGET_HEIGHT,
    URI,
)
from utils.control import (
    emergency_stop,
    reset_height_limiter,
    reset_velocity_limiter,
    send_arming_request,
    send_velocity_limited,
)
from utils.helpers import clamp, in_goal_distribution

from .reader import RangePoseReader


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
