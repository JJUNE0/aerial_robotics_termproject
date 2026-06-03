"""
Autonomous Level-3 round-trip mission for Crazyflie 2.1 Brushless.

State sequence:
  TAKEOFF → ROTATION_SCAN → NAV_TO_LANDING → LANDING_REGION_SCAN
  → PAD_CONFIRM → LANDING_ON_PAD → TAKEOFF_FROM_PAD → TURN_AROUND
  → NAV_TO_START → LANDING_ON_START → DONE
"""

import math
import threading
import time

import numpy as np
from typing import Optional, Tuple

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import config
import controller
from gui import MissionGUI
from logger import FlightLogger
from mapping import OccupancyGrid, HeightMap
from navigator import FrontierNavigator, LawnmowerNavigator, astar, simplify_path
from sensors import SensorData, SensorHub
from shared_state import SharedState


class EmergencyException(Exception):
    pass


class PadNotFoundException(Exception):
    pass


_last_map_pose: Optional[Tuple[float, float, float]] = None
_map_freeze_until: float = 0.0


# ------------------------------------------------------------------ helpers

def _range_contact_risk(data: SensorData) -> bool:
    r = data.ranges
    return any(d is not None and d < config.COLLISION_THRESHOLD
               for d in (r.front, r.back, r.left, r.right))


def _map_update_allowed(data: SensorData) -> bool:
    """Freeze map writes briefly when EKF pose is implausible after contact."""
    global _last_map_pose, _map_freeze_until

    now = time.time()
    x, y, _, _ = data.pose

    allowed = now >= _map_freeze_until
    if _last_map_pose is not None:
        px, py, pt = _last_map_pose
        dt = max(1e-3, now - pt)
        jump = math.hypot(x - px, y - py)
        speed = jump / dt
        if jump > config.POSE_JUMP_THRESHOLD or speed > config.POSE_SPEED_THRESHOLD:
            _map_freeze_until = now + config.MAP_FREEZE_TIME
            allowed = False

    if _range_contact_risk(data):
        _map_freeze_until = now + config.MAP_FREEZE_TIME
        allowed = False

    _last_map_pose = (x, y, now)
    return allowed

def _step(shared: SharedState, hub: SensorHub,
          occ: OccupancyGrid, hmap: HeightMap) -> SensorData:
    """Read sensors, update maps, push state to SharedState. Raise on emergency."""
    if shared.emergency_flag:
        raise EmergencyException()

    data = hub.read()
    x, y, z, yaw = data.pose

    shared.pose = data.pose
    shared.battery_pct = data.battery_pct
    hub.set_state(shared.current_state)
    if shared.target_pos is not None:
        hub.set_target(*shared.target_pos)
    else:
        hub.clear_target()
    if shared.local_scan_region is not None:
        hub.set_local_scan_region(shared.local_scan_region)
    else:
        hub.clear_local_scan_region()

    if _map_update_allowed(data):
        occ.update_all_rays(x, y, yaw, data.ranges)

    while not hub.edge_queue.empty():
        ev = hub.edge_queue.get_nowait()
        if shared.current_state in ('LANDING_REGION_SCAN',
                                    'LOCAL_EDGE_SCAN',
                                    'PAD_CONFIRM'):
            if ev.kind == 'entry':
                hmap.add_entry(ev.x, ev.y, ev.timestamp)
            else:
                hmap.add_exit(ev.x, ev.y, ev.timestamp)
            shared.add_height_map_edge(ev)

    shared.occupancy_grid = occ.snapshot()
    shared.pad_pairs = hmap.get_pairs()
    shared.pad_candidates = hmap.get_candidates()

    return data


def _obstacle_in_direction(data: SensorData, dx: float, dy: float,
                           yaw_deg: float) -> bool:
    """Check if range sensors block intended motion.

    Sensors aligned with the motion use a larger margin; perpendicular
    sensors use the minimum side margin.
    """
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return False

    motion_deg = math.degrees(math.atan2(dy, dx))
    rel = (motion_deg - yaw_deg + 360) % 360

    tf = config.NAV_FORWARD_MARGIN
    ts = config.NAV_SIDE_MARGIN
    r = data.ranges
    Ff = r.front is not None and r.front < tf
    Bf = r.back  is not None and r.back  < tf
    Lf = r.left  is not None and r.left  < tf
    Rf = r.right is not None and r.right < tf
    Fs = r.front is not None and r.front < ts
    Bs = r.back  is not None and r.back  < ts
    Ls = r.left  is not None and r.left  < ts
    Rs = r.right is not None and r.right < ts

    if   rel < 22.5  or rel >= 337.5: return Ff or Ls or Rs
    elif rel < 67.5:                  return Ff or Lf
    elif rel < 112.5:                 return Lf or Fs or Bs
    elif rel < 157.5:                 return Bf or Lf
    elif rel < 202.5:                 return Bf or Ls or Rs
    elif rel < 247.5:                 return Bf or Rf
    elif rel < 292.5:                 return Rf or Fs or Bs
    else:                             return Ff or Rf


def _clamp_to_arena(x: float, y: float) -> Tuple[float, float]:
    m = config.BOUNDARY_MARGIN
    return (
        max(config.ekf_arena_x_min() + m,
            min(config.ekf_arena_x_max() - m, x)),
        max(config.ekf_arena_y_min() + m,
            min(config.ekf_arena_y_max() - m, y)),
    )


def _boundary_avoid_vector(x: float, y: float) -> Tuple[float, float]:
    m = config.BOUNDARY_MARGIN
    vx = vy = 0.0
    if x < config.ekf_arena_x_min() + m:
        vx += 1.0
    if x > config.ekf_arena_x_max() - m:
        vx -= 1.0
    if y < config.ekf_arena_y_min() + m:
        vy += 1.0
    if y > config.ekf_arena_y_max() - m:
        vy -= 1.0
    return vx, vy


def _boundary_too_close(data: SensorData) -> bool:
    x, y, _, _ = data.pose
    bx, by = _boundary_avoid_vector(x, y)
    return abs(bx) > 1e-6 or abs(by) > 1e-6


def _moving_out_of_bounds(data: SensorData, dx: float, dy: float) -> bool:
    x, y, _, _ = data.pose
    bx, by = _boundary_avoid_vector(x, y)
    if abs(bx) < 1e-6 and abs(by) < 1e-6:
        return False
    return dx * bx + dy * by < 0.0


def _too_close_any(data: SensorData) -> bool:
    r = data.ranges
    t = config.STOP_THRESHOLD
    return (_boundary_too_close(data) or
            any(d is not None and d < t
                for d in (r.front, r.back, r.left, r.right)))


def _nudge_away_from_obstacles(cf, shared, hub, occ, hmap,
                               data: SensorData) -> bool:
    """Move about SAFETY_NUDGE_DIST away from any close horizontal range."""
    x, y, _, yaw_deg = data.pose
    yaw = math.radians(yaw_deg)
    heading = (math.cos(yaw), math.sin(yaw))
    left = (-math.sin(yaw), math.cos(yaw))

    bx, by = _boundary_avoid_vector(x, y)
    vx, vy = bx * 2.0, by * 2.0
    t = config.STOP_THRESHOLD
    ranges = data.ranges
    checks = [
        (ranges.front, -heading[0], -heading[1]),
        (ranges.back,   heading[0],  heading[1]),
        (ranges.left,   -left[0],    -left[1]),
        (ranges.right,   left[0],     left[1]),
    ]
    for dist, ax, ay in checks:
        if dist is not None and dist < t:
            weight = max(0.2, (t - dist) / t)
            vx += ax * weight
            vy += ay * weight

    mag = math.hypot(vx, vy)
    if mag < 1e-6:
        closest = min(
            ((dist, ax, ay) for dist, ax, ay in checks if dist is not None),
            default=None)
        if closest is None:
            return False
        _, vx, vy = closest
        mag = math.hypot(vx, vy)
        if mag < 1e-6:
            return False

    vx /= mag
    vy /= mag
    speed = config.SAFETY_NUDGE_SPEED
    duration = config.SAFETY_NUDGE_DIST / speed
    end_t = time.time() + duration

    while time.time() < end_t:
        data = _step(shared, hub, occ, hmap)
        _, _, _, cyaw = data.pose
        vx_b, vy_b = controller.vel_to_body(vx * speed, vy * speed, cyaw)
        yaw_rate = _search_yaw_hold_rate(shared, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
        time.sleep(config.DT)

    cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
    time.sleep(0.1)
    return True


# ------------------------------------------------------------------ states

def _hold_hover(cf, duration: float, z: float = config.FLIGHT_Z):
    end_t = time.time() + duration
    while time.time() < end_t:
        cf.commander.send_hover_setpoint(0, 0, 0, z)
        time.sleep(config.DT)


def _yaw_error_deg(target: float, current: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _search_yaw_hold_rate(shared: SharedState, current_yaw: float) -> float:
    if shared.current_state not in ('LANDING_REGION_SCAN',
                                    'LOCAL_EDGE_SCAN',
                                    'PAD_CONFIRM'):
        return 0.0

    err = _yaw_error_deg(shared.home_yaw, current_yaw)
    if abs(err) < config.YAW_ALIGN_TOL:
        return 0.0
    return max(-config.YAW_ALIGN_RATE,
               min(config.YAW_ALIGN_RATE, err * config.YAW_HOLD_KP))


def _align_yaw(cf, shared, hub, occ, hmap, target_yaw: float):
    shared.current_state = 'YAW_ALIGN'
    end_t = time.time() + config.YAW_ALIGN_TIMEOUT
    while time.time() < end_t:
        data = _step(shared, hub, occ, hmap)
        err = _yaw_error_deg(target_yaw, data.pose[3])
        if abs(err) < config.YAW_ALIGN_TOL:
            break
        yaw_rate = max(-config.YAW_ALIGN_RATE,
                       min(config.YAW_ALIGN_RATE, err * 1.5))
        cf.commander.send_hover_setpoint(0, 0, yaw_rate, config.FLIGHT_Z)
        time.sleep(config.DT)
    cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
    _hold_hover(cf, 0.3)


def do_takeoff(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF'
    shared.start_timer()
    controller.init_ekf(cf)
    controller.takeoff_vel(cf, config.FLIGHT_Z)
    data = _step(shared, hub, occ, hmap)

    # Record stabilised hover position as home — EKF may have drifted during takeoff
    home_x, home_y = data.pose[0], data.pose[1]
    shared.home_pos = (home_x, home_y)
    shared.home_yaw = data.pose[3]
    print(f'[mission] home recorded: ({home_x:.3f}, {home_y:.3f})')


def do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg: float = config.SCAN_ROTATE_ANGLE):
    """Rotate with yaw-rate control while holding the scan start position."""
    shared.current_state = 'ROTATION_SCAN'

    data = _step(shared, hub, occ, hmap)
    hold_x, hold_y = data.pose[0], data.pose[1]

    duration = abs(angle_deg) / config.SCAN_ROTATE_RATE
    yaw_rate = math.copysign(config.SCAN_ROTATE_RATE, angle_deg)  # deg/s

    hold_kp = 2.0
    max_hold_speed = min(0.10, config.NAV_SPEED)

    end_t = time.time() + duration
    while time.time() < end_t:
        data = _step(shared, hub, occ, hmap)
        if _too_close_any(data):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            _nudge_away_from_obstacles(cf, shared, hub, occ, hmap, data)
            data = _step(shared, hub, occ, hmap)
            hold_x, hold_y = data.pose[0], data.pose[1]
            continue

        cx, cy, _, cyaw = data.pose
        ex, ey = hold_x - cx, hold_y - cy

        vx_w = max(-max_hold_speed, min(max_hold_speed, ex * hold_kp))
        vy_w = max(-max_hold_speed, min(max_hold_speed, ey * hold_kp))
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)

        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
        time.sleep(config.DT)

    cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
    _step(shared, hub, occ, hmap)


def _scan_navigate(cf, shared, hub, occ, hmap, tx, ty, speed=config.SCAN_SPEED):
    """Navigate at scan speed and stop immediately when entry event fires.
    Returns ('entry', pos), ('arrived', None), or ('blocked', None)."""
    KP = 3.0
    tx, ty = _clamp_to_arena(tx, ty)
    prev_seq = hmap.entry_seq

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, cyaw = data.pose

        if hmap.entry_seq > prev_seq:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            time.sleep(0.3)
            return 'entry', hmap.last_entry_pos

        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)

        if dist < config.NAV_ARRIVE_THRESHOLD:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            return 'arrived', None

        if _too_close_any(data):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            _nudge_away_from_obstacles(cf, shared, hub, occ, hmap, data)
            return 'blocked', None

        if _moving_out_of_bounds(data, dx, dy):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            return 'blocked', None

        if _obstacle_in_direction(data, dx, dy, cyaw):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            return 'blocked', None

        v = min(speed, dist * KP)
        vx_b, vy_b = controller.vel_to_body((dx / dist) * v, (dy / dist) * v, cyaw)
        yaw_rate = _search_yaw_hold_rate(shared, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
        time.sleep(config.DT)


def _edge_seq(hmap: HeightMap, kind: str) -> int:
    return hmap.entry_seq if kind == 'entry' else hmap.exit_seq


def _last_edge_pos(hmap: HeightMap, kind: str) -> Optional[Tuple[float, float]]:
    return hmap.last_entry_pos if kind == 'entry' else hmap.last_exit_pos


def _move_until_edge(cf, shared, hub, occ, hmap,
                     tx: float, ty: float, edge_kind: str,
                     speed: float) -> Optional[Tuple[float, float]]:
    """Move toward target and stop on the requested edge event."""
    KP = 3.0
    tx, ty = _clamp_to_arena(tx, ty)
    prev_seq = _edge_seq(hmap, edge_kind)
    shared.target_pos = (tx, ty)

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, cyaw = data.pose

        if _edge_seq(hmap, edge_kind) > prev_seq:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            shared.target_pos = None
            time.sleep(0.2)
            return _last_edge_pos(hmap, edge_kind)

        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)
        if dist < config.NAV_ARRIVE_THRESHOLD:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            shared.target_pos = None
            return None

        if _too_close_any(data):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            _nudge_away_from_obstacles(cf, shared, hub, occ, hmap, data)
            shared.target_pos = None
            return None

        if _moving_out_of_bounds(data, dx, dy):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            shared.target_pos = None
            return None

        if _obstacle_in_direction(data, dx, dy, cyaw):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            shared.target_pos = None
            return None

        v = min(speed, dist * KP)
        vx_b, vy_b = controller.vel_to_body((dx / dist) * v,
                                             (dy / dist) * v, cyaw)
        yaw_rate = _search_yaw_hold_rate(shared, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
        time.sleep(config.DT)


def do_square_pad_verify(cf, shared, hub, occ, hmap,
                         entry_x: float, entry_y: float,
                         dir_x: float, dir_y: float) -> Optional[Tuple[float, float]]:
    """Verify a square pad by crossing two perpendicular sides."""
    mag = math.hypot(dir_x, dir_y)
    if mag < 1e-6:
        return None
    ux, uy = dir_x / mag, dir_y / mag
    px, py = -uy, ux

    shared.current_state = 'PAD_CONFIRM'
    hmap.reset()
    hub.reset_edge_detector()
    hmap.add_entry(entry_x, entry_y)

    half = config.PAD_SIZE / 2.0
    margin = config.SQUARE_VERIFY_MARGIN
    speed = config.LOCAL_SCAN_SPEED

    center_guess = (entry_x + ux * half, entry_y + uy * half)
    _navigate_to(cf, shared, hub, occ, hmap,
                 center_guess[0], center_guess[1], config.FLIGHT_Z, speed)
    _hold_hover(cf, config.PAD_CONFIRM_TIME)

    exit1_target = (entry_x + ux * (config.PAD_SIZE + margin),
                    entry_y + uy * (config.PAD_SIZE + margin))
    exit1 = _move_until_edge(cf, shared, hub, occ, hmap,
                             exit1_target[0], exit1_target[1], 'exit', speed)
    if exit1 is None:
        shared.current_state = 'LANDING_REGION_SCAN'
        return None

    side1 = math.hypot(exit1[0] - entry_x, exit1[1] - entry_y)
    if abs(side1 - config.PAD_SIZE) > config.PAD_SIDE_TOL:
        shared.current_state = 'LANDING_REGION_SCAN'
        return None

    reentry1_target = (exit1[0] - ux * (side1 + margin),
                       exit1[1] - uy * (side1 + margin))
    reentry1 = _move_until_edge(cf, shared, hub, occ, hmap,
                                reentry1_target[0], reentry1_target[1],
                                'entry', speed)
    if reentry1 is None:
        shared.current_state = 'LANDING_REGION_SCAN'
        return None

    center_x = (entry_x + exit1[0]) / 2.0
    center_y = (entry_y + exit1[1]) / 2.0
    _navigate_to(cf, shared, hub, occ, hmap,
                 center_x, center_y, config.FLIGHT_Z, speed)
    _hold_hover(cf, config.PAD_CONFIRM_TIME)

    exit2_target = (center_x + px * (side1 / 2.0 + margin),
                    center_y + py * (side1 / 2.0 + margin))
    exit2 = _move_until_edge(cf, shared, hub, occ, hmap,
                             exit2_target[0], exit2_target[1], 'exit', speed)
    if exit2 is None:
        shared.current_state = 'LANDING_REGION_SCAN'
        return None

    side2 = 2.0 * math.hypot(exit2[0] - center_x, exit2[1] - center_y)
    if abs(side2 - side1) > config.PAD_SIDE_TOL:
        shared.current_state = 'LANDING_REGION_SCAN'
        return None

    reentry2_target = (exit2[0] - px * (side2 / 2.0 + margin),
                       exit2[1] - py * (side2 / 2.0 + margin))
    reentry2 = _move_until_edge(cf, shared, hub, occ, hmap,
                                reentry2_target[0], reentry2_target[1],
                                'entry', speed)
    if reentry2 is None:
        shared.current_state = 'LANDING_REGION_SCAN'
        return None

    _navigate_to(cf, shared, hub, occ, hmap,
                 center_x, center_y, config.FLIGHT_Z, speed)
    shared.current_state = 'LANDING_REGION_SCAN'
    return (center_x, center_y)


def _verify_or_local_retry(cf, shared, hub, occ, hmap,
                           entry_x: float, entry_y: float,
                           dir_x: float, dir_y: float) -> Optional[Tuple[float, float]]:
    """Search locally around an edge first, then verify square geometry."""
    candidates = do_local_edge_scan(cf, shared, hub, occ, hmap, entry_x, entry_y)
    local_pairs = hmap.get_pairs()
    for pair in local_pairs:
        if len(pair) < 6:
            continue
        _, _, enx, eny, exx, exy = pair
        vx, vy = exx - enx, exy - eny
        pad_pos = do_square_pad_verify(
            cf, shared, hub, occ, hmap, enx, eny, vx, vy)
        if pad_pos is not None:
            return pad_pos

    half = config.PAD_SIZE / 2.0
    for cand in candidates:
        for vx, vy in ((dir_x, dir_y), (1.0, 0.0), (-1.0, 0.0),
                       (0.0, 1.0), (0.0, -1.0)):
            mag = math.hypot(vx, vy)
            if mag < 1e-6:
                continue
            ux, uy = vx / mag, vy / mag
            pad_pos = do_square_pad_verify(
                cf, shared, hub, occ, hmap,
                cand.cx - ux * half, cand.cy - uy * half,
                ux, uy)
            if pad_pos is not None:
                return pad_pos

    return None


def do_precise_pad_scan(cf, shared, hub, occ, hmap, rough_x, rough_y):
    """From rough entry position, sweep ±PRECISE_SWEEP_DIST in 4 directions.
    Collects entry-exit pairs to confirm and refine pad position."""
    shared.current_state = 'PAD_CONFIRM'
    hmap.reset()
    hub.reset_edge_detector()

    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        # Return to rough position
        _navigate_to(cf, shared, hub, occ, hmap,
                     rough_x, rough_y, config.FLIGHT_Z, config.SCAN_SPEED)
        hub.reset_edge_detector()
        time.sleep(config.EDGE_COOLDOWN)

        # Record start as entry
        hmap.add_entry(rough_x, rough_y)
        prev_exit_seq = hmap.exit_seq

        tx = rough_x + ddx * config.PRECISE_SWEEP_DIST
        ty = rough_y + ddy * config.PRECISE_SWEEP_DIST
        KP = 3.0

        while True:
            data = _step(shared, hub, occ, hmap)
            cx, cy, _, cyaw = data.pose

            if _too_close_any(data):
                cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
                _nudge_away_from_obstacles(cf, shared, hub, occ, hmap, data)
                break

            if hmap.exit_seq > prev_exit_seq:
                ep = hmap.last_exit_pos
                mx = ep[0] + ddx * config.PRECISE_EXIT_MARGIN
                my = ep[1] + ddy * config.PRECISE_EXIT_MARGIN
                _navigate_to(cf, shared, hub, occ, hmap,
                             mx, my, config.FLIGHT_Z, config.SCAN_SPEED)
                time.sleep(config.EDGE_COOLDOWN)
                break

            ex, ey = tx - cx, ty - cy
            dist = math.hypot(ex, ey)
            if dist < 0.05:
                break

            v = min(config.SCAN_SPEED, dist * KP)
            vx_b, vy_b = controller.vel_to_body((ex/dist)*v, (ey/dist)*v, cyaw)
            yaw_rate = _search_yaw_hold_rate(shared, cyaw)
            cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
            time.sleep(config.DT)

    return hmap.get_candidates()


def do_local_edge_scan(cf, shared, hub, occ, hmap,
                       edge_x: float, edge_y: float):
    """Slow dense lawnmower inside a circular area around the first edge."""
    shared.current_state = 'LOCAL_EDGE_SCAN'

    radius = config.LOCAL_SCAN_RADIUS
    x_min = max(config.ekf_landing_region_start() + 0.10,
                edge_x - radius)
    x_max = min(config.ekf_arena_x_max() - 0.10,
                edge_x + radius)
    y_min = max(config.ekf_arena_y_min() + 0.10,
                edge_y - radius)
    y_max = min(config.ekf_arena_y_max() - 0.10,
                edge_y + radius)
    shared.local_scan_region = (
        edge_x - radius, edge_x + radius,
        edge_y - radius, edge_y + radius)

    hmap.reset()
    hub.reset_edge_detector()

    y = y_min
    going_right = True
    while y <= y_max + 1e-6:
        dy = y - edge_y
        if abs(dy) > radius:
            y += config.LOCAL_SCAN_ROW_SPACING
            continue

        x_span = math.sqrt(max(0.0, radius * radius - dy * dy))
        row_x_min = max(x_min, edge_x - x_span)
        row_x_max = min(x_max, edge_x + x_span)
        if row_x_max - row_x_min < config.NAV_ARRIVE_THRESHOLD:
            y += config.LOCAL_SCAN_ROW_SPACING
            continue

        x_start = row_x_min if going_right else row_x_max
        x_target = row_x_max if going_right else row_x_min

        if not _navigate_to(cf, shared, hub, occ, hmap,
                            x_start, y, config.FLIGHT_Z, config.LOCAL_SCAN_SPEED):
            y += config.LOCAL_SCAN_ROW_SPACING
            going_right = not going_right
            continue

        shared.current_state = 'LOCAL_EDGE_SCAN'
        hub.reset_edge_detector()
        scan_status, _ = _scan_navigate(
            cf, shared, hub, occ, hmap,
            x_target, y, speed=config.LOCAL_SCAN_SPEED)
        if scan_status == 'blocked':
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)

        y += config.LOCAL_SCAN_ROW_SPACING
        going_right = not going_right

    shared.current_state = 'LANDING_REGION_SCAN'
    shared.local_scan_region = None
    return hmap.get_candidates()


def _navigate_to(cf, shared, hub, occ, hmap,
                 tx: float, ty: float, tz: float,
                 speed: float) -> bool:
    """P-controller velocity navigation. Returns True if arrived, False if blocked."""
    KP = 3.0
    tx, ty = _clamp_to_arena(tx, ty)
    shared.target_pos = (tx, ty)
    tz = min(tz, config.CEILING_LIMIT - 0.10)

    last_progress_t = time.time()
    last_dist = float('inf')
    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, cyaw = data.pose
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)

        if dist < config.NAV_ARRIVE_THRESHOLD:
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            shared.target_pos = None
            return True

        if dist < last_dist - 0.02:
            last_dist = dist
            last_progress_t = time.time()

        if time.time() - last_progress_t > config.NAV_SETTLE_TIMEOUT:
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            shared.target_pos = None
            return dist < 0.15  # close enough counts as arrived

        if _too_close_any(data):
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            _nudge_away_from_obstacles(cf, shared, hub, occ, hmap, data)
            shared.target_pos = None
            return False

        if _moving_out_of_bounds(data, dx, dy):
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            shared.target_pos = None
            return False

        if _obstacle_in_direction(data, dx, dy, cyaw):
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            time.sleep(0.3)
            shared.target_pos = None
            return False

        v = min(speed, dist * KP)
        vx_b, vy_b = controller.vel_to_body((dx / dist) * v,
                                             (dy / dist) * v, cyaw)
        yaw_rate = _search_yaw_hold_rate(shared, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, tz)
        time.sleep(config.DT)


def _retreat_from_target(cf, shared, hub, occ, hmap,
                         blocked_tx: float, blocked_ty: float) -> bool:
    """Back away from a blocked target before selecting the next frontier."""
    data = _step(shared, hub, occ, hmap)
    cx, cy, _, _ = data.pose
    dx, dy = blocked_tx - cx, blocked_ty - cy
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return False

    step = config.REPLAN_RETREAT_DIST
    rx = cx - (dx / dist) * step
    ry = cy - (dy / dist) * step
    rx = max(config.ekf_arena_x_min() + 0.10,
             min(config.ekf_arena_x_max() - 0.10, rx))
    ry = max(config.ekf_arena_y_min() + 0.10,
             min(config.ekf_arena_y_max() - 0.10, ry))

    return _navigate_to(cf, shared, hub, occ, hmap,
                        rx, ry, config.FLIGHT_Z, config.SCAN_SPEED)


def _navigate_via_grid(cf, shared, hub, occ, hmap,
                       tx: float, ty: float, speed: float,
                       nav_state: Optional[str] = None,
                       max_replans: int = 1) -> bool:
    """Navigate to a point through the inflated occupancy grid."""
    tx, ty = _clamp_to_arena(tx, ty)
    old_state = shared.current_state
    if nav_state is not None:
        shared.current_state = nav_state

    try:
        for _ in range(max_replans + 1):
            data = _step(shared, hub, occ, hmap)
            cx, cy, _, _ = data.pose
            grid = occ.snapshot()
            path = astar(grid,
                         occ.world_to_cell(cx, cy),
                         occ.world_to_cell(tx, ty))

            if path is None or len(path) <= 1:
                return _navigate_to(cf, shared, hub, occ, hmap,
                                    tx, ty, config.FLIGHT_Z, speed)

            waypoints = [_clamp_to_arena(*occ.cell_to_world(r, c))
                         for r, c in simplify_path(path, grid)[1:]]
            blocked = False
            for wx, wy in waypoints:
                if not _navigate_to(cf, shared, hub, occ, hmap,
                                    wx, wy, config.FLIGHT_Z, speed):
                    blocked = True
                    _retreat_from_target(cf, shared, hub, occ, hmap, wx, wy)
                    break

            if not blocked:
                return True

        return False
    finally:
        if nav_state is not None:
            shared.current_state = old_state


def do_nav_to_landing(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_LANDING'
    frontier = FrontierNavigator(occ)
    target_x = config.ekf_landing_region_start()
    y_center = config.ARENA_Y / 2.0 - config.TAKEOFF_PAD_Y   # Y 중앙 (EKF 좌표)

    failed_plans = 0    # consecutive planning failures
    MAX_FAILED = 5
    blocked_targets = []

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, cz, cyaw = data.pose

        if cx >= target_x:
            break

        # Within 20 cm of the boundary — navigate directly rather than re-planning
        if target_x - cx <= 0.20:
            _navigate_to(cf, shared, hub, occ, hmap,
                         target_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            break

        target = frontier.find_max_x_target(
            cx, cy, x_limit=target_x,
            avoid_targets=blocked_targets,
            avoid_radius=config.TARGET_BLOCK_RADIUS)
        if target is not None:
            target = _clamp_to_arena(*target)
        if target is None or target[0] <= cx + 0.1:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                blocked_targets.clear()
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                # +X 막힐 때마다 Y 중앙 방향으로 10cm씩 이동 후 즉시 재확인
                step = 0.10 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        grid = occ.snapshot()
        path = astar(grid,
                     occ.world_to_cell(cx, cy),
                     occ.world_to_cell(target[0], target[1]))

        if path is None or len(path) <= 1:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                blocked_targets.clear()
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                step = 0.10 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        # Follow the planned cells instead of cutting straight through obstacles.
        waypoints = [occ.cell_to_world(r, c)
                     for r, c in simplify_path(path, grid)[1:]]
        blocked = False
        for wx, wy in waypoints:
            if not _navigate_to(cf, shared, hub, occ, hmap,
                                wx, wy, config.FLIGHT_Z, config.NAV_SPEED):
                blocked = True
                break

        if blocked:
            failed_plans += 1
            blocked_targets.append(target)
            blocked_targets = blocked_targets[-8:]
            _retreat_from_target(cf, shared, hub, occ, hmap,
                                 target[0], target[1])
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                blocked_targets.clear()
                failed_plans = 0
        else:
            failed_plans = 0


def do_landing_region_scan(cf, shared, hub, occ, hmap) -> Optional[Tuple[float, float]]:
    """Row-based X sweep lawnmower. Advances in Y, sweeps in X.
    If blocked, stop that row and continue with the next scan row.
    """
    shared.current_state = 'LANDING_REGION_SCAN'
    hmap.reset()
    hub.reset_edge_detector()

    x_entry  = config.ekf_landing_region_start() + 0.10
    x_end    = config.ekf_arena_x_max() - 0.10
    y_min    = config.ekf_arena_y_min() + 0.10
    y_max    = config.ekf_arena_y_max() - 0.10
    y_center = config.ARENA_Y / 2.0 - config.TAKEOFF_PAD_Y

    def _scan_move(tx, ty, speed=config.SCAN_SPEED):
        shared.current_state = 'LANDING_REGION_SCAN'
        return _navigate_to(cf, shared, hub, occ, hmap,
                            tx, ty, config.FLIGHT_Z, speed)

    # Decide Y start and step direction based on current position
    data = _step(shared, hub, occ, hmap)
    cy = data.pose[1]
    if cy > y_center:
        y = max(y_min, min(y_max, cy))
        y_step = -config.SCAN_ROW_SPACING
    else:
        y = max(y_min, min(y_max, cy))
        y_step = config.SCAN_ROW_SPACING

    # Start the search from the far side, so the first active sweep is -X.
    shared.current_state = 'NAV_TO_SCAN_START'
    _navigate_via_grid(
        cf, shared, hub, occ, hmap, x_end, y, config.NAV_SPEED,
        nav_state='NAV_TO_SCAN_START',
        max_replans=config.LANDING_AVOID_REPLANS)
    _align_yaw(cf, shared, hub, occ, hmap, shared.home_yaw)
    shared.current_state = 'LANDING_REGION_SCAN'
    hmap.reset()
    hub.reset_edge_detector()

    going_right = False

    while y_min <= y <= y_max:
        # 1. Set Y position for this row
        data = _step(shared, hub, occ, hmap)
        prev_entry_seq = hmap.entry_seq
        row_start = (data.pose[0], data.pose[1])
        if not _scan_move(data.pose[0], y):
            y += y_step
            continue
        if hmap.entry_seq > prev_entry_seq:
            entry_pos = hmap.last_entry_pos
            dir_x = 0.0
            dir_y = y - row_start[1]
            if abs(dir_y) < 1e-6:
                dir_y = y_step
            pad_pos = _verify_or_local_retry(
                cf, shared, hub, occ, hmap,
                entry_pos[0], entry_pos[1], dir_x, dir_y)
            if pad_pos is not None:
                shared.current_state = 'LANDING_REGION_SCAN'
                return pad_pos
        hub.reset_edge_detector()

        # 2. X sweep — stop immediately on entry, then search locally
        x_target = x_end if going_right else x_entry
        while True:
            scan_status, entry_pos = _scan_navigate(
                cf, shared, hub, occ, hmap, x_target, y)

            if scan_status == 'entry':
                scan_dir_x = 1.0 if going_right else -1.0
                pad_pos = _verify_or_local_retry(
                    cf, shared, hub, occ, hmap,
                    entry_pos[0], entry_pos[1], scan_dir_x, 0.0)
                if pad_pos is not None:
                    shared.current_state = 'LANDING_REGION_SCAN'
                    return pad_pos

                # Not confirmed — resume scan from current position
                shared.current_state = 'LANDING_REGION_SCAN'
                hmap.reset()
                hub.reset_edge_detector()
                data = _step(shared, hub, occ, hmap)
                x_target = x_end if going_right else x_entry
                continue

            if scan_status == 'blocked':
                shared.current_state = 'LANDING_REGION_SCAN'
                hmap.reset()
                hub.reset_edge_detector()
                break

            # Arrived at row end.
            data = _step(shared, hub, occ, hmap)
            if math.hypot(x_target - data.pose[0], y - data.pose[1]) > 0.10:
                break
            break

        # 3. Advance Y and flip direction
        y += y_step
        going_right = not going_right

    return None


def do_pad_confirm(cf, shared, hub, occ, hmap) -> Optional[Tuple[float, float]]:
    shared.current_state = 'PAD_CONFIRM'
    candidates = hmap.get_candidates()

    if not candidates:
        return None

    for cand in candidates:
        arrived = _navigate_to(cf, shared, hub, occ, hmap,
                               cand.cx, cand.cy, config.FLIGHT_Z, config.SCAN_SPEED)
        if not arrived:
            continue

        # Hover and count confirmations from z-ranger
        confirm_needed = int(config.PAD_CONFIRM_TIME / config.DT)
        confirm_count = 0
        t_end = time.time() + config.PAD_CONFIRM_TIME * 3

        while time.time() < t_end:
            data = _step(shared, hub, occ, hmap)
            zd = data.ranges.down
            # Over a 10 cm pad at 50 cm altitude → z_down ≈ 40 cm
            if zd is not None and zd < (config.FLIGHT_Z - config.PAD_HEIGHT + 0.06):
                confirm_count += 1
            else:
                confirm_count = max(0, confirm_count - 1)

            if confirm_count >= confirm_needed:
                return (cand.cx, cand.cy)
            time.sleep(config.DT)

    return None


def do_land_on_pad(cf, shared, hub, occ, hmap, pad_x: float, pad_y: float):
    shared.current_state = 'LANDING_ON_PAD'
    _navigate_to(cf, shared, hub, occ, hmap,
                 pad_x, pad_y, config.FLIGHT_Z, config.SCAN_SPEED)
    controller.land_vel(cf, hub)


def do_takeoff_from_pad(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF_FROM_PAD'
    controller.arm(cf)
    controller.takeoff_vel(cf, config.FLIGHT_Z)
    _step(shared, hub, occ, hmap)


def do_turn_around(cf, shared, hub, occ, hmap):
    shared.current_state = 'TURN_AROUND'
    do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)


def do_nav_to_start(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_START'
    data = _step(shared, hub, occ, hmap)
    cx, cy, cz, _ = data.pose

    hx, hy = shared.home_pos
    gr, gc = occ.world_to_cell(hx, hy)

    def _plan(ox, oy):
        g = occ.snapshot()
        sr, sc = occ.world_to_cell(ox, oy)
        p = astar(g, (sr, sc), (gr, gc))
        if p is not None and len(p) > 1:
            p = simplify_path(p, g)
            return [occ.cell_to_world(r, c) for r, c in p[1:]]
        return [(hx, hy)]

    waypoints = _plan(cx, cy)
    max_replans = 3
    replans = 0
    i = 0

    while i < len(waypoints):
        wx, wy = waypoints[i]
        arrived = _navigate_to(cf, shared, hub, occ, hmap,
                               wx, wy, config.FLIGHT_Z, config.NAV_SPEED)
        if arrived:
            i += 1
            continue

        do_rotation_scan(cf, shared, hub, occ, hmap)
        replans += 1
        if replans > max_replans:
            break

        data = _step(shared, hub, occ, hmap)
        cx, cy, _, _ = data.pose
        new_wps = _plan(cx, cy)
        if new_wps != [(0.0, 0.0)] or (cx ** 2 + cy ** 2) > 0.25:
            waypoints = new_wps
            i = 0
        else:
            i += 1   # can't replan — skip to next waypoint


def do_land_on_start(cf, shared, hub, occ, hmap):
    shared.current_state = 'LANDING_ON_START'
    hx, hy = shared.home_pos
    _navigate_to(cf, shared, hub, occ, hmap,
                 hx, hy, config.FLIGHT_Z, config.SCAN_SPEED)
    controller.land_vel(cf, hub)
    shared.current_state = 'DONE'


# ------------------------------------------------------------------ mission

def run_mission(cf, shared: SharedState):
    occ = OccupancyGrid()
    hmap = HeightMap()
    logger = FlightLogger()
    hub = SensorHub(cf, logger=logger)
    hub.start()

    try:
        do_takeoff(cf, shared, hub, occ, hmap)
        do_rotation_scan(cf, shared, hub, occ, hmap)
        _align_yaw(cf, shared, hub, occ, hmap, shared.home_yaw)
        do_nav_to_landing(cf, shared, hub, occ, hmap)
        pad_pos = do_landing_region_scan(cf, shared, hub, occ, hmap)
        if pad_pos is None:
            raise PadNotFoundException()

        shared.landing_target = pad_pos
        do_land_on_pad(cf, shared, hub, occ, hmap, pad_pos[0], pad_pos[1])
        do_takeoff_from_pad(cf, shared, hub, occ, hmap)
        do_turn_around(cf, shared, hub, occ, hmap)
        do_nav_to_start(cf, shared, hub, occ, hmap)
        do_land_on_start(cf, shared, hub, occ, hmap)

    except EmergencyException:
        shared.current_state = 'EMERGENCY_LAND'
        try:
            controller.land_vel(cf, hub)
        except Exception:
            pass

    except PadNotFoundException:
        shared.current_state = 'PAD_NOT_FOUND'
        shared.landing_target = None
        print('[mission] pad not found — returning to start instead of landing on a fake pad')
        try:
            do_nav_to_start(cf, shared, hub, occ, hmap)
            do_land_on_start(cf, shared, hub, occ, hmap)
        except Exception as exc:
            print(f'[mission] return after pad-not-found failed: {exc}')
            try:
                controller.land_vel(cf, hub)
            except Exception:
                pass

    except Exception as exc:
        shared.current_state = 'ERROR'
        print(f'[mission] unhandled exception: {exc}')
        try:
            controller.land_vel(cf, hub)
        except Exception:
            pass

    finally:
        hub.stop()
        try:
            np.savez(logger.occ_path,
                     grid=occ.snapshot(),
                     res=np.array([config.OCCUPANCY_GRID_RES]),
                     x_min=np.array([occ.x_min]),
                     y_min=np.array([occ.y_min]))
            print(f'[mission] occ map saved → {logger.occ_path}')
        except Exception as e:
            print(f'[mission] occ save failed: {e}')
        logger.close()
        controller.disarm(cf)


# ------------------------------------------------------------------ entry

def main():
    if config.TAKEOFF_PAD_X is None or config.TAKEOFF_PAD_Y is None:
        raise ValueError(
            'Set TAKEOFF_PAD_X and TAKEOFF_PAD_Y in config.py before running.'
        )

    cflib.crtp.init_drivers()
    shared = SharedState()

    for attempt in range(1, 6):
        try:
            print(f'[main] connecting to {config.RADIO_URI}  (attempt {attempt}/5)')
            with SyncCrazyflie(config.RADIO_URI,
                               cf=Crazyflie(rw_cache='./cache')) as scf:
                mission_thread = threading.Thread(
                    target=run_mission, args=(scf.cf, shared), daemon=True)
                mission_thread.start()
                MissionGUI(shared).start()
            break
        except Exception as exc:
            print(f'[main] connection error: {exc}')
            if attempt < 5:
                print('[main] retrying in 3 s  (move drone closer / check dongle)')
                time.sleep(3.0)
            else:
                print('[main] gave up after 5 attempts')
                raise


if __name__ == '__main__':
    main()
