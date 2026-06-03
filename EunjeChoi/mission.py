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
from mapping import OccupancyGrid, HeightMap, find_pad_from_diff
from navigator import FrontierNavigator, LawnmowerNavigator, astar, simplify_path
from sensors import SensorData, SensorHub
from shared_state import SharedState


class EmergencyException(Exception):
    pass


# ------------------------------------------------------------------ helpers

def _step(shared: SharedState, hub: SensorHub,
          occ: OccupancyGrid, hmap: HeightMap,
          update_occ: bool = True) -> SensorData:
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

    if update_occ:
        occ.update_all_rays(x, y, yaw, data.ranges)

    while not hub.edge_queue.empty():
        ev = hub.edge_queue.get_nowait()
        if shared.current_state == 'LANDING_REGION_SCAN':
            if ev.kind == 'entry':
                hmap.add_entry(ev.x, ev.y)
            else:
                hmap.add_exit(ev.x, ev.y)
            shared.add_height_map_edge(ev)

    shared.occupancy_grid = occ.snapshot()
    shared.pad_pairs = hmap.get_pairs()
    shared.pad_candidates = hmap.get_candidates()

    return data


def _obstacle_in_direction(data: SensorData, dx: float, dy: float,
                           yaw_deg: float) -> bool:
    """Check if any sensor facing the motion direction is blocked.

    Uses 8-sector classification so diagonal movements check two sensors.
    """
    motion_deg = math.degrees(math.atan2(dy, dx))
    rel = (motion_deg - yaw_deg + 360) % 360

    t = config.STOP_THRESHOLD
    r = data.ranges
    F = r.front is not None and r.front < t
    B = r.back  is not None and r.back  < t
    L = r.left  is not None and r.left  < t
    R = r.right is not None and r.right < t

    if   rel < 22.5  or rel >= 337.5: return F
    elif rel < 67.5:                  return F or L
    elif rel < 112.5:                 return L
    elif rel < 157.5:                 return B or L
    elif rel < 202.5:                 return B
    elif rel < 247.5:                 return B or R
    elif rel < 292.5:                 return R
    else:                             return F or R


# ------------------------------------------------------------------ states

def do_takeoff(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF'
    shared.start_timer()
    controller.init_ekf(cf)
    controller.takeoff_vel(cf, config.FLIGHT_Z)
    data = _step(shared, hub, occ, hmap)

    # Record stabilised hover position as home — EKF may have drifted during takeoff
    home_x, home_y = data.pose[0], data.pose[1]
    shared.home_pos = (home_x, home_y)
    print(f'[mission] home recorded: ({home_x:.3f}, {home_y:.3f})')


def do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg: float = config.SCAN_ROTATE_ANGLE,
                     occ_target: Optional[OccupancyGrid] = None,
                     scan_z: Optional[float] = None,
                     freeze_occ: bool = False):
    """Rotate with yaw-rate control while holding the scan start position.

    occ_target: if set, sensor rays are also recorded into this grid.
    scan_z: altitude to hold (defaults to FLIGHT_Z).
    """
    shared.current_state = 'ROTATION_SCAN'
    z = scan_z if scan_z is not None else config.FLIGHT_Z

    data = _step(shared, hub, occ, hmap, update_occ=not freeze_occ)
    hold_x, hold_y = data.pose[0], data.pose[1]

    duration = abs(angle_deg) / config.SCAN_ROTATE_RATE
    yaw_rate = math.copysign(config.SCAN_ROTATE_RATE, angle_deg)

    hold_kp = 1.5
    max_hold_speed = min(0.08, config.NAV_SPEED)

    end_t = time.time() + duration
    while time.time() < end_t:
        data = _step(shared, hub, occ, hmap, update_occ=not freeze_occ)
        if occ_target is not None:
            x, y, _, yaw_v = data.pose
            occ_target.update_all_rays(x, y, yaw_v, data.ranges)

        cx, cy, _, cyaw = data.pose
        ex, ey = hold_x - cx, hold_y - cy
        vx_w = max(-max_hold_speed, min(max_hold_speed, ex * hold_kp))
        vy_w = max(-max_hold_speed, min(max_hold_speed, ey * hold_kp))
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, z)
        time.sleep(config.DT)

    cf.commander.send_hover_setpoint(0, 0, 0, z)
    _step(shared, hub, occ, hmap)


def _scan_navigate(cf, shared, hub, occ, hmap, tx, ty):
    """Navigate at SCAN_SPEED and stop immediately when entry event fires.
    Returns entry (x, y) if detected, None if arrived or blocked."""
    KP = 3.0
    prev_seq = hmap.entry_seq

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, cyaw = data.pose

        if hmap.entry_seq > prev_seq:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            time.sleep(0.3)
            return hmap.last_entry_pos

        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)

        if dist < config.NAV_ARRIVE_THRESHOLD:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            return None

        if _obstacle_in_direction(data, dx, dy, cyaw):
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            return None

        v = min(config.SCAN_SPEED, dist * KP)
        vx_b, vy_b = controller.vel_to_body((dx / dist) * v, (dy / dist) * v, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0, config.FLIGHT_Z)
        time.sleep(config.DT)


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
            cf.commander.send_hover_setpoint(vx_b, vy_b, 0, config.FLIGHT_Z)
            time.sleep(config.DT)

    return hmap.get_candidates()


def _navigate_to(cf, shared, hub, occ, hmap,
                 tx: float, ty: float, tz: float,
                 speed: float) -> bool:
    """P-controller velocity navigation. Returns True if arrived, False if blocked."""
    KP = 3.0
    shared.target_pos = (tx, ty)
    tz = min(tz, config.CEILING_LIMIT - 0.10)

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, cyaw = data.pose
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)

        if dist < config.NAV_ARRIVE_THRESHOLD:
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            shared.target_pos = None
            return True

        if _obstacle_in_direction(data, dx, dy, cyaw):
            cf.commander.send_hover_setpoint(0, 0, 0, tz)
            time.sleep(0.3)
            shared.target_pos = None
            return False

        v = min(speed, dist * KP)
        vx_b, vy_b = controller.vel_to_body((dx / dist) * v,
                                             (dy / dist) * v, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0, tz)
        time.sleep(config.DT)


def _nav_to_x(cf, shared, hub, occ, hmap, target_x: float):
    """Frontier + A* navigation toward target_x, doing rotation scans when stuck."""
    frontier = FrontierNavigator(occ)
    y_center = config.ARENA_Y / 2.0 - config.TAKEOFF_PAD_Y
    failed_plans = 0
    MAX_FAILED = 5

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, _ = data.pose

        if cx >= target_x:
            break

        if target_x - cx <= 0.20:
            _navigate_to(cf, shared, hub, occ, hmap,
                         target_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            break

        target = frontier.find_max_x_target(cx, cy, x_limit=target_x)
        if target is None or target[0] <= cx + 0.1:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                step = 0.20 * (1 if y_center > cy else -1)
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
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                step = 0.20 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        failed_plans = 0
        path = simplify_path(path, grid)
        nav_ok = True
        for pr, pc in path[1:]:
            pwx, pwy = occ.cell_to_world(pr, pc)
            if not _navigate_to(cf, shared, hub, occ, hmap,
                                pwx, pwy, config.FLIGHT_Z, config.NAV_SPEED):
                nav_ok = False
                break
        if not nav_ok:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                failed_plans = 0


def do_nav_to_landing(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_LANDING'
    _nav_to_x(cf, shared, hub, occ, hmap, config.ekf_landing_region_start())


def do_landing_region_scan(cf, shared, hub, occ, hmap):
    """Two-pass dual-altitude scan to locate the landing pad.

    Navigate to x=LANDING_SCAN_X (keeping current y), then:
    Pass 1 (FLIGHT_Z): 360° rotation → occ_high
    Pass 2 (LOW_SCAN_Z): 360° rotation → occ_low
    Diff map → BFS → square bounding box = pad.
    Returns EKF (x, y) of pad, or None.
    """
    shared.current_state = 'LANDING_REGION_SCAN'

    # Navigate to scan x using frontier + A*
    sx = config.LANDING_SCAN_X - config.TAKEOFF_PAD_X
    _nav_to_x(cf, shared, hub, occ, hmap, sx)

    data = _step(shared, hub, occ, hmap)
    print(f'[scan] scanning from x={data.pose[0]:.2f} (target sx={sx:.2f})')

    # ── Pass 1: FLIGHT_Z — dedicated scan map (independent from nav occ)
    occ_scan_high = OccupancyGrid()
    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=90.0,
                     occ_target=occ_scan_high)
    shared.occ_scan_high_grid = occ_scan_high.snapshot()

    # ── Pass 2: LOW_SCAN_Z
    occ_low = OccupancyGrid()
    controller.takeoff_vel(cf, config.LOW_SCAN_Z, speed=0.15, settle=0.5)

    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=90.0,
                     occ_target=occ_low,
                     scan_z=config.LOW_SCAN_Z,
                     freeze_occ=True)
    shared.occ_low_grid = occ_low.snapshot()

    controller.takeoff_vel(cf, config.FLIGHT_Z, speed=0.15, settle=0.5)

    pad_pos, diff_grid = find_pad_from_diff(occ_scan_high, occ_low)
    shared.occ_diff_grid = diff_grid
    if pad_pos is not None:
        print(f'[scan] pad found at EKF ({pad_pos[0]:.3f}, {pad_pos[1]:.3f})')
    else:
        print('[scan] pad not found from diff map')
    return pad_pos


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
    # Re-arm after landing on pad
    for _ in range(20):
        if cf.supervisor.can_be_armed:
            break
        time.sleep(0.1)
    cf.supervisor.send_arming_request(True)
    time.sleep(0.5)
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
        do_nav_to_landing(cf, shared, hub, occ, hmap)
        pad_pos = do_landing_region_scan(cf, shared, hub, occ, hmap)
        if pad_pos is not None:
            shared.landing_target = pad_pos
            do_land_on_pad(cf, shared, hub, occ, hmap, pad_pos[0], pad_pos[1])
            do_takeoff_from_pad(cf, shared, hub, occ, hmap)
        else:
            print('[mission] pad not found — skipping pad landing, returning home')
        do_turn_around(cf, shared, hub, occ, hmap)
        do_nav_to_start(cf, shared, hub, occ, hmap)
        do_land_on_start(cf, shared, hub, occ, hmap)

    except EmergencyException:
        shared.current_state = 'EMERGENCY_LAND'
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
        for label, path, grid_fn in [
            ('occ_scan_high', logger.occ_scan_high_path, lambda: shared.occ_scan_high_grid),
            ('occ_low',       logger.occ_low_path,       lambda: shared.occ_low_grid),
            ('occ_diff',      logger.occ_diff_path,      lambda: shared.occ_diff_grid),
        ]:
            try:
                g = grid_fn()
                if g is not None:
                    np.savez(path,
                             grid=g,
                             res=np.array([config.OCCUPANCY_GRID_RES]),
                             x_min=np.array([occ.x_min]),
                             y_min=np.array([occ.y_min]))
                    print(f'[mission] {label} saved → {path}')
            except Exception as e:
                print(f'[mission] {label} save failed: {e}')
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
