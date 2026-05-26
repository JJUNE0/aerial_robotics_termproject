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


# ------------------------------------------------------------------ helpers

def _step(shared: SharedState, hub: SensorHub,
          occ: OccupancyGrid, hmap: HeightMap) -> SensorData:
    """Read sensors, update maps, push state to SharedState. Raise on emergency."""
    if shared.emergency_flag:
        raise EmergencyException()

    data = hub.read()
    x, y, z, yaw = data.pose

    shared.pose = data.pose
    shared.battery_pct = data.battery_pct

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
    cf.high_level_commander.takeoff(config.FLIGHT_Z, 2.0)
    time.sleep(2.5)
    _step(shared, hub, occ, hmap)


def do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg: float = config.SCAN_ROTATE_ANGLE):
    """Rotate in place by angle_deg in one go_to command while mapping."""
    shared.current_state = 'ROTATION_SCAN'

    data = _step(shared, hub, occ, hmap)
    cx, cy = data.pose[0], data.pose[1]
    _, _, _, current_yaw_deg = data.pose

    target_deg = (current_yaw_deg + angle_deg + 180.0) % 360.0 - 180.0
    duration = abs(angle_deg) / config.SCAN_ROTATE_RATE
    cf.high_level_commander.go_to(cx, cy, config.FLIGHT_Z,
                                  math.radians(target_deg), duration)

    end_t = time.time() + duration
    while time.time() < end_t:
        _step(shared, hub, occ, hmap)
        time.sleep(config.DT)

    _step(shared, hub, occ, hmap)


def _navigate_to(cf, shared, hub, occ, hmap,
                 tx: float, ty: float, tz: float,
                 speed: float) -> bool:
    """Send go_to and monitor until arrival or obstacle. Returns True if arrived."""
    shared.target_pos = (tx, ty)

    data = _step(shared, hub, occ, hmap)
    cx, cy, cz, cyaw = data.pose

    tz = min(tz, config.CEILING_LIMIT - 0.10)
    duration = controller.go_to_nonblocking(cf, tx, ty, tz, cyaw, speed, cx, cy, cz)
    elapsed = 0.0

    while elapsed < duration:
        data = _step(shared, hub, occ, hmap)
        cx, cy, cz, cyaw = data.pose
        dx, dy = tx - cx, ty - cy

        if _obstacle_in_direction(data, dx, dy, cyaw):
            # Hold current position — do NOT call hlc.stop() which terminates
            # HLC control and causes the drone to fall on firmware without a
            # low-level setpoint fallback.
            cf.high_level_commander.go_to(cx, cy, cz, math.radians(cyaw), 0.5)
            time.sleep(0.5)
            shared.target_pos = None
            return False

        if math.hypot(dx, dy) < 0.15:
            break   # close — exit timing loop and wait for full settle below

        time.sleep(config.DT)
        elapsed += config.DT

    # Wait until drone settles within arrival threshold (or timeout)
    settle_end = time.time() + config.NAV_SETTLE_TIMEOUT
    while time.time() < settle_end:
        data = _step(shared, hub, occ, hmap)
        cx, cy = data.pose[0], data.pose[1]
        if math.hypot(tx - cx, ty - cy) < config.NAV_ARRIVE_THRESHOLD:
            break
        time.sleep(config.DT)

    shared.target_pos = None
    return True


def do_nav_to_landing(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_LANDING'
    frontier = FrontierNavigator(occ)
    target_x = config.ekf_landing_region_start()

    failed_plans = 0    # consecutive planning failures — triggers rescan
    MAX_FAILED = 3

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

        target = frontier.find_max_x_target(cx, cy, x_limit=target_x)
        if target is None or target[0] <= cx + 0.1:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                failed_plans = 0
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
            continue

        failed_plans = 0
        path = simplify_path(path, grid)
        for pr, pc in path[1:]:
            pwx, pwy = occ.cell_to_world(pr, pc)
            if not _navigate_to(cf, shared, hub, occ, hmap,
                                pwx, pwy, config.FLIGHT_Z, config.NAV_SPEED):
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                break   # re-plan from new position in next loop iteration


def do_landing_region_scan(cf, shared, hub, occ, hmap):
    shared.current_state = 'LANDING_REGION_SCAN'
    hmap.reset()
    hub.reset_edge_detector()

    x_start = config.ekf_landing_region_start()
    x_end = config.ekf_arena_x_max() - 0.15
    y_min = config.ekf_arena_y_min() + 0.25
    y_max = config.ekf_arena_y_max() - 0.25

    waypoints = LawnmowerNavigator().generate(x_start, x_end, y_min, y_max)

    for wx, wy in waypoints:
        if hmap.get_candidates():
            break   # pad confirmed — skip remaining scan

        arrived = _navigate_to(cf, shared, hub, occ, hmap,
                               wx, wy, config.FLIGHT_Z, config.SCAN_SPEED)
        if arrived:
            continue

        # Blocked: map the obstacle with a rotation scan, then A* around it
        do_rotation_scan(cf, shared, hub, occ, hmap)

        data = _step(shared, hub, occ, hmap)
        cx, cy, _, _ = data.pose
        grid = occ.snapshot()
        path = astar(grid,
                     occ.world_to_cell(cx, cy),
                     occ.world_to_cell(wx, wy))
        if path is not None and len(path) > 1:
            path = simplify_path(path, grid)
            for pr, pc in path[1:]:
                pwx, pwy = occ.cell_to_world(pr, pc)
                if not _navigate_to(cf, shared, hub, occ, hmap,
                                    pwx, pwy, config.FLIGHT_Z, config.SCAN_SPEED):
                    break   # still blocked — skip to next lawnmower waypoint


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
    data = _step(shared, hub, occ, hmap)
    cx, cy, cz, _ = data.pose
    _navigate_to(cf, shared, hub, occ, hmap,
                 pad_x, pad_y, config.FLIGHT_Z, config.SCAN_SPEED)
    cf.high_level_commander.land(0.0, 2.0)
    time.sleep(2.8)


def do_takeoff_from_pad(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF_FROM_PAD'
    cf.high_level_commander.takeoff(config.FLIGHT_Z, 2.0)
    time.sleep(2.5)
    _step(shared, hub, occ, hmap)


def do_turn_around(cf, shared, hub, occ, hmap):
    shared.current_state = 'TURN_AROUND'
    do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)


def do_nav_to_start(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_START'
    data = _step(shared, hub, occ, hmap)
    cx, cy, cz, _ = data.pose

    gr, gc = occ.world_to_cell(0.0, 0.0)   # EKF origin = takeoff pad

    def _plan(ox, oy):
        g = occ.snapshot()
        sr, sc = occ.world_to_cell(ox, oy)
        p = astar(g, (sr, sc), (gr, gc))
        if p is not None and len(p) > 1:
            p = simplify_path(p, g)
            return [occ.cell_to_world(r, c) for r, c in p[1:]]
        return [(0.0, 0.0)]

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
    _navigate_to(cf, shared, hub, occ, hmap,
                 0.0, 0.0, config.FLIGHT_Z, config.SCAN_SPEED)
    cf.high_level_commander.land(0.0, 2.0)
    time.sleep(2.8)
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
        do_landing_region_scan(cf, shared, hub, occ, hmap)

        pad_pos = do_pad_confirm(cf, shared, hub, occ, hmap)
        if pad_pos is None:
            candidates = hmap.get_candidates()
            if candidates:
                pad_pos = (candidates[0].cx, candidates[0].cy)
            else:
                data = hub.read()
                pad_pos = (data.pose[0], data.pose[1])

        shared.landing_target = pad_pos
        do_land_on_pad(cf, shared, hub, occ, hmap, pad_pos[0], pad_pos[1])
        do_takeoff_from_pad(cf, shared, hub, occ, hmap)
        do_turn_around(cf, shared, hub, occ, hmap)
        do_nav_to_start(cf, shared, hub, occ, hmap)
        do_land_on_start(cf, shared, hub, occ, hmap)

    except EmergencyException:
        shared.current_state = 'EMERGENCY_LAND'
        try:
            cf.high_level_commander.land(0.0, 2.0)
        except Exception:
            pass
        time.sleep(3.0)

    except Exception as exc:
        shared.current_state = 'ERROR'
        print(f'[mission] unhandled exception: {exc}')
        try:
            cf.high_level_commander.land(0.0, 3.0)
        except Exception:
            pass

    finally:
        hub.stop()
        logger.close()
        controller.disarm(cf)   # leave firmware in a clean state for next run


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
