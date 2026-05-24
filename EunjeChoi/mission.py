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
        if ev.kind == 'entry':
            hmap.add_entry(ev.x, ev.y)
        else:
            hmap.add_exit(ev.x, ev.y)
        shared.add_height_map_edge(ev)

    shared.occupancy_grid = occ.snapshot()
    shared.pad_candidates = hmap.get_candidates()

    return data


def _obstacle_in_direction(data: SensorData, dx: float, dy: float,
                           yaw_deg: float) -> bool:
    """Check if the sensor facing the motion direction is blocked."""
    # Compute angle from current heading to motion direction
    motion_deg = math.degrees(math.atan2(dy, dx))
    rel = (motion_deg - yaw_deg + 360) % 360

    threshold = config.STOP_THRESHOLD
    r = data.ranges

    if rel < 45 or rel >= 315:              # forward
        return r.front is not None and r.front < threshold
    elif 45 <= rel < 135:                   # left
        return r.left is not None and r.left < threshold
    elif 135 <= rel < 225:                  # back
        return r.back is not None and r.back < threshold
    else:                                   # right
        return r.right is not None and r.right < threshold


# ------------------------------------------------------------------ states

def do_takeoff(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF'
    shared.start_timer()
    controller.init_ekf(cf)
    controller.takeoff(cf, config.FLIGHT_Z, duration=2.0)
    time.sleep(2.5)
    _step(shared, hub, occ, hmap)


def do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg: float = config.SCAN_ROTATE_ANGLE):
    """Rotate in place while continuously mapping."""
    shared.current_state = 'ROTATION_SCAN'
    duration = angle_deg / config.SCAN_ROTATE_RATE
    end_t = time.time() + duration

    while time.time() < end_t:
        data = _step(shared, hub, occ, hmap)
        controller.send_hover(cf, 0.0, 0.0, config.SCAN_ROTATE_RATE, config.FLIGHT_Z)
        time.sleep(config.DT)

    controller.stop_motion(cf)
    time.sleep(0.3)
    _step(shared, hub, occ, hmap)


def _navigate_to(cf, shared, hub, occ, hmap,
                 tx: float, ty: float, tz: float,
                 speed: float) -> bool:
    """Send go_to and monitor until arrival or obstacle. Returns True if arrived."""
    data = _step(shared, hub, occ, hmap)
    cx, cy, cz, cyaw = data.pose

    duration = controller.go_to_nonblocking(cf, tx, ty, tz, 0.0, speed, cx, cy, cz)
    elapsed = 0.0

    while elapsed < duration:
        data = _step(shared, hub, occ, hmap)
        cx, cy, cz, cyaw = data.pose
        dx, dy = tx - cx, ty - cy

        if _obstacle_in_direction(data, dx, dy, cyaw):
            cf.high_level_commander.stop()
            time.sleep(0.15)
            return False

        if math.hypot(dx, dy) < 0.15:
            return True

        controller.clamp_z(cf, cz)
        time.sleep(config.DT)
        elapsed += config.DT

    return True


def do_nav_to_landing(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_LANDING'
    frontier = FrontierNavigator(occ)
    target_x = config.ekf_landing_region_start()
    stall_count = 0

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, cz, cyaw = data.pose

        if cx >= target_x:
            break

        target = frontier.find_max_x_target(cx, cy, x_limit=target_x)
        if target is None or target[0] <= cx + 0.1:
            # No progress possible — scan and retry
            do_rotation_scan(cf, shared, hub, occ, hmap)
            stall_count += 1
            if stall_count > 4:
                # Force a small step in +x and continue
                _navigate_to(cf, shared, hub, occ, hmap,
                             cx + 0.3, cy, config.FLIGHT_Z, config.NAV_SPEED)
                stall_count = 0
            continue

        stall_count = 0
        arrived = _navigate_to(cf, shared, hub, occ, hmap,
                               target[0], target[1], config.FLIGHT_Z,
                               config.NAV_SPEED)
        if not arrived:
            do_rotation_scan(cf, shared, hub, occ, hmap)


def do_landing_region_scan(cf, shared, hub, occ, hmap):
    shared.current_state = 'LANDING_REGION_SCAN'

    x_start = config.ekf_landing_region_start()
    x_end = config.ekf_arena_x_max() - 0.15
    y_min = config.ekf_arena_y_min() + 0.25
    y_max = config.ekf_arena_y_max() - 0.25

    waypoints = LawnmowerNavigator().generate(x_start, x_end, y_min, y_max)

    for wx, wy in waypoints:
        arrived = _navigate_to(cf, shared, hub, occ, hmap,
                               wx, wy, config.FLIGHT_Z, config.SCAN_SPEED)
        if not arrived:
            # Simple side-step to get around obstacle
            data = hub.read()
            cx, cy = data.pose[0], data.pose[1]
            _navigate_to(cf, shared, hub, occ, hmap,
                         cx, cy + 0.3, config.FLIGHT_Z, config.NAV_SPEED)

        if len(hmap.get_candidates()) >= 1:
            # Finish the current row before stopping
            pass   # scan all rows for reliability


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
    controller.land(cf, 0.0, 2.0)
    time.sleep(2.8)


def do_takeoff_from_pad(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF_FROM_PAD'
    controller.takeoff(cf, config.FLIGHT_Z, 2.0)
    time.sleep(2.5)
    _step(shared, hub, occ, hmap)


def do_turn_around(cf, shared, hub, occ, hmap):
    shared.current_state = 'TURN_AROUND'
    do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=180.0)


def do_nav_to_start(cf, shared, hub, occ, hmap):
    shared.current_state = 'NAV_TO_START'
    data = _step(shared, hub, occ, hmap)
    cx, cy, cz, _ = data.pose

    grid = occ.snapshot()
    sr, sc = occ.world_to_cell(cx, cy)
    gr, gc = occ.world_to_cell(0.0, 0.0)   # EKF origin = takeoff pad

    path = astar(grid, (sr, sc), (gr, gc))
    if path is not None and len(path) > 1:
        path = simplify_path(path, grid)
        waypoints = [occ.cell_to_world(r, c) for r, c in path[1:]]
    else:
        # Fallback: direct line
        waypoints = [(0.0, 0.0)]

    for wx, wy in waypoints:
        arrived = _navigate_to(cf, shared, hub, occ, hmap,
                               wx, wy, config.FLIGHT_Z, config.NAV_SPEED)
        if not arrived:
            do_rotation_scan(cf, shared, hub, occ, hmap)
            # Replan
            data = _step(shared, hub, occ, hmap)
            cx, cy, _, _ = data.pose
            grid = occ.snapshot()
            sr, sc = occ.world_to_cell(cx, cy)
            path = astar(grid, (sr, sc), (gr, gc))
            if path is not None:
                path = simplify_path(path, grid)
                waypoints = [occ.cell_to_world(r, c) for r, c in path[1:]]
            break


def do_land_on_start(cf, shared, hub, occ, hmap):
    shared.current_state = 'LANDING_ON_START'
    _navigate_to(cf, shared, hub, occ, hmap,
                 0.0, 0.0, config.FLIGHT_Z, config.SCAN_SPEED)
    controller.land(cf, 0.0, 2.0)
    time.sleep(2.8)
    shared.current_state = 'DONE'


# ------------------------------------------------------------------ mission

def run_mission(cf, shared: SharedState):
    occ = OccupancyGrid()
    hmap = HeightMap()
    hub = SensorHub(cf)
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

        do_land_on_pad(cf, shared, hub, occ, hmap, pad_pos[0], pad_pos[1])
        do_takeoff_from_pad(cf, shared, hub, occ, hmap)
        do_turn_around(cf, shared, hub, occ, hmap)
        do_nav_to_start(cf, shared, hub, occ, hmap)
        do_land_on_start(cf, shared, hub, occ, hmap)

    except EmergencyException:
        shared.current_state = 'EMERGENCY_LAND'
        try:
            controller.land(cf, 0.0, 2.0)
        except Exception:
            pass
        time.sleep(3.0)

    except Exception as exc:
        shared.current_state = f'ERROR'
        print(f'[mission] unhandled exception: {exc}')
        try:
            controller.land(cf, 0.0, 3.0)
        except Exception:
            pass

    finally:
        hub.stop()


# ------------------------------------------------------------------ entry

def main():
    if config.TAKEOFF_PAD_X is None or config.TAKEOFF_PAD_Y is None:
        raise ValueError(
            'Set TAKEOFF_PAD_X and TAKEOFF_PAD_Y in config.py before running.'
        )

    cflib.crtp.init_drivers()
    shared = SharedState()

    with SyncCrazyflie(config.RADIO_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        mission_thread = threading.Thread(
            target=run_mission,
            args=(scf.cf, shared),
            daemon=True,
        )
        mission_thread.start()

        MissionGUI(shared).start()   # blocks until window closed


if __name__ == '__main__':
    main()
