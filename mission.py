"""
Autonomous Level-3 round-trip mission for Crazyflie 2.1 Brushless.

State sequence:
  TAKEOFF
  → ROTATION_SCAN
  → NAV_TO_LANDING
      NAV_FRONTIER   : BFS로 다음 frontier 탐색 중
      NAV_WAYPOINT   : A* waypoint 이동 중
      NAV_RECOVER_Y  : 막혀서 Y방향 복구 이동
      NAV_RECOVER_X  : 막혀서 X방향 nudge 이동
      ROTATION_SCAN  : 막힘 복구용 스캔
  → SCAN_NAV_FRONTIER / SCAN_NAV_WAYPOINT / ...  (스캔 위치까지 이동)
  → SCAN_HIGH        : 고고도 180° 회전 스캔
  → SCAN_LOW         : 저고도 180° 회전 스캔
  → LANDING_REGION_SCAN (스캔 완료 후 diff 계산)
  → LAND_X_ALIGN     : pad X 중앙으로 정렬
  → LAND_APPROACH    : pad Y 방향 접근
  → LAND_ENTRY       : entry edge 감지됨, 계속 전진
  → LAND_HOVER       : 착지 전 1s 호버
  → LAND_YAW_ALIGN   : home_yaw+180° 방향으로 회전 후 착지
  → TAKEOFF_FROM_PAD
  → YAW_ALIGN        : home_yaw+180°로 IMU drift 보정 회전
  → TURN_AROUND
  → NAV_TO_START
      RET_FRONTIER / RET_WAYPOINT / RET_RECOVER_Y / RET_RECOVER_X
  → LANDING_ON_START
  → DONE
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
          update_occ: bool = True,
          consume_edges: bool = True) -> SensorData:
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

    if consume_edges:
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

    # Record stabilised hover position and yaw as home reference
    home_x, home_y = data.pose[0], data.pose[1]
    shared.home_pos = (home_x, home_y)
    shared.home_yaw = data.pose[3]
    print(f'[takeoff] home pos=({home_x:.2f}, {home_y:.2f})  yaw={data.pose[3]:.1f}°')
    print(f'[mission] home recorded: ({home_x:.3f}, {home_y:.3f})')


def do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg: float = config.SCAN_ROTATE_ANGLE,
                     occ_target: Optional[OccupancyGrid] = None,
                     scan_z: Optional[float] = None,
                     freeze_occ: bool = False,
                     state: str = 'ROTATION_SCAN'):
    """Rotate with yaw-rate control while holding the scan start position.

    occ_target: if set, sensor rays are also recorded into this grid.
    scan_z: altitude to hold (defaults to FLIGHT_Z).
    state: shared.current_state label to use during this scan.
    """
    shared.current_state = state
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


def _nav_to_x(cf, shared, hub, occ, hmap, target_x: float, prefix: str = 'NAV'):
    """Frontier + A* navigation toward target_x, doing rotation scans when stuck."""
    frontier = FrontierNavigator(occ)
    y_center = config.ARENA_Y / 2.0 - config.TAKEOFF_PAD_Y
    failed_plans = 0
    MAX_FAILED = 2

    while True:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, _ = data.pose

        if cx >= target_x:
            break

        if target_x - cx <= 0.20:
            shared.current_state = f'{prefix}_WAYPOINT'
            _navigate_to(cf, shared, hub, occ, hmap,
                         target_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            break

        shared.current_state = f'{prefix}_FRONTIER'
        target = frontier.find_max_x_target(cx, cy, x_limit=min(target_x, cx + 1.1))
        if target is None or target[0] <= cx + 0.1:
            shared.frontier_target = None
            shared.nav_waypoints = []
            hub.clear_frontier()
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                shared.current_state = f'{prefix}_RECOVER_Y'
                step = 0.20 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            else:
                shared.current_state = f'{prefix}_RECOVER_X'
                nudge_x = min(cx + 0.20, target_x)
                _navigate_to(cf, shared, hub, occ, hmap,
                             nudge_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        shared.frontier_target = target
        hub.set_frontier(target[0], target[1])

        grid = occ.snapshot()
        path = astar(grid,
                     occ.world_to_cell(cx, cy),
                     occ.world_to_cell(target[0], target[1]))

        if path is None or len(path) <= 1:
            shared.nav_waypoints = []
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                shared.current_state = f'{prefix}_RECOVER_Y'
                step = 0.20 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            else:
                shared.current_state = f'{prefix}_RECOVER_X'
                nudge_x = min(cx + 0.20, target_x)
                _navigate_to(cf, shared, hub, occ, hmap,
                             nudge_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        failed_plans = 0
        path = simplify_path(path, grid)
        shared.nav_waypoints = [occ.cell_to_world(pr, pc) for pr, pc in path[1:]]
        shared.current_state = f'{prefix}_WAYPOINT'
        nav_ok = True
        for pr, pc in path[1:]:
            pwx, pwy = occ.cell_to_world(pr, pc)
            if not _navigate_to(cf, shared, hub, occ, hmap,
                                pwx, pwy, config.FLIGHT_Z, config.NAV_SPEED):
                nav_ok = False
                break
        shared.nav_waypoints = []
        if nav_ok:
            data = _step(shared, hub, occ, hmap)
            if data.pose[0] < target_x - 0.20:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
        else:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)
                failed_plans = 0

    shared.frontier_target = None
    shared.nav_waypoints = []
    hub.clear_frontier()


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

    # Navigate to scan position (arena X=4.5, Y=0.5) using frontier + A* then fine approach
    sx = config.LANDING_SCAN_X - config.TAKEOFF_PAD_X
    sy = config.LANDING_SCAN_Y - config.TAKEOFF_PAD_Y
    _nav_to_x(cf, shared, hub, occ, hmap, sx, prefix='SCAN_NAV')
    shared.current_state = 'LANDING_REGION_SCAN'
    _navigate_to(cf, shared, hub, occ, hmap, sx, sy, config.FLIGHT_Z, config.NAV_SPEED)

    data = _step(shared, hub, occ, hmap)
    print(f'[scan] scanning from ({data.pose[0]:.2f}, {data.pose[1]:.2f})'
          f'  target=({sx:.2f}, {sy:.2f})')

    # ── Pass 1: FLIGHT_Z — dedicated scan map (independent from nav occ)
    occ_scan_high = OccupancyGrid()
    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=180.0,
                     occ_target=occ_scan_high,
                     state='SCAN_HIGH')
    shared.occ_scan_high_grid = occ_scan_high.snapshot(filter_outliers=False)

    # ── Pass 2: LOW_SCAN_Z
    occ_low = OccupancyGrid()
    controller.takeoff_vel(cf, config.LOW_SCAN_Z, speed=0.15, settle=0.5)

    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=180.0,
                     occ_target=occ_low,
                     scan_z=config.LOW_SCAN_Z,
                     freeze_occ=True,
                     state='SCAN_LOW')
    shared.occ_low_grid = occ_low.snapshot(filter_outliers=False)

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


def do_land_on_pad(cf, shared, hub, occ, hmap, pad_x: float, pad_y: float,
                   yaw_align: bool = True,
                   entry_overshoot: float = None):
    if entry_overshoot is None:
        entry_overshoot = config.PAD_LAND_ENTRY_OVERSHOOT
    shared.current_state = 'LANDING_ON_PAD'

    # Step 1: align X with cluster centre, keeping current Y
    shared.current_state = 'LAND_X_ALIGN'
    data = _step(shared, hub, occ, hmap)
    _, cy, _, _ = data.pose
    shared.landing_align_pos = (pad_x, cy)
    _navigate_to(cf, shared, hub, occ, hmap,
                 pad_x, cy, config.FLIGHT_Z, config.PAD_LAND_X_SPEED)

    # Hover 1s to stabilise before approaching pad
    end_hover = time.time() + 1.0
    while time.time() < end_hover:
        cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
        time.sleep(config.DT)

    # Step 2: search past the estimated entry edge; if entry is detected,
    # move a fixed distance from that measured edge before landing.
    hub.reset_edge_detector()
    while not hub.edge_queue.empty():
        hub.edge_queue.get_nowait()

    data = _step(shared, hub, occ, hmap, consume_edges=False)
    _, cy, _, _ = data.pose
    approach_dy = 1.0 if pad_y >= cy else -1.0
    start_y = cy
    estimated_edge_y = pad_y - approach_dy * (config.PAD_SIZE / 2.0)
    edge_dist = approach_dy * (estimated_edge_y - start_y)
    search_dist = max(0.0, edge_dist) + config.PAD_LAND_SEARCH_MARGIN
    search_target_y = start_y + approach_dy * search_dist
    shared.target_pos = (pad_x, search_target_y)

    shared.current_state = 'LAND_APPROACH'
    entry_pos = None
    entry_target_y = None
    second_edge_pos = None
    land_reason = None
    print(f'[land] searching to estimated edge + {config.PAD_LAND_SEARCH_MARGIN:.2f} m'
          f'  target=({pad_x:.3f}, {search_target_y:.3f})')

    while True:
        data = _step(shared, hub, occ, hmap, consume_edges=False)
        cx, cy, _, cyaw = data.pose

        while not hub.edge_queue.empty():
            ev = hub.edge_queue.get_nowait()
            shared.add_height_map_edge(ev)
            if ev.kind == 'entry' and entry_pos is None:
                entry_pos = (ev.x, ev.y)
                entry_target_y = ev.y + approach_dy * entry_overshoot
                shared.target_pos = (pad_x, entry_target_y)
                shared.current_state = 'LAND_ENTRY'
                print(f'[land] edge entry at ({ev.x:.3f}, {ev.y:.3f}),'
                      f' target +{entry_overshoot:.2f} m'
                      f' -> ({pad_x:.3f}, {entry_target_y:.3f})')
            elif entry_pos is not None and second_edge_pos is None:
                second_edge_pos = (ev.x, ev.y)
                land_reason = f'{ev.kind}_edge'
                print(f'[land] second edge ({ev.kind}) at ({ev.x:.3f}, {ev.y:.3f}),'
                      ' landing after hover')

        if second_edge_pos is not None:
            break

        if entry_pos is not None:
            pushed_dist = approach_dy * (cy - entry_pos[1])
            if pushed_dist >= entry_overshoot:
                land_reason = 'distance'
                break
        else:
            search_progress = approach_dy * (cy - start_y)
            if search_progress >= search_dist:
                land_reason = 'estimated_edge_distance'
                print('[land] no entry detected; reached estimated edge'
                      f' + {config.PAD_LAND_SEARCH_MARGIN:.2f} m, landing after hover')
                break

        vx_w = max(-0.05, min(0.05, (pad_x - cx) * 2.0))
        vy_w = approach_dy * config.PAD_LAND_PUSH_SPEED
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0, config.FLIGHT_Z)
        time.sleep(config.DT)

    if entry_pos is not None:
        data = _step(shared, hub, occ, hmap, consume_edges=False)
        dx = data.pose[0] - entry_pos[0]
        dy = data.pose[1] - entry_pos[1]
        print(f'[land] pushed {math.hypot(dx, dy):.3f} m after entry'
              f'  y_axis={approach_dy * dy:.3f} m'
              f'  reason={land_reason}')

    shared.current_state = 'LAND_HOVER'
    cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
    time.sleep(1.0)

    if yaw_align:
        land_yaw = shared.home_yaw
        shared.current_state = 'LAND_YAW_ALIGN'
        data = _step(shared, hub, occ, hmap)
        current_yaw = data.pose[3]
        hold_x, hold_y = data.pose[0], data.pose[1]
        delta = ((land_yaw - current_yaw) + 180.0) % 360.0 - 180.0
        print(f'[land] yaw align: target={land_yaw:.1f}°  current={current_yaw:.1f}°  delta={delta:+.1f}°')
        if abs(delta) > 1.0:
            hold_kp = 2.0
            max_xy  = 0.10
            timeout = time.time() + abs(delta) / config.SCAN_ROTATE_RATE * 4.0
            while time.time() < timeout:
                data = _step(shared, hub, occ, hmap)
                cx, cy, _, cyaw = data.pose
                delta = ((land_yaw - cyaw) + 180.0) % 360.0 - 180.0
                if abs(delta) < 1.0:
                    break
                yaw_rate = math.copysign(config.SCAN_ROTATE_RATE, delta)
                vx_w = max(-max_xy, min(max_xy, (hold_x - cx) * hold_kp))
                vy_w = max(-max_xy, min(max_xy, (hold_y - cy) * hold_kp))
                vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, cyaw)
                cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
                time.sleep(config.DT)
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            time.sleep(0.5)

    shared.target_pos = None
    controller.land_vel(cf, hub)


def do_takeoff_from_pad(cf, shared, hub, occ, hmap):
    shared.current_state = 'TAKEOFF_FROM_PAD'
    # Wait for firmware to allow re-arm after landing
    time.sleep(1.0)
    armed = False
    for i in range(50):   # up to 5 s
        if cf.supervisor.can_be_armed:
            armed = True
            print(f'[takeoff_pad] can_be_armed=True (attempt {i+1})')
            break
        time.sleep(0.1)
    if not armed:
        print('[takeoff_pad] WARNING: can_be_armed never True — attempting anyway')
    cf.supervisor.send_arming_request(True)
    time.sleep(0.5)
    controller.takeoff_vel(cf, config.FLIGHT_Z)
    _step(shared, hub, occ, hmap)


# ------------------------------------------------------------------ yaw alignment

def do_yaw_align(cf, shared, hub, occ, hmap, target_yaw: float):
    """Rotate to recover target_yaw saved before landing.

    target_yaw: circular mean of the last N yaw readings before landing (deg).
    Corrects IMU drift that accumulates while the drone is disarmed on the pad.
    """
    shared.current_state = 'YAW_ALIGN'
    data = _step(shared, hub, occ, hmap)
    current_yaw = data.pose[3]
    delta = ((target_yaw - current_yaw) + 180.0) % 360.0 - 180.0  # [-180, 180]
    print(f'[yaw_align] target={target_yaw:.1f}°  current={current_yaw:.1f}°  delta={delta:+.1f}°')

    if abs(delta) < 1.0:
        print('[yaw_align] within ±1°, no correction needed')
        return

    hold_x, hold_y = data.pose[0], data.pose[1]
    hold_kp = 2.0
    max_xy  = 0.10
    timeout = time.time() + abs(delta) / config.SCAN_ROTATE_RATE * 4.0
    while time.time() < timeout:
        data = _step(shared, hub, occ, hmap)
        cx, cy, _, current_yaw = data.pose
        delta = ((target_yaw - current_yaw) + 180.0) % 360.0 - 180.0
        if abs(delta) < 1.0:
            break
        yaw_rate = math.copysign(config.SCAN_ROTATE_RATE, delta)
        vx_w = max(-max_xy, min(max_xy, (hold_x - cx) * hold_kp))
        vy_w = max(-max_xy, min(max_xy, (hold_y - cy) * hold_kp))
        vx_b, vy_b = controller.vel_to_body(vx_w, vy_w, current_yaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, yaw_rate, config.FLIGHT_Z)
        time.sleep(config.DT)
    cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
    time.sleep(0.5)
    data = _step(shared, hub, occ, hmap)
    print(f'[yaw_align] yaw after correction: {data.pose[3]:.1f}°')


def do_turn_around(cf, shared, hub, occ, hmap):
    shared.current_state = 'TURN_AROUND'
    do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=90.0)


def _nav_to_home(cf, shared, hub, occ_return: OccupancyGrid, hmap, home_x: float):
    """Navigate in -X direction with a fresh map, mirroring _nav_to_x."""
    frontier = FrontierNavigator(occ_return)
    y_center = config.ARENA_Y / 2.0 - config.TAKEOFF_PAD_Y
    failed_plans = 0
    MAX_FAILED = 2

    while True:
        data = _step(shared, hub, occ_return, hmap)
        cx, cy, _, _ = data.pose

        if cx <= home_x:
            break

        if cx - home_x <= 0.20:
            _navigate_to(cf, shared, hub, occ_return, hmap,
                         home_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            break

        shared.current_state = 'RET_FRONTIER'
        target = frontier.find_min_x_target(cx, cy, x_limit=max(home_x, cx - 1.1))
        if target is None or target[0] >= cx - 0.1:
            shared.frontier_target = None
            shared.nav_waypoints = []
            hub.clear_frontier()
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ_return, hmap, angle_deg=90.0)
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                shared.current_state = 'RET_RECOVER_Y'
                step = 0.20 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ_return, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            else:
                shared.current_state = 'RET_RECOVER_X'
                nudge_x = max(cx - 0.20, home_x)
                _navigate_to(cf, shared, hub, occ_return, hmap,
                             nudge_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        shared.frontier_target = target
        hub.set_frontier(target[0], target[1])

        grid_snap = occ_return.snapshot()
        path = astar(grid_snap,
                     occ_return.world_to_cell(cx, cy),
                     occ_return.world_to_cell(target[0], target[1]))

        if path is None or len(path) <= 1:
            shared.nav_waypoints = []
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ_return, hmap, angle_deg=90.0)
                failed_plans = 0
            elif abs(cy - y_center) > 0.10:
                shared.current_state = 'RET_RECOVER_Y'
                step = 0.20 * (1 if y_center > cy else -1)
                _navigate_to(cf, shared, hub, occ_return, hmap,
                             cx, cy + step, config.FLIGHT_Z, config.NAV_SPEED)
            else:
                shared.current_state = 'RET_RECOVER_X'
                nudge_x = max(cx - 0.20, home_x)
                _navigate_to(cf, shared, hub, occ_return, hmap,
                             nudge_x, cy, config.FLIGHT_Z, config.NAV_SPEED)
            continue

        failed_plans = 0
        path = simplify_path(path, grid_snap)
        shared.nav_waypoints = [occ_return.cell_to_world(pr, pc) for pr, pc in path[1:]]
        shared.current_state = 'RET_WAYPOINT'
        nav_ok = True
        for pr, pc in path[1:]:
            pwx, pwy = occ_return.cell_to_world(pr, pc)
            if not _navigate_to(cf, shared, hub, occ_return, hmap,
                                pwx, pwy, config.FLIGHT_Z, config.NAV_SPEED):
                nav_ok = False
                break
        shared.nav_waypoints = []
        if nav_ok:
            do_rotation_scan(cf, shared, hub, occ_return, hmap, angle_deg=90.0)
        else:
            failed_plans += 1
            if failed_plans >= MAX_FAILED:
                do_rotation_scan(cf, shared, hub, occ_return, hmap, angle_deg=90.0)
                failed_plans = 0

    shared.frontier_target = None
    shared.nav_waypoints = []
    hub.clear_frontier()


def do_nav_to_start(cf, shared, hub, _occ, hmap):
    shared.current_state = 'NAV_TO_START'
    occ_return = OccupancyGrid()
    do_rotation_scan(cf, shared, hub, occ_return, hmap)
    home_x = config.RETURN_SCAN_X - config.TAKEOFF_PAD_X   # arena X=0.5 → EKF -0.5
    _nav_to_home(cf, shared, hub, occ_return, hmap, home_x)
    shared.current_state = 'NAV_TO_START'
    shared.occ_return_grid = occ_return.snapshot()


def do_return_region_scan(cf, shared, hub, occ, hmap):
    """Two-pass dual-altitude scan to locate the takeoff pad on return."""
    shared.current_state = 'RETURN_REGION_SCAN'

    sx = config.RETURN_SCAN_X - config.TAKEOFF_PAD_X
    sy = config.RETURN_SCAN_Y - config.TAKEOFF_PAD_Y
    _navigate_to(cf, shared, hub, occ, hmap, sx, sy, config.FLIGHT_Z, config.NAV_SPEED)

    data = _step(shared, hub, occ, hmap)
    print(f'[return_scan] scanning from ({data.pose[0]:.2f}, {data.pose[1]:.2f})'
          f'  target=({sx:.2f}, {sy:.2f})')

    occ_scan_high = OccupancyGrid()
    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=180.0,
                     occ_target=occ_scan_high,
                     state='SCAN_HIGH')
    shared.occ_scan_high_grid = occ_scan_high.snapshot(filter_outliers=False)

    occ_low = OccupancyGrid()
    controller.takeoff_vel(cf, config.LOW_SCAN_Z, speed=0.15, settle=0.5)
    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=180.0,
                     occ_target=occ_low,
                     scan_z=config.LOW_SCAN_Z,
                     freeze_occ=True,
                     state='SCAN_LOW')
    shared.occ_low_grid = occ_low.snapshot(filter_outliers=False)
    controller.takeoff_vel(cf, config.FLIGHT_Z, speed=0.15, settle=0.5)

    pad_pos, diff_grid = find_pad_from_diff(occ_scan_high, occ_low)
    shared.occ_diff_grid = diff_grid
    if pad_pos is not None:
        print(f'[return_scan] takeoff pad at EKF ({pad_pos[0]:.3f}, {pad_pos[1]:.3f})')
    else:
        print('[return_scan] takeoff pad not found')
    return pad_pos


def do_land_on_start(cf, shared, hub, occ, hmap):
    shared.current_state = 'LANDING_ON_START'
    hx, hy = shared.home_pos
    do_land_on_pad(cf, shared, hub, occ, hmap, hx, hy,
                   yaw_align=False, entry_overshoot=0.08)
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
            ('occ_scan_high', logger.occ_scan_high_path,  lambda: shared.occ_scan_high_grid),
            ('occ_low',       logger.occ_low_path,        lambda: shared.occ_low_grid),
            ('occ_diff',      logger.occ_diff_path,       lambda: shared.occ_diff_grid),
            ('occ_return',    logger.occ_return_path,     lambda: shared.occ_return_grid),
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
