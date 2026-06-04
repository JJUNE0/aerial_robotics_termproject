"""
Landing-only test: takeoff → scan → land on pad.
Scan position is a test-specific offset from the takeoff/home position.
"""

import threading
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import config
import controller
from gui import MissionGUI
from logger import FlightLogger
from mapping import OccupancyGrid, HeightMap, find_pad_from_diff
from mission import (
    EmergencyException,
    _navigate_to,
    _nav_to_x,
    _step,
    do_rotation_scan,
    do_takeoff,
    do_land_on_pad,
)
from sensors import SensorHub
from shared_state import SharedState


TEST_SCAN_OFFSET_X = 0.4


def do_test_landing_scan(cf, shared, hub, occ, hmap):
    """Scan for the landing pad from the test-specific scan point."""
    home_x, home_y = shared.home_pos
    sx = home_x
    sy = home_y + TEST_SCAN_OFFSET_X

    print(f'[test] scan target from home ({home_x:.2f}, {home_y:.2f})'
          f' -> ({sx:.2f}, {sy:.2f})')

    shared.current_state = 'LANDING_REGION_SCAN'
    _nav_to_x(cf, shared, hub, occ, hmap, sx)
    _navigate_to(cf, shared, hub, occ, hmap, sx, sy,
                 config.FLIGHT_Z, config.NAV_SPEED)

    data = _step(shared, hub, occ, hmap)
    print(f'[scan] scanning from ({data.pose[0]:.2f}, {data.pose[1]:.2f})'
          f'  target=({sx:.2f}, {sy:.2f})')

    occ_scan_high = OccupancyGrid()
    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=180.0,
                     occ_target=occ_scan_high)
    shared.occ_scan_high_grid = occ_scan_high.snapshot()

    occ_low = OccupancyGrid()
    controller.takeoff_vel(cf, config.LOW_SCAN_Z, speed=0.15, settle=0.5)

    do_rotation_scan(cf, shared, hub, occ, hmap,
                     angle_deg=180.0,
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


def run_test(cf, shared: SharedState):
    occ = OccupancyGrid()
    hmap = HeightMap()
    logger = FlightLogger()
    hub = SensorHub(cf, logger=logger)
    hub.start()

    try:
        do_takeoff(cf, shared, hub, occ, hmap)

        pad_pos = do_test_landing_scan(cf, shared, hub, occ, hmap)

        if pad_pos is not None:
            shared.landing_target = pad_pos
            do_land_on_pad(cf, shared, hub, occ, hmap, pad_pos[0], pad_pos[1])
        else:
            print('[test] pad not found — landing in place')
            shared.current_state = 'LANDING_ON_START'
            controller.land_vel(cf, hub)

        shared.current_state = 'DONE'

    except EmergencyException:
        shared.current_state = 'EMERGENCY_LAND'
        try:
            controller.land_vel(cf, hub)
        except Exception:
            pass

    except Exception as exc:
        shared.current_state = 'ERROR'
        print(f'[test] unhandled exception: {exc}')
        try:
            controller.land_vel(cf, hub)
        except Exception:
            pass

    finally:
        hub.stop()
        controller.disarm(cf)


def main():
    cflib.crtp.init_drivers()
    shared = SharedState()

    for attempt in range(1, 6):
        try:
            print(f'[main] connecting to {config.RADIO_URI}  (attempt {attempt}/5)')
            with SyncCrazyflie(config.RADIO_URI,
                               cf=Crazyflie(rw_cache='./cache')) as scf:
                t = threading.Thread(target=run_test, args=(scf.cf, shared), daemon=True)
                t.start()
                MissionGUI(shared).start()
            break
        except Exception as exc:
            print(f'[main] connection error: {exc}')
            if attempt < 5:
                time.sleep(3.0)
            else:
                raise


if __name__ == '__main__':
    main()
