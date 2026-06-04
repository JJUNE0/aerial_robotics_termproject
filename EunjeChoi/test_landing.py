"""
Landing-only test: takeoff → scan → land on pad.
No navigation to landing region; drone scans from wherever it takes off.
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
    _step,
    do_takeoff,
    do_landing_region_scan,
    do_land_on_pad,
)
from sensors import SensorHub
from shared_state import SharedState


def run_test(cf, shared: SharedState):
    occ = OccupancyGrid()
    hmap = HeightMap()
    logger = FlightLogger()
    hub = SensorHub(cf, logger=logger)
    hub.start()

    try:
        do_takeoff(cf, shared, hub, occ, hmap)

        pad_pos = do_landing_region_scan(cf, shared, hub, occ, hmap)

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
