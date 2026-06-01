"""
Simple back-and-forth test flight.
Takeoff → hover → move +X → return → land.

Usage:
    python test_hover.py
"""

import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import config
import controller
from logger import FlightLogger
from sensors import SensorHub

FORWARD_DIST = 0.45 * 2  # m to travel in +X
SPEED        = 0.1   # m/s
HOVER_TIME   = 1     # s pause at each end
PAUSE        = config.EDGE_COOLDOWN   # hover duration on entry/exit (= detection cooldown)


def _travel(hlc, hub, tx, ty, speed, label='move'):
    """go_to (tx, ty) at speed, pausing on every entry or exit event."""
    pose = hub.read().pose
    dist = ((tx - pose[0]) ** 2 + (ty - pose[1]) ** 2) ** 0.5
    duration = max(dist / speed, 0.5)
    print(f'[test] {label} → ({tx:.3f}, {ty:.3f})  {duration:.1f}s')
    hlc.go_to(tx, ty, config.FLIGHT_Z, 0.0, duration)

    deadline = time.time() + duration
    while time.time() < deadline:
        try:
            event = hub.edge_queue.get_nowait()
            if event.kind in ('entry', 'exit'):
                data = hub.read()
                px, py = data.pose[0], data.pose[1]
                vx, vy = data.velocity[0], data.velocity[1]
                stop_x = px + vx * PAUSE * 0.5
                stop_y = py + vy * PAUSE * 0.5
                print(f'[test] {event.kind} at ({event.x:.3f}, {event.y:.3f}) '
                      f'vel=({vx:.3f},{vy:.3f}) → stop ({stop_x:.3f}, {stop_y:.3f})')
                hlc.go_to(stop_x, stop_y, config.FLIGHT_Z, 0.0, PAUSE)
                time.sleep(PAUSE)
                remaining = max(deadline - time.time(), 0.5)
                hlc.go_to(tx, ty, config.FLIGHT_Z, 0.0, remaining)
        except Exception:
            pass
        time.sleep(config.DT)


def run(cf):
    hlc = cf.high_level_commander
    logger = FlightLogger()
    hub = SensorHub(cf, logger=logger)

    hub.start()
    controller.init_ekf(cf)

    try:
        print('[test] takeoff')
        hlc.takeoff(0.5, 1.0)    # punch through ground effect fast (cfclient style)
        time.sleep(1.5)
        hlc.go_to(0.0, 0.0, config.FLIGHT_Z, 0.0, 1.0)   # descend to cruise altitude
        time.sleep(2.0)

        # Record stabilised hover position as home — EKF may have drifted during takeoff
        pose = hub.read().pose
        home_x, home_y = pose[0], pose[1]
        print(f'[test] home recorded: ({home_x:.3f}, {home_y:.3f})')

        _travel(hlc, hub, home_x + FORWARD_DIST, home_y, SPEED, 'move forward')
        time.sleep(HOVER_TIME)

        _travel(hlc, hub, home_x, home_y, SPEED, 'return home')
        time.sleep(HOVER_TIME)

        print('[test] land')
        hlc.land(0.0, 2.0)
        time.sleep(2.5)

    except KeyboardInterrupt:
        print('[test] interrupted — emergency land')
        hlc.land(0.0, 2.0)
        time.sleep(2.5)

    finally:
        hub.stop()
        logger.close()
        controller.disarm(cf)
        print('[test] done')


def main():
    cflib.crtp.init_drivers()
    print(f'[test] connecting to {config.RADIO_URI}')
    with SyncCrazyflie(config.RADIO_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        run(scf.cf)


if __name__ == '__main__':
    main()
