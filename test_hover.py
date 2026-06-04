"""
Simple back-and-forth test flight using velocity control.
Takeoff → hover → move +X → return → land.

Usage:
    python test_hover.py
"""

import math
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import config
import controller
from logger import FlightLogger
from sensors import SensorHub

FORWARD_DIST = 0.45 * 2  # m to travel in +X
SPEED        = 0.1        # m/s
HOVER_TIME   = 1.0        # s pause at each end
PAUSE        = config.EDGE_COOLDOWN   # hover on entry/exit (= detection cooldown)
KP           = 3.0        # position P-gain


def _travel(cf, hub, tx, ty, speed, label='move'):
    """Velocity P-controller to (tx, ty), pausing on entry/exit events."""
    print(f'[test] {label} → ({tx:.3f}, {ty:.3f})')

    while True:
        pose = hub.read().pose
        cx, cy, _, cyaw = pose
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy)

        if dist < config.NAV_ARRIVE_THRESHOLD:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            return

        # Check edge events
        try:
            event = hub.edge_queue.get_nowait()
            if event.kind in ('entry', 'exit'):
                vel = hub.read().velocity
                vx, vy = vel[0], vel[1]
                stop_x = cx + vx * PAUSE * 0.5
                stop_y = cy + vy * PAUSE * 0.5
                print(f'[test] {event.kind} at ({event.x:.3f}, {event.y:.3f}) '
                      f'vel=({vx:.3f},{vy:.3f}) → stop ({stop_x:.3f}, {stop_y:.3f})')
                # Hover at predicted stop position
                end_t = time.time() + PAUSE
                while time.time() < end_t:
                    sx = stop_x - cx
                    sy = stop_y - cy
                    sd = math.hypot(sx, sy)
                    if sd > 0.02:
                        sv = min(speed, sd * KP)
                        vxb, vyb = controller.vel_to_body(
                            (sx/sd)*sv, (sy/sd)*sv, cyaw)
                        cf.commander.send_hover_setpoint(vxb, vyb, 0, config.FLIGHT_Z)
                    else:
                        cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
                    time.sleep(config.DT)
                    pose = hub.read().pose
                    cx, cy, _, cyaw = pose
        except Exception:
            pass

        v = min(speed, dist * KP)
        vx_b, vy_b = controller.vel_to_body((dx/dist)*v, (dy/dist)*v, cyaw)
        cf.commander.send_hover_setpoint(vx_b, vy_b, 0, config.FLIGHT_Z)
        time.sleep(config.DT)


def run(cf):
    logger = FlightLogger()
    hub = SensorHub(cf, logger=logger)

    hub.start()
    controller.init_ekf(cf)

    try:
        print('[test] takeoff')
        controller.takeoff_vel(cf, config.FLIGHT_Z)

        pose = hub.read().pose
        home_x, home_y = pose[0], pose[1]
        print(f'[test] home recorded: ({home_x:.3f}, {home_y:.3f})')

        _travel(cf, hub, home_x + FORWARD_DIST, home_y, SPEED, 'move forward')
        end_t = time.time() + HOVER_TIME
        while time.time() < end_t:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            time.sleep(config.DT)

        _travel(cf, hub, home_x, home_y, SPEED, 'return home')
        end_t = time.time() + HOVER_TIME
        while time.time() < end_t:
            cf.commander.send_hover_setpoint(0, 0, 0, config.FLIGHT_Z)
            time.sleep(config.DT)

        print('[test] land')
        controller.land_vel(cf, hub)

    except KeyboardInterrupt:
        print('[test] interrupted — emergency land')
        controller.land_vel(cf, hub)

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
