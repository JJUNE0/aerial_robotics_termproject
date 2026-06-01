import math
import threading
import time

from cflib.crazyflie.log import LogConfig

import config


def disarm(cf):
    """Stop motors and disarm."""
    try:
        cf.commander.send_stop_setpoint()
    except Exception:
        pass
    try:
        cf.supervisor.send_arming_request(False)
    except Exception:
        pass


def _wait_for_ekf(cf, timeout: float = 10.0,
                  var_threshold: float = 0.001,
                  history_len: int = 10):
    """Block until kalman XY variances have stabilised."""
    hist_x = [1000.0] * history_len
    hist_y = [1000.0] * history_len
    done = threading.Event()

    def _cb(_, data, __):
        hist_x.append(data['kalman.varPX'])
        hist_x.pop(0)
        hist_y.append(data['kalman.varPY'])
        hist_y.pop(0)
        if (max(hist_x) - min(hist_x) < var_threshold and
                max(hist_y) - min(hist_y) < var_threshold):
            done.set()

    cfg = LogConfig('ekf_var', period_in_ms=100)
    cfg.add_variable('kalman.varPX', 'float')
    cfg.add_variable('kalman.varPY', 'float')
    cfg.data_received_cb.add_callback(_cb)

    cf.log.add_config(cfg)
    cfg.start()
    converged = done.wait(timeout=timeout)
    cfg.stop()
    cfg.delete()

    if converged:
        print('[init_ekf] EKF converged')
    else:
        print(f'[init_ekf] EKF did not converge within {timeout}s — proceeding anyway')


def init_ekf(cf):
    """Reset EKF, wait for convergence, then arm. No HLC — velocity control only."""
    print('[init_ekf] disabling HLC...')
    cf.param.set_value('commander.enHighLevel', '0')

    print('[init_ekf] resetting EKF...')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    print('[init_ekf] waiting for EKF to converge...')
    _wait_for_ekf(cf)

    armed_ready = False
    for i in range(20):
        if cf.supervisor.can_be_armed:
            armed_ready = True
            print(f'[init_ekf] can_be_armed=True (attempt {i+1})')
            break
        time.sleep(0.1)

    if armed_ready:
        print('[init_ekf] new firmware — arming via supervisor')
    else:
        print('[init_ekf] legacy firmware — sending arming request anyway')

    cf.supervisor.send_arming_request(True)
    time.sleep(0.5)
    print('[init_ekf] armed')


def hover(cf, z: float):
    """Send a single hover-in-place setpoint."""
    cf.commander.send_hover_setpoint(0, 0, 0, z)


def takeoff_vel(cf, target_z: float, speed: float = 0.3, settle: float = 1.5):
    """Ramp altitude from 0 to target_z using velocity control, then settle."""
    print(f'[takeoff] ascending to {target_z:.2f} m')
    z_cmd = 0.05
    while z_cmd < target_z:
        z_cmd = min(z_cmd + speed * config.DT, target_z)
        cf.commander.send_hover_setpoint(0, 0, 0, z_cmd)
        time.sleep(config.DT)

    end_t = time.time() + settle
    while time.time() < end_t:
        cf.commander.send_hover_setpoint(0, 0, 0, target_z)
        time.sleep(config.DT)
    print('[takeoff] stable')


def land_vel(cf, hub, speed: float = 0.2):
    """Ramp altitude down to 0 using velocity control, then stop motors."""
    print('[land] descending')
    z_cmd = hub.read().pose[2]
    while z_cmd > 0.05:
        z_cmd = max(z_cmd - speed * config.DT, 0.0)
        cf.commander.send_hover_setpoint(0, 0, 0, z_cmd)
        time.sleep(config.DT)
    cf.commander.send_stop_setpoint()
    print('[land] landed')


def vel_to_body(vx_w: float, vy_w: float, yaw_deg: float):
    """Convert world-frame XY velocity to body frame."""
    yaw_rad = math.radians(yaw_deg)
    vx_b =  vx_w * math.cos(yaw_rad) + vy_w * math.sin(yaw_rad)
    vy_b = -vx_w * math.sin(yaw_rad) + vy_w * math.cos(yaw_rad)
    return vx_b, vy_b
