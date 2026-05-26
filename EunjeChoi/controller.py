import math
import threading
import time

from cflib.crazyflie.log import LogConfig

import config


def disarm(cf):
    """Disarm motors and disable HLC — leaves firmware in a clean state."""
    try:
        cf.supervisor.send_arming_request(False)
    except Exception:
        pass
    try:
        cf.param.set_value('commander.enHighLevel', '0')
    except Exception:
        pass


def _wait_for_ekf(cf, timeout: float = 10.0,
                  var_threshold: float = 0.001,
                  history_len: int = 10):
    """Block until kalman XY variances have stabilised.

    Convergence criterion: the range (max-min) of the last `history_len`
    variance samples is below `var_threshold` for both X and Y.
    """
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
    """Disarm leftover state, reset EKF, wait for convergence, then re-arm."""
    print('[init_ekf] disarming...')
    disarm(cf)
    time.sleep(0.5)

    print('[init_ekf] resetting EKF...')
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    print('[init_ekf] waiting for EKF to converge...')
    _wait_for_ekf(cf)

    cf.param.set_value('commander.enHighLevel', '1')
    time.sleep(0.1)

    # Check if supervisor-based arming is supported (requires CRTP v12+)
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
    time.sleep(1.0)
    print('[init_ekf] arm request sent')


def go_to_nonblocking(cf, tx: float, ty: float, tz: float,
                      yaw_deg: float, speed: float,
                      current_x: float, current_y: float,
                      current_z: float) -> float:
    """Send HL go_to and return expected flight duration in seconds."""
    dist = math.sqrt((tx - current_x) ** 2 +
                     (ty - current_y) ** 2 +
                     (tz - current_z) ** 2)
    duration = max(dist / max(speed, 0.01), 0.3)
    cf.high_level_commander.go_to(tx, ty, tz, math.radians(yaw_deg), duration)
    return duration
