"""Crazyflie control helpers: velocity/height rate limiters + arming + e-stop."""
import time
import warnings

from .config import MAX_HEIGHT_COMMAND, MAX_HEIGHT_STEP_UP, MAX_VELOCITY_STEP


_last_height_command = 0.0
_last_vx_command = 0.0
_last_vy_command = 0.0


def reset_height_limiter(h=0.0):
    global _last_height_command
    _last_height_command = h


def reset_velocity_limiter(vx=0.0, vy=0.0):
    global _last_vx_command, _last_vy_command
    _last_vx_command = vx
    _last_vy_command = vy


def limited_velocity(rvx, rvy):
    global _last_vx_command, _last_vy_command
    dvx = max(-MAX_VELOCITY_STEP, min(MAX_VELOCITY_STEP, rvx - _last_vx_command))
    dvy = max(-MAX_VELOCITY_STEP, min(MAX_VELOCITY_STEP, rvy - _last_vy_command))
    _last_vx_command += dvx
    _last_vy_command += dvy
    return _last_vx_command, _last_vy_command


def limited_height(rh, max_h=MAX_HEIGHT_COMMAND):
    global _last_height_command
    h = min(rh, max_h)
    if h > _last_height_command:
        h = min(h, _last_height_command + MAX_HEIGHT_STEP_UP)
    _last_height_command = h
    return h


def send_velocity_limited(cf, vx, vy, rh, max_h=MAX_HEIGHT_COMMAND):
    h = limited_height(rh, max_h)
    vx_l, vy_l = limited_velocity(vx, vy)
    cf.commander.send_hover_setpoint(vx_l, vy_l, 0.0, h)
    return h, vx_l, vy_l


def send_arming_request(cf, do_arm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cf.supervisor.send_arming_request(do_arm)
            return
        except Exception:
            pass
        try:
            cf.platform.send_arming_request(do_arm)
        except Exception:
            pass


def emergency_stop(cf):
    if cf is None:
        return
    for _ in range(3):
        try:
            cf.commander.send_stop_setpoint()
        except Exception:
            pass
        send_arming_request(cf, False)
        time.sleep(0.05)
