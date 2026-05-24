import math
import time

import config


def init_ekf(cf):
    """Reset Kalman filter and allow 2 s for variance to settle."""
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2.0)


def takeoff(cf, height: float, duration: float):
    """Non-blocking takeoff via HighLevelCommander."""
    cf.high_level_commander.takeoff(height, duration)


def land(cf, target_z: float, duration: float):
    """Non-blocking land via HighLevelCommander."""
    cf.high_level_commander.land(target_z, duration)


def go_to_nonblocking(cf, tx: float, ty: float, tz: float,
                      yaw_deg: float, speed: float,
                      current_x: float, current_y: float, current_z: float) -> float:
    """Send HL go_to and return expected flight duration in seconds."""
    dist = math.sqrt((tx - current_x) ** 2 +
                     (ty - current_y) ** 2 +
                     (tz - current_z) ** 2)
    duration = max(dist / max(speed, 0.01), 0.3)
    cf.high_level_commander.go_to(tx, ty, tz, math.radians(yaw_deg), duration)
    return duration


def set_velocity_world(cf, vx: float, vy: float, vz: float, yaw_rate: float):
    """Send velocity setpoint in world frame."""
    cf.commander.send_velocity_world_setpoint(vx, vy, vz, yaw_rate)


def send_hover(cf, vx_body: float, vy_body: float,
               yaw_rate: float, z: float):
    """Send hover setpoint (body-frame velocity, yaw rate, abs z above floor)."""
    cf.commander.send_hover_setpoint(vx_body, vy_body, yaw_rate, z)


def stop_motion(cf):
    """Hover in place at cruise altitude."""
    cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, config.FLIGHT_Z)


def near_ceiling(z: float) -> bool:
    return z > config.CEILING_LIMIT - 0.05


def clamp_z(cf, z: float):
    """If close to ceiling, command a lower altitude."""
    if near_ceiling(z):
        cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, config.CEILING_LIMIT - 0.15)
