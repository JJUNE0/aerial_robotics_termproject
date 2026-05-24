import math

RADIO_URI = 'radio://0/80/2M/E7E7E7E7E5'

# Arena dimensions (meters)
ARENA_X = 5.0
ARENA_Y = 3.0
START_REGION_X = 1.5
MIDDLE_REGION_X = 2.0
LANDING_REGION_X = 1.5

# Takeoff pad location — provided 2026-06-01; fill before running
TAKEOFF_PAD_X = None   # meters from arena west wall along +x
TAKEOFF_PAD_Y = None   # meters from arena south wall along +y

# Drone body spec (including protective frame)
DRONE_BODY_SIZE = 0.15
DRONE_HALF_DIAGONAL = (DRONE_BODY_SIZE / 2) * math.sqrt(2)   # ~0.106 m

# Flight
FLIGHT_Z = 0.5           # cruise altitude (m)
CEILING_LIMIT = 1.20     # hard ceiling (m)
NAV_SPEED = 0.3          # navigation speed (m/s)
SCAN_SPEED = 0.20        # lawnmower speed (m/s)
AVOID_THRESHOLD = 0.40   # begin avoidance (m)
STOP_THRESHOLD = 0.25    # stop threshold (m)

# Rotation scan
SCAN_ROTATE_RATE = 30.0    # deg/s
SCAN_ROTATE_ANGLE = 90.0   # deg

# Map resolution
OCCUPANCY_GRID_RES = 0.05   # meters / cell
HEIGHT_MAP_RES = 0.05
INFLATION_RADIUS = DRONE_HALF_DIAGONAL + 0.05   # ~0.156 m

# Landing pad detection
PAD_SIZE = 0.30               # 30 × 30 cm
PAD_HEIGHT = 0.10             # ~10 cm above floor
SCAN_ROW_SPACING = 0.25       # ㄹ pattern row gap (m)
EDGE_THRESHOLD = 0.05         # z-ranger drop to trigger edge (m)
EDGE_BASELINE_ALPHA = 0.02    # EMA smoothing factor
PAD_MIN_WIDTH = 0.20          # minimum footprint width to be a pad (m)
PAD_MAX_WIDTH = 0.45          # maximum footprint width to be a pad (m)
PAD_CONFIRM_TIME = 0.6        # hover-confirm duration (s)

# Mission timer (display only — no forced actions)
MISSION_TIME_LIMIT = 180.0

# Battery display
BATTERY_V_MAX = 4.20
BATTERY_V_MIN = 3.00

# Control loop period
DT = 0.05   # 20 Hz


# EKF coordinate frame helpers
# After kalman.resetEstimation the EKF origin is the takeoff pad on the floor.
# +x toward landing region, +y left, +z up.

def ekf_landing_region_start():
    """EKF x where the landing region begins."""
    return START_REGION_X + MIDDLE_REGION_X - TAKEOFF_PAD_X


def ekf_arena_x_min():
    return -TAKEOFF_PAD_X


def ekf_arena_x_max():
    return ARENA_X - TAKEOFF_PAD_X


def ekf_arena_y_min():
    return -TAKEOFF_PAD_Y


def ekf_arena_y_max():
    return ARENA_Y - TAKEOFF_PAD_Y
