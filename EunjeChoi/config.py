import math

RADIO_URI = 'radio://0/80/2M/E7E7E7E7E5'

# Arena dimensions (meters)
ARENA_X = 3
ARENA_Y = 1
START_REGION_X = 1.5
LANDING_REGION_X = 1.5
MIDDLE_REGION_X = ARENA_X - (START_REGION_X + LANDING_REGION_X) # 2 m

# Takeoff pad location — provided: fill before running
TAKEOFF_PAD_X = 0.5   # meters from arena west wall along +x
TAKEOFF_PAD_Y = 0.5   # meters from arena south wall along +y

# Drone body spec (including protective frame)
DRONE_BODY_SIZE = 0.15
DRONE_HALF_DIAGONAL = (DRONE_BODY_SIZE / 2) * math.sqrt(2)   # ~0.106 m

# Flight
FLIGHT_Z = 0.3           # cruise altitude (m)
CEILING_LIMIT = 1.20     # hard ceiling (m)
NAV_SPEED = 0.2          # navigation speed (m/s)
SCAN_SPEED = 0.20        # lawnmower speed (m/s)
AVOID_THRESHOLD = 0.40        # begin avoidance (m)
STOP_THRESHOLD = 0.25         # stop threshold (m)
NAV_ARRIVE_THRESHOLD = 0.03   # arrival radius — drone must settle within this (m)
NAV_SETTLE_TIMEOUT = 2.0      # max extra wait after motion for settling (s)

# Rotation scan
SCAN_ROTATE_RATE = 20.0    # deg/s
SCAN_ROTATE_STEP = 45.0    # deg per go_to command (smaller = less overshoot)
SCAN_ROTATE_ANGLE = 90.0   # default total scan angle

# Map resolution
OCCUPANCY_GRID_RES = 0.03  # meters / cell
HEIGHT_MAP_RES = 0.05
INFLATION_RADIUS = DRONE_HALF_DIAGONAL + 0.03   # ~0.156 m

# Landing pad detection
PAD_SIZE = 0.30               # 30 × 30 cm
PAD_HEIGHT = 0.10             # ~10 cm above floor
SCAN_ROW_SPACING = 0.15        # column spacing for Y-sweep lawnmower (m)
PAIR_SAME_COL_TOL = 0.06      # max |entry_x - exit_x| to count as same column (m)
PAIR_MIN_Y_SPAN   = 0.20      # min |entry_y - exit_y| for a valid pad crossing (m)
EDGE_THRESHOLD       = 0.05   # m — z_down drop from baseline to trigger entry
EDGE_BASELINE_ALPHA  = 0.1    # EMA coefficient for baseline update (frozen while over pad)
EDGE_EXIT_DEBOUNCE   = 3      # consecutive samples below exit threshold before exit fires
PAD_MIN_CLUSTER_SPAN = 0.15   # min X-span of paired cluster to be a pad (m)
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
