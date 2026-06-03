import math

RADIO_URI = 'radio://0/80/2M/E7E7E7E7E5'

# Arena dimensions (meters)
ARENA_X = 5
ARENA_Y = 3
START_REGION_X = 1.5
LANDING_REGION_X = 1.5
MIDDLE_REGION_X = ARENA_X - (START_REGION_X + LANDING_REGION_X) # 2 m

# Takeoff pad location — provided: fill before running
TAKEOFF_PAD_X = 1.0   # meters from arena west wall along +x
TAKEOFF_PAD_Y = 2.5   # meters from arena south wall along +y

# Drone body spec (including protective frame)
DRONE_BODY_SIZE = 0.15
DRONE_HALF_DIAGONAL = (DRONE_BODY_SIZE / 2) * math.sqrt(2)   # ~0.106 m

# Flight
FLIGHT_Z = 0.2           # cruise altitude (m)
CEILING_LIMIT = 1.20     # hard ceiling (m)
NAV_SPEED = 0.3          # navigation speed (m/s)
SCAN_SPEED = 0.1        # lawnmower speed (m/s)
AVOID_THRESHOLD = 0.40        # begin avoidance (m)
STOP_THRESHOLD = 0.25         # stop threshold (m)
BOUNDARY_MARGIN = 0.05        # keep this far inside arena boundary (m)
NAV_FORWARD_MARGIN = 0.35     # obstacle margin along current motion direction (m)
NAV_SIDE_MARGIN = STOP_THRESHOLD  # obstacle margin perpendicular to motion (m)
COLLISION_THRESHOLD = 0.12    # treat as contact risk; freeze map update (m)
SAFETY_NUDGE_DIST = 0.05      # move away this far from any too-close obstacle (m)
SAFETY_NUDGE_SPEED = 0.08     # speed for the short safety nudge (m/s)
NAV_ARRIVE_THRESHOLD = 0.03   # arrival radius — drone must settle within this (m)
NAV_SETTLE_TIMEOUT = 2.0      # stop/replan after this long without progress (s)
REPLAN_RETREAT_DIST = 0.15    # back up this far before choosing a new target (m)
TARGET_BLOCK_RADIUS = 0.25    # temporarily avoid targets near a blocked point (m)
POSE_JUMP_THRESHOLD = 0.25    # ignore map updates after an EKF XY jump (m)
POSE_SPEED_THRESHOLD = 1.20   # ignore map updates above plausible XY speed (m/s)
MAP_FREEZE_TIME = 0.75        # seconds to trust sensors over map after a jump/contact
LANDING_AVOID_REPLANS = 2     # A* retries for landing-region obstacle detours

# Rotation scan
SCAN_ROTATE_RATE = 15.0    # deg/s
SCAN_ROTATE_STEP = 45.0    # deg per go_to command (smaller = less overshoot)
SCAN_ROTATE_ANGLE = 90.0   # default total scan angle
YAW_ALIGN_RATE = 20.0      # deg/s for aligning yaw before pad search
YAW_ALIGN_TOL = 3.0        # acceptable yaw error before searching (deg)
YAW_ALIGN_TIMEOUT = 8.0    # max seconds for yaw alignment
YAW_HOLD_KP = 1.5          # yaw-rate gain while holding takeoff yaw in search

# Map resolution
OCCUPANCY_GRID_RES = 0.02  # meters / cell
HEIGHT_MAP_RES = 0.05
INFLATION_RADIUS = DRONE_HALF_DIAGONAL + 0.01  # body radius + safety margin (m)

# Landing pad detection
PAD_SIZE = 0.30               # 30 × 30 cm
PAD_HEIGHT = 0.10             # ~10 cm above floor
PAD_SIDE_TOL = 0.08           # allowed side-length mismatch for square check (m)
SCAN_ROW_SPACING = 0.1        # column spacing for Y-sweep lawnmower (m)
LOCAL_SCAN_SPEED = 0.06       # slow speed for focused search near an edge (m/s)
LOCAL_SCAN_RADIUS = 0.40      # focused circular scan radius around first edge (m)
LOCAL_SCAN_ROW_SPACING = 0.06 # denser rows inside focused scan area (m)
PAD_PAIR_MIN = PAD_SIZE - PAD_SIDE_TOL  # reject crossings shorter than a pad side
PAD_PAIR_MAX = PAD_SIZE + PAD_SIDE_TOL  # reject long-box crossings
PAIR_CROSS_AXIS_TOL = 0.08      # max sideways drift during one edge crossing (m)
EDGE_BASELINE        = FLIGHT_Z   # fixed baseline = cruise altitude (m)
EDGE_ENTRY_DIP       = 0.05       # valley must be ≥5 cm below baseline → entry
EDGE_EXIT_RISE       = 0.05       # peak must be ≥5 cm above baseline  → exit
EDGE_MIN_DZ          = 0.008      # minimum |dz| per sample to register a direction change (m)
EDGE_COOLDOWN        = 1.0        # seconds to suppress detection after entry or exit fires

# Precise pad scan
PRECISE_SWEEP_DIST   = 0.30       # max sweep in each direction from entry point (m)
PRECISE_EXIT_MARGIN  = 0.05       # extra distance beyond exit before reversing (m)
SQUARE_VERIFY_MARGIN = 0.08       # travel beyond expected side during square check (m)

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
