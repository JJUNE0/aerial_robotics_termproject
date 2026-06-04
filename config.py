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
FLIGHT_Z = 0.3           # cruise altitude (m)
LAND_HOVER_TIME  = 1.0   # hover duration before descending (s)
LAND_SPEED       = 0.10  # descent speed (m/s)
LAND_CUTOFF_Z    = 0.03  # altitude at which motors are cut (m)
CEILING_LIMIT = 1.20     # hard ceiling (m)
NAV_SPEED = 0.3          # navigation speed (m/s)
SCAN_SPEED = 0.1        # lawnmower speed (m/s)
AVOID_THRESHOLD = 0.40        # begin avoidance (m)
STOP_THRESHOLD = 0.15         # stop threshold (m)
NAV_ARRIVE_THRESHOLD = 0.03   # arrival radius — drone must settle within this (m)
NAV_SETTLE_TIMEOUT = 2.0      # max extra wait after motion for settling (s)
ARENA_NAV_MARGIN = DRONE_HALF_DIAGONAL  # safe margin from arena boundary (m)

# Rotation scan
SCAN_ROTATE_RATE = 15.0    # deg/s
SCAN_ROTATE_STEP = 45.0    # deg per go_to command (smaller = less overshoot)
SCAN_ROTATE_ANGLE = 90.0   # default total scan angle

# Map resolution
OCCUPANCY_GRID_RES = 0.02  # meters / cell
HEIGHT_MAP_RES = 0.05

# A* clearance penalty — keeps path away from obstacles beyond inflation radius
A_STAR_CLEARANCE_RADIUS = 4   # cell radius to scan for nearby obstacles (4 cells = 8 cm)
A_STAR_CLEARANCE_WEIGHT = 1.0 # penalty weight per obstacle cell (inverse-distance weighted)
INFLATION_RADIUS = DRONE_HALF_DIAGONAL #  m
OCC_MIN_COMPONENT_CELLS = 3  # OCCUPIED clusters smaller than this are treated as noise

# Landing pad detection
PAD_SIZE = 0.30               # 30 × 30 cm
PAD_HEIGHT = 0.10             # ~10 cm above floor
SCAN_ROW_SPACING = 0.1        # column spacing for Y-sweep lawnmower (m)
PAIR_SAME_COL_TOL = 0.06      # max |entry_x - exit_x| to count as same column (m)
PAIR_MIN_Y_SPAN   = 0.20      # min |entry_y - exit_y| for a valid pad crossing (m)
EDGE_BASELINE        = FLIGHT_Z   # fixed baseline = cruise altitude (m)
EDGE_ENTRY_DIP       = 0.03       # valley must be ≥5 cm below baseline → entry
EDGE_EXIT_RISE       = 0.06       # peak must be ≥5 cm above baseline  → exit
EDGE_MIN_DZ          = 0.008      # minimum |dz| per sample to register a direction change (m)
EDGE_COOLDOWN        = 1.0        # seconds to suppress detection after entry or exit fires

# Precise pad scan
PRECISE_SWEEP_DIST   = 0.30       # max sweep in each direction from entry point (m)
PRECISE_EXIT_MARGIN  = 0.05       # extra distance beyond exit before reversing (m)

PAD_MIN_CLUSTER_SPAN = 0.15   # min X-span of paired cluster to be a pad (m)
PAD_CONFIRM_TIME = 0.6        # hover-confirm duration (s)

# Dual-altitude pad detection
LOW_SCAN_Z      = 0.08   # low-altitude scan height — below pad/bar (10 cm)
LANDING_SCAN_X  = 4.7    # scan setpoint x in arena coords (m)
LANDING_SCAN_Y  = 1.5    # scan setpoint y in arena coords (m)
RETURN_SCAN_X   = 0.5    # return scan setpoint x in arena coords (m)
RETURN_SCAN_Y   = 2.0    # return scan setpoint y in arena coords (m)
LANDING_SCAN_Y1 = 0.5    # scan setpoint y1 in arena coords (m)
LANDING_SCAN_Y2 = 2.5    # scan setpoint y2 in arena coords (m)
PAD_LAND_OVERSHOOT = 0.08  # approach target: this far past pad centre in +X (m)
PAD_LAND_SEARCH_MARGIN = 0.15  # max travel past estimated entry edge without detection (m)
PAD_LAND_ENTRY_OVERSHOOT = 0.30  # max travel after detected entry edge (m)
PAD_LAND_X_SPEED    = 0.15  # speed for X-axis alignment before pad approach (m/s)
PAD_LAND_PUSH_SPEED = 0.25   # Y-axis approach speed toward pad (m/s)
PAD_BBOX_MIN    = 0.20   # min short side of bounding box (m)
PAD_BBOX_MAX    = 0.50   # max long  side of bounding box (m)

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
