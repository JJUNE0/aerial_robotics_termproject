import threading
import time


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = 'INIT'
        self._pose = (0.0, 0.0, 0.0, 0.0)   # x, y, z, yaw_deg
        self._battery_pct = 0.0
        self._emergency_flag = False
        self._occupancy_grid = None           # numpy array snapshot
        self._height_map_edges = []           # list of EdgeEvent
        self._pad_pairs = []                  # list of (cx, cy) matched pairs
        self._pad_candidates = []             # list of PadCandidate
        self._target_pos = None               # current nav target (x, y) or None
        self._landing_target = None           # confirmed landing pad position (x, y) or None
        self._start_time = None

    # ------------------------------------------------------------------ timer

    def start_timer(self):
        with self._lock:
            self._start_time = time.time()

    @property
    def elapsed_time(self):
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time

    # ------------------------------------------------------------------ state

    @property
    def current_state(self):
        with self._lock:
            return self._state

    @current_state.setter
    def current_state(self, v):
        with self._lock:
            self._state = v

    # ------------------------------------------------------------------ pose

    @property
    def pose(self):
        with self._lock:
            return self._pose

    @pose.setter
    def pose(self, v):
        with self._lock:
            self._pose = v

    # -------------------------------------------------------------- battery

    @property
    def battery_pct(self):
        with self._lock:
            return self._battery_pct

    @battery_pct.setter
    def battery_pct(self, v):
        with self._lock:
            self._battery_pct = v

    # ----------------------------------------------------------- emergency

    @property
    def emergency_flag(self):
        with self._lock:
            return self._emergency_flag

    def trigger_emergency(self):
        with self._lock:
            self._emergency_flag = True

    # ------------------------------------------------------- occupancy grid

    @property
    def occupancy_grid(self):
        with self._lock:
            return self._occupancy_grid

    @occupancy_grid.setter
    def occupancy_grid(self, v):
        with self._lock:
            self._occupancy_grid = v

    # ------------------------------------------------------- height map

    @property
    def height_map_edges(self):
        with self._lock:
            return list(self._height_map_edges)

    def add_height_map_edge(self, edge):
        with self._lock:
            self._height_map_edges.append(edge)

    # ------------------------------------------------------- nav target

    @property
    def target_pos(self):
        with self._lock:
            return self._target_pos

    @target_pos.setter
    def target_pos(self, v):
        with self._lock:
            self._target_pos = v

    # ------------------------------------------------------- landing target

    @property
    def landing_target(self):
        with self._lock:
            return self._landing_target

    @landing_target.setter
    def landing_target(self, v):
        with self._lock:
            self._landing_target = v

    # ------------------------------------------------------- pad pairs

    @property
    def pad_pairs(self):
        with self._lock:
            return list(self._pad_pairs)

    @pad_pairs.setter
    def pad_pairs(self, v):
        with self._lock:
            self._pad_pairs = list(v)

    # ------------------------------------------------------- pad candidates

    @property
    def pad_candidates(self):
        with self._lock:
            return list(self._pad_candidates)

    @pad_candidates.setter
    def pad_candidates(self, v):
        with self._lock:
            self._pad_candidates = list(v)
