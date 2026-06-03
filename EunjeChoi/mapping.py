import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import config

FREE = 0
UNKNOWN = 1
OCCUPIED = 2
INFLATED = 3


class OccupancyGrid:
    """2-D occupancy grid with Bresenham ray-casting and obstacle inflation."""

    def __init__(self):
        res = config.OCCUPANCY_GRID_RES
        # Grid bounds add 0.5 m margin around the arena (EKF frame)
        self._x_min = config.ekf_arena_x_min() - 0.5
        self._y_min = config.ekf_arena_y_min() - 0.5
        x_max = config.ekf_arena_x_max() + 0.5
        y_max = config.ekf_arena_y_max() + 0.5

        self._res = res
        cols = int((x_max - self._x_min) / res) + 2
        rows = int((y_max - self._y_min) / res) + 2

        self._grid = np.full((rows, cols), UNKNOWN, dtype=np.int8)
        self._lock = threading.Lock()
        self._inflation_cells = max(1, int(config.INFLATION_RADIUS / res))

    # -------------------------------------------------------- coord helpers

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int((x - self._x_min) / self._res)
        row = int((y - self._y_min) / self._res)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self._x_min + col * self._res
        y = self._y_min + row * self._res
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._grid.shape[0] and 0 <= col < self._grid.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self._grid.shape

    @property
    def res(self) -> float:
        return self._res

    @property
    def x_min(self) -> float:
        return self._x_min

    @property
    def y_min(self) -> float:
        return self._y_min

    # -------------------------------------------------------- Bresenham

    @staticmethod
    def _bresenham(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return cells

    # -------------------------------------------------------- update

    def _inflate(self, row: int, col: int):
        """Mark cells within inflation radius as INFLATED (no lock — caller holds it)."""
        rad = self._inflation_cells
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr * dr + dc * dc <= rad * rad:
                    nr, nc = row + dr, col + dc
                    if self.in_bounds(nr, nc) and self._grid[nr, nc] == FREE:
                        self._grid[nr, nc] = INFLATED

    def update_ray(self, drone_x: float, drone_y: float,
                   ray_angle_deg: float, sensor_dist: Optional[float],
                   max_range: float = 3.5):
        """Update grid with one sensor ray."""
        r0, c0 = self.world_to_cell(drone_x, drone_y)
        ang = math.radians(ray_angle_deg)

        if sensor_dist is not None:
            hx = drone_x + sensor_dist * math.cos(ang)
            hy = drone_y + sensor_dist * math.sin(ang)
        else:
            hx = drone_x + max_range * math.cos(ang)
            hy = drone_y + max_range * math.sin(ang)

        r1, c1 = self.world_to_cell(hx, hy)
        cells = self._bresenham(r0, c0, r1, c1)

        with self._lock:
            if sensor_dist is not None:
                # Mark path free, endpoint occupied
                for r, c in cells[:-1]:
                    if self.in_bounds(r, c) and self._grid[r, c] == UNKNOWN:
                        self._grid[r, c] = FREE
                if cells:
                    er, ec = cells[-1]
                    if self.in_bounds(er, ec):
                        self._grid[er, ec] = OCCUPIED
                        self._inflate(er, ec)
            else:
                # No detection — mark entire ray as free
                for r, c in cells:
                    if self.in_bounds(r, c) and self._grid[r, c] == UNKNOWN:
                        self._grid[r, c] = FREE

    def update_all_rays(self, drone_x: float, drone_y: float,
                        yaw_deg: float, ranges):
        """Update with all 4 horizontal sensors compensated for current yaw."""
        # Body-frame offsets from drone heading
        directions = [
            (0.0,   ranges.front),   # front
            (-90.0, ranges.right),   # right
            (180.0, ranges.back),    # back
            (90.0,  ranges.left),    # left
        ]
        for offset, dist in directions:
            self.update_ray(drone_x, drone_y, yaw_deg + offset, dist)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._grid.copy()


# ---------------------------------------------------------------- HeightMap


@dataclass
class PadCandidate:
    cx: float
    cy: float
    pair_count: int = 0


@dataclass
class EdgeMark:
    x: float
    y: float
    stamp: float
    seq: int


class HeightMap:
    """Detects landing pad from z-ranger edge events.

    Strategy: pair each exit with the nearest earlier unpaired entry, reject
    crossings that are too short/long for a square pad side, then group valid
    pair centers spatially.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[EdgeMark] = []
        self._exits: List[EdgeMark] = []
        self._pairs: List[Tuple[float, float, float, float, float, float]] = []
        self._candidates: List[PadCandidate] = []
        self._entry_seq: int = 0
        self._exit_seq: int = 0
        self._edge_seq: int = 0
        self._last_entry_pos: Optional[Tuple[float, float]] = None
        self._last_exit_pos: Optional[Tuple[float, float]] = None

    @property
    def entry_seq(self) -> int:
        with self._lock:
            return self._entry_seq

    @property
    def exit_seq(self) -> int:
        with self._lock:
            return self._exit_seq

    @property
    def last_entry_pos(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._last_entry_pos

    @property
    def last_exit_pos(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._last_exit_pos

    def add_entry(self, x: float, y: float, stamp: Optional[float] = None):
        with self._lock:
            self._edge_seq += 1
            self._entries.append(EdgeMark(x, y, self._edge_seq if stamp is None else stamp,
                                          self._edge_seq))
            self._entry_seq += 1
            self._last_entry_pos = (x, y)

    def add_exit(self, x: float, y: float, stamp: Optional[float] = None):
        with self._lock:
            self._edge_seq += 1
            self._exits.append(EdgeMark(x, y, self._edge_seq if stamp is None else stamp,
                                        self._edge_seq))
            self._exit_seq += 1
            self._last_exit_pos = (x, y)
        self._recompute_candidates()

    @staticmethod
    def _valid_pad_pair(entry: EdgeMark, exit_: EdgeMark) -> bool:
        dx = exit_.x - entry.x
        dy = exit_.y - entry.y
        along = max(abs(dx), abs(dy))
        cross = min(abs(dx), abs(dy))
        return (config.PAD_PAIR_MIN <= along <= config.PAD_PAIR_MAX and
                cross <= config.PAIR_CROSS_AXIS_TOL)

    def _recompute_candidates(self):
        with self._lock:
            # Pair each exit with the most recent unpaired entry that happened
            # before it. Invalid crossings are consumed and discarded.
            pairs = []
            used_entries = set()

            for exit_ in self._exits:
                best_j = None
                for j in range(len(self._entries) - 1, -1, -1):
                    if j not in used_entries and self._entries[j].seq < exit_.seq:
                        best_j = j
                        break

                if best_j is None:
                    continue

                entry = self._entries[best_j]
                used_entries.add(best_j)
                if not self._valid_pad_pair(entry, exit_):
                    continue

                cx = (entry.x + exit_.x) / 2.0
                cy = (entry.y + exit_.y) / 2.0
                pairs.append((cx, cy, entry.x, entry.y, exit_.x, exit_.y))

            self._pairs = list(pairs)

            # Merge pairs within PAD_SIZE of each other → one candidate per group
            used = set()
            candidates = []
            for i, (cx, cy, *_) in enumerate(pairs):
                if i in used:
                    continue
                group = [i]
                used.add(i)
                for j in range(i + 1, len(pairs)):
                    if j in used:
                        continue
                    if math.hypot(pairs[j][0] - cx, pairs[j][1] - cy) < config.PAD_SIZE:
                        group.append(j)
                        used.add(j)
                gcx = sum(pairs[k][0] for k in group) / len(group)
                gcy = sum(pairs[k][1] for k in group) / len(group)
                candidates.append(PadCandidate(cx=gcx, cy=gcy,
                                               pair_count=len(group)))

            self._candidates = candidates

    def get_pairs(self) -> List[Tuple[float, float]]:
        with self._lock:
            return list(self._pairs)

    def get_candidates(self) -> List[PadCandidate]:
        with self._lock:
            return list(self._candidates)

    def reset(self):
        with self._lock:
            self._entries.clear()
            self._exits.clear()
            self._pairs.clear()
            self._candidates.clear()
            self._entry_seq = 0
            self._exit_seq = 0
            self._edge_seq = 0
            self._last_entry_pos = None
            self._last_exit_pos = None
            self._exits.clear()
            self._pairs.clear()
            self._candidates.clear()
