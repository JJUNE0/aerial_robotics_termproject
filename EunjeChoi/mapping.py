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


class HeightMap:
    """Detects landing pad from z-ranger edge events.

    Strategy: pair entry/exit events on the same scan column (X proximity),
    then group pairs spatially. A pad (30 cm) generates 2+ pairs with an
    X-span >= PAD_MIN_CLUSTER_SPAN; a narrow bar (13 cm) generates at most 1.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: List[Tuple[float, float]] = []   # (x, y)
        self._exits: List[Tuple[float, float]] = []
        self._pairs: List[Tuple[float, float]] = []     # matched pair centres
        self._candidates: List[PadCandidate] = []

    def add_entry(self, x: float, y: float):
        with self._lock:
            self._entries.append((x, y))

    def add_exit(self, x: float, y: float):
        with self._lock:
            self._exits.append((x, y))
        self._recompute_candidates()

    def _recompute_candidates(self):
        with self._lock:
            # Step 1: pair each exit with the nearest unpaired entry on the
            # same scan column (close X, any Y — Y-sweep pattern)
            # pairs: (cx, cy, entry_x, entry_y, exit_x, exit_y)
            pairs = []
            used_entries = set()

            for exx, exy in self._exits:
                best_j = None
                best_d = config.PAIR_SAME_COL_TOL   # max X distance to count as same column
                for en_j, (enx, eny) in enumerate(self._entries):
                    if en_j in used_entries:
                        continue
                    d = abs(enx - exx)
                    if d < best_d:
                        best_d = d
                        best_j = en_j

                if best_j is None:
                    continue

                enx, eny = self._entries[best_j]
                # Reject if Y span is too small — not a full pad crossing
                if abs(eny - exy) < config.PAIR_MIN_Y_SPAN:
                    continue

                used_entries.add(best_j)
                pairs.append(((enx + exx) / 2.0, (eny + exy) / 2.0,
                               enx, eny, exx, exy))

            self._pairs = list(pairs)

            # Step 2: group pairs whose centres are within PAD_SIZE of each other
            used = set()
            groups: List[List[int]] = []
            for i in range(len(pairs)):
                if i in used:
                    continue
                group = [i]
                used.add(i)
                for j in range(i + 1, len(pairs)):
                    if j in used:
                        continue
                    if math.hypot(pairs[i][0] - pairs[j][0],
                                  pairs[i][1] - pairs[j][1]) < config.PAD_SIZE:
                        group.append(j)
                        used.add(j)
                groups.append(group)

            # Step 3: any group with ≥1 pair becomes a candidate.
            # A single pair (wider row spacing) is sufficient to estimate pad centre.
            candidates = []
            for group in groups:
                cx = sum(pairs[i][0] for i in group) / len(group)
                cy = sum(pairs[i][1] for i in group) / len(group)
                candidates.append(PadCandidate(cx=cx, cy=cy))

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
