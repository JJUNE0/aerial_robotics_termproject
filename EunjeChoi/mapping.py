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

        # Pre-mark arena boundary band as INFLATED so frontier never
        # selects targets near the edge.
        margin_cells = max(1, int(config.ARENA_NAV_MARGIN / res))
        ax0 = max(0, int((config.ekf_arena_x_min() - self._x_min) / res))
        ax1 = min(cols, int((config.ekf_arena_x_max() - self._x_min) / res) + 1)
        ay0 = max(0, int((config.ekf_arena_y_min() - self._y_min) / res))
        ay1 = min(rows, int((config.ekf_arena_y_max() - self._y_min) / res) + 1)
        # bottom / top Y bands
        self._grid[ay0:ay0 + margin_cells, ax0:ax1] = INFLATED
        self._grid[ay1 - margin_cells:ay1, ax0:ax1] = INFLATED
        # left / right X bands
        self._grid[ay0:ay1, ax0:ax0 + margin_cells] = INFLATED
        self._grid[ay0:ay1, ax1 - margin_cells:ax1] = INFLATED

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


def find_pad_from_diff(occ_high: 'OccupancyGrid',
                       occ_low: 'OccupancyGrid'
                       ) -> Tuple[Optional[Tuple[float, float]], np.ndarray]:
    """Compare two occupancy maps at different altitudes to locate the landing pad.

    Cells OCCUPIED in occ_low but FREE in occ_high are elevated objects
    (height between LOW_SCAN_Z and FLIGHT_Z) — bars or the landing pad.
    BFS connected components → bounding box aspect ratio → pick the square one.
    Returns (EKF (x, y) or None, diff boolean array).
    """
    high = occ_high.snapshot()
    low  = occ_low.snapshot()

    diff = (low == OCCUPIED) & (high != OCCUPIED)
    if not diff.any():
        return None, diff.astype(np.int8)

    rows, cols = diff.shape
    visited = np.zeros_like(diff, dtype=bool)
    best = None

    for sr in range(rows):
        for sc in range(cols):
            if not diff[sr, sc] or visited[sr, sc]:
                continue
            component = []
            queue = [(sr, sc)]
            visited[sr, sc] = True
            while queue:
                r, c = queue.pop()
                component.append((r, c))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and diff[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rs = [p[0] for p in component]
            cs = [p[1] for p in component]
            h_m = (max(rs) - min(rs)) * occ_low.res
            w_m = (max(cs) - min(cs)) * occ_low.res
            if h_m < 1e-3 or w_m < 1e-3:
                continue
            short, long_ = min(h_m, w_m), max(h_m, w_m)
            if (long_ / short < config.PAD_ASPECT_MAX
                    and config.PAD_BBOX_MIN < short
                    and long_ < config.PAD_BBOX_MAX):
                cx_row = (max(rs) + min(rs)) / 2
                cx_col = (max(cs) + min(cs)) / 2
                wx, wy = occ_low.cell_to_world(int(cx_row), int(cx_col))
                best = (wx, wy)
                break

    return best, diff.astype(np.int8)


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
        self._entries: List[Tuple[float, float]] = []
        self._exits: List[Tuple[float, float]] = []
        self._pairs: List[Tuple[float, float]] = []
        self._candidates: List[PadCandidate] = []
        self._entry_seq: int = 0
        self._exit_seq: int = 0
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

    def add_entry(self, x: float, y: float):
        with self._lock:
            self._entries.append((x, y))
            self._entry_seq += 1
            self._last_entry_pos = (x, y)

    def add_exit(self, x: float, y: float):
        with self._lock:
            self._exits.append((x, y))
            self._exit_seq += 1
            self._last_exit_pos = (x, y)
        self._recompute_candidates()

    def _recompute_candidates(self):
        with self._lock:
            # Pair each exit with the most recent unpaired entry (time-based).
            # Valid pair: Euclidean distance ≥ PAIR_MIN_Y_SPAN (20 cm).
            pairs = []
            used_entries = set()

            for exx, exy in self._exits:
                # Find most recent unpaired entry
                best_j = None
                for j in range(len(self._entries) - 1, -1, -1):
                    if j not in used_entries:
                        best_j = j
                        break

                if best_j is None:
                    continue

                enx, eny = self._entries[best_j]
                if math.hypot(enx - exx, eny - exy) < config.PAIR_MIN_Y_SPAN:
                    continue   # too close — not a valid pad crossing

                used_entries.add(best_j)
                cx = (enx + exx) / 2.0
                cy = (eny + exy) / 2.0
                pairs.append((cx, cy, enx, eny, exx, exy))

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
                candidates.append(PadCandidate(cx=gcx, cy=gcy))

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
            self._last_entry_pos = None
            self._last_exit_pos = None
            self._exits.clear()
            self._pairs.clear()
            self._candidates.clear()
