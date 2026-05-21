"""Log-odds occupancy grid with Bresenham ray casting."""
import math
import threading

import numpy as np

from utils.config import (
    GOAL_SEARCH_RADIUS,
    LANDING_CLEAR_RADIUS,
    LANDING_OCCUPIED_LIMIT,
    LOG_ODDS_FREE,
    LOG_ODDS_MAX,
    LOG_ODDS_MIN,
    LOG_ODDS_OCC,
    MAP_RESOLUTION,
    RAY_MAX_RANGE,
)
from utils.helpers import clamp, gaussian_weight


class OccupancyGrid:
    """
    Log-odds occupancy grid.

    World coordinate system:
      x : 0 .. width_m
      y : 0 .. height_m
      origin is map corner (0, 0). y-axis points "up" in plotting.

    Cell indexing:
      col = floor(x / res)
      row = floor(y / res)
    """

    def __init__(self, width_m, height_m, resolution_m=MAP_RESOLUTION):
        self.width_m = width_m
        self.height_m = height_m
        self.res = resolution_m
        self.cols = max(1, int(round(width_m / resolution_m)))
        self.rows = max(1, int(round(height_m / resolution_m)))
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.lock = threading.Lock()

    def world_to_cell(self, x, y):
        col = int(x / self.res)
        row = int(y / self.res)
        return row, col

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    @staticmethod
    def _bresenham(r0, c0, r1, c1):
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dc - dr
        r, c = r0, c0
        # safety cap to avoid runaway loops
        max_cells = (dr + dc) * 2 + 4
        while max_cells > 0:
            cells.append((r, c))
            if r == r1 and c == c1:
                break
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr
            max_cells -= 1
        return cells

    def _update_ray(self, x0, y0, x1, y1, hit_endpoint):
        r0, c0 = self.world_to_cell(x0, y0)
        r1, c1 = self.world_to_cell(x1, y1)
        cells = self._bresenham(r0, c0, r1, c1)
        if not cells:
            return
        with self.lock:
            for i, (r, c) in enumerate(cells):
                if not self.in_bounds(r, c):
                    continue
                last = (i == len(cells) - 1)
                if last and hit_endpoint:
                    self.log_odds[r, c] = min(
                        LOG_ODDS_MAX, self.log_odds[r, c] + LOG_ODDS_OCC
                    )
                else:
                    self.log_odds[r, c] = max(
                        LOG_ODDS_MIN, self.log_odds[r, c] + LOG_ODDS_FREE
                    )

    # body-frame ray offsets (CCW from front)
    _RAY_ANGLES = {
        "front": 0.0,
        "left":  math.pi / 2,
        "back":  math.pi,
        "right": -math.pi / 2,
    }

    def update_from_ranges(self, pose_x, pose_y, pose_yaw_rad, ranges_dict):
        """
        Multi-ranger F/L/R/B 빔을 ray-cast해 log-odds 업데이트.
        pose_x, pose_y: world frame meters.
        pose_yaw_rad: yaw in radians (CCW positive).
        ranges_dict: {"front": m, "left": m, "right": m, "back": m}, None=OUT.
        """
        for name, dist in ranges_dict.items():
            if name not in self._RAY_ANGLES:
                continue
            angle = pose_yaw_rad + self._RAY_ANGLES[name]
            if dist is None or dist >= RAY_MAX_RANGE:
                end_x = pose_x + RAY_MAX_RANGE * math.cos(angle)
                end_y = pose_y + RAY_MAX_RANGE * math.sin(angle)
                self._update_ray(pose_x, pose_y, end_x, end_y, hit_endpoint=False)
            else:
                end_x = pose_x + dist * math.cos(angle)
                end_y = pose_y + dist * math.sin(angle)
                self._update_ray(pose_x, pose_y, end_x, end_y, hit_endpoint=True)

    def snapshot(self):
        with self.lock:
            return self.log_odds.copy()

    def cell_center(self, row, col):
        return (col + 0.5) * self.res, (row + 0.5) * self.res

    def is_landing_clear(self, x, y, clear_radius=LANDING_CLEAR_RADIUS):
        """Return True when the local OGM patch has no likely occupied cells."""
        row, col = self.world_to_cell(x, y)
        radius_cells = max(1, int(math.ceil(clear_radius / self.res)))
        with self.lock:
            for rr in range(row - radius_cells, row + radius_cells + 1):
                for cc in range(col - radius_cells, col + radius_cells + 1):
                    if not self.in_bounds(rr, cc):
                        return False
                    cx, cy = self.cell_center(rr, cc)
                    if math.hypot(cx - x, cy - y) > clear_radius:
                        continue
                    if self.log_odds[rr, cc] >= LANDING_OCCUPIED_LIMIT:
                        return False
        return True

    def best_landing_target(self, goal_xy):
        """
        Pick a landing cell inside the Gaussian goal region.

        The score favors cells near the goal mean and cells observed as free.
        Occupied cells, map edges, and cells without enough clearance are rejected.
        """
        best = None
        best_score = -float("inf")
        search_cells = max(1, int(math.ceil(GOAL_SEARCH_RADIUS / self.res)))
        grow, gcol = self.world_to_cell(*goal_xy)
        with self.lock:
            log_odds = self.log_odds.copy()

        for row in range(grow - search_cells, grow + search_cells + 1):
            for col in range(gcol - search_cells, gcol + search_cells + 1):
                if not self.in_bounds(row, col):
                    continue
                x, y = self.cell_center(row, col)
                dist = math.hypot(x - goal_xy[0], y - goal_xy[1])
                if dist > GOAL_SEARCH_RADIUS:
                    continue
                lo = float(log_odds[row, col])
                if lo >= LANDING_OCCUPIED_LIMIT:
                    continue
                if not self.is_landing_clear(x, y):
                    continue

                goal_score = gaussian_weight(x, y, goal_xy)
                free_bonus = clamp(-lo / abs(LOG_ODDS_MIN), 0.0, 1.0)
                unknown_penalty = 0.10 if abs(lo) < 0.05 else 0.0
                score = goal_score + 0.35 * free_bonus - unknown_penalty
                if score > best_score:
                    best_score = score
                    best = (x, y)

        return best if best is not None else tuple(goal_xy)
