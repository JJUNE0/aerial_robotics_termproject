import heapq
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

import config
from mapping import FREE, UNKNOWN, OCCUPIED, INFLATED, OccupancyGrid


def _passable(grid: np.ndarray, row: int, col: int) -> bool:
    rows, cols = grid.shape
    return (0 <= row < rows and 0 <= col < cols and
            grid[row, col] not in (OCCUPIED, INFLATED))


def astar(grid: np.ndarray,
          start: Tuple[int, int],
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* path on an occupancy grid snapshot. Returns list of (row, col) or None."""
    sr, sc = start
    gr, gc = goal

    if not _passable(grid, gr, gc):
        return None

    def h(r: int, c: int) -> float:
        return math.hypot(r - gr, c - gc)

    open_heap: list = []
    heapq.heappush(open_heap, (h(sr, sc), 0.0, sr, sc))
    came_from: dict = {}
    g_score: dict = {(sr, sc): 0.0}

    while open_heap:
        _, g, r, c = heapq.heappop(open_heap)

        if (r, c) == (gr, gc):
            path = []
            node = (r, c)
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not _passable(grid, nr, nc):
                    continue
                step = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                ng = g + step
                if ng < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))

    return None


def simplify_path(path: List[Tuple[int, int]],
                  grid: np.ndarray) -> List[Tuple[int, int]]:
    """Line-of-sight pruning: remove intermediate waypoints when LOS exists."""
    if len(path) <= 2:
        return path

    def los(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0
        while (r, c) != (r1, c1):
            if not _passable(grid, r, c):
                return False
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return True

    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if los(path[i], path[j]):
                break
            j -= 1
        simplified.append(path[j])
        i = j
    return simplified


class FrontierNavigator:
    """Finds the farthest reachable FREE cell in the +x direction via BFS."""

    def __init__(self, occ_grid: OccupancyGrid):
        self._occ = occ_grid

    def find_max_x_target(self, drone_x: float, drone_y: float,
                          x_limit: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """Return world (x, y) of the farthest reachable cell in +x, or None."""
        grid = self._occ.snapshot()
        rows, cols = grid.shape
        sr, sc = self._occ.world_to_cell(drone_x, drone_y)

        if not (0 <= sr < rows and 0 <= sc < cols):
            return None

        # If we have an x limit, compute the column limit
        limit_col = cols - 1
        if x_limit is not None:
            lc = int((x_limit - self._occ.x_min) / self._occ.res)
            limit_col = min(cols - 1, lc)

        visited = np.zeros((rows, cols), dtype=bool)
        queue = deque([(sr, sc)])
        visited[sr, sc] = True
        best_col = sc
        best_row = sr

        while queue:
            r, c = queue.popleft()
            if c > best_col:
                best_col = c
                best_row = r
            elif c == best_col and abs(r - rows // 2) < abs(best_row - rows // 2):
                best_row = r

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nc > limit_col:
                        continue
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and
                            grid[nr, nc] == FREE):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        if best_col == sc:
            return None   # no progress possible

        return self._occ.cell_to_world(best_row, best_col)


class LawnmowerNavigator:
    """Generates ㄹ-pattern waypoints for the landing region scan."""

    def generate(self, x_start: float, x_end: float,
                 y_min: float, y_max: float) -> List[Tuple[float, float]]:
        waypoints: List[Tuple[float, float]] = []
        y = y_min
        forward = True
        while y <= y_max + 0.01:
            if forward:
                waypoints.append((x_start, y))
                waypoints.append((x_end, y))
            else:
                waypoints.append((x_end, y))
                waypoints.append((x_start, y))
            y += config.SCAN_ROW_SPACING
            forward = not forward
        return waypoints
