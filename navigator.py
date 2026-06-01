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


def _passable_los(grid: np.ndarray, row: int, col: int) -> bool:
    """Thick passability check used only for LOS pruning.

    Checks the cell and its 8 neighbours so that a straight-line shortcut
    is only accepted when the drone body — not just its centre — clears all
    inflated zones.  INFLATION_RADIUS already adds DRONE_HALF_DIAGONAL around
    each obstacle; one extra cell here guards against diagonal corner-cutting.
    """
    rows, cols = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] in (OCCUPIED, INFLATED):
                    return False
    return True


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
                # Block diagonal moves through tight corners: both cardinal
                # neighbours must also be free so the drone body doesn't clip.
                if dr != 0 and dc != 0:
                    if not _passable(grid, r + dr, c) or not _passable(grid, r, c + dc):
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
            if not _passable_los(grid, r, c):
                return False
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return _passable_los(grid, r1, c1)  # also verify endpoint

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
        found = False

        while queue:
            r, c = queue.popleft()

            # Only FREE cells count as valid destinations
            if grid[r, c] == FREE and c > sc:
                if not found or c > best_col or (
                        c == best_col and
                        abs(r - rows // 2) < abs(best_row - rows // 2)):
                    best_col = c
                    best_row = r
                    found = True

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nc > limit_col:
                        continue
                    # Traverse FREE and UNKNOWN — UNKNOWN gaps should not block
                    # exploration toward confirmed-FREE cells ahead
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and
                            grid[nr, nc] not in (OCCUPIED, INFLATED)):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        if not found:
            return None

        return self._occ.cell_to_world(best_row, best_col)


class LawnmowerNavigator:
    """Generates column-wise (Y-sweep, +X advance) ㄹ-pattern waypoints.

    Pattern: descend to y_min → advance +X → sweep +Y → advance +X → sweep -Y → …
    """

    def generate(self, x_start: float, x_end: float,
                 y_min: float, y_max: float) -> List[Tuple[float, float]]:
        waypoints: List[Tuple[float, float]] = []

        # First: drop to y_min at the region entry
        waypoints.append((x_start, y_min))

        x = x_start
        going_up = True   # first Y-sweep is +Y

        while x < x_end - 0.01:
            x_next = min(x + config.SCAN_ROW_SPACING, x_end)
            y_cur = y_min if going_up else y_max   # current edge (where we advance X)
            y_far = y_max if going_up else y_min   # far edge (sweep target)

            waypoints.append((x_next, y_cur))   # advance +X at current Y edge
            waypoints.append((x_next, y_far))   # sweep Y to far edge

            going_up = not going_up
            x = x_next

        return waypoints
