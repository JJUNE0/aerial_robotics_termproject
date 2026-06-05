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
    """A* on occupancy grid — 8-directional, clearance-weighted cost.

    Diagonal moves (cost √2) reduce staircase patterns so simplify_path
    collapses them into far fewer waypoints.  Corner-cutting through
    obstacles is prevented: both cardinal neighbours of a diagonal step
    must be passable.  Clearance cost biases paths toward open corridors.
    """
    sr, sc = start
    gr, gc = goal
    rows, cols = grid.shape

    if not _passable(grid, gr, gc):
        return None

    def h(r: int, c: int) -> float:
        return abs(r - gr) + abs(c - gc)  # Manhattan

    def clearance_penalty(r: int, c: int) -> float:
        pen = 0.0
        rad = config.A_STAR_CLEARANCE_RADIUS
        w   = config.A_STAR_CLEARANCE_WEIGHT
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                        grid[nr, nc] in (OCCUPIED, INFLATED)):
                    pen += w / math.sqrt(dr * dr + dc * dc)
        return pen

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

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not _passable(grid, nr, nc):
                continue
            ng = g + 1.0 + clearance_penalty(nr, nc)
            if ng < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))

    return None


def _segment_clear(grid: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Check every cell on the horizontal or vertical line from a to b."""
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            if not _passable(grid, r0, c):
                return False
    else:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            if not _passable(grid, r, c1):
                return False
    return True


def _l_via(grid: np.ndarray,
           a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Return a via-point for an L-shaped path from a to b, or None if impossible.

    Tries H-then-V and V-then-H.  Returns the via-point, or b itself if a
    direct (same-row/col) segment is clear.
    """
    ra, ca = a
    rb, cb = b
    if ra == rb or ca == cb:
        return b if _segment_clear(grid, a, b) else None
    via_hv = (ra, cb)   # go horizontal first
    if _segment_clear(grid, a, via_hv) and _segment_clear(grid, via_hv, b):
        return via_hv
    via_vh = (rb, ca)   # go vertical first
    if _segment_clear(grid, a, via_vh) and _segment_clear(grid, via_vh, b):
        return via_vh
    return None


def simplify_path(path: List[Tuple[int, int]],
                  grid: np.ndarray) -> List[Tuple[int, int]]:
    """Collinear merge + greedy L-shape skip.

    After collapsing collinear steps, greedily skip from the current
    anchor to the farthest waypoint reachable via a single L-shaped
    path (H-then-V or V-then-H).  This turns many small staircase
    steps into one large L-shaped move whenever the corridor is free.
    """
    if len(path) <= 2:
        return path

    # Step 1: collinear merge
    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        d2 = (path[i+1][0] - path[i][0],  path[i+1][1] - path[i][1])
        if d1 != d2:
            simplified.append(path[i])
    simplified.append(path[-1])
    path = simplified

    # Step 2: greedy L-shape skip
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Search backwards for the farthest reachable waypoint via L-shape
        j = len(path) - 1
        while j > i + 1:
            if _l_via(grid, path[i], path[j]) is not None:
                break
            j -= 1
        via = _l_via(grid, path[i], path[j])
        if via is not None and via != path[j]:
            result.append(via)
        result.append(path[j])
        i = j

    return result


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
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and
                            grid[nr, nc] == FREE):
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        if not found:
            return None

        return self._occ.cell_to_world(best_row, best_col)

    def find_min_x_target(self, drone_x: float, drone_y: float,
                          x_limit: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """Return world (x, y) of the farthest reachable cell in -x, or None."""
        grid = self._occ.snapshot()
        rows, cols = grid.shape
        sr, sc = self._occ.world_to_cell(drone_x, drone_y)

        if not (0 <= sr < rows and 0 <= sc < cols):
            return None

        limit_col = 0
        if x_limit is not None:
            lc = int((x_limit - self._occ.x_min) / self._occ.res)
            limit_col = max(0, lc)

        visited = np.zeros((rows, cols), dtype=bool)
        queue = deque([(sr, sc)])
        visited[sr, sc] = True
        best_col = sc
        best_row = sr
        found = False

        while queue:
            r, c = queue.popleft()

            if grid[r, c] == FREE and c < sc:
                if not found or c < best_col or (
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
                    if nc < limit_col:
                        continue
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not visited[nr, nc] and
                            grid[nr, nc] == FREE):
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
