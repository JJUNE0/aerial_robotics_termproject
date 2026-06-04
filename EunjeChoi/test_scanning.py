"""
Scanning-only test: takeoff -> 180 deg rotation scan -> land.

After landing, save the raw occupancy map and a morphology-closed copy so the
two maps can be compared without changing the live planner map.
"""

import threading
import time

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

import config
import controller
from gui import MissionGUI
from logger import FlightLogger
from mapping import FREE, UNKNOWN, OCCUPIED, INFLATED, OccupancyGrid, HeightMap
from mission import EmergencyException, do_takeoff, do_rotation_scan
from sensors import SensorHub
from shared_state import SharedState


CLOSING_RADIUS_CELLS = 2
CLOSING_RADIUS_M = CLOSING_RADIUS_CELLS * config.OCCUPANCY_GRID_RES


def _disk_offsets(radius_cells: int):
    offsets = []
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc <= radius_cells * radius_cells:
                offsets.append((dr, dc))
    return offsets


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    rows, cols = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    src_r0 = max(0, -dr)
    src_r1 = rows - max(0, dr)
    src_c0 = max(0, -dc)
    src_c1 = cols - max(0, dc)

    dst_r0 = max(0, dr)
    dst_r1 = rows - max(0, -dr)
    dst_c0 = max(0, dc)
    dst_c1 = cols - max(0, -dc)

    if src_r0 < src_r1 and src_c0 < src_c1:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _binary_dilate(mask: np.ndarray, offsets) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dr, dc in offsets:
        out |= _shift_mask(mask, dr, dc)
    return out


def _binary_erode(mask: np.ndarray, offsets) -> np.ndarray:
    out = np.ones_like(mask, dtype=bool)
    for dr, dc in offsets:
        out &= _shift_mask(mask, dr, dc)
    return out


def morphologically_close_occupancy(grid: np.ndarray,
                                    radius_m: float = CLOSING_RADIUS_M):
    """Return a planner-style copy with small blocked-region concavities closed."""
    radius_cells = max(1, int(round(radius_m / config.OCCUPANCY_GRID_RES)))
    offsets = _disk_offsets(radius_cells)

    blocked = (grid == OCCUPIED) | (grid == INFLATED)
    closed_blocked = _binary_erode(_binary_dilate(blocked, offsets), offsets)
    closed_blocked |= blocked

    closed = grid.copy()
    closed[closed_blocked & (closed != OCCUPIED)] = INFLATED
    return closed, closed_blocked, radius_cells


def _cell_counts(grid: np.ndarray):
    labels = {
        FREE: 'free',
        UNKNOWN: 'unknown',
        OCCUPIED: 'occupied',
        INFLATED: 'inflated',
    }
    values, counts = np.unique(grid, return_counts=True)
    found = {labels.get(int(v), str(int(v))): int(c)
             for v, c in zip(values, counts)}
    for name in labels.values():
        found.setdefault(name, 0)
    return found


def _print_map_comparison(raw: np.ndarray, closed: np.ndarray, radius_cells: int):
    res = config.OCCUPANCY_GRID_RES
    changed = raw != closed
    new_inflated = (closed == INFLATED) & (raw != INFLATED)

    print('[scan-test] occupancy resolution unchanged:')
    print(f'  res={res:.3f} m/cell  shape={raw.shape[1]}x{raw.shape[0]} cells')
    print(f'  closing radius={radius_cells} cells ({radius_cells * res:.3f} m)')
    print(f'  raw counts={_cell_counts(raw)}')
    print(f'  closed counts={_cell_counts(closed)}')
    print(f'  changed cells={int(changed.sum())}'
          f'  area={changed.sum() * res * res:.4f} m^2')
    print(f'  newly inflated cells={int(new_inflated.sum())}'
          f'  area={new_inflated.sum() * res * res:.4f} m^2')


def _save_maps(logger: FlightLogger, occ: OccupancyGrid):
    raw = occ.snapshot()
    closed, closed_blocked, radius_cells = morphologically_close_occupancy(raw)
    changed = raw != closed

    raw_path = logger.occ_path
    closed_path = raw_path.replace('_occ.npz', '_occ_closed.npz')
    diff_path = raw_path.replace('_occ.npz', '_occ_closing_diff.npz')
    png_path = raw_path.replace('_occ.npz', '_occ_compare.png')

    common = {
        'res': np.array([config.OCCUPANCY_GRID_RES]),
        'x_min': np.array([occ.x_min]),
        'y_min': np.array([occ.y_min]),
        'closing_radius_m': np.array([CLOSING_RADIUS_M]),
        'closing_radius_cells': np.array([radius_cells]),
    }
    np.savez(raw_path, grid=raw, **common)
    np.savez(closed_path, grid=closed, closed_blocked=closed_blocked, **common)
    np.savez(diff_path, changed=changed, raw=raw, closed=closed, **common)

    _print_map_comparison(raw, closed, radius_cells)
    print(f'[scan-test] raw occ saved -> {raw_path}')
    print(f'[scan-test] closed occ saved -> {closed_path}')
    print(f'[scan-test] closing diff saved -> {diff_path}')

    try:
        _save_compare_png(raw, closed, changed, png_path)
        print(f'[scan-test] comparison png saved -> {png_path}')
    except Exception as exc:
        print(f'[scan-test] comparison png skipped: {exc}')


def _rgb_image(grid: np.ndarray):
    rgb = np.zeros((*grid.shape, 3), dtype=float)
    colors = {
        FREE: np.array([1.0, 1.0, 1.0]),
        UNKNOWN: np.array([0.7, 0.7, 0.7]),
        OCCUPIED: np.array([0.1, 0.1, 0.1]),
        INFLATED: np.array([0.45, 0.45, 0.45]),
    }
    for value, color in colors.items():
        rgb[grid == value] = color
    return rgb


def _save_compare_png(raw: np.ndarray, closed: np.ndarray,
                      changed: np.ndarray, path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    diff_img = np.zeros((*changed.shape, 3), dtype=float)
    diff_img[changed] = np.array([1.0, 0.1, 0.1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, title, img in [
            (axes[0], 'Raw occupancy', _rgb_image(raw)),
            (axes[1], 'Morphological closing', _rgb_image(closed)),
            (axes[2], 'Changed cells', diff_img)]:
        ax.imshow(img, origin='lower', interpolation='nearest')
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_scan_test(cf, shared: SharedState):
    occ = OccupancyGrid()
    hmap = HeightMap()
    logger = FlightLogger()
    hub = SensorHub(cf, logger=logger)
    hub.start()

    try:
        do_takeoff(cf, shared, hub, occ, hmap)
        do_rotation_scan(cf, shared, hub, occ, hmap, angle_deg=180.0)

        shared.current_state = 'LANDING_ON_START'
        controller.land_vel(cf, hub)
        shared.current_state = 'DONE'

    except EmergencyException:
        shared.current_state = 'EMERGENCY_LAND'
        try:
            controller.land_vel(cf, hub)
        except Exception:
            pass

    except Exception as exc:
        shared.current_state = 'ERROR'
        print(f'[scan-test] unhandled exception: {exc}')
        try:
            controller.land_vel(cf, hub)
        except Exception:
            pass

    finally:
        try:
            _save_maps(logger, occ)
        except Exception as exc:
            print(f'[scan-test] map save failed: {exc}')
        hub.stop()
        logger.close()
        controller.disarm(cf)


def main():
    cflib.crtp.init_drivers()
    shared = SharedState()

    for attempt in range(1, 6):
        try:
            print(f'[main] connecting to {config.RADIO_URI}  (attempt {attempt}/5)')
            with SyncCrazyflie(config.RADIO_URI,
                               cf=Crazyflie(rw_cache='./cache')) as scf:
                t = threading.Thread(
                    target=run_scan_test, args=(scf.cf, shared), daemon=True)
                t.start()
                MissionGUI(shared).start()
            break
        except Exception as exc:
            print(f'[main] connection error: {exc}')
            if attempt < 5:
                time.sleep(3.0)
            else:
                raise


if __name__ == '__main__':
    main()
