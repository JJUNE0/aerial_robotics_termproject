import csv
import os
import time
from datetime import datetime
from typing import Optional


class FlightLogger:
    """Logs flight sensor data to log/<YYYYMMDD_HHMMSS>.csv."""

    _FLUSH_EVERY = 10   # flush to disk every N rows (~1 s at 10 Hz)

    def __init__(self, log_dir: str = 'log'):
        os.makedirs(log_dir, exist_ok=True)
        fname = datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
        self._path = os.path.join(log_dir, fname)
        self._f = open(self._path, 'w', newline='')
        self._writer = csv.writer(self._f)
        self._writer.writerow([
            'time_s', 'state', 'x_m', 'y_m', 'z_down_m',
            'edge_event', 'edge_x_m', 'edge_y_m',
            'yaw_deg', 'roll_deg', 'pitch_deg', 'yaw_ref_deg',
            'vx_ms', 'vy_ms', 'vz_ms',
            'range_front_m', 'range_back_m', 'range_left_m', 'range_right_m', 'range_up_m',
            'target_x_m', 'target_y_m',
            'local_x_min_m', 'local_x_max_m', 'local_y_min_m', 'local_y_max_m',
        ])
        self._t0 = time.time()
        self._row_count = 0
        self._state = ''
        self._target = None
        self._local_scan_region = None
        print(f'[logger] {self._path}')

    @property
    def occ_path(self) -> str:
        return self._path.replace('.csv', '_occ.npz')

    def set_state(self, state: str):
        self._state = state

    def set_target(self, x, y):
        self._target = (x, y)

    def clear_target(self):
        self._target = None

    def set_local_scan_region(self, region):
        self._local_scan_region = region

    def clear_local_scan_region(self):
        self._local_scan_region = None

    def log(self, x: float, y: float, z_down: Optional[float],
            yaw: float = 0.0, roll: float = 0.0, pitch: float = 0.0,
            yaw_ref: float = 0.0,
            vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
            front: Optional[float] = None, back: Optional[float] = None,
            left: Optional[float] = None, right: Optional[float] = None,
            up: Optional[float] = None,
            edge_event: str = '', edge_x: Optional[float] = None,
            edge_y: Optional[float] = None):
        t = round(time.time() - self._t0, 3)
        def _f(v): return '' if v is None else round(v, 4)
        tx = '' if self._target is None else round(self._target[0], 4)
        ty = '' if self._target is None else round(self._target[1], 4)
        if self._local_scan_region is None:
            lx0 = lx1 = ly0 = ly1 = ''
        else:
            lx0, lx1, ly0, ly1 = (
                round(self._local_scan_region[0], 4),
                round(self._local_scan_region[1], 4),
                round(self._local_scan_region[2], 4),
                round(self._local_scan_region[3], 4),
            )
        self._writer.writerow([
            t, self._state, round(x, 4), round(y, 4), _f(z_down),
            edge_event, _f(edge_x), _f(edge_y),
            round(yaw, 2), round(roll, 2), round(pitch, 2), round(yaw_ref, 2),
            round(vx, 4), round(vy, 4), round(vz, 4),
            _f(front), _f(back), _f(left), _f(right), _f(up),
            tx, ty, lx0, lx1, ly0, ly1,
        ])
        self._row_count += 1
        if self._row_count % self._FLUSH_EVERY == 0:
            self._f.flush()

    def close(self):
        self._f.flush()
        self._f.close()
        print(f'[logger] saved {self._row_count} rows → {self._path}')
