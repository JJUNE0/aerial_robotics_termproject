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
        self._writer.writerow(['time_s', 'x_m', 'y_m', 'z_down_m'])
        self._t0 = time.time()
        self._row_count = 0
        print(f'[logger] {self._path}')

    def log(self, x: float, y: float, z_down: Optional[float]):
        t = round(time.time() - self._t0, 3)
        val = '' if z_down is None else round(z_down, 4)
        self._writer.writerow([t, round(x, 4), round(y, 4), val])
        self._row_count += 1
        if self._row_count % self._FLUSH_EVERY == 0:
            self._f.flush()

    def close(self):
        self._f.flush()
        self._f.close()
        print(f'[logger] saved {self._row_count} rows → {self._path}')
