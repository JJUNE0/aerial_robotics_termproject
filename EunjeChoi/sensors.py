import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Optional

from cflib.crazyflie.log import LogConfig

import config


@dataclass
class RangeData:
    front: Optional[float] = None
    back: Optional[float] = None
    left: Optional[float] = None
    right: Optional[float] = None
    up: Optional[float] = None
    down: Optional[float] = None


@dataclass
class SensorData:
    pose: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))
    ranges: RangeData = field(default_factory=RangeData)
    battery_pct: float = 0.0
    timestamp: float = 0.0


@dataclass
class EdgeEvent:
    kind: str          # 'entry' or 'exit'
    x: float
    y: float
    timestamp: float


class EdgeDetector:
    """Baseline-drop detector for the down-facing z-ranger.

    Maintains an EMA baseline of z_down (frozen while over a surface).
    Entry fires when the drop exceeds EDGE_THRESHOLD (5 cm).
    Exit fires after EDGE_EXIT_DEBOUNCE consecutive samples where the
    drop has returned below half the threshold.
    """

    def __init__(self):
        self._baseline: Optional[float] = None
        self._in_drop: bool = False
        self._exit_count: int = 0

    def reset(self):
        self._baseline = None
        self._in_drop = False
        self._exit_count = 0

    def update(self, z_down: float, drone_x: float, drone_y: float) -> Optional[EdgeEvent]:
        if z_down is None:
            return None

        if self._baseline is None:
            self._baseline = z_down
            return None

        drop = self._baseline - z_down
        event = None

        if not self._in_drop:
            if drop > config.EDGE_THRESHOLD:
                self._in_drop = True
                self._exit_count = 0
                event = EdgeEvent('entry', drone_x, drone_y, time.time())
        else:
            if drop < config.EDGE_THRESHOLD * 0.5:
                self._exit_count += 1
                if self._exit_count >= config.EDGE_EXIT_DEBOUNCE:
                    self._in_drop = False
                    self._exit_count = 0
                    event = EdgeEvent('exit', drone_x, drone_y, time.time())
            else:
                self._exit_count = 0

        if not self._in_drop:
            self._baseline = (config.EDGE_BASELINE_ALPHA * z_down +
                              (1.0 - config.EDGE_BASELINE_ALPHA) * self._baseline)

        return event


class SensorHub:
    """Manages all cflib log streams and produces SensorData + EdgeEvents."""

    def __init__(self, cf, logger=None):
        self._cf = cf
        self._logger = logger
        self._lock = threading.Lock()

        self._pose = (0.0, 0.0, 0.0, 0.0)
        self._ranges = RangeData()
        self._battery_pct = 0.0
        self._timestamp = 0.0

        self.edge_queue: Queue = Queue()
        self._edge_detector = EdgeDetector()

        self._range_cfg = self._build_range_config()
        self._pose_cfg = self._build_pose_config()
        self._battery_cfg = self._build_battery_config()

    # ---------------------------------------------------------------- builders

    def _build_range_config(self) -> LogConfig:
        cfg = LogConfig('ranges', period_in_ms=50)
        cfg.add_variable('range.front', 'uint16_t')
        cfg.add_variable('range.back', 'uint16_t')
        cfg.add_variable('range.left', 'uint16_t')
        cfg.add_variable('range.right', 'uint16_t')
        cfg.add_variable('range.up', 'uint16_t')
        cfg.add_variable('range.zrange', 'uint16_t')
        cfg.data_received_cb.add_callback(self._on_ranges)
        return cfg

    def _build_pose_config(self) -> LogConfig:
        cfg = LogConfig('pose', period_in_ms=50)
        cfg.add_variable('stateEstimate.x', 'float')
        cfg.add_variable('stateEstimate.y', 'float')
        cfg.add_variable('stateEstimate.z', 'float')
        cfg.add_variable('stateEstimate.yaw', 'float')
        cfg.data_received_cb.add_callback(self._on_pose)
        return cfg

    def _build_battery_config(self) -> LogConfig:
        cfg = LogConfig('battery', period_in_ms=1000)
        cfg.add_variable('pm.vbat', 'float')
        cfg.data_received_cb.add_callback(self._on_battery)
        return cfg

    # ------------------------------------------------------------ callbacks

    @staticmethod
    def _mm_to_m(val: int) -> Optional[float]:
        if val >= 8000:
            return None
        return val / 1000.0

    def _on_ranges(self, timestamp, data, logconf):
        r = RangeData(
            front=self._mm_to_m(data['range.front']),
            back=self._mm_to_m(data['range.back']),
            left=self._mm_to_m(data['range.left']),
            right=self._mm_to_m(data['range.right']),
            up=self._mm_to_m(data['range.up']),
            down=self._mm_to_m(data['range.zrange']),
        )
        with self._lock:
            self._ranges = r
            self._timestamp = time.time()
            pose = self._pose

        if self._logger is not None:
            self._logger.log(pose[0], pose[1], r.down)

        event = self._edge_detector.update(r.down, pose[0], pose[1])
        if event is not None:
            self.edge_queue.put(event)

    def _on_pose(self, timestamp, data, logconf):
        with self._lock:
            self._pose = (
                data['stateEstimate.x'],
                data['stateEstimate.y'],
                data['stateEstimate.z'],
                data['stateEstimate.yaw'],
            )

    def _on_battery(self, timestamp, data, logconf):
        v = data['pm.vbat']
        pct = (v - config.BATTERY_V_MIN) / (config.BATTERY_V_MAX - config.BATTERY_V_MIN) * 100.0
        pct = max(0.0, min(100.0, pct))
        with self._lock:
            self._battery_pct = pct

    # --------------------------------------------------------- lifecycle

    def reset_edge_detector(self):
        self._edge_detector.reset()
        # Drain any stale events accumulated before the scan
        while not self.edge_queue.empty():
            try:
                self.edge_queue.get_nowait()
            except Exception:
                break

    def start(self):
        self._cf.log.add_config(self._range_cfg)
        self._cf.log.add_config(self._pose_cfg)
        self._cf.log.add_config(self._battery_cfg)
        self._range_cfg.start()
        self._pose_cfg.start()
        self._battery_cfg.start()

    def stop(self):
        try:
            self._range_cfg.delete()
        except Exception:
            pass
        try:
            self._pose_cfg.delete()
        except Exception:
            pass
        try:
            self._battery_cfg.delete()
        except Exception:
            pass

    # ----------------------------------------------------------- read

    def read(self) -> SensorData:
        with self._lock:
            r = self._ranges
            return SensorData(
                pose=self._pose,
                ranges=RangeData(
                    front=r.front, back=r.back,
                    left=r.left,  right=r.right,
                    up=r.up,      down=r.down,
                ),
                battery_pct=self._battery_pct,
                timestamp=self._timestamp,
            )
