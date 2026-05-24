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
    """Detects entry/exit edges in the z-ranger signal via EMA baseline."""

    def __init__(self):
        self._baseline: Optional[float] = None
        self._in_drop: bool = False

    def reset(self):
        self._baseline = None
        self._in_drop = False

    def update(self, z_down: float, drone_x: float, drone_y: float) -> Optional[EdgeEvent]:
        if z_down is None:
            return None

        if self._baseline is None:
            self._baseline = z_down
            return None

        self._baseline = (config.EDGE_BASELINE_ALPHA * z_down +
                          (1.0 - config.EDGE_BASELINE_ALPHA) * self._baseline)

        drop = self._baseline - z_down
        event = None

        if not self._in_drop and drop > config.EDGE_THRESHOLD:
            self._in_drop = True
            event = EdgeEvent('entry', drone_x, drone_y, time.time())
        elif self._in_drop and drop < config.EDGE_THRESHOLD * 0.4:
            self._in_drop = False
            event = EdgeEvent('exit', drone_x, drone_y, time.time())

        return event


class SensorHub:
    """Manages all cflib log streams and produces SensorData + EdgeEvents."""

    def __init__(self, cf):
        self._cf = cf
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
        cfg = LogConfig('ranges', period_in_ms=100)
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
