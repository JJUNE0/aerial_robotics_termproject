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
    velocity: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))   # vx, vy, vz (m/s)
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
    """Direct peak/valley detector using a fixed cruise-altitude baseline.

    ENTRY fires at the valley when z_down dips ≥ EDGE_ENTRY_DIP below baseline.
    EXIT  fires at the peak  when z_down rises ≥ EDGE_EXIT_RISE above baseline.
    """

    def __init__(self):
        self._prev_z: Optional[float] = None
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._last_dir: int = 0   # +1 rising / -1 falling

    def reset(self):
        self._prev_z = None
        self._last_dir = 0

    def update(self, z_down: float, drone_x: float, drone_y: float) -> Optional[EdgeEvent]:
        if z_down is None:
            return None

        if self._prev_z is None:
            self._prev_z = z_down
            self._prev_x = drone_x
            self._prev_y = drone_y
            return None

        dz = z_down - self._prev_z
        event = None

        if abs(dz) > config.EDGE_MIN_DZ:
            cur_dir = 1 if dz > 0 else -1

            if self._last_dir < 0 and cur_dir > 0:
                # VALLEY at previous sample — fire ENTRY if deep enough
                if self._prev_z <= config.EDGE_BASELINE - config.EDGE_ENTRY_DIP:
                    event = EdgeEvent('entry', self._prev_x, self._prev_y, time.time())

            elif self._last_dir > 0 and cur_dir < 0:
                # PEAK at previous sample — fire EXIT if high enough
                if self._prev_z >= config.EDGE_BASELINE + config.EDGE_EXIT_RISE:
                    event = EdgeEvent('exit', self._prev_x, self._prev_y, time.time())

            self._last_dir = cur_dir

        self._prev_z = z_down
        self._prev_x = drone_x
        self._prev_y = drone_y

        return event


class SensorHub:
    """Manages all cflib log streams and produces SensorData + EdgeEvents."""

    def __init__(self, cf, logger=None):
        self._cf = cf
        self._logger = logger
        self._lock = threading.Lock()

        self._pose = (0.0, 0.0, 0.0, 0.0)   # x, y, z, yaw_deg
        self._attitude = (0.0, 0.0)           # roll_deg, pitch_deg
        self._yaw_ref = 0.0                   # ctrltarget.yaw (deg)
        self._velocity = (0.0, 0.0, 0.0)
        self._ranges = RangeData()
        self._battery_pct = 0.0
        self._timestamp = 0.0

        self.edge_queue: Queue = Queue()
        self._edge_detector = EdgeDetector()

        self._range_cfg = self._build_range_config()
        self._pose_cfg = self._build_pose_config()
        self._velocity_cfg = self._build_velocity_config()
        self._setpoint_cfg = self._build_setpoint_config()
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
        cfg.add_variable('stabilizer.roll', 'float')
        cfg.add_variable('stabilizer.pitch', 'float')
        cfg.data_received_cb.add_callback(self._on_pose)
        return cfg

    def _build_velocity_config(self) -> LogConfig:
        cfg = LogConfig('velocity', period_in_ms=50)
        cfg.add_variable('stateEstimate.vx', 'float')
        cfg.add_variable('stateEstimate.vy', 'float')
        cfg.add_variable('stateEstimate.vz', 'float')
        cfg.data_received_cb.add_callback(self._on_velocity)
        return cfg

    def _build_setpoint_config(self) -> LogConfig:
        cfg = LogConfig('setpoint', period_in_ms=50)
        cfg.add_variable('ctrltarget.yaw', 'float')
        cfg.data_received_cb.add_callback(self._on_setpoint)
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
            with self._lock:
                vel = self._velocity
                att = self._attitude
                yaw_ref = self._yaw_ref
            self._logger.log(
                pose[0], pose[1], r.down,
                pose[3], att[0], att[1], yaw_ref,
                vel[0], vel[1], vel[2],
                r.front, r.back, r.left, r.right, r.up,
            )

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
            self._attitude = (
                data['stabilizer.roll'],
                data['stabilizer.pitch'],
            )

    def _on_setpoint(self, timestamp, data, logconf):
        with self._lock:
            self._yaw_ref = data['ctrltarget.yaw']

    def _on_velocity(self, timestamp, data, logconf):
        with self._lock:
            self._velocity = (
                data['stateEstimate.vx'],
                data['stateEstimate.vy'],
                data['stateEstimate.vz'],
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
        self._cf.log.add_config(self._velocity_cfg)
        self._cf.log.add_config(self._setpoint_cfg)
        self._cf.log.add_config(self._battery_cfg)
        self._range_cfg.start()
        self._pose_cfg.start()
        self._velocity_cfg.start()
        self._setpoint_cfg.start()
        self._battery_cfg.start()

    def stop(self):
        for cfg in (self._range_cfg, self._pose_cfg,
                    self._velocity_cfg, self._setpoint_cfg,
                    self._battery_cfg):
            try:
                cfg.delete()
            except Exception:
                pass

    # ----------------------------------------------------------- read

    def read(self) -> SensorData:
        with self._lock:
            r = self._ranges
            return SensorData(
                pose=self._pose,
                velocity=self._velocity,
                ranges=RangeData(
                    front=r.front, back=r.back,
                    left=r.left,  right=r.right,
                    up=r.up,      down=r.down,
                ),
                battery_pct=self._battery_pct,
                timestamp=self._timestamp,
            )
