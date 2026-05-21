"""Multi-ranger + Kalman pose log streamer."""
import threading

from cflib.crazyflie.log import LogConfig

from utils.config import LOG_PERIOD_MS, RANGE_FILTER_ALPHA
from utils.helpers import mm_to_m


class RangePoseReader:
    """
    두 LogConfig를 동시에 운용:
      - range.front/left/right/back/up/zrange + pm.vbat  (50ms)
      - stateEstimate.x/y/yaw                            (50ms)
    snapshot() 으로 thread-safe하게 최신값 dict 반환.
    """

    def __init__(self, cf):
        self._cf = cf
        self._lock = threading.Lock()
        self._latest = {
            "front": None, "left": None, "right": None, "back": None,
            "up": None, "zrange": None, "zrange_raw": None, "vbat": None,
            "x": None, "y": None, "z": None, "yaw": None,
        }
        self._range_log = None
        self._state_log = None

    def __enter__(self):
        rl = LogConfig(name="Ranges", period_in_ms=LOG_PERIOD_MS)
        rl.add_variable("range.front", "uint16_t")
        rl.add_variable("range.left", "uint16_t")
        rl.add_variable("range.right", "uint16_t")
        rl.add_variable("range.back", "uint16_t")
        rl.add_variable("range.up", "uint16_t")
        rl.add_variable("range.zrange", "uint16_t")
        rl.add_variable("pm.vbat", "float")
        self._cf.log.add_config(rl)
        rl.data_received_cb.add_callback(self._range_cb)
        rl.start()
        self._range_log = rl

        sl = LogConfig(name="Pose", period_in_ms=LOG_PERIOD_MS)
        sl.add_variable("stateEstimate.x", "float")
        sl.add_variable("stateEstimate.y", "float")
        sl.add_variable("stateEstimate.z", "float")
        sl.add_variable("stateEstimate.yaw", "float")
        self._cf.log.add_config(sl)
        sl.data_received_cb.add_callback(self._state_cb)
        sl.start()
        self._state_log = sl
        return self

    def __exit__(self, *_a):
        self.stop()

    def stop(self):
        for log in (self._range_log, self._state_log):
            if log is not None:
                try:
                    log.stop()
                except Exception:
                    pass
        self._range_log = None
        self._state_log = None

    def _range_cb(self, _ts, data, _conf):
        zrange_raw = mm_to_m(data.get("range.zrange"))
        raw = {
            "front":  mm_to_m(data.get("range.front")),
            "left":   mm_to_m(data.get("range.left")),
            "right":  mm_to_m(data.get("range.right")),
            "back":   mm_to_m(data.get("range.back")),
            "up":     mm_to_m(data.get("range.up")),
            "zrange": zrange_raw,
            "zrange_raw": zrange_raw,
            "vbat":   data.get("pm.vbat"),
        }
        with self._lock:
            for k, v in raw.items():
                if k in ("vbat", "zrange_raw"):
                    self._latest[k] = v
                    continue
                prev = self._latest.get(k)
                if v is None:
                    pass  # keep previous
                elif prev is None:
                    self._latest[k] = v
                else:
                    self._latest[k] = (
                        RANGE_FILTER_ALPHA * v + (1.0 - RANGE_FILTER_ALPHA) * prev
                    )

    def _state_cb(self, _ts, data, _conf):
        with self._lock:
            self._latest["x"] = data.get("stateEstimate.x")
            self._latest["y"] = data.get("stateEstimate.y")
            self._latest["z"] = data.get("stateEstimate.z")
            self._latest["yaw"] = data.get("stateEstimate.yaw")  # degrees

    def snapshot(self):
        with self._lock:
            return dict(self._latest)
