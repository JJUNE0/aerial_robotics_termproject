"""
Entry point for the Crazyflie OGM + Gaussian goal landing app.

Run:
    python main.py [--uri radio://0/80/2M/E7E7E7E7E5] [--target-height 0.40]

The original monolithic ogm.py has been split into:
    utils/   — config, helpers, control (rate limiters + arming)
    module/  — OccupancyGrid, RangePoseReader, CrazyflieWorker, MapView, MainWindow
"""
import argparse
import logging
import signal
import sys
import warnings

try:
    from PyQt5 import QtWidgets
    PYQT_AVAILABLE = True
except ImportError:
    QtWidgets = None
    PYQT_AVAILABLE = False

from utils.config import TARGET_HEIGHT, URI


logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*supervisor.*", category=UserWarning)


_active_worker = None


def signal_handler(_sn, _fr):
    if _active_worker is not None:
        _active_worker.request_emergency_stop()


def parse_args():
    p = argparse.ArgumentParser(
        description="Crazyflie OGM Gaussian-goal landing"
    )
    p.add_argument("--uri", default=URI)
    p.add_argument("--target-height", type=float, default=TARGET_HEIGHT)
    return p.parse_args()


def main():
    args = parse_args()
    if not PYQT_AVAILABLE:
        print("PyQt5 not installed. Try: pip install PyQt5")
        return 1

    from module.main_window import MainWindow  # imported lazily (requires PyQt5)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args)
    win.resize(1080, 740)
    win.show()
    global _active_worker
    _active_worker = win.worker
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
