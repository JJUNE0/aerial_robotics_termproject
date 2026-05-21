# range_test.py
"""
Flow deck v2 ToF + Multi-ranger 5방향 거리 통합 측정
드론을 손에 들고 벽/책상에 가까이/멀리 움직이며 측정
"""
import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E5')
logging.basicConfig(level=logging.ERROR)


def fmt(val_mm):
    """mm 값을 보기 좋게 변환. 4000 이상은 'OUT'으로 표시"""
    if val_mm >= 4000:
        return "  OUT "
    return f"{val_mm:>4.0f}mm"


def log_callback(timestamp, data, logconf):
    # Flow deck (아래)
    z_down = data['range.zrange']
    
    # Multi-ranger (5방향)
    front = data['range.front']
    back = data['range.back']
    left = data['range.left']
    right = data['range.right']
    up = data['range.up']
    
    print(f"[{timestamp:>7}]  "
          f"↓={fmt(z_down)}  "
          f"↑={fmt(up)}  "
          f"F={fmt(front)}  "
          f"B={fmt(back)}  "
          f"L={fmt(left)}  "
          f"R={fmt(right)}")


def main():
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache='./cache')

    with SyncCrazyflie(URI, cf=cf) as scf:
        print("Connected.")
        print("=" * 60)
        print("드론을 손에 들고 다음을 해보세요:")
        print("  1. 책상 위 → 위로 들어올리기 (↓ 값 변화)")
        print("  2. 벽/사람에게 가까이/멀리 (F/B/L/R 값 변화)")
        print("  3. 천장 아래 → 천장과의 거리 (↑ 값)")
        print("=" * 60)
        time.sleep(2)

        # Multi-ranger는 한 LogConfig에 변수가 많으면 문제 생길 수 있어
        # 두 개로 나눠서 만듦 (cflib 권장)
        log_conf = LogConfig(name='Ranges', period_in_ms=100)
        log_conf.add_variable('range.zrange', 'uint16_t')
        log_conf.add_variable('range.front', 'uint16_t')
        log_conf.add_variable('range.back', 'uint16_t')
        log_conf.add_variable('range.left', 'uint16_t')
        log_conf.add_variable('range.right', 'uint16_t')
        log_conf.add_variable('range.up', 'uint16_t')

        scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(log_callback)
        log_conf.start()

        print("\n20초간 측정 시작...\n")
        time.sleep(20)

        log_conf.stop()

    print("\nDone.")


if __name__ == '__main__':
    main()