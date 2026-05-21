# log_test.py
"""
Step 1: 비행 없이 통신 + EKF + 센서 검증
드론 켜고 평평한 곳에 그냥 두기 (모터 회전 없음)
"""
import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

# 통신 주소 (cfclient에서 확인한 값)
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E5')

logging.basicConfig(level=logging.ERROR)


def log_callback(timestamp, data, logconf):
    """100ms마다 호출되는 콜백 — 센서 데이터 출력"""
    print(f"[{timestamp:>8}] "
          f"roll={data['stabilizer.roll']:7.2f}  "
          f"pitch={data['stabilizer.pitch']:7.2f}  "
          f"yaw={data['stabilizer.yaw']:7.2f}  "
          f"batt={data['pm.vbat']:.2f}V")


def main():
    cflib.crtp.init_drivers()

    print(f"Connecting to {URI}...")
    cf = Crazyflie(rw_cache='./cache')

    # SyncCrazyflie를 context manager로 사용
    with SyncCrazyflie(URI, cf=cf) as scf:
        print("Connected!")
        time.sleep(1)

        # 어떤 변수를 로깅할지 정의
        log_conf = LogConfig(name='Stabilizer', period_in_ms=100)
        log_conf.add_variable('stabilizer.roll', 'float')
        log_conf.add_variable('stabilizer.pitch', 'float')
        log_conf.add_variable('stabilizer.yaw', 'float')
        log_conf.add_variable('pm.vbat', 'float')

        scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(log_callback)
        log_conf.start()

        print("Logging for 10 seconds... (드론을 손으로 살짝 기울여보세요)")
        time.sleep(10)

        log_conf.stop()

    print("Done.")


if __name__ == '__main__':
    main()