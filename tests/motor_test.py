# motor_test.py
"""
Step 2: 모터 명령 검증
프로펠러 제거 + 고정 상태에서만 실행
모터 4개에 매우 약한 신호를 보내서 회전하는지 확인
"""
import time
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E5')

logging.basicConfig(level=logging.ERROR)


def main():
    cflib.crtp.init_drivers()

    print("=" * 50)
    print("⚠️  안전 체크")
    print("=" * 50)
    print("1. 프로펠러 4개 모두 제거되었습니까?")
    print("2. 드론이 단단히 고정되어 있습니까?")
    print("3. Ctrl+C 누를 준비 되었습니까?")
    print("=" * 50)
    answer = input("모두 OK이면 'yes' 입력: ")
    if answer.strip().lower() != 'yes':
        print("중단합니다.")
        return

    cf = Crazyflie(rw_cache='./cache')
    
    with SyncCrazyflie(URI, cf=cf) as scf:
        print("\nConnected. 3초 후 모터 테스트 시작...")
        time.sleep(3)

        # 안전 잠금 해제 (이게 있어야 모터 명령이 받아들여짐)
        scf.cf.platform.send_arming_request(True)
        time.sleep(1)

        # 매우 약한 thrust 값들
        # 0~65535 범위 / 호버 추력은 보통 ~36000~42000
        # 우리는 안전을 위해 10000 (약 15%)만 사용
        thrust_levels = [0, 5000, 10000, 20000, 30000, 40000, 30000, 20000, 10000, 5000, 0]
        
        for thrust in thrust_levels:
            print(f"  Thrust = {thrust} ({thrust/65535*100:.1f}%)")
            # roll=0, pitch=0, yawrate=0, thrust=값
            # 첫 호출은 thrust=0 unlock 시퀀스
            scf.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            scf.cf.commander.send_setpoint(0, 0, 0, thrust)
            time.sleep(2)

        # 모터 정지
        print("\n모터 정지...")
        scf.cf.commander.send_setpoint(0, 0, 0, 0)
        scf.cf.commander.send_stop_setpoint()
        time.sleep(1)

    print("\nDone. 4개 모터가 모두 천천히 → 빠르게 → 천천히 → 정지 했나요?")


if __name__ == '__main__':
    main()