# Crazyflie Level-3 Autonomous Mission

POSTECH MECH701A-01 텀프로젝트 — Crazyflie 2.1 Brushless를 이용한 Level-3 자율비행 미션

---

## 미션 개요

5×3m 아레나에서 이륙 패드(Start Region)를 출발해 장애물을 회피하며 Landing Region까지 이동, 랜딩 패드 위에 자율 착지한 뒤 다시 출발 패드로 복귀하여 착지합니다.

```
이륙 → 회전 스캔 → 장애물 회피 이동(+X) → 랜딩 패드 스캔
    → 랜딩 패드 착지 → 재이륙 → 복귀 이동(-X) → 출발 패드 착지
```

---

## 하드웨어 / 펌웨어

| 항목 | 사양 |
|------|------|
| 기체 | Crazyflie 2.1 Brushless |
| 센서 | Multi-ranger Deck (front/back/left/right/up), Flow Deck v2 (XY odometry + 하부 ToF) |
| 통신 | Crazyradio PA (2.4 GHz) |
| cflib | 0.1.32 |

---

## 코드 구조

```
.
├── mission.py        # 미션 메인 로직 (상태 시퀀스 전체)
├── config.py         # 전체 파라미터 설정
├── controller.py     # 저수준 비행 제어 (이/착륙, velocity 명령)
├── mapping.py        # Occupancy Grid 맵핑 + 랜딩 패드 검출
├── navigator.py      # A* 경로 계획 + Frontier 탐색
├── sensors.py        # 센서 데이터 수집 + Edge 검출
├── shared_state.py   # 스레드 간 공유 상태 (thread-safe)
├── gui.py            # PyQt5 실시간 시각화
├── logger.py         # CSV + NPZ 비행 로그 저장
├── plot_log.py       # 비행 로그 분석 및 시각화
└── log/              # 비행 로그 파일 (CSV + NPZ)
```

---

## 파일별 알고리즘 설명

### `mapping.py` — Occupancy Grid + 패드 검출

- **OccupancyGrid**: 2cm/셀 해상도의 2D 점유 격자. 각 레인저 방향으로 Bresenham ray-casting을 적용해 FREE/OCCUPIED/UNKNOWN/INFLATED 셀을 갱신. INFLATED는 드론 대각선 반경(`DRONE_HALF_DIAGONAL`)만큼 팽창시킨 안전 마진.
- **find_pad_from_diff**: 고고도(30cm) 스캔 맵과 저고도(8cm) 스캔 맵의 차분(diff)을 계산. 저고도에서만 검출되는 셀(= 바닥보다 높은 물체)을 8-방향 BFS로 클러스터링해 랜딩 패드 위치를 추정.
- **HeightMap / EdgeDetector**: z_down 레인저가 순항고도(30cm) 기준으로 일정 이상 낮아지면(패드 edge) entry/exit 이벤트 발생. valley/peak 방향 전환 + 임계값(27cm / 36cm) 조건으로 검출.

### `navigator.py` — 경로 계획

- **A\* (4방향)**: Manhattan 휴리스틱 + clearance penalty. 장애물 근처 셀에 반비례 거중치(`1/dist`, radius=4 cells)를 추가해 경로가 장애물 중앙을 선호하게 유도.
- **simplify_path**: 두 단계 경로 단순화. ① 연속 방향이 같은 waypoint 합치기(collinear merge) ② greedy L-shape skip — 현재 위치에서 가장 먼 도달 가능 waypoint까지 H→V 또는 V→H 한 번의 꺾임으로 직행할 수 있으면 중간 waypoint를 건너뜀.
- **FrontierNavigator**: BFS로 맵 상에서 이동 가능한(FREE) 셀 중 +X 방향(출발) 또는 -X 방향(복귀)으로 가장 먼 frontier를 탐색. 탐색 한계를 현재 위치 기준 최대 1.1m로 제한.

### `sensors.py` — 센서 수집

- **SensorHub**: `LogConfig`를 이용해 EKF pose(x, y, z, yaw), 5방향 레인저, 배터리를 20Hz로 수집. 별도 스레드에서 실행.
- **EdgeDetector**: 슬라이딩 윈도우로 z_down의 방향 전환(valley/peak)을 감지. 쿨다운(1s)으로 연속 오검출 억제.

### `controller.py` — 비행 제어

- **send_hover_setpoint 전용**: High-Level Commander 비사용. 모든 이동은 `(vx_body, vy_body, yaw_rate, z)` velocity setpoint로 제어.
- **init_ekf**: `kalman.resetEstimation`으로 EKF 초기화 후 XY 분산이 수렴할 때까지 대기. 이후 supervisor arming.
- **land_vel**: 착지 전 위치 P-제어기(`kp=2.0`)로 XY 홀딩하면서 z를 `LAND_CUTOFF_Z(3cm)`까지 서서히 하강.
- **vel_to_body**: world frame 속도 벡터를 현재 yaw로 회전변환해 body frame으로 변환.

### `mission.py` — 미션 시퀀스

- **_navigate_to**: P-제어기(`KP=3.0`) velocity navigation. 목표까지의 거리 오차에 비례한 속도 명령 + 진행 방향 레인저가 STOP_THRESHOLD(15cm) 이하이면 즉시 정지.
- **_nav_to_x / _nav_to_home**: Frontier → A* → waypoint 이동 루프. 2회 연속 계획 실패 시 90° rotation scan으로 맵 갱신 후 재시도.
- **do_landing_region_scan**: 스캔 위치(arena 4.5, 1.5)에서 고/저 고도 180° 회전 스캔 2회 수행. diff 맵에서 패드 위치 추정.
- **do_land_on_pad**: X-align → 1s 호버 → Y방향 접근(edge detection) → entry 감지 후 `entry_overshoot`만큼 전진 → yaw align(선택) → 착지.
- **LAND_YAW_ALIGN**: 착지 직전 feedback 기반 yaw 제어. `home_yaw(0°)` 기준으로 ±1° 이내가 될 때까지 회전 + XY position hold.

### `gui.py` — 실시간 시각화

- PyQt5 + Matplotlib. 20Hz로 갱신되는 4개 맵 패널(Navigation Map, High-Alt Scan, Low-Alt Scan, Diff Map) + 상태/배터리/타이머 표시.
- 출발 경로 waypoint: cyan, 복귀 경로 waypoint: orange로 구분.

### `plot_log.py` — 로그 분석

- `z_down vs Time` 그래프: entry/exit 검출 임계선 + 검출 시점 마커.
- `XY Trajectory` 그래프: 출발 경로(Reds colormap), 복귀 경로(파란색)로 구분. Occupancy Map 오버레이.

---

## 환경 구축

### Python 패키지 설치

```bash
pip install cflib numpy scipy pyqt5 matplotlib pandas
```

### Crazyradio 드라이버 (Linux)

```bash
# USB 권한 설정
sudo groupadd plugdev
sudo usermod -aG plugdev $USER
# udev rules
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1915", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-crazyradio.rules
sudo udevadm control --reload-rules
```

로그아웃 후 재로그인 필요.

---

## 실행 방법

### 1. Radio URI 설정

`config.py`에서 드론 주소 확인:

```python
RADIO_URI = 'radio://0/80/2M/E7E7E7E7E5'
```

### 2. 이륙 패드 위치 설정

`config.py`에서 아레나 내 이륙 패드 좌표(단위: m) 입력:

```python
TAKEOFF_PAD_X = 1.0   # 서쪽 벽에서 +x 방향
TAKEOFF_PAD_Y = 2.5   # 남쪽 벽에서 +y 방향
```

### 3. 미션 실행

```bash
python3 mission.py
```

실행 시 PyQt5 GUI가 열립니다.  
키보드 `Q` 또는 GUI 우상단 **STOP** 버튼으로 긴급 정지.

### 4. 로그 분석

```bash
python3 plot_log.py
```

`log/` 폴더의 최신 비행 로그를 자동으로 불러와 그래프를 표시합니다.

---

## 주요 파라미터 (`config.py`)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `FLIGHT_Z` | 0.30 m | 순항 고도 |
| `NAV_SPEED` | 0.30 m/s | 이동 속도 |
| `LANDING_SCAN_X/Y` | 4.5 / 1.5 m | 랜딩 패드 스캔 위치 (arena 좌표) |
| `PAD_LAND_X_SPEED` | 0.15 m/s | 패드 X-align 속도 |
| `PAD_LAND_PUSH_SPEED` | 0.25 m/s | 패드 Y 접근 속도 |
| `A_STAR_CLEARANCE_RADIUS` | 4 cells | A* clearance penalty 반경 |
| `EDGE_ENTRY_DIP` | 0.03 m | edge entry 검출 임계값 |
| `EDGE_EXIT_RISE` | 0.06 m | edge exit 검출 임계값 |

---

## EKF 좌표계

EKF reset 후 이륙 패드 위치가 원점 (0, 0, 0).  
+X: Landing Region 방향, +Y: 왼쪽(북쪽), +Z: 위쪽.

```
arena 좌표 → EKF 좌표 변환:
  ekf_x = arena_x - TAKEOFF_PAD_X
  ekf_y = arena_y - TAKEOFF_PAD_Y
```

---

## 미션 상태 시퀀스

```
TAKEOFF
  → ROTATION_SCAN           초기 환경 스캔
  → NAV_FRONTIER            Frontier BFS 탐색
  → NAV_WAYPOINT            A* waypoint 이동
  → SCAN_NAV_*              스캔 위치까지 이동
  → SCAN_HIGH / SCAN_LOW    이중 고도 패드 스캔
  → LAND_X_ALIGN            패드 X 정렬
  → LAND_APPROACH           Y 방향 edge 탐색 접근
  → LAND_ENTRY              entry edge 감지 후 전진
  → LAND_HOVER              착지 전 호버링
  → LAND_YAW_ALIGN          yaw 0° 정렬
  → TAKEOFF_FROM_PAD        재이륙
  → RET_FRONTIER            복귀 Frontier 탐색
  → RET_WAYPOINT            복귀 A* waypoint 이동
  → LANDING_ON_START        출발 패드 착지
  → DONE
```
