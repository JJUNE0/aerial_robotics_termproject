# Crazyflie Level-3 Autonomous Mission
**POSTECH MECH701A-01 항공로봇공학 텀프로젝트**

Crazyflie 2.1 Brushless + Multi-ranger Deck + Flow Deck을 이용한 Level-3 자율비행 구현.
이륙 → 장애물 회피 탐색 → Landing region 스캔 → Pad 검출 및 착지 → 복귀 → 착지의 전체 왕복 미션을 수행합니다.

---

## Hardware

| 항목 | 사양 |
|------|------|
| 기체 | Crazyflie 2.1 Brushless |
| 거리 센서 | Multi-ranger Deck (front / back / left / right / up / down) |
| 위치 추정 | Flow Deck v2 (optical flow + ToF) |
| 통신 | Crazyradio PA (2.4 GHz) |

---

## Arena 구성

```
0        1.5 m      (1.5 + 0 m)    3.0 m
│← Start →│←── Middle ──→│← Landing →│
│  Region  │  (obstacle)  │  Region   │
│          │              │  [PAD]    │
└──────────┴──────────────┴───────────┘
          Arena Y: 1.0 m
```

- **Start region** (0 ~ 1.5 m): 이륙 패드 위치, 복귀 착지 지점
- **Middle region**: 장애물 포함 구간, frontier 탐색으로 통과
- **Landing region** (1.5 m ~ 3.0 m): 30 × 30 cm 착지 패드 위치

---

## Software Architecture

```
main thread                mission thread
──────────                 ──────────────
MissionGUI ←─SharedState─→ run_mission()
 (matplotlib)               │
                            ├─ SensorHub (cflib log callbacks)
                            │   ├─ OccupancyGrid (Bresenham ray-cast)
                            │   ├─ EdgeDetector  (z-ranger drop detection)
                            │   └─ FlightLogger  (CSV)
                            │
                            └─ state machine
                                TAKEOFF → ROTATION_SCAN → NAV_TO_LANDING
                                → LANDING_REGION_SCAN → PAD_CONFIRM
                                → LANDING_ON_PAD → TAKEOFF_FROM_PAD
                                → TURN_AROUND → NAV_TO_START
                                → LANDING_ON_START → DONE
```

### 파일 구성

| 파일 | 역할 |
|------|------|
| `mission.py` | 미션 상태 머신 진입점 |
| `controller.py` | EKF 초기화, arming, `go_to_nonblocking` |
| `sensors.py` | cflib log 수신, `EdgeDetector`, `SensorHub` |
| `mapping.py` | `OccupancyGrid` (점유 격자), `HeightMap` (패드 후보 관리) |
| `navigator.py` | A\*, LOS 경로 단순화, `FrontierNavigator`, `LawnmowerNavigator` |
| `shared_state.py` | 미션 스레드 ↔ GUI 스레드 간 thread-safe 상태 공유 |
| `gui.py` | 실시간 점유 맵 + 엣지 이벤트 시각화 (matplotlib) |
| `logger.py` | 비행 데이터 CSV 저장 (`log/YYYYMMDD_HHMMSS.csv`) |
| `plot_log.py` | 비행 후 로그 시각화 (z vs time, XY trajectory) |
| `config.py` | 모든 튜닝 파라미터 중앙 관리 |

---

## 주요 알고리즘

### 장애물 회피 (Middle region)
- **OccupancyGrid**: 3 cm/cell 해상도, Bresenham ray-casting으로 4방향 거리 센서를 격자에 반영
- **Inflation**: 드론 외접원 반경 + 3 cm 마진을 INFLATED 셀로 표시
- **A\***: 8방향 이동, 대각선 코너 clipping 방지
- **LOS 단순화**: 두꺼운 LOS 검사 (3×3 이웃 포함) 로 waypoint 수 최소화
- **FrontierNavigator**: BFS로 +X 방향 가장 먼 FREE 셀 탐색

### Landing Pad 검출
- **EdgeDetector**: EMA baseline 대비 z_down 낙하가 5 cm 초과 시 `entry`, 2.5 cm 미만 3샘플 연속 시 `exit`
- **HeightMap**: entry-exit 쌍 매칭 조건:
  - 같은 열 판정: `|entry_x - exit_x| < PAIR_SAME_COL_TOL (6 cm)`
  - 유효 패드 통과 판정: `|entry_y - exit_y| ≥ PAIR_MIN_Y_SPAN (20 cm)`
- **PadCandidate**: 매칭된 쌍들을 PAD_SIZE(30 cm) 반경으로 그룹핑 → 중심 좌표 추정
- **PAD_CONFIRM**: 후보 위치 호버링 중 `z_down < FLIGHT_Z - PAD_HEIGHT + 6 cm` 지속 확인

### EKF 수렴 대기
이륙 전 `kalman.varPX`, `kalman.varPY`를 100 ms 주기로 모니터링하여 최근 10샘플 범위가 0.001 미만일 때 이륙 허용 (최대 10 초 대기).

---

## 설정 (config.py)

```python
# 아레나
ARENA_X = 3              # m
ARENA_Y = 1              # m
TAKEOFF_PAD_X = 0.5      # 이륙 패드 X (아레나 좌표계, m)
TAKEOFF_PAD_Y = 0.5      # 이륙 패드 Y (아레나 좌표계, m)

# 비행
FLIGHT_Z = 0.3           # 순항 고도 (m)
NAV_SPEED = 0.2          # 탐색 속도 (m/s)
SCAN_SPEED = 0.2         # 스캔 속도 (m/s)

# 격자 맵
OCCUPANCY_GRID_RES = 0.03   # m/cell
INFLATION_RADIUS = DRONE_HALF_DIAGONAL + 0.03   # ~0.136 m

# 패드 검출
EDGE_THRESHOLD = 0.05        # entry 판별 낙하량 (m)
EDGE_BASELINE_ALPHA = 0.1    # EMA 계수
EDGE_EXIT_DEBOUNCE = 3       # exit 확정 샘플 수
PAIR_SAME_COL_TOL = 0.06     # 같은 열 판정 X 허용 오차 (m)
PAIR_MIN_Y_SPAN = 0.20       # 유효 통과 최소 Y span (m)
SCAN_ROW_SPACING = 0.15      # 롤모어 열 간격 (m)
```

실행 전 반드시 `TAKEOFF_PAD_X`, `TAKEOFF_PAD_Y`와 `RADIO_URI`를 환경에 맞게 설정하세요.

---

## 설치

```bash
pip install cflib numpy matplotlib pandas
```

Python 3.9 이상, cflib 0.1.32 (CRTP v8 legacy firmware) 기준입니다.

---

## 실행

```bash
python mission.py
```

GUI 창이 열리고 미션이 자동 시작됩니다. **EMERGENCY LAND** 버튼으로 언제든 즉시 착지할 수 있습니다.

---

## 비행 로그 분석

```bash
python plot_log.py              # 최신 로그 자동 선택
python plot_log.py log/파일.csv  # 특정 파일 지정
```

좌측: z_down vs 시간 (entry/exit 이벤트 표시)  
우측: XY 궤적 (색상 = z_down 값, entry/exit 마커, pair 연결선)

---

## 센서 좌표계

EKF 원점은 `kalman.resetEstimation` 시점의 이륙 패드 위치입니다.

```
EKF 좌표  →  아레나 좌표
x_arena = x_ekf + TAKEOFF_PAD_X
y_arena = y_ekf + TAKEOFF_PAD_Y
```

- +X: 아레나 동쪽 (landing region 방향)
- +Y: 아레나 북쪽
- +Z: 위

---

## 비상 착지

GUI의 **EMERGENCY LAND** 버튼 또는 `SharedState.trigger_emergency()` 호출 시 즉시 `hlc.land(0.0, 2.0)` 실행 후 미션 종료.
