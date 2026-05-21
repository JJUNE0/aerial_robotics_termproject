"""Configuration constants for the OGM / Gaussian goal landing app."""
from pathlib import Path

from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E5")

TARGET_HEIGHT = 0.40
SPEED_X = 0.3
SPEED_Y = 0.2
ARRIVAL_RADIUS = 0.10            # axis 도착 판정 반경

# Goal distribution / landing search
GOAL_SIGMA = 0.30                # goal point 주변 Gaussian 표준편차
GOAL_SEARCH_RADIUS = 0.75        # landing 후보를 찾을 최대 반경
LANDING_CLEAR_RADIUS = 0.18      # 착륙 후보 주변 장애물 여유 반경
LANDING_OCCUPIED_LIMIT = 0.75    # 이 log-odds 이상이면 landing 후보 제외
LANDING_Z_DELTA = 0.045          # 주변 바닥 대비 zrange 변화량
LANDING_Z_STEP_DELTA = 0.025     # 이전 스텝 대비 zrange 급변 감지 기준
LANDING_Z_STABLE_COUNT = 3       # landing 높이 후보 연속 검출 횟수
LANDING_START_RADIUS = 0.15      # goal 평균점에 이만큼 접근한 뒤 착륙 탐색 시작
GOAL_SCAN_RADIUS = 0.35          # 분포 내부 z-search 순회 반경
GOAL_SCAN_SPEED = 0.08

# Safety: 4 ranger 10 cm 유지
SAFETY_DIST = 0.10
LANDING_DESCENT_MIN_DIST = 0.12  # goal 내부 착륙 하강을 허용할 최소 수평 거리
SAFETY_BRAKE_DIST = 0.25         # 25cm 안으로 들어오면 비례 감속 시작
FRONT_AVOID_DIST = 0.35          # front가 이보다 가까우면 좌/우 회피
SIDE_RETURN_DIST = 0.18          # 선택한 side가 가까워지면 goal 분포로 복귀
SIDE_STEP_SPEED = 0.12
GOAL_SEEK_SPEED = 0.16
SIDE_AVOID_MAX_S = 2.0

# OGM
MAP_RESOLUTION = 0.10
DEFAULT_MAP_W = 3.0
DEFAULT_MAP_H = 5.0
DEFAULT_START = (0.5, 0.5)
DEFAULT_GOAL = (2.5, 4.5)

LOG_ODDS_OCC = 0.85
LOG_ODDS_FREE = -0.4
LOG_ODDS_MAX = 5.0
LOG_ODDS_MIN = -5.0
RAY_MAX_RANGE = 3.5              # ranger OUT 처리 시 free로 그릴 거리

# 기존 보존
CONTROL_DT = 0.05
LOG_PERIOD_MS = 50
RANGE_FILTER_ALPHA = 0.35
MAX_VELOCITY_STEP = 0.025
MAX_HEIGHT_COMMAND = TARGET_HEIGHT
MAX_HEIGHT_STEP_UP = 0.01
TAKEOFF_STEP_M = 0.02
TAKEOFF_STEP_S = 0.08
LANDING_STEP_M = 0.02
LANDING_STEP_S = 0.10
OUT_OF_RANGE_MM = 4000
SENSOR_LOG_DIR = Path("logs")
