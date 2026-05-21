"""Pure helpers: unit conversion, formatting, scoring."""
import math

from .config import GOAL_SIGMA, OUT_OF_RANGE_MM


def mm_to_m(value_mm):
    if value_mm is None:
        return None
    try:
        v = float(value_mm)
    except (TypeError, ValueError):
        return None
    if v <= 0 or v >= OUT_OF_RANGE_MM or math.isinf(v):
        return None
    return v / 1000.0


def fmt_distance(v):
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return "OUT"
    return f"{v:.2f} m"


def battery_percent(vbat):
    if vbat is None:
        return 0
    return int(max(0.0, min(100.0, (float(vbat) - 3.2) / (4.2 - 3.2) * 100.0)))


def battery_level_name(vbat):
    if vbat is None:
        return "unknown"
    if vbat < 3.4:
        return "critical"
    if vbat < 3.6:
        return "low"
    return "good"


def fmt_battery(vbat):
    if vbat is None:
        return "Battery --"
    return f"Battery {vbat:.2f} V  ·  {battery_percent(vbat)}%"


def clamp(value, low, high):
    return max(low, min(high, value))


def gaussian_weight(x, y, mean_xy, sigma=GOAL_SIGMA):
    if sigma <= 0:
        return 1.0 if (x, y) == tuple(mean_xy) else 0.0
    dx = x - mean_xy[0]
    dy = y - mean_xy[1]
    return math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def goal_distribution_threshold():
    return gaussian_weight(GOAL_SIGMA, 0.0, (0.0, 0.0))


def in_goal_distribution(x, y, goal_xy):
    return gaussian_weight(x, y, goal_xy) >= goal_distribution_threshold()
