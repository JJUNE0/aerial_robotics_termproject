"""Qt widget that renders the OGM + start/goal markers + drone trace."""
from PyQt5 import QtCore, QtGui, QtWidgets

from utils.config import GOAL_SEARCH_RADIUS, GOAL_SIGMA, LOG_ODDS_MAX


class MapView(QtWidgets.QWidget):
    """Render OGM + start/goal markers + drone trace."""

    def __init__(self):
        super().__init__()
        self.ogm = None
        self.drone_world = None
        self.start_world = None
        self.goal_world = None
        self.landing_world = None
        self.path_world = []
        self.setMinimumSize(360, 480)
        self.setStyleSheet(
            "background: white; border: 1px solid #d0d6e0; border-radius: 8px;"
        )

    def set_map(self, ogm, start_xy, goal_xy):
        self.ogm = ogm
        self.start_world = start_xy
        self.goal_world = goal_xy
        self.landing_world = None
        self.path_world = []
        self.update()

    def set_landing_marker(self, world_x, world_y):
        self.landing_world = (world_x, world_y)
        self.update()

    def update_drone(self, world_x, world_y):
        self.drone_world = (world_x, world_y)
        self.path_world.append((world_x, world_y))
        if len(self.path_world) > 4000:
            self.path_world = self.path_world[-4000:]
        self.update()

    def map_changed(self):
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        if self.ogm is None:
            p.setPen(QtGui.QColor("#9ca3af"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       "Press 'Build Map' (after Connect) to start mapping")
            return

        # 비대칭 margin: 화면 좌측에 x 라벨, 화면 하단에 y 라벨
        # (drone forward = 화면 위, drone left = 화면 왼쪽 으로 회전 표시)
        margin_l = 36
        margin_b = 24
        margin_t = 14
        margin_r = 14
        avail_w = w - margin_l - margin_r
        avail_h = h - margin_t - margin_b
        # world x(=forward)는 화면 세로, world y(=left)는 화면 가로에 매핑
        scale = min(avail_w / self.ogm.height_m, avail_h / self.ogm.width_m)

        def w2s(wx, wy):
            # world +x (drone forward) → screen up  (sy 감소)
            # world +y (drone left)    → screen left (sx 감소)
            sx = (w - margin_r) - wy * scale
            sy = (h - margin_b) - wx * scale
            return sx, sy

        # ---- 격자 (0.5m 점선) ----
        grid_step = 0.5
        p.setPen(QtGui.QPen(QtGui.QColor(220, 224, 232), 1, QtCore.Qt.DotLine))
        gx = 0.0
        while gx <= self.ogm.width_m + 1e-6:
            sx_a, sy_a = w2s(gx, 0)
            sx_b, sy_b = w2s(gx, self.ogm.height_m)
            p.drawLine(QtCore.QPointF(sx_a, sy_a), QtCore.QPointF(sx_b, sy_b))
            gx += grid_step
        gy = 0.0
        while gy <= self.ogm.height_m + 1e-6:
            sx_a, sy_a = w2s(0, gy)
            sx_b, sy_b = w2s(self.ogm.width_m, gy)
            p.drawLine(QtCore.QPointF(sx_a, sy_a), QtCore.QPointF(sx_b, sy_b))
            gy += grid_step

        # ---- 눈금 라벨 ----
        p.setPen(QtGui.QColor("#6b7280"))
        font = p.font()
        font.setPointSize(8)
        p.setFont(font)
        # x 라벨 → 화면 좌측 (각 wx 격자선의 왼쪽 끝)
        gx = 0.0
        while gx <= self.ogm.width_m + 1e-6:
            sx, sy = w2s(gx, self.ogm.height_m)
            p.drawText(QtCore.QPointF(sx - 30, sy + 4), f"{gx:.1f}")
            gx += grid_step
        # y 라벨 → 화면 하단 (각 wy 격자선의 아래쪽 끝)
        gy = 0.0
        while gy <= self.ogm.height_m + 1e-6:
            sx, sy = w2s(0, gy)
            p.drawText(QtCore.QPointF(sx - 8, sy + 16), f"{gy:.1f}")
            gy += grid_step

        # boundary
        sx0, sy0 = w2s(0, 0)
        sx1, sy1 = w2s(self.ogm.width_m, self.ogm.height_m)
        p.setPen(QtGui.QPen(QtGui.QColor("#9ca3af"), 1))
        rect = QtCore.QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0).normalized()
        p.drawRect(rect)

        # ---- 원점 좌표축 화살표 (x=빨강 위로, y=초록 왼쪽으로) ----
        arrow_len_world = 0.35  # meters
        ox, oy = w2s(0, 0)
        exs, eys = w2s(arrow_len_world, 0)   # x축 끝점 (화면 위)
        yxs, yys = w2s(0, arrow_len_world)   # y축 끝점 (화면 왼쪽)
        # x축 (빨강, 위로)
        p.setPen(QtGui.QPen(QtGui.QColor("#ef4444"), 2))
        p.drawLine(QtCore.QPointF(ox, oy), QtCore.QPointF(exs, eys))
        p.drawLine(QtCore.QPointF(exs, eys), QtCore.QPointF(exs - 4, eys + 6))
        p.drawLine(QtCore.QPointF(exs, eys), QtCore.QPointF(exs + 4, eys + 6))
        font_b = p.font()
        font_b.setBold(True)
        font_b.setPointSize(10)
        p.setFont(font_b)
        p.setPen(QtGui.QColor("#ef4444"))
        p.drawText(QtCore.QPointF(exs + 6, eys + 4), "x (fwd)")
        # y축 (초록, 왼쪽으로)
        p.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 2))
        p.drawLine(QtCore.QPointF(ox, oy), QtCore.QPointF(yxs, yys))
        p.drawLine(QtCore.QPointF(yxs, yys), QtCore.QPointF(yxs + 6, yys - 4))
        p.drawLine(QtCore.QPointF(yxs, yys), QtCore.QPointF(yxs + 6, yys + 4))
        p.setPen(QtGui.QColor("#10b981"))
        p.drawText(QtCore.QPointF(yxs - 32, yys - 4), "y (left)")
        p.setFont(font)

        # cells (vectorised lookup but simple iteration for clarity)
        log_odds = self.ogm.snapshot()
        cell_w = self.ogm.res * scale
        for row in range(self.ogm.rows):
            for col in range(self.ogm.cols):
                lo = log_odds[row, col]
                if abs(lo) < 0.05:
                    continue
                cx = col * self.ogm.res
                cy = row * self.ogm.res
                px, py = w2s(cx, cy + self.ogm.res)
                if lo > 0:
                    a = int(min(220, lo / LOG_ODDS_MAX * 220 + 35))
                    color = QtGui.QColor(220, 60, 60, a)
                else:
                    a = int(min(140, -lo / LOG_ODDS_MAX * 120 + 20))
                    color = QtGui.QColor(100, 180, 100, a)
                p.fillRect(QtCore.QRectF(px, py, cell_w, cell_w), color)

        # commanded reference line (start -> goal mean)
        if self.start_world and self.goal_world:
            p.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180, 160), 1, QtCore.Qt.DashLine))
            sxs, sys_ = w2s(*self.start_world)
            gxs, gys = w2s(*self.goal_world)
            p.drawLine(QtCore.QPointF(sxs, sys_), QtCore.QPointF(gxs, gys))

        # Gaussian goal distribution: 1-sigma filled area and search boundary.
        if self.goal_world:
            gxs, gys = w2s(*self.goal_world)
            sigma_px = GOAL_SIGMA * scale
            search_px = GOAL_SEARCH_RADIUS * scale
            p.setBrush(QtGui.QColor(245, 158, 11, 42))
            p.setPen(QtGui.QPen(QtGui.QColor(245, 158, 11, 150), 1.5))
            p.drawEllipse(QtCore.QPointF(gxs, gys), sigma_px, sigma_px)
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(245, 158, 11, 95), 1, QtCore.Qt.DashLine))
            p.drawEllipse(QtCore.QPointF(gxs, gys), search_px, search_px)

        # measured trajectory
        if len(self.path_world) >= 2:
            p.setPen(QtGui.QPen(QtGui.QColor("#3b82f6"), 1.8))
            pts = [QtCore.QPointF(*w2s(x, y)) for x, y in self.path_world]
            path = QtGui.QPainterPath()
            path.moveTo(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.drawPath(path)

        def marker(wx, wy, color, label):
            sx, sy = w2s(wx, wy)
            p.setBrush(QtGui.QColor(color))
            p.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
            p.drawEllipse(QtCore.QPointF(sx, sy), 6, 6)
            p.drawText(QtCore.QPointF(sx + 9, sy + 4), label)

        if self.start_world:
            marker(*self.start_world, color="#10b981", label="S")
        if self.goal_world:
            marker(*self.goal_world, color="#f59e0b", label="Gμ")

        if self.landing_world:
            sx, sy = w2s(*self.landing_world)
            p.setPen(QtGui.QPen(QtGui.QColor("#dc2626"), 3))
            p.drawLine(QtCore.QPointF(sx - 8, sy - 8), QtCore.QPointF(sx + 8, sy + 8))
            p.drawLine(QtCore.QPointF(sx - 8, sy + 8), QtCore.QPointF(sx + 8, sy - 8))
            p.drawText(QtCore.QPointF(sx + 10, sy - 8), "L")

        if self.drone_world:
            sx, sy = w2s(*self.drone_world)
            p.setBrush(QtGui.QColor("#1e3a8a"))
            p.setPen(QtGui.QPen(QtGui.QColor("#1e3a8a"), 2))
            p.drawEllipse(QtCore.QPointF(sx, sy), 5, 5)
