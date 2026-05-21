"""Main Qt window: wiring GUI controls to CrazyflieWorker + MapView."""
from PyQt5 import QtWidgets

from utils.config import DEFAULT_GOAL, DEFAULT_MAP_H, DEFAULT_MAP_W, DEFAULT_START
from utils.helpers import battery_level_name, battery_percent, fmt_battery, fmt_distance

from .map_view import MapView
from .occupancy_grid import OccupancyGrid
from .worker import CrazyflieWorker


class MainWindow(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Crazyflie OGM + Gaussian Goal Landing")
        self.setStyleSheet(self._stylesheet())

        # widgets
        self.uri_edit = QtWidgets.QLineEdit(args.uri)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setObjectName("connectButton")
        self.disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("dangerButton")
        self.disconnect_btn.setEnabled(False)

        self.map_w_spin = self._spin(0.5, 10.0, DEFAULT_MAP_W, 0.5)
        self.map_h_spin = self._spin(0.5, 10.0, DEFAULT_MAP_H, 0.5)
        self.start_x_spin = self._spin(0.0, 10.0, DEFAULT_START[0], 0.1)
        self.start_y_spin = self._spin(0.0, 10.0, DEFAULT_START[1], 0.1)
        self.goal_x_spin = self._spin(0.0, 10.0, DEFAULT_GOAL[0], 0.1)
        self.goal_y_spin = self._spin(0.0, 10.0, DEFAULT_GOAL[1], 0.1)
        self.height_spin = self._spin(0.20, 0.80, args.target_height, 0.05)

        self.start_btn = QtWidgets.QPushButton("Start Goal Landing")
        self.land_btn = QtWidgets.QPushButton("Land")
        self.estop_btn = QtWidgets.QPushButton("Emergency Stop")
        self.estop_btn.setObjectName("dangerButton")
        self.start_btn.setEnabled(False)
        self.land_btn.setEnabled(False)
        self.estop_btn.setEnabled(False)

        # Mapping buttons (Build = toggle)
        self.build_btn = QtWidgets.QPushButton("Build Map")
        self.build_btn.setObjectName("mapButton")
        self.build_btn.setCheckable(True)
        self.reset_btn = QtWidgets.QPushButton("Reset Map")
        self.build_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setObjectName("statusLabel")
        self.battery_label = QtWidgets.QLabel("Battery --")
        self.battery_label.setObjectName("metric")
        self.battery_bar = QtWidgets.QProgressBar()
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setTextVisible(False)
        self._set_battery_style(None)

        self.range_labels = {k: QtWidgets.QLabel("--") for k in
                             ("front", "left", "right", "back", "up", "zrange")}
        self.pose_label = QtWidgets.QLabel("x: --  y: --  yaw: --")
        self.pose_label.setObjectName("metric")

        self.error_label = QtWidgets.QLabel("Press Start to fly. Drift report will appear here.")
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)

        self.map_view = MapView()

        # worker wiring
        self.worker = CrazyflieWorker()
        self.worker.sensor_updated.connect(self.on_sensor)
        self.worker.connection_state.connect(self.on_connection_state)
        self.worker.status_text.connect(self.status_label.setText)
        self.worker.map_changed.connect(self.map_view.map_changed)
        self.worker.flight_result.connect(self.on_flight_result)
        self.worker.landing_found.connect(self.map_view.set_landing_marker)

        self.connect_btn.clicked.connect(self.on_connect)
        self.disconnect_btn.clicked.connect(self.on_disconnect)
        self.start_btn.clicked.connect(self.on_start_flight)
        self.land_btn.clicked.connect(self.on_land)
        self.estop_btn.clicked.connect(self.on_estop)
        self.build_btn.toggled.connect(self.on_build_toggled)
        self.reset_btn.clicked.connect(self.on_reset_map)

        self.ogm = None
        self._build_layout()

    def _stylesheet(self):
        return """
        QWidget { background: #f6f7fb; color: #1f2937; font-size: 13px; }
        QGroupBox { background: #ffffff; border: 1px solid #d9dee8;
                    border-radius: 8px; margin-top: 12px; padding: 12px;
                    font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
        QLineEdit, QDoubleSpinBox { background: #ffffff; border: 1px solid #cfd6e3;
                                    border-radius: 6px; padding: 4px 6px; }
        QPushButton { background: #2563eb; color: white; border: none;
                      border-radius: 7px; padding: 8px 12px; font-weight: 600; }
        QPushButton:hover { background: #1d4ed8; }
        QPushButton:disabled { background: #aeb7c7; }
        QPushButton#dangerButton { background: #dc2626; }
        QPushButton#dangerButton:hover { background: #b91c1c; }
        QPushButton#connectButton { background: #16a34a; }
        QPushButton#connectButton:hover { background: #15803d; }
        QPushButton#mapButton { background: #7c3aed; }
        QPushButton#mapButton:hover { background: #6d28d9; }
        QPushButton#mapButton:checked { background: #5b21b6; }
        QLabel#statusLabel { background: #111827; color: white;
                             border-radius: 8px; padding: 10px 12px; font-weight: 600; }
        QLabel#metric { color: #4b5563; padding: 2px 0; }
        QLabel#errorLabel { background: #fef3c7; color: #92400e;
                            border: 1px solid #fbbf24; border-radius: 6px;
                            padding: 8px; font-family: monospace; }
        """

    def _spin(self, lo, hi, val, step):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(2)
        return s

    def _set_battery_style(self, vbat):
        level = battery_level_name(vbat)
        color = {"good": "#16a34a", "low": "#f59e0b",
                 "critical": "#dc2626", "unknown": "#9ca3af"}[level]
        self.battery_bar.setStyleSheet(
            f"QProgressBar {{ background: #e5e7eb; border: none; "
            f"  border-radius: 5px; height: 10px; min-width: 150px; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 5px; }}"
        )

    def _build_layout(self):
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()

        # Connection
        conn_box = QtWidgets.QGroupBox("Connection")
        conn = QtWidgets.QFormLayout()
        conn.addRow("URI", self.uri_edit)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.connect_btn)
        row.addWidget(self.disconnect_btn)
        conn.addRow(row)
        conn_box.setLayout(conn)
        left.addWidget(conn_box)

        # Map setup
        map_box = QtWidgets.QGroupBox("Map setup (world coords, meters)")
        mform = QtWidgets.QFormLayout()
        wh = QtWidgets.QHBoxLayout()
        wh.addWidget(QtWidgets.QLabel("W"))
        wh.addWidget(self.map_w_spin)
        wh.addWidget(QtWidgets.QLabel("H"))
        wh.addWidget(self.map_h_spin)
        mform.addRow("Size", wh)
        s = QtWidgets.QHBoxLayout()
        s.addWidget(QtWidgets.QLabel("x"))
        s.addWidget(self.start_x_spin)
        s.addWidget(QtWidgets.QLabel("y"))
        s.addWidget(self.start_y_spin)
        mform.addRow("Start", s)
        g = QtWidgets.QHBoxLayout()
        g.addWidget(QtWidgets.QLabel("x"))
        g.addWidget(self.goal_x_spin)
        g.addWidget(QtWidgets.QLabel("y"))
        g.addWidget(self.goal_y_spin)
        mform.addRow("Goal", g)
        mform.addRow("Height", self.height_spin)
        map_btn_row = QtWidgets.QHBoxLayout()
        map_btn_row.addWidget(self.build_btn)
        map_btn_row.addWidget(self.reset_btn)
        mform.addRow(map_btn_row)
        map_box.setLayout(mform)
        left.addWidget(map_box)

        # Flight buttons
        fl_box = QtWidgets.QGroupBox("Flight")
        fl = QtWidgets.QVBoxLayout()
        fl.addWidget(self.start_btn)
        fl.addWidget(self.land_btn)
        fl.addWidget(self.estop_btn)
        fl_box.setLayout(fl)
        left.addWidget(fl_box)

        # Sensors
        sens_box = QtWidgets.QGroupBox("Sensors")
        sens = QtWidgets.QGridLayout()
        for row_i, k in enumerate(("front", "left", "right", "back", "up", "zrange")):
            sens.addWidget(QtWidgets.QLabel(k), row_i, 0)
            sens.addWidget(self.range_labels[k], row_i, 1)
        sens_box.setLayout(sens)
        left.addWidget(sens_box)

        left.addWidget(self.pose_label)

        # Battery
        bat = QtWidgets.QHBoxLayout()
        bat.addWidget(self.battery_label)
        bat.addWidget(self.battery_bar)
        left.addLayout(bat)

        left.addWidget(self.status_label)
        left.addWidget(self.error_label)
        left.addStretch(1)

        # Right: map
        right = QtWidgets.QVBoxLayout()
        map_title = QtWidgets.QLabel(
            "Occupancy Grid (top-down, drone forward = ↑)   "
            "red=occupied · green=free · blue=path · orange=goal distribution · x↑ red, y← green"
        )
        map_title.setStyleSheet("font-weight: 600; padding: 4px;")
        right.addWidget(map_title)
        right.addWidget(self.map_view, 1)

        root.addLayout(left, 0)
        root.addLayout(right, 1)

    # ---- handlers ----
    def on_connect(self):
        self.status_label.setText("Connecting…")
        self.connect_btn.setEnabled(False)
        self.worker.request_connect(self.uri_edit.text().strip())

    def on_disconnect(self):
        self.status_label.setText("Disconnecting…")
        self.worker.request_disconnect()

    def on_start_flight(self):
        # OGM이 이미 있으면 (매핑 중이었으면) 재사용, 없으면 새로 생성
        if self.ogm is None:
            self.ogm = OccupancyGrid(
                self.map_w_spin.value(), self.map_h_spin.value()
            )
        start_xy = (self.start_x_spin.value(), self.start_y_spin.value())
        goal_xy = (self.goal_x_spin.value(), self.goal_y_spin.value())
        self.map_view.set_map(self.ogm, start_xy, goal_xy)
        self.error_label.setText("Flight in progress…")
        self.worker.request_fly(
            self.ogm, start_xy, goal_xy, self.height_spin.value()
        )

    def on_build_toggled(self, checked):
        if checked:
            # 매핑 시작
            if self.ogm is None:
                self.ogm = OccupancyGrid(
                    self.map_w_spin.value(), self.map_h_spin.value()
                )
            start_xy = (self.start_x_spin.value(), self.start_y_spin.value())
            goal_xy = (self.goal_x_spin.value(), self.goal_y_spin.value())
            self.map_view.set_map(self.ogm, start_xy, goal_xy)
            self.worker.request_build_map(self.ogm, start_xy, goal_xy)
            self.build_btn.setText("Stop Mapping")
            self.reset_btn.setEnabled(True)
            # 매핑 중에는 사이즈 변경 잠금 (논리 일관성)
            for w in (self.map_w_spin, self.map_h_spin,
                      self.start_x_spin, self.start_y_spin):
                w.setEnabled(False)
        else:
            self.worker.request_stop_mapping()
            self.build_btn.setText("Build Map")
            for w in (self.map_w_spin, self.map_h_spin,
                      self.start_x_spin, self.start_y_spin):
                w.setEnabled(True)

    def on_reset_map(self):
        self.worker.request_reset_map()
        self.map_view.path_world = []
        self.map_view.landing_world = None
        self.map_view.update()

    def on_land(self):
        self.worker.request_land()
        self.status_label.setText("Landing requested")

    def on_estop(self):
        self.worker.request_emergency_stop()
        self.status_label.setText("E-STOP")

    def on_connection_state(self, state):
        connected = state in ("connected", "flying")
        self.connect_btn.setEnabled(state == "disconnected")
        self.disconnect_btn.setEnabled(connected)
        self.start_btn.setEnabled(state == "connected")
        self.land_btn.setEnabled(state == "flying")
        self.estop_btn.setEnabled(connected)
        # Build Map은 connected에서만 (flying 중에는 자동 매핑이라 토글 무의미)
        self.build_btn.setEnabled(state == "connected")
        # Disconnect 시 매핑 토글도 풀기
        if state == "disconnected" and self.build_btn.isChecked():
            self.build_btn.blockSignals(True)
            self.build_btn.setChecked(False)
            self.build_btn.setText("Build Map")
            self.build_btn.blockSignals(False)
            self.reset_btn.setEnabled(False)
            for w in (self.map_w_spin, self.map_h_spin,
                      self.start_x_spin, self.start_y_spin):
                w.setEnabled(True)

    def on_sensor(self, snap):
        for k in self.range_labels:
            self.range_labels[k].setText(fmt_distance(snap.get(k)))
        x = snap.get("x"); y = snap.get("y"); yaw = snap.get("yaw")
        if x is not None and y is not None:
            self.pose_label.setText(
                f"x: {x:+.3f}  y: {y:+.3f}  yaw: {yaw if yaw is not None else 0.0:+.1f}°"
            )
            if self.ogm is not None:
                start = (self.start_x_spin.value(), self.start_y_spin.value())
                self.map_view.update_drone(start[0] + x, start[1] + y)
        vbat = snap.get("vbat")
        self.battery_label.setText(fmt_battery(vbat))
        self.battery_bar.setValue(battery_percent(vbat))
        self._set_battery_style(vbat)

    def on_flight_result(self, result):
        ex = result.get("error_x", 0.0)
        ey = result.get("error_y", 0.0)
        en = result.get("error_norm", 0.0)
        cd = result.get("commanded_drone", [0, 0])
        md = result.get("measured_drone", [0, 0])
        lw = result.get("landing_world")
        sensor_log = result.get("sensor_log")
        landing_text = (
            f"\nLanding world:      ({lw[0]:+.3f}, {lw[1]:+.3f}) m"
            if lw else ""
        )
        log_text = f"\nSensor log: {sensor_log}" if sensor_log else ""
        text = (
            f"Status:   {result.get('status', '?')}\n"
            f"Cmd (drone-frame):  ({cd[0]:+.3f}, {cd[1]:+.3f}) m\n"
            f"Meas (drone-frame): ({md[0]:+.3f}, {md[1]:+.3f}) m\n"
            f"Error:    dx={ex:+.3f}  dy={ey:+.3f}   ||e||={en:.3f} m"
            f"{landing_text}"
            f"{log_text}"
        )
        self.error_label.setText(text)

    def closeEvent(self, ev):
        self.worker.request_emergency_stop()
        self.worker.request_quit()
        self.worker.wait(2000)
        ev.accept()
