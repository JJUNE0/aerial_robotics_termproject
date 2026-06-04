import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import Button

import numpy as np

import config
from mapping import FREE, UNKNOWN, OCCUPIED, INFLATED, compute_diff_clusters
from shared_state import SharedState

_CELL_RGB = {
    FREE:     [1.0, 1.0, 1.0],   # white
    UNKNOWN:  [0.7, 0.7, 0.7],   # light grey
    OCCUPIED: [0.1, 0.1, 0.1],   # near-black
    INFLATED: [0.45, 0.45, 0.45], # dark grey
}


class MissionGUI:
    def __init__(self, shared: SharedState):
        self._shared = shared

    def start(self):
        """Build the figure and block on plt.show() — call from main thread."""
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#1a1a1a')
        self._fig = fig

        gs = gridspec.GridSpec(4, 2, figure=fig,
                               height_ratios=[0.08, 0.43, 0.43, 0.06],
                               hspace=0.40, wspace=0.3)

        # ---- status bar (top row, spans both columns)
        ax_status = fig.add_subplot(gs[0, :])
        ax_status.set_axis_off()
        self._status_txt = ax_status.text(
            0.01, 0.5, '',
            transform=ax_status.transAxes,
            color='white', fontsize=10.5, va='center',
            fontfamily='monospace',
        )

        # ---- occupancy map (nav, top-left)
        self._ax_occ = fig.add_subplot(gs[1, 0])
        self._ax_occ.set_title('Navigation Map', color='white', fontsize=10)

        # ---- high-altitude scan map (top-right)
        self._ax_scan_high = fig.add_subplot(gs[1, 1])
        self._ax_scan_high.set_title('High-Alt Scan Map (0.30 m)', color='white', fontsize=10)

        # ---- low-altitude scan map (bottom-left)
        self._ax_hm = fig.add_subplot(gs[2, 0])
        self._ax_hm.set_title('Low-Alt Scan Map (0.08 m)', color='white', fontsize=10)

        # ---- diff map (bottom-right)
        self._ax_diff = fig.add_subplot(gs[2, 1])
        self._ax_diff.set_title('Diff Map (elevated objects)', color='white', fontsize=10)

        # ---- emergency button (bottom row)
        btn_axes = plt.axes([0.38, 0.01, 0.24, 0.045])
        self._btn = Button(btn_axes, 'EMERGENCY LAND',
                           color='#aa0000', hovercolor='#ff2222')
        self._btn.label.set_color('white')
        self._btn.label.set_fontsize(10)
        self._btn.label.set_fontweight('bold')
        self._btn.on_clicked(self._on_emergency)

        self._ani = FuncAnimation(fig, self._update, interval=250,
                                  cache_frame_data=False)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()

    # ---------------------------------------------------------------- callbacks

    def _on_emergency(self, _event):
        self._shared.trigger_emergency()

    def _update(self, _frame):
        shared = self._shared
        x, y, z, yaw = shared.pose
        state = shared.current_state
        elapsed = shared.elapsed_time
        batt = shared.battery_pct
        remaining = max(0.0, config.MISSION_TIME_LIMIT - elapsed)

        self._status_txt.set_text(
            f'State: {state:<22}  '
            f'Elapsed: {int(elapsed):3d}s  Remaining: {int(remaining):3d}s  '
            f'Battery: {batt:5.1f}%  '
            f'Pos: ({x:5.2f}, {y:5.2f}, {z:5.2f})  Yaw: {yaw:6.1f}°'
        )

        self._draw_occupancy(x, y, shared.landing_target)
        self._draw_scan_map(self._ax_scan_high, shared.occ_scan_high_grid,
                            'High-Alt Scan Map (0.30 m)', shared)
        self._draw_scan_map(self._ax_hm, shared.occ_low_grid,
                            'Low-Alt Scan Map (0.08 m)', shared)
        self._draw_diff(shared)

        return []

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _setup_map_grid(ax):
        """Add 10 cm minor grid lines to a map axes."""
        step = 0.1 / config.OCCUPANCY_GRID_RES
        ax.xaxis.set_minor_locator(MultipleLocator(step))
        ax.yaxis.set_minor_locator(MultipleLocator(step))
        ax.grid(True, which='minor', color='#2e2e2e', linewidth=0.3, zorder=2)
        ax.grid(True, which='major', color='#444444', linewidth=0.5, zorder=2)

    def _draw_region_lines(self, ax, wx_to_col, wy_to_row):
        """Draw start/middle/landing region dividers with labels on any map axes."""
        x1 = config.START_REGION_X - config.TAKEOFF_PAD_X
        x2 = config.START_REGION_X + config.MIDDLE_REGION_X - config.TAKEOFF_PAD_X
        y_top = config.ekf_arena_y_max()

        for xv in (x1, x2):
            ax.axvline(wx_to_col(xv), color='yellow', linewidth=0.8,
                       linestyle='--', alpha=0.7, zorder=4)

        # Region label centres (arena x midpoints → EKF)
        regions = [
            ((config.ekf_arena_x_min() + x1) / 2, 'START'),
            ((x1 + x2) / 2,                        'MIDDLE'),
            ((x2 + config.ekf_arena_x_max()) / 2,  'LANDING'),
        ]
        for rx, label in regions:
            ax.text(wx_to_col(rx), wy_to_row(y_top) - 2,
                    label, color='yellow', fontsize=6,
                    ha='center', va='top', alpha=0.85, zorder=5)

    # ---------------------------------------------------------------- occupancy

    def _draw_occupancy(self, drone_x: float, drone_y: float, landing_target=None):
        ax = self._ax_occ
        ax.clear()
        ax.set_title('Occupancy Map', color='white', fontsize=10)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')

        grid = self._shared.occupancy_grid
        if grid is None:
            return

        rows, cols = grid.shape
        img = np.zeros((rows, cols, 3), dtype=float)
        for val, rgb in _CELL_RGB.items():
            mask = grid == val
            img[mask] = rgb

        ax.imshow(img, origin='lower', aspect='equal', interpolation='nearest')

        if config.TAKEOFF_PAD_X is None:
            return

        res = config.OCCUPANCY_GRID_RES
        x_min = config.ekf_arena_x_min() - 0.5
        y_min = config.ekf_arena_y_min() - 0.5

        def wx_to_col(wx):
            return (wx - x_min) / res

        def wy_to_row(wy):
            return (wy - y_min) / res

        # ---- Clip view to exact arena bounds (5 m × 3 m)
        ax.set_xlim(wx_to_col(config.ekf_arena_x_min()),
                    wx_to_col(config.ekf_arena_x_max()))
        ax.set_ylim(wy_to_row(config.ekf_arena_y_min()),
                    wy_to_row(config.ekf_arena_y_max()))

        # Ticks in arena frame: (0,0) = arena bottom-left corner
        # Arena coord i  →  EKF coord (i - TAKEOFF_PAD_X / _Y)
        ax.set_xticks([wx_to_col(i - config.TAKEOFF_PAD_X)
                       for i in range(int(config.ARENA_X) + 1)])
        ax.set_xticklabels([str(i) for i in range(int(config.ARENA_X) + 1)],
                           fontsize=7)
        ax.set_xlabel('x (m)', color='white', fontsize=8)

        ax.set_yticks([wy_to_row(i - config.TAKEOFF_PAD_Y)
                       for i in range(int(config.ARENA_Y) + 1)])
        ax.set_yticklabels([str(i) for i in range(int(config.ARENA_Y) + 1)],
                           fontsize=7)
        ax.set_ylabel('y (m)', color='white', fontsize=8)
        self._setup_map_grid(ax)

        # Drone marker (EKF pos → same col/row mapping)
        ax.plot(wx_to_col(drone_x), wy_to_row(drone_y),
                'co', markersize=7, zorder=5, label='Drone')

        # Target marker — drawn only while navigating
        target = self._shared.target_pos
        if target is not None:
            tx, ty = target
            ax.plot(wx_to_col(tx), wy_to_row(ty),
                    'rx', markersize=9, markeredgewidth=2, zorder=6,
                    label='Target')
            ax.annotate('', xy=(wx_to_col(tx), wy_to_row(ty)),
                        xytext=(wx_to_col(drone_x), wy_to_row(drone_y)),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=1.2, alpha=0.7),
                        zorder=5)

        # Landing target — shown once confirmed, persists until mission ends
        if landing_target is not None:
            lx, ly = landing_target
            ax.plot(wx_to_col(lx), wy_to_row(ly),
                    'D', color='#ff8800', markersize=9, zorder=7,
                    label='Landing target')
            ax.add_patch(plt.Circle(
                (wx_to_col(lx), wy_to_row(ly)),
                config.PAD_SIZE / 2 / config.OCCUPANCY_GRID_RES,
                color='#ff8800', fill=False, linewidth=1.5, zorder=6))

        # Takeoff-pad marker at EKF origin
        ax.plot(wx_to_col(0.0), wy_to_row(0.0),
                'y^', markersize=6, zorder=5, label='Takeoff pad')

        # Frontier target (yellow star)
        frontier = self._shared.frontier_target
        if frontier is not None:
            fx, fy = frontier
            ax.plot(wx_to_col(fx), wy_to_row(fy),
                    '*', color='yellow', markersize=14, zorder=8, label='Frontier')

        # A* waypoints — cyan for outward, orange for return
        waypoints = self._shared.nav_waypoints
        if waypoints:
            ret_states = {'RET_WAYPOINT', 'NAV_TO_START'}
            wp_color = '#ff8c00' if self._shared.current_state in ret_states else 'cyan'
            wc = [wx_to_col(wx) for wx, _ in waypoints]
            wr = [wy_to_row(wy) for _, wy in waypoints]
            all_c = [wx_to_col(drone_x)] + wc
            all_r = [wy_to_row(drone_y)] + wr
            ax.plot(all_c, all_r, color=wp_color, lw=0.9,
                    linestyle='--', alpha=0.7, zorder=4)
            ax.scatter(wc, wr, c=wp_color, s=12, zorder=5, marker='o')

        # Region dividers + labels
        self._draw_region_lines(ax, wx_to_col, wy_to_row)

        # Legend patches
        patches = [
            mpatches.Patch(color='white',   label='Free'),
            mpatches.Patch(color='#b3b3b3', label='Unknown'),
            mpatches.Patch(color='#1a1a1a', label='Occupied'),
            mpatches.Patch(color='#737373', label='Inflated'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=6,
                  facecolor='#333333', labelcolor='white', framealpha=0.8)

    # ---------------------------------------------------------------- scan maps

    def _draw_scan_map(self, ax, grid, title: str, shared: SharedState):
        ax.clear()
        ax.set_title(title, color='white', fontsize=10)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')

        if grid is None:
            ax.text(0.5, 0.5, 'Not yet available',
                    transform=ax.transAxes, color='#888888',
                    fontsize=10, ha='center', va='center')
            return

        rows, cols = grid.shape
        img = np.zeros((rows, cols, 3), dtype=float)
        for val, rgb in _CELL_RGB.items():
            img[grid == val] = rgb
        ax.imshow(img, origin='lower', aspect='equal', interpolation='nearest')

        res = config.OCCUPANCY_GRID_RES
        x_min = config.ekf_arena_x_min() - 0.5
        y_min = config.ekf_arena_y_min() - 0.5

        def wx_to_col(wx): return (wx - x_min) / res
        def wy_to_row(wy): return (wy - y_min) / res

        ax.set_xlim(wx_to_col(config.ekf_arena_x_min()),
                    wx_to_col(config.ekf_arena_x_max()))
        ax.set_ylim(wy_to_row(config.ekf_arena_y_min()),
                    wy_to_row(config.ekf_arena_y_max()))
        ax.set_xticks([wx_to_col(i - config.TAKEOFF_PAD_X)
                       for i in range(int(config.ARENA_X) + 1)])
        ax.set_xticklabels([str(i) for i in range(int(config.ARENA_X) + 1)],
                           fontsize=7)
        ax.set_yticks([wy_to_row(i - config.TAKEOFF_PAD_Y)
                       for i in range(int(config.ARENA_Y) + 1)])
        ax.set_yticklabels([str(i) for i in range(int(config.ARENA_Y) + 1)],
                           fontsize=7)
        ax.set_xlabel('x (m)', color='white', fontsize=8)
        ax.set_ylabel('y (m)', color='white', fontsize=8)
        self._setup_map_grid(ax)

        self._draw_region_lines(ax, wx_to_col, wy_to_row)

        drone_x, drone_y, _, _ = shared.pose
        ax.plot(wx_to_col(drone_x), wy_to_row(drone_y),
                'co', markersize=6, zorder=5)

        landing_target = shared.landing_target
        if landing_target is not None:
            lx, ly = landing_target
            ax.plot(wx_to_col(lx), wy_to_row(ly),
                    'D', color='#ff8800', markersize=8, zorder=7)
            ax.add_patch(plt.Circle(
                (wx_to_col(lx), wy_to_row(ly)),
                config.PAD_SIZE / 2 / res,
                color='#ff8800', fill=False, linewidth=1.5, zorder=6))

        align_pos = shared.landing_align_pos
        if align_pos is not None:
            ax.axvline(wx_to_col(align_pos[0]), color='#00ccff',
                       linewidth=1.2, linestyle='--', alpha=0.85, zorder=8)

        patches = [
            mpatches.Patch(color='white',   label='Free'),
            mpatches.Patch(color='#b3b3b3', label='Unknown'),
            mpatches.Patch(color='#1a1a1a', label='Occupied'),
            mpatches.Patch(color='#737373', label='Inflated'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=6,
                  facecolor='#333333', labelcolor='white', framealpha=0.8)

    # ---------------------------------------------------------------- diff map

    def _draw_diff(self, shared: SharedState):
        ax = self._ax_diff
        ax.clear()
        ax.set_title('Diff Map (elevated objects)', color='white', fontsize=10)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')

        diff = shared.occ_diff_grid
        if diff is None:
            ax.text(0.5, 0.5, 'Not yet available',
                    transform=ax.transAxes, color='#888888',
                    fontsize=10, ha='center', va='center')
            return

        rows, cols = diff.shape
        # White = elevated object (diff=1), dark = background (diff=0)
        img = np.zeros((rows, cols, 3), dtype=float)
        img[diff == 0] = [0.15, 0.15, 0.15]
        img[diff == 1] = [1.0,  0.6,  0.1]   # orange = elevated cell

        ax.imshow(img, origin='lower', aspect='equal', interpolation='nearest')

        res = config.OCCUPANCY_GRID_RES
        x_min = config.ekf_arena_x_min() - 0.5
        y_min = config.ekf_arena_y_min() - 0.5

        def wx_to_col(wx): return (wx - x_min) / res
        def wy_to_row(wy): return (wy - y_min) / res

        ax.set_xlim(wx_to_col(config.ekf_arena_x_min()),
                    wx_to_col(config.ekf_arena_x_max()))
        ax.set_ylim(wy_to_row(config.ekf_arena_y_min()),
                    wy_to_row(config.ekf_arena_y_max()))
        ax.set_xticks([wx_to_col(i - config.TAKEOFF_PAD_X)
                       for i in range(int(config.ARENA_X) + 1)])
        ax.set_xticklabels([str(i) for i in range(int(config.ARENA_X) + 1)],
                           fontsize=7)
        ax.set_yticks([wy_to_row(i - config.TAKEOFF_PAD_Y)
                       for i in range(int(config.ARENA_Y) + 1)])
        ax.set_yticklabels([str(i) for i in range(int(config.ARENA_Y) + 1)],
                           fontsize=7)
        ax.set_xlabel('x (m)', color='white', fontsize=8)
        ax.set_ylabel('y (m)', color='white', fontsize=8)
        self._setup_map_grid(ax)

        self._draw_region_lines(ax, wx_to_col, wy_to_row)

        drone_x, drone_y, _, _ = shared.pose
        ax.plot(wx_to_col(drone_x), wy_to_row(drone_y),
                'co', markersize=6, zorder=5)

        landing_target = shared.landing_target
        if landing_target is not None:
            lx, ly = landing_target
            ax.plot(wx_to_col(lx), wy_to_row(ly),
                    'D', color='#ff8800', markersize=8, zorder=7)
            ax.add_patch(plt.Circle(
                (wx_to_col(lx), wy_to_row(ly)),
                config.PAD_SIZE / 2 / res,
                color='#ff8800', fill=False, linewidth=1.5, zorder=6))

        align_pos = shared.landing_align_pos
        if align_pos is not None:
            ax.axvline(wx_to_col(align_pos[0]), color='#00ccff',
                       linewidth=1.2, linestyle='--', alpha=0.85,
                       zorder=8, label='X-align')

        # Cluster bounding boxes
        clusters = compute_diff_clusters(diff, res)
        first_pad = True
        first_rej = True
        for cl in clusters:
            color = '#00ff88' if cl['is_pad'] else '#ff4444'
            lbl = None
            if cl['is_pad'] and first_pad:
                lbl = 'Pad cand.'
                first_pad = False
            elif not cl['is_pad'] and first_rej:
                lbl = 'Rejected'
                first_rej = False
            ax.add_patch(mpatches.Rectangle(
                (cl['col_min'], cl['row_min']),
                cl['col_max'] - cl['col_min'],
                cl['row_max'] - cl['row_min'],
                linewidth=1.5, edgecolor=color,
                facecolor='none', zorder=8, label=lbl))
            ax.text(cl['col_min'], cl['row_max'] + 1,
                    f'X={cl["w_m"]*100:.0f} Y={cl["h_m"]*100:.0f}cm',
                    color=color, fontsize=5.5, zorder=9, va='bottom')

        patches = [
            mpatches.Patch(color='#ff9900', label='Elevated (pad/bar)'),
            mpatches.Patch(color='#262626', label='Background'),
            mpatches.Patch(color='#00ff88', fill=False, label='Pad cand.'),
            mpatches.Patch(color='#ff4444', fill=False, label='Rejected'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=6,
                  facecolor='#333333', labelcolor='white', framealpha=0.8)
