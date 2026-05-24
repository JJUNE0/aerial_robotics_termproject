import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import numpy as np

import config
from mapping import FREE, UNKNOWN, OCCUPIED, INFLATED
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
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('#1a1a1a')
        self._fig = fig

        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[0.10, 0.82, 0.08],
                               hspace=0.35, wspace=0.3)

        # ---- status bar (top row, spans both columns)
        ax_status = fig.add_subplot(gs[0, :])
        ax_status.set_axis_off()
        self._status_txt = ax_status.text(
            0.01, 0.5, '',
            transform=ax_status.transAxes,
            color='white', fontsize=10.5, va='center',
            fontfamily='monospace',
        )

        # ---- occupancy map (left)
        self._ax_occ = fig.add_subplot(gs[1, 0])
        self._ax_occ.set_title('Occupancy Map', color='white', fontsize=10)

        # ---- height map / edge scatter (right)
        self._ax_hm = fig.add_subplot(gs[1, 1])
        self._ax_hm.set_title('Edge Map & Pad Candidates', color='white', fontsize=10)

        # ---- emergency button (bottom row)
        btn_axes = plt.axes([0.38, 0.01, 0.24, 0.055])
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

        self._draw_occupancy(x, y)
        self._draw_edges(shared)

        return []

    # ---------------------------------------------------------------- occupancy

    def _draw_occupancy(self, drone_x: float, drone_y: float):
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

        # Drone marker
        dc = wx_to_col(drone_x)
        dr = wy_to_row(drone_y)
        ax.plot(dc, dr, 'co', markersize=7, zorder=5)

        # Region dividers
        for region_x in [config.START_REGION_X - config.TAKEOFF_PAD_X,
                          config.START_REGION_X + config.MIDDLE_REGION_X - config.TAKEOFF_PAD_X]:
            ax.axvline(wx_to_col(region_x), color='yellow', linewidth=0.8,
                       linestyle='--', alpha=0.6)

        # Legend patches
        patches = [
            mpatches.Patch(color='white', label='Free'),
            mpatches.Patch(color='#b3b3b3', label='Unknown'),
            mpatches.Patch(color='#1a1a1a', label='Occupied'),
            mpatches.Patch(color='#737373', label='Inflated'),
        ]
        ax.legend(handles=patches, loc='upper right', fontsize=6,
                  facecolor='#333333', labelcolor='white', framealpha=0.8)

    # ---------------------------------------------------------------- edges

    def _draw_edges(self, shared: SharedState):
        ax = self._ax_hm
        ax.clear()
        ax.set_title('Edge Map & Pad Candidates', color='white', fontsize=10)
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')

        edges = shared.height_map_edges
        candidates = shared.pad_candidates

        if edges:
            ex = [e.x for e in edges if e.kind == 'entry']
            ey = [e.y for e in edges if e.kind == 'entry']
            ox = [e.x for e in edges if e.kind == 'exit']
            oy = [e.y for e in edges if e.kind == 'exit']
            if ex:
                ax.scatter(ex, ey, c='#ff4444', s=12, label='Entry', zorder=3)
            if ox:
                ax.scatter(ox, oy, c='#4444ff', s=12, label='Exit', zorder=3)

        for cand in candidates:
            ax.plot(cand.cx, cand.cy, 'g*', markersize=16, zorder=5, label='Pad candidate')
            circle = plt.Circle((cand.cx, cand.cy), config.PAD_SIZE / 2,
                                 color='lime', fill=False, linewidth=1.2, zorder=4)
            ax.add_patch(circle)

        if edges or candidates:
            ax.legend(loc='upper right', fontsize=7,
                      facecolor='#333333', labelcolor='white', framealpha=0.8)

        ax.set_xlabel('x (m)', color='white', fontsize=8)
        ax.set_ylabel('y (m)', color='white', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
