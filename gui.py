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

        self._draw_occupancy(x, y, shared.landing_target)
        self._draw_edges(shared)

        return []

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

        # Region dividers
        for region_x in [config.START_REGION_X - config.TAKEOFF_PAD_X,
                          config.START_REGION_X + config.MIDDLE_REGION_X - config.TAKEOFF_PAD_X]:
            ax.axvline(wx_to_col(region_x), color='yellow', linewidth=0.8,
                       linestyle='--', alpha=0.6)

        # Legend patches
        patches = [
            mpatches.Patch(color='white',   label='Free'),
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

        if config.TAKEOFF_PAD_X is None:
            return

        px, py = config.TAKEOFF_PAD_X, config.TAKEOFF_PAD_Y
        drone_x, drone_y, _, _ = shared.pose

        edges = shared.height_map_edges
        pairs = shared.pad_pairs
        candidates = shared.pad_candidates

        # Edge events (EKF → arena coords)
        if edges:
            ex = [e.x + px for e in edges if e.kind == 'entry']
            ey = [e.y + py for e in edges if e.kind == 'entry']
            ox = [e.x + px for e in edges if e.kind == 'exit']
            oy = [e.y + py for e in edges if e.kind == 'exit']
            if ex:
                ax.scatter(ex, ey, c='#ff4444', s=12, label='Entry', zorder=3)
            if ox:
                ax.scatter(ox, oy, c='#4444ff', s=12, label='Exit', zorder=3)

        # Matched entry-exit pairs: line from entry → exit + midpoint dot
        for i, (cx, cy, enx, eny, exx, exy) in enumerate(pairs):
            lbl = f'Pair ({len(pairs)})' if i == 0 else ''
            ax.plot([enx + px, exx + px], [eny + py, exy + py],
                    color='#ffaa00', lw=1.2, zorder=4, label=lbl)

        # Landing target (arena coords) — must be read before the candidate block
        landing_target = shared.landing_target

        # Pad candidates — hidden once landing target is confirmed
        if landing_target is None:
            for cand in candidates:
                cx_a, cy_a = cand.cx + px, cand.cy + py
                ax.plot(cx_a, cy_a, 'g*', markersize=16, zorder=5,
                        label='Pad candidate')
                ax.add_patch(plt.Circle((cx_a, cy_a), config.PAD_SIZE / 2,
                                        color='lime', fill=False,
                                        linewidth=1.2, zorder=4))
        if landing_target is not None:
            lx, ly = landing_target[0] + px, landing_target[1] + py
            ax.plot(lx, ly, 'D', color='#ff8800', markersize=10, zorder=7,
                    label='Landing target')
            ax.add_patch(plt.Circle((lx, ly), config.PAD_SIZE / 2,
                                    color='#ff8800', fill=False,
                                    linewidth=1.5, zorder=6))

        # Drone marker
        ax.plot(drone_x + px, drone_y + py,
                'co', markersize=7, zorder=5, label='Drone')

        # Landing region only
        land_x0 = config.START_REGION_X + config.MIDDLE_REGION_X
        ax.set_xlim(land_x0 - 0.1, config.ARENA_X + 0.1)
        ax.set_ylim(-0.1, config.ARENA_Y + 0.1)
        x_ticks = [x for x in range(int(land_x0), int(config.ARENA_X) + 1)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks], fontsize=7)
        ax.set_yticks(range(int(config.ARENA_Y) + 1))
        ax.set_yticklabels([str(y) for y in range(int(config.ARENA_Y) + 1)],
                           fontsize=7)
        ax.set_xlabel('x (m)', color='white', fontsize=8)
        ax.set_ylabel('y (m)', color='white', fontsize=8)
        ax.set_aspect('equal', adjustable='box')

        if edges or candidates or landing_target is not None:
            ax.legend(loc='upper right', fontsize=6,
                      facecolor='#333333', labelcolor='white', framealpha=0.8)
