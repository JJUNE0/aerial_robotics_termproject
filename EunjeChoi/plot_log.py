"""
Usage:
  python plot_log.py              # prompts for filename, defaults to latest
  python plot_log.py log/file.csv # specific file
"""

import sys
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
from matplotlib.ticker import MultipleLocator
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import config
from mapping import compute_diff_clusters as _compute_diff_clusters


def compute_diff_clusters(diff_npz) -> list:
    """Wrapper: accepts an npz dict and delegates to mapping.compute_diff_clusters."""
    if diff_npz is None:
        return []
    return _compute_diff_clusters(diff_npz['grid'], float(diff_npz['res'][0]))


# ------------------------------------------------------------------ file I/O

def pick_file() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]

    log_dir = 'log'
    files = sorted(glob.glob(os.path.join(log_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f'No CSV files in {log_dir}/')

    print('Available log files:')
    for i, f in enumerate(files):
        print(f'  [{i}] {os.path.basename(f)}')

    raw = input(f'Select file number (Enter = latest [{len(files)-1}]): ').strip()
    idx = int(raw) if raw else len(files) - 1
    return files[idx]


def load_occ(csv_path: str):
    """Load occupancy map saved alongside the CSV. Returns dict or None."""
    occ_path = csv_path.replace('.csv', '_occ.npz')
    if not os.path.exists(occ_path):
        return None
    return np.load(occ_path)


def load_maps(csv_path: str) -> dict:
    """Load all saved map npz files for this log. Keys: occ, scan_high, occ_low, diff."""
    suffixes = {
        'occ':       '_occ.npz',
        'scan_high': '_occ_scan_high.npz',
        'occ_low':   '_occ_low.npz',
        'diff':      '_occ_diff.npz',
    }
    result = {}
    for key, suffix in suffixes.items():
        path = csv_path.replace('.csv', suffix)
        if os.path.exists(path):
            result[key] = np.load(path)
            print(f'  {key} map loaded: {result[key]["grid"].shape}')
        else:
            result[key] = None
            print(f'  {key} map: not found')
    return result


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['z_down_m'] = pd.to_numeric(df['z_down_m'], errors='coerce')
    if 'state' not in df.columns:
        df['state'] = ''
    for col in ('yaw_deg', 'roll_deg', 'pitch_deg', 'yaw_ref_deg',
                'vx_ms', 'vy_ms', 'vz_ms',
                'range_front_m', 'range_back_m',
                'range_left_m', 'range_right_m', 'range_up_m',
                'target_x_m', 'target_y_m'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
    return df.dropna(subset=['z_down_m'])


# ------------------------------------------------------------------ edge detection

def detect_edges(df: pd.DataFrame,
                 baseline: float = config.EDGE_BASELINE,
                 entry_dip: float = config.EDGE_ENTRY_DIP,
                 exit_rise: float = config.EDGE_EXIT_RISE,
                 min_dz: float = config.EDGE_MIN_DZ,
                 cooldown: float = config.EDGE_COOLDOWN,
                 states: set = None):
    """Return (entries, exits) each as list of (time, x, y).

    states: if given, only detect edges while row['state'] is in that set.
            Detector resets at every state boundary to avoid cross-state transitions.
    """
    entries, exits = [], []

    prev_z = prev_x = prev_y = prev_t = None
    last_dir = 0
    cooldown_until = -1.0

    for _, row in df.iterrows():
        z = row['z_down_m']
        t, x, y = row['time_s'], row['x_m'], row['y_m']

        # State filter: skip rows outside target states, reset on boundary
        if states is not None and row.get('state', '') not in states:
            prev_z = None
            last_dir = 0
            continue

        if prev_z is None:
            prev_z, prev_x, prev_y, prev_t = z, x, y, t
            continue

        dz = z - prev_z

        if abs(dz) > min_dz:
            cur_dir = 1 if dz > 0 else -1

            if t > cooldown_until:
                if last_dir < 0 and cur_dir > 0:
                    if prev_z <= baseline - entry_dip:
                        entries.append((prev_t, prev_x, prev_y))
                        cooldown_until = prev_t + cooldown

                elif last_dir > 0 and cur_dir < 0:
                    if prev_z >= baseline + exit_rise:
                        exits.append((prev_t, prev_x, prev_y))
                        cooldown_until = prev_t + cooldown

            last_dir = cur_dir

        prev_z, prev_x, prev_y, prev_t = z, x, y, t

    return entries, exits


# ------------------------------------------------------------------ pair matching

def compute_pairs(entries, exits):
    """Match entry-exit pairs: returns (cx, cy, enx, eny, exx, exy)."""
    used = set()
    pairs = []
    for _, exx, exy in exits:
        best_j, best_d = None, config.SCAN_ROW_SPACING
        for j, (_, enx, eny) in enumerate(entries):
            if j in used:
                continue
            d = abs(enx - exx)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            continue
        used.add(best_j)
        _, enx, eny = entries[best_j]
        pairs.append(((enx + exx) / 2.0, (eny + exy) / 2.0,
                       enx, eny, exx, exy))
    return pairs


# ------------------------------------------------------------------ plot

def plot(df: pd.DataFrame, title: str, occ_data=None):
    t = df['time_s'].values
    # Convert EKF frame → arena frame
    x = df['x_m'].values + config.TAKEOFF_PAD_X
    y = df['y_m'].values + config.TAKEOFF_PAD_Y
    z = df['z_down_m'].values

    # Target positions (unique, EKF → arena)
    targets = []
    if 'target_x_m' in df.columns:
        tdf = df[df['target_x_m'] != ''][['target_x_m', 'target_y_m']].drop_duplicates()
        if len(tdf):
            tdf = tdf.apply(pd.to_numeric, errors='coerce').dropna()
            targets = list(zip(
                tdf['target_x_m'] + config.TAKEOFF_PAD_X,
                tdf['target_y_m'] + config.TAKEOFF_PAD_Y,
            ))

    entries, exits = detect_edges(df, states={'LANDING_ON_PAD'})
    # Apply arena offset to edge event positions
    entries = [(te, ex + config.TAKEOFF_PAD_X, ey + config.TAKEOFF_PAD_Y)
               for te, ex, ey in entries]
    exits   = [(te, ex + config.TAKEOFF_PAD_X, ey + config.TAKEOFF_PAD_Y)
               for te, ex, ey in exits]
    pairs = compute_pairs(entries, exits)

    # ---- figure layout: white background throughout
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(title, color='black', fontsize=11)

    # ---- left: z_down vs time
    ax1.set_facecolor('white')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#aaa')
    ax1.tick_params(axis='both', which='both', colors='black', labelcolor='black')

    ax1.plot(t, z, color='#1a6faf', lw=1.2, zorder=2, label='z_down')

    # baseline
    ax1.axhline(config.EDGE_BASELINE, color='#888888', lw=1.0, linestyle='--',
                alpha=0.7, label=f'baseline ({config.EDGE_BASELINE:.2f} m)')

    for te, *_ in entries:
        ax1.axvline(te, color='#cc2222', lw=1.2, linestyle='--', alpha=0.9,
                    label='entry' if te == entries[0][0] else '')
    for te, *_ in exits:
        ax1.axvline(te, color='#2244cc', lw=1.2, linestyle=':', alpha=0.9,
                    label='exit' if te == exits[0][0] else '')

    ax1.set_xlabel('time (s)', color='black')
    ax1.set_ylabel('z_down (m)', color='black')
    ax1.set_title('z-ranger (down) vs Time', color='black')
    ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.grid(True, which='major', color='#cccccc', linewidth=0.7, zorder=0)
    ax1.grid(True, which='minor', color='#eeeeee', linewidth=0.4, zorder=0)
    ax1.legend(fontsize=8, facecolor='white', labelcolor='black', framealpha=0.9)

    # ---- right: XY scatter
    ax2.set_facecolor('white')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#aaa')
    ax2.tick_params(axis='both', which='both', colors='black', labelcolor='black')

    # ---- occupancy map overlay
    if occ_data is not None:
        grid  = occ_data['grid']          # shape (rows=Y, cols=X)
        res   = float(occ_data['res'][0])
        x_min = float(occ_data['x_min'][0]) + config.TAKEOFF_PAD_X  # EKF → arena
        y_min = float(occ_data['y_min'][0]) + config.TAKEOFF_PAD_Y
        n_rows, n_cols = grid.shape
        x_max = x_min + n_cols * res
        y_max = y_min + n_rows * res

        # RGBA: FREE=transparent, UNKNOWN=light gray, OCCUPIED=dark, INFLATED=mid gray
        rgba = np.zeros((*grid.shape, 4), dtype=float)
        rgba[grid == 1] = [0.80, 0.80, 0.80, 0.40]   # UNKNOWN
        rgba[grid == 2] = [0.15, 0.15, 0.15, 0.75]   # OCCUPIED
        rgba[grid == 3] = [0.55, 0.55, 0.55, 0.40]   # INFLATED

        ax2.imshow(rgba, origin='lower',
                   extent=[x_min, x_max, y_min, y_max],
                   aspect='auto', zorder=1)

    sc = ax2.scatter(x, y, c=z, cmap='Reds',
                     s=20, vmin=z.min(), vmax=z.max(), zorder=2)
    cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('z_down (m)', color='black')
    cb.ax.yaxis.set_tick_params(color='black')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')

    # target positions
    if targets:
        tx_arr, ty_arr = zip(*targets)
        ax2.scatter(tx_arr, ty_arr, marker='*', color='gold', s=80,
                    edgecolors='orange', linewidths=0.5,
                    zorder=6, label=f'target ({len(targets)})')

    # entry/exit markers on XY
    if entries:
        ax2.scatter([e[1] for e in entries], [e[2] for e in entries],
                    color='red', marker='+', s=80, linewidths=2,
                    zorder=5, label=f'entry ({len(entries)})')
    if exits:
        ax2.scatter([e[1] for e in exits], [e[2] for e in exits],
                    color='blue', marker='x', s=80, linewidths=2,
                    zorder=5, label=f'exit ({len(exits)})')

    # Matched pairs: entry → exit line + midpoint diamond
    for i, (_, _, enx, eny, exx, exy) in enumerate(pairs):
        lbl = f'Pair ({len(pairs)})' if i == 0 else ''
        ax2.plot([enx, exx], [eny, exy],
                 color='#ff8800', lw=1.5, zorder=4, label=lbl)

    ax2.plot(x[0],  y[0],  '^', color='green', ms=9, zorder=6, label='Start')
    ax2.plot(x[-1], y[-1], 's', color='black', ms=9, zorder=6, label='End')
    ax2.set_xlabel('x (m)', color='black')
    ax2.set_ylabel('y (m)', color='black')
    ax2.set_title('XY Trajectory  (colour = z_down)', color='black')
    ax2.set_xlim(0, config.ARENA_X)
    ax2.set_ylim(0, config.ARENA_Y)
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(axis='both', which='both', labelsize=8,
                    colors='black', labelcolor='black')
    ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax2.grid(True, which='major', color='#cccccc', linewidth=0.7, zorder=0)
    ax2.grid(True, which='minor', color='#eeeeee', linewidth=0.4, zorder=0)
    ax2.legend(fontsize=8, facecolor='white', labelcolor='black',
               framealpha=0.9, edgecolor='#aaa')

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ 4-map view

_CELL_RGB = {
    0: [1.0, 1.0, 1.0],          # FREE  — white
    1: [0.75, 0.75, 0.75],       # UNKNOWN — light grey
    2: [0.12, 0.12, 0.12],       # OCCUPIED — near-black
    3: [0.50, 0.50, 0.50],       # INFLATED — mid grey
}


def _draw_map_ax(ax, npz, title: str,
                 df: pd.DataFrame, is_diff: bool = False,
                 align_x_col=None):
    """Render one map panel (occupancy or diff) onto ax."""
    import matplotlib.patches as mpatches

    ax.set_facecolor('#1a1a1a')
    ax.set_title(title, color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=7)

    res  = config.OCCUPANCY_GRID_RES
    x_min_map = config.ekf_arena_x_min() - 0.5
    y_min_map = config.ekf_arena_y_min() - 0.5

    def wx_to_col(wx): return (wx - x_min_map) / res
    def wy_to_row(wy): return (wy - y_min_map) / res

    if npz is not None:
        grid = npz['grid']
        rows, cols = grid.shape
        if is_diff:
            img = np.zeros((rows, cols, 3), dtype=float)
            img[grid == 0] = [0.15, 0.15, 0.15]
            img[grid == 1] = [1.0,  0.6,  0.1]
        else:
            img = np.zeros((rows, cols, 3), dtype=float)
            for val, rgb in _CELL_RGB.items():
                img[grid == val] = rgb
        ax.imshow(img, origin='lower', aspect='equal', interpolation='nearest')
    else:
        ax.text(0.5, 0.5, 'Not available', transform=ax.transAxes,
                color='#888888', fontsize=9, ha='center', va='center')

    # Arena bounds
    ax.set_xlim(wx_to_col(config.ekf_arena_x_min()),
                wx_to_col(config.ekf_arena_x_max()))
    ax.set_ylim(wy_to_row(config.ekf_arena_y_min()),
                wy_to_row(config.ekf_arena_y_max()))

    ax.set_xticks([wx_to_col(i - config.TAKEOFF_PAD_X)
                   for i in range(int(config.ARENA_X) + 1)])
    ax.set_xticklabels([str(i) for i in range(int(config.ARENA_X) + 1)],
                       fontsize=7, color='white')
    ax.set_yticks([wy_to_row(i - config.TAKEOFF_PAD_Y)
                   for i in range(int(config.ARENA_Y) + 1)])
    ax.set_yticklabels([str(i) for i in range(int(config.ARENA_Y) + 1)],
                       fontsize=7, color='white')
    ax.set_xlabel('x (m)', color='white', fontsize=8)
    ax.set_ylabel('y (m)', color='white', fontsize=8)

    # 10 cm minor grid
    minor_step = 0.1 / res
    ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.grid(True, which='minor', color='#3a3a3a', linewidth=0.3, zorder=2)
    ax.grid(True, which='major', color='#555555', linewidth=0.6, zorder=2)

    # Region dividers
    x1_ekf = config.START_REGION_X - config.TAKEOFF_PAD_X
    x2_ekf = config.START_REGION_X + config.MIDDLE_REGION_X - config.TAKEOFF_PAD_X
    y_top  = config.ekf_arena_y_max()
    for xv in (x1_ekf, x2_ekf):
        ax.axvline(wx_to_col(xv), color='yellow', lw=0.8, ls='--', alpha=0.7)
    for rx, lbl in [
        ((config.ekf_arena_x_min() + x1_ekf) / 2, 'START'),
        ((x1_ekf + x2_ekf) / 2,                   'MIDDLE'),
        ((x2_ekf + config.ekf_arena_x_max()) / 2,  'LANDING'),
    ]:
        ax.text(wx_to_col(rx), wy_to_row(y_top) - 2, lbl,
                color='yellow', fontsize=6, ha='center', va='top', alpha=0.85)

    # X-alignment line (cluster centre X used for landing approach)
    if align_x_col is not None:
        ax.axvline(align_x_col, color='#00ccff', linewidth=1.2,
                   linestyle='--', alpha=0.85, zorder=7, label='X-align')

    # Diff cluster bounding boxes
    if is_diff and npz is not None:
        import matplotlib.patches as rect_patches
        clusters = compute_diff_clusters(npz)
        first_pad = True
        first_rej = True
        for cl in clusters:
            is_pad = cl['is_pad']
            color  = '#00ff88' if is_pad else '#ff4444'
            label  = None
            if is_pad and first_pad:
                label = f'Pad (X={cl["w_m"]*100:.0f}cm)'
                first_pad = False
            elif not is_pad and first_rej:
                label = 'Rejected'
                first_rej = False
            rect = rect_patches.Rectangle(
                (cl['col_min'], cl['row_min']),
                cl['col_max'] - cl['col_min'],
                cl['row_max'] - cl['row_min'],
                linewidth=1.5, edgecolor=color,
                facecolor='none', zorder=8, label=label)
            ax.add_patch(rect)
            ax.text(cl['col_min'], cl['row_max'] + 1,
                    f'X={cl["w_m"]*100:.0f}  Y={cl["h_m"]*100:.0f}cm',
                    color=color, fontsize=5.5, zorder=9, va='bottom')
        ax.legend(loc='upper left', fontsize=6,
                  facecolor='#1a1a1a', labelcolor='white', framealpha=0.8)

    # Flight trajectory (EKF coords → col/row)
    tx = df['x_m'].values
    ty = df['y_m'].values
    ax.plot([wx_to_col(x) for x in tx],
            [wy_to_row(y) for y in ty],
            color='cyan', lw=0.8, alpha=0.6, zorder=4)
    if len(tx):
        ax.plot(wx_to_col(tx[0]),  wy_to_row(ty[0]),
                '^', color='lime',  ms=6, zorder=6)
        ax.plot(wx_to_col(tx[-1]), wy_to_row(ty[-1]),
                's', color='white', ms=6, zorder=6)


def plot_maps(csv_path: str, df: pd.DataFrame, maps: dict, title: str):
    """Show final-state 2×2 map view matching the mission GUI layout."""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(f'Final Map State — {title}', color='white', fontsize=11)

    # Compute X-alignment column from best pad cluster in diff map
    align_x_col = None
    diff_npz = maps.get('diff')
    if diff_npz is not None:
        clusters = _compute_diff_clusters(diff_npz['grid'], float(diff_npz['res'][0]))
        pad_clusters = [c for c in clusters if c['is_pad']]
        if pad_clusters:
            best = min(pad_clusters, key=lambda c: abs(c['w_m'] - config.PAD_SIZE))
            align_x_col = (best['col_min'] + best['col_max']) / 2.0

    _draw_map_ax(axes[0, 0], maps.get('occ'),       'Navigation Map',              df,
                 align_x_col=align_x_col)
    _draw_map_ax(axes[0, 1], maps.get('scan_high'),  'High-Alt Scan Map (0.30 m)', df,
                 align_x_col=align_x_col)
    _draw_map_ax(axes[1, 0], maps.get('occ_low'),    'Low-Alt Scan Map (0.08 m)',  df,
                 align_x_col=align_x_col)
    _draw_map_ax(axes[1, 1], maps.get('diff'),       'Diff Map (elevated objects)', df,
                 is_diff=True, align_x_col=align_x_col)

    plt.tight_layout()


# ------------------------------------------------------------------ main

if __name__ == '__main__':
    path = pick_file()
    print(f'Loading: {path}')
    df = load(path)
    maps = load_maps(path)
    print(f'  {len(df)} rows  |  '
          f't=[{df.time_s.min():.2f}s, {df.time_s.max():.2f}s]  |  '
          f'z_down=[{df.z_down_m.min():.3f}, {df.z_down_m.max():.3f}] m')
    base = os.path.basename(path)
    plot(df, title=base, occ_data=maps.get('occ'))
    plot_maps(path, df, maps, title=base)
    plt.show()
