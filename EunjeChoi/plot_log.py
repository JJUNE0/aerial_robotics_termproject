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
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import config


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


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['z_down_m'] = pd.to_numeric(df['z_down_m'], errors='coerce')
    if 'state' not in df.columns:
        df['state'] = ''
    if 'edge_event' not in df.columns:
        df['edge_event'] = ''
    for col in ('yaw_deg', 'roll_deg', 'pitch_deg', 'yaw_ref_deg',
                'vx_ms', 'vy_ms', 'vz_ms',
                'range_front_m', 'range_back_m',
                'range_left_m', 'range_right_m', 'range_up_m'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0
    for col in ('target_x_m', 'target_y_m',
                'local_x_min_m', 'local_x_max_m',
                'local_y_min_m', 'local_y_max_m',
                'edge_x_m', 'edge_y_m'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    return df.dropna(subset=['z_down_m'])


# ------------------------------------------------------------------ edge detection

def detect_edges(df: pd.DataFrame,
                 baseline: float = config.EDGE_BASELINE,
                 entry_dip: float = config.EDGE_ENTRY_DIP,
                 exit_rise: float = config.EDGE_EXIT_RISE,
                 min_dz: float = config.EDGE_MIN_DZ,
                 cooldown: float = config.EDGE_COOLDOWN):
    """Return (entries, exits) each as list of (time, x, y)."""
    if 'edge_event' in df.columns:
        actual = df[df['edge_event'].isin(['entry', 'exit'])]
        if len(actual):
            entries = [
                (row['time_s'], row['edge_x_m'], row['edge_y_m'])
                for _, row in actual[actual['edge_event'] == 'entry'].iterrows()
            ]
            exits = [
                (row['time_s'], row['edge_x_m'], row['edge_y_m'])
                for _, row in actual[actual['edge_event'] == 'exit'].iterrows()
            ]
            return entries, exits

    entries, exits = [], []

    prev_z = prev_x = prev_y = prev_t = None
    last_dir = 0
    cooldown_until = -1.0   # suppress detection until this time_s

    for _, row in df.iterrows():
        z = row['z_down_m']
        t, x, y = row['time_s'], row['x_m'], row['y_m']

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


def local_scan_regions(df: pd.DataFrame):
    cols = ['local_x_min_m', 'local_x_max_m', 'local_y_min_m', 'local_y_max_m']
    if any(col not in df.columns for col in cols):
        return []
    rdf = df.dropna(subset=cols)[cols].drop_duplicates()
    regions = []
    for _, row in rdf.iterrows():
        regions.append((
            row['local_x_min_m'] + config.TAKEOFF_PAD_X,
            row['local_x_max_m'] + config.TAKEOFF_PAD_X,
            row['local_y_min_m'] + config.TAKEOFF_PAD_Y,
            row['local_y_max_m'] + config.TAKEOFF_PAD_Y,
        ))
    return regions


def local_scan_intervals(df: pd.DataFrame):
    if 'state' not in df.columns:
        return []
    intervals = []
    start_t = None
    last_t = None
    for _, row in df.iterrows():
        active = row['state'] == 'LOCAL_EDGE_SCAN'
        t = row['time_s']
        if active and start_t is None:
            start_t = t
        elif not active and start_t is not None:
            intervals.append((start_t, last_t))
            start_t = None
        last_t = t
    if start_t is not None:
        intervals.append((start_t, last_t))
    return intervals


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
        tdf = df.dropna(subset=['target_x_m', 'target_y_m'])[
            ['target_x_m', 'target_y_m']].drop_duplicates()
        if len(tdf):
            targets = list(zip(
                tdf['target_x_m'] + config.TAKEOFF_PAD_X,
                tdf['target_y_m'] + config.TAKEOFF_PAD_Y,
            ))

    entries, exits = detect_edges(df)
    # Apply arena offset to edge event positions
    entries = [(te, ex + config.TAKEOFF_PAD_X, ey + config.TAKEOFF_PAD_Y)
               for te, ex, ey in entries]
    exits   = [(te, ex + config.TAKEOFF_PAD_X, ey + config.TAKEOFF_PAD_Y)
               for te, ex, ey in exits]
    pairs = compute_pairs(entries, exits)
    local_regions = local_scan_regions(df)
    local_intervals = local_scan_intervals(df)

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

    for i, (t0, t1) in enumerate(local_intervals):
        ax1.axvspan(t0, t1, color='#00cc88', alpha=0.14, zorder=1,
                    label='local scan' if i == 0 else '')

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

    for i, (lx0, lx1, ly0, ly1) in enumerate(local_regions):
        cx = (lx0 + lx1) / 2.0
        cy = (ly0 + ly1) / 2.0
        radius = min(lx1 - lx0, ly1 - ly0) / 2.0
        ax2.add_patch(plt.Circle(
            (cx, cy), radius,
            fill=False, edgecolor='#00aa66', linewidth=1.8,
            linestyle='--', zorder=6,
            label='local scan region' if i == 0 else ''))

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


# ------------------------------------------------------------------ main

if __name__ == '__main__':
    path = pick_file()
    print(f'Loading: {path}')
    df = load(path)
    occ_data = load_occ(path)
    if occ_data is not None:
        print(f'  occ map loaded: {occ_data["grid"].shape}')
    else:
        print('  occ map: not found')
    print(f'  {len(df)} rows  |  '
          f't=[{df.time_s.min():.2f}s, {df.time_s.max():.2f}s]  |  '
          f'z_down=[{df.z_down_m.min():.3f}, {df.z_down_m.max():.3f}] m')
    plot(df, title=os.path.basename(path), occ_data=occ_data)
