"""
Usage:
  python plot_log.py              # prompts for filename, defaults to latest
  python plot_log.py log/file.csv # specific file
"""

import sys
import os
import glob

import pandas as pd
import matplotlib
import matplotlib.ticker
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


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['z_down_m'] = pd.to_numeric(df['z_down_m'], errors='coerce')
    return df.dropna(subset=['z_down_m'])


# ------------------------------------------------------------------ edge detection

def detect_edges(df: pd.DataFrame,
                 threshold: float = config.EDGE_THRESHOLD,
                 baseline_alpha: float = config.EDGE_BASELINE_ALPHA,
                 exit_debounce: int = config.EDGE_EXIT_DEBOUNCE):
    """Return (entries, exits) each as list of (time, x, y).

    Mirrors the baseline-drop algorithm in EdgeDetector (sensors.py).
    """
    entries, exits = [], []
    baseline = None
    in_drop = False
    exit_count = 0

    for _, row in df.iterrows():
        z = row['z_down_m']
        t, x, y = row['time_s'], row['x_m'], row['y_m']

        if baseline is None:
            baseline = z
            continue

        drop = baseline - z

        if not in_drop:
            if drop > threshold:
                in_drop = True
                exit_count = 0
                entries.append((t, x, y))
        else:
            if drop < threshold * 0.5:
                exit_count += 1
                if exit_count >= exit_debounce:
                    in_drop = False
                    exit_count = 0
                    exits.append((t, x, y))
            else:
                exit_count = 0

        if not in_drop:
            baseline = baseline_alpha * z + (1 - baseline_alpha) * baseline

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

def plot(df: pd.DataFrame, title: str):
    t = df['time_s'].values
    # Convert EKF frame → arena frame
    x = df['x_m'].values + config.TAKEOFF_PAD_X
    y = df['y_m'].values + config.TAKEOFF_PAD_Y
    z = df['z_down_m'].values

    entries, exits = detect_edges(df)
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

    ax1.plot(t, z, color='#1a6faf', lw=1.2, zorder=2)
    for te, *_ in entries:
        ax1.axvline(te, color='#cc2222', lw=1.0, linestyle='--', alpha=0.9,
                    label='entry' if te == entries[0][0] else '')
    for te, *_ in exits:
        ax1.axvline(te, color='#2244cc', lw=1.0, linestyle=':', alpha=0.9,
                    label='exit' if te == exits[0][0] else '')
    ax1.set_xlabel('time (s)', color='black')
    ax1.set_ylabel('z_down (m)', color='black')
    ax1.set_title('z-ranger (down) vs Time', color='black')
    ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.grid(True, which='major', color='#cccccc', linewidth=0.7, zorder=0)
    ax1.grid(True, which='minor', color='#eeeeee', linewidth=0.4, zorder=0)
    if entries or exits:
        ax1.legend(fontsize=8, facecolor='white', labelcolor='black', framealpha=0.9)

    # ---- right: XY scatter
    ax2.set_facecolor('white')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#aaa')
    ax2.tick_params(axis='both', which='both', colors='black', labelcolor='black')

    sc = ax2.scatter(x, y, c=z, cmap='Reds',
                     s=20, vmin=z.min(), vmax=z.max(), zorder=2)
    cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label('z_down (m)', color='black')
    cb.ax.yaxis.set_tick_params(color='black')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='black')

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
    for i, (cx, cy, enx, eny, exx, exy) in enumerate(pairs):
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


# ------------------------------------------------------------------ main

if __name__ == '__main__':
    path = pick_file()
    print(f'Loading: {path}')
    df = load(path)
    print(f'  {len(df)} rows  |  '
          f't=[{df.time_s.min():.2f}s, {df.time_s.max():.2f}s]  |  '
          f'z_down=[{df.z_down_m.min():.3f}, {df.z_down_m.max():.3f}] m')
    plot(df, title=os.path.basename(path))
