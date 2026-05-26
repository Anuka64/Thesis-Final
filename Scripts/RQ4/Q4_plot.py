"""
RQ4_combined_strategy.py — fixed with broken y-axis on Panel 2

Usage:
    python RQ4_combined_strategy.py <q1_csv> <q3_csv> <q6_csv>

Example:
    python RQ4_combined_strategy.py q1_results_SF5.csv q3_results_SF5.csv q6_results.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

IGPU_PEAK_BW_GBps = 68.0
C1 = '#1f77b4'
C3 = '#d62728'
C6 = '#2ca02c'


# ═════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_q1(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'achieved_selectivity': 'achieved_s',
        'overhead_percentage':  'overhead_pct',
        'bandwidth_GB_per_sec': 'bw_GBps',
    })
    df['sel_pct'] = df['achieved_s'] * 100
    return df[['sel_pct', 'overhead_pct', 'bw_GBps']].copy()


def load_q3(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'achieved_selectivity': 'achieved_s',
        'overhead_percentage':  'overhead_pct',
        'bandwidth_GB_per_sec': 'bw_GBps',
    })
    df['sel_pct'] = df['achieved_s'] * 100
    return df[['sel_pct', 'overhead_pct', 'bw_GBps']].copy()


def load_q6(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'achieved_selectivity': 'achieved_s',
        'overhead_percentage':  'overhead_pct',
        'bandwidth_GB_per_sec': 'bw_GBps',
    })
    df['sel_pct'] = df['achieved_s'] * 100
    df = df.drop_duplicates(subset=['sel_pct'], keep='first')
    return df[['sel_pct', 'overhead_pct', 'bw_GBps']].copy()


# ═════════════════════════════════════════════════════════════════════════════
# Main plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_rq4(df1, df3, df6, out_file='RQ4_combined_strategy.png'):

    # Layout: 1 row, 3 columns
    # col 0      → Panel 1 (overhead, single axes)
    # col 1+2    → Panel 2 (bandwidth, broken y-axis = top + bottom axes)
    fig = plt.figure(figsize=(18, 7))
    gs  = GridSpec(2, 3,
                   figure=fig,
                   width_ratios=[1.15, 0.85, 0.001],   # col2 is invisible spacer
                   height_ratios=[2, 1],                # top axes taller than bottom
                   hspace=0.07,                         # small gap = broken axis feel
                   wspace=0.32)

    ax1    = fig.add_subplot(gs[:, 0])      # Panel 1 spans both rows, col 0
    ax2top = fig.add_subplot(gs[0, 1])      # Panel 2 upper (Q3 and Q6 region)
    ax2bot = fig.add_subplot(gs[1, 1])      # Panel 2 lower (Q1 region)

    fig.suptitle(
        'iGPU Execution Strategy Under Varying Selectivity\n'
        'Overhead and Bandwidth Bottleneck Analysis Across Query Types',
        fontsize=13, fontweight='bold'
    )

    MSTYLE = dict(markersize=8, linewidth=2.2)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Overhead % vs Selectivity
    # ═════════════════════════════════════════════════════════════════════════

    ax1.plot(df1['sel_pct'], df1['overhead_pct'],
             'o-', color=C1, label='Q1  multi-group aggregation', **MSTYLE)
    ax1.plot(df3['sel_pct'], df3['overhead_pct'],
             's-', color=C3, label='Q3  filter + join reduction', **MSTYLE)
    ax1.plot(df6['sel_pct'], df6['overhead_pct'],
             '^-', color=C6, label='Q6  pure filter + reduction', **MSTYLE)

    # Q1 annotation
    ax1.annotate(
        'Q1 (~5.7%): overhead stable\nacross all selectivity levels.\n'
        'Bottleneck is inside the kernel.\n'
        '→ Redesign aggregation structure',
        xy=(df1['sel_pct'].iloc[-1], df1['overhead_pct'].iloc[-1]),
        xytext=(-195, 20), textcoords='offset points',
        fontsize=8, color=C1,
        arrowprops=dict(arrowstyle='->', color=C1, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C1, alpha=0.85)
    )

    # Q3 annotation
    ax1.annotate(
        'Q3 (25–29%): persistently high\nregardless of selectivity.\n'
        'Join overhead dominates.\n'
        '→ Pre-filter before GPU dispatch',
        xy=(df3['sel_pct'].iloc[0], df3['overhead_pct'].iloc[0]),
        xytext=(12, -72), textcoords='offset points',
        fontsize=8, color=C3,
        arrowprops=dict(arrowstyle='->', color=C3, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C3, alpha=0.85)
    )

    # Q6 annotation
    ax1.annotate(
        'Q6 (16–25%): highest at low\nselectivity, decreases as more\n'
        'rows pass the filter predicate.\n'
        '→ Batch low-selectivity queries',
        xy=(df6['sel_pct'].iloc[-1], df6['overhead_pct'].iloc[-1]),
        xytext=(-200, -55), textcoords='offset points',
        fontsize=8, color=C6,
        arrowprops=dict(arrowstyle='->', color=C6, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C6, alpha=0.85)
    )

    ax1.set_xscale('log')
    ax1.set_ylim(0, 38)
    ax1.set_xlabel('Selectivity (%)', fontsize=11)
    ax1.set_ylabel('Non-Kernel Overhead (%)', fontsize=11)
    ax1.set_title(
        'Dispatch Overhead vs Selectivity\n'
        '(Higher overhead = fixed GPU launch cost dominates total time)',
        fontsize=10
    )
    ax1.legend(fontsize=9, loc='center right')
    ax1.grid(True, alpha=0.3)

    # ═════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Broken y-axis bandwidth plot
    #
    # Top axes  (ax2top): 35–80 GB/s — shows Q3 and Q6 with real separation
    # Bottom axes (ax2bot):  0–2 GB/s — shows Q1's structural bottleneck
    #
    # Standard broken-axis technique used in scientific papers when two
    # groups of values are far apart and a log scale would compress one group.
    # ═════════════════════════════════════════════════════════════════════════

    for ax in (ax2top, ax2bot):
        ax.plot(df1['sel_pct'], df1['bw_GBps'],
                'o-', color=C1, label='Q1  multi-group aggregation', **MSTYLE)
        ax.plot(df3['sel_pct'], df3['bw_GBps'],
                's-', color=C3, label='Q3  filter + join reduction', **MSTYLE)
        ax.plot(df6['sel_pct'], df6['bw_GBps'],
                '^-', color=C6, label='Q6  pure filter + reduction', **MSTYLE)
        ax.axhline(y=IGPU_PEAK_BW_GBps, color='black', linestyle='--',
                   linewidth=1.5,
                   label=f'Est. DRAM peak (~{IGPU_PEAK_BW_GBps:.0f} GB/s)')
        ax.axhspan(IGPU_PEAK_BW_GBps, IGPU_PEAK_BW_GBps * 1.06,
                   alpha=0.08, color='gray')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    # Top axes shows Q3 and Q6
    ax2top.set_ylim(35, 80)
    ax2top.set_ylabel('Theoretical Throughput (GB/s)', fontsize=10)
    ax2top.xaxis.set_visible(False)   # hide x-axis on top — shared with bottom

    # Bottom axes shows Q1
    ax2bot.set_ylim(0, 2)
    ax2bot.set_xlabel('Selectivity (%)', fontsize=11)
    ax2bot.set_ylabel('GB/s', fontsize=9)

    # ── Broken axis diagonal markers ──────────────────────────────────────────
    # Standard visual convention: small diagonal lines show where axis is cut
    d = 0.015
    kwargs = dict(transform=ax2top.transAxes, color='black',
                  clip_on=False, linewidth=1.2)
    ax2top.plot((-d, +d), (-d, +d), **kwargs)
    ax2top.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2bot.transAxes)
    ax2bot.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2bot.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    # ── Annotations on top axes (Q3 and Q6 region) ───────────────────────────
    ax2top.annotate(
        f'Q3: {df3["bw_GBps"].min():.0f}–{df3["bw_GBps"].max():.0f} GB/s\n'
        f'Q6: {df6["bw_GBps"].min():.0f}–{df6["bw_GBps"].max():.0f} GB/s\n'
        'Both memory-bound.\n'
        'Values near/above peak =\n'
        'LLC cache acceleration.\n'
        '→ Optimise memory coalescing',
        xy=(df3['sel_pct'].iloc[2], df3['bw_GBps'].iloc[2]),
        xytext=(12, -100), textcoords='offset points',
        fontsize=8, color='#444444',
        arrowprops=dict(arrowstyle='->', color='#444444', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='gray', alpha=0.85)
    )

    # ── Annotation on bottom axes (Q1 region) ────────────────────────────────
    q1_mean = df1['bw_GBps'].mean()
    ax2bot.annotate(
        f'Q1: ~{q1_mean:.2f} GB/s avg\n'
        f'({q1_mean/IGPU_PEAK_BW_GBps*100:.1f}% of est. peak)\n'
        'Structural bottleneck —\n'
        'local memory bank conflicts.\n'
        'Selectivity cannot fix this.\n'
        '→ Redesign aggregation kernel',
        xy=(df1['sel_pct'].iloc[3], df1['bw_GBps'].iloc[3]),
        xytext=(10, 15), textcoords='offset points',
        fontsize=8, color=C1,
        arrowprops=dict(arrowstyle='->', color=C1, lw=1.2),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor=C1, alpha=0.9)
    )

    # Legend on top axes only
    ax2top.legend(fontsize=8, loc='lower left')

    # Shared title spanning both sub-axes
    ax2top.set_title(
        'Bandwidth Utilisation vs Selectivity\n'
        '(Broken y-axis: top = Q3/Q6 region, bottom = Q1 region)',
        fontsize=10
    )

    # ── Bottom notes ──────────────────────────────────────────────────────────
    fig.text(
        0.5, -0.04,
        'Note 1: Each query covers a different selectivity range due to dataset '
        'and predicate constraints: Q1 (0.01–100%), Q3 (0.23–2.53%), Q6 (0.01–24.8%). '
        'Comparison shows bottleneck zone per query type, not same-selectivity comparison.     '
        'Note 2: Q6 selectivity capped at ~24.8% (multi-predicate constraint).     '
        'Note 3: Bandwidth is theoretical throughput (bytes/kernel_ms); '
        'values above DRAM peak reflect LLC cache effects.',
        ha='center', fontsize=7.5, color='#555555', style='italic'
    )

    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {out_file}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Console summary
# ═════════════════════════════════════════════════════════════════════════════

def print_strategy_summary(df1, df3, df6):
    sep = '=' * 72
    print(f"\n{sep}")
    print("RQ4 — iGPU EXECUTION STRATEGY SUMMARY")
    print(sep)

    q1_mean_bw = df1['bw_GBps'].mean()
    q1_mean_oh = df1['overhead_pct'].mean()

    print(f"\nQ1  Overhead: ~{q1_mean_oh:.1f}% (flat)  |  "
          f"BW: ~{q1_mean_bw:.2f} GB/s ({q1_mean_bw/IGPU_PEAK_BW_GBps*100:.1f}% of est. peak)")
    print("    Bottleneck : structural — local memory bank conflicts")
    print("    Finding    : selectivity has NO effect on Q1 performance")
    print("    Strategy   : redesign aggregation kernel architecture")

    print(f"\nQ3  Overhead: {df3['overhead_pct'].min():.1f}–{df3['overhead_pct'].max():.1f}%  |  "
          f"BW: {df3['bw_GBps'].min():.1f}–{df3['bw_GBps'].max():.1f} GB/s")
    print("    Bottleneck : join dispatch overhead + near-peak bandwidth")
    print("    Finding    : overhead high and stable regardless of selectivity")
    print("    Strategy   : pre-filter before GPU dispatch")

    print(f"\nQ6  Overhead: {df6['overhead_pct'].min():.1f}–{df6['overhead_pct'].max():.1f}%  |  "
          f"BW: {df6['bw_GBps'].min():.1f}–{df6['bw_GBps'].max():.1f} GB/s")
    print("    Bottleneck : dispatch overhead at low sel.; bandwidth at high sel.")
    print("    Finding    : most selectivity-sensitive of the three queries")
    print("    Strategy   : batch low-selectivity queries")
    print(f"    Ceiling    : selectivity capped at ~24.8% (multi-predicate)")

    print(f"\n{'─'*72}")
    print("KEY FINDING: Bottleneck type is determined by query structure,")
    print("not selectivity level. Selectivity shifts position within each")
    print("bottleneck zone but never moves a query from one zone to another.")
    print(f"{sep}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python RQ4_combined_strategy.py <q1_csv> <q3_csv> <q6_csv>")
        sys.exit(1)

    df1 = load_q1(sys.argv[1])
    df3 = load_q3(sys.argv[2])
    df6 = load_q6(sys.argv[3])

    print("Generating RQ4 combined strategy plot...")
    plot_rq4(df1, df3, df6)
    print_strategy_summary(df1, df3, df6)