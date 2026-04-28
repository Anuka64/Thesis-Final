import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_file='q1_results.csv'):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    column_mapping = {
        'achieved_selectivity': 'achieved_s',
        'kernel_ms_median': 'kernel_ms_med',
        'kernel_ms_min': 'kernel_ms_min',
        'kernel_ms_max': 'kernel_ms_max',
        'total_execution_time_median': 'wall_ms_med',
        'overhead_percentage': 'overhead_pct',
        'thread_utilization_percentage': 'thread_utilization_pct',
        'data_efficiency_percentage': 'data_efficiency_pct',
        'bandwidth_GB_per_sec': 'theoritical_GBps_med',
        'num_groups': 'num_groups',
        'useful_data_MB': 'useful_data_MB',
        'total_data_MB': 'total_data_MB'
    }
    df = df.rename(columns=column_mapping)
    df['sel_pct'] = df['achieved_s'] * 100
    return df

def RQ1_divergence_and_bandwidth(df):
    """
    Left: predicate pass rate vs aggregation-idle threads.
    Right: useful data (MB) vs total data scanned — shows absolute memory waste.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['sel_pct'], df['thread_utilization_pct'], 'o-', linewidth=2, markersize=8,
             label='Pred. pass rate')
    ax1.plot(df['sel_pct'], 100 - df['thread_utilization_pct'], 's--', linewidth=2, markersize=8,
             label='Aggregation-idle threads')
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Thread fraction (%)')
    ax1.set_title('Predicate Pass Rate vs Selectivity\n(All threads read; only qualifying threads aggregate)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    total_mb = df['total_data_MB'].iloc[0]
    ax2.plot(df['sel_pct'], df['useful_data_MB'], 'o-', linewidth=2, markersize=8,
             label='Useful data read (MB)')
    ax2.axhline(y=total_mb, color='r', linestyle='--', linewidth=2,
                label=f'Total data scanned ({total_mb:.0f} MB)')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Data (MB)')
    ax2.set_title('Useful vs Total Data Read\n(Total always scanned — waste grows at low selectivity)')
    ax2.set_xscale('log')
    ax2.set_ylim(0, total_mb * 1.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('Q1_RQ1_divergence_bandwidth.png', dpi=300)
    print("Saved: Q1_RQ1_divergence_bandwidth.png")
    plt.close()

def RQ2_end_to_end(df):
    """
    Shows kernel vs total time + overhead breakdown
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['sel_pct'], df['kernel_ms_med'], 'o-', linewidth=2, label='Kernel Only', markersize=8)
    ax1.plot(df['sel_pct'], df['wall_ms_med'], 's-', linewidth=2, label='Total Time', markersize=8)
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Kernel vs End-to-End Time\n(Near-flat kernel = scan dominates; small rise from aggregation work)')
    ax1.set_xscale('log')
    ax1.set_ylim(0, df['wall_ms_med'].max() * 1.15)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(df['sel_pct'], df['overhead_pct'], 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=10, color='r', linestyle='--', linewidth=2, label='10% threshold')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Overhead (%)')
    ax2.set_title('Non-kernel overhead\n(D2H transfer + CPU finalization)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('Q1_RQ2_end_to_end.png', dpi=300)
    print("Saved: Q1_RQ2_end_to_end.png")
    plt.close()

def RQ3_kernel_scaling_and_bandwidth(df):
    """
    Left: kernel time with min-max band.
    Right: Memory bandwidth.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['sel_pct'], df['kernel_ms_med'], 'o-', linewidth=2, markersize=8)
    ax1.fill_between(df['sel_pct'], df['kernel_ms_min'], df['kernel_ms_max'],
                     alpha=0.3, label='Min-Max range')
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Kernel time (ms)')
    ax1.set_title('Kernel performance scaling\n(Near-flat: small rise due to aggregation work at higher selectivity)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    igpu_peak_bw = 51.0
    bw_min = df['theoritical_GBps_med'].min()
    bw_max = df['theoritical_GBps_med'].max()
    bw_margin = (bw_max - bw_min) * 3

    ax2.plot(df['sel_pct'], df['theoritical_GBps_med'], 'o-', linewidth=2,
             color='green', markersize=8, label='Observed BW')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Memory Bandwidth Utilization\n (iGPU peak = 51 GB/s; observed <1% of peak)')
    ax2.set_xscale('log')
    ax2.set_ylim(bw_min - bw_margin, bw_max + bw_margin)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.98, 0.95, f'iGPU peak: {igpu_peak_bw} GB/s\nObserved: ~{bw_max:.3f} GB/s\n(<1% of peak)',
             transform=ax2.transAxes, ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.legend()

    plt.tight_layout()
    plt.savefig('Q1_RQ3_scaling_bandwidth.png', dpi=300)
    print("Saved: Q1_RQ3_scaling_bandwidth.png")
    plt.close()

def RQ4_strategy_zones(df):
    """
    No selectivity-dependent strategy zones for Q1.
    All data points fall in the same cost band.
    Group cardinality (2 vs 4) is annotated directly on data points.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
 
    ax.axvspan(df['sel_pct'].min() * 0.5, df['sel_pct'].max() * 1.5,
               alpha=0.08, color='green')
 
    ax.plot(df['sel_pct'], df['kernel_ms_med'], 'o-', linewidth=3,
            markersize=10, color='black', label='Kernel Time')
 
    ymin = df['kernel_ms_med'].min()
    ymax = df['kernel_ms_med'].max()
    band_margin = (ymax - ymin) * 0.5
    ax.axhspan(ymin - band_margin, ymax + band_margin,
               alpha=0.12, color='blue',
               label=f'Constant cost band ({ymin:.0f}–{ymax:.0f} ms)')
 
    for i, (_, row) in enumerate(df.iterrows()):
        offset = 14 if i % 2 == 0 else -22
        ax.annotate(f"{int(row['num_groups'])} groups",
                    xy=(row['sel_pct'], row['kernel_ms_med']),
                    xytext=(0, offset),
                    textcoords='offset points',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='gray', alpha=0.8))
 
    ax.set_xlim(df['sel_pct'].min() * 0.5, df['sel_pct'].max() * 1.5)
    ax.set_ylim(ymin - 10, ymax + 15)
    ax.set_xlabel('Selectivity (%)', fontsize=12)
    ax.set_ylabel('Kernel Time (ms)', fontsize=12)
    ax.set_title('Q1 Execution Strategy\n'
                 '(Aggregation-Heavy: GPU cost is selectivity-independent — viable at all selectivities)',
                 fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('Q1_rq4_strategy_zones.png', dpi=300, bbox_inches='tight')
    print("Saved: Q1_rq4_strategy_zones.png")
    plt.close()

def Q1_active_groups(df):
    """
    Shows how many GROUP BY combinations are active at each selectivity level
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    positions = range(len(df))
    ax.bar(positions, df['num_groups'], color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{v:.1f}" for v in df['sel_pct']])
    ax.set_xlabel('Selectivity (%)', fontsize=12)
    ax.set_ylabel('Number of Active Groups', fontsize=12)
    ax.set_title('Q1: Active GROUP BY Groups vs Selectivity',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3, axis='y')

    for pos, (_, row) in zip(positions, df.iterrows()):
        ax.text(pos, row['num_groups'] + 0.15, f"{int(row['num_groups'])}",
                ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('Q1_active_groups.png', dpi=300, bbox_inches='tight')
    print("Saved: Q1_active_groups.png")
    plt.close()

def generate_summary_table(df):
    """
    Create CSV table with key metrics for easy reference
    """
    summary = pd.DataFrame({
        'Selectivity (%)': df['sel_pct'].round(3),
        'Kernel (ms)': df['kernel_ms_med'].round(3),
        'Total (ms)': df['wall_ms_med'].round(3),
        'Overhead (%)': df['overhead_pct'].round(1),
        'Thread Util (%)': df['thread_utilization_pct'].round(1),
        'Data Efficiency (%)': df['data_efficiency_pct'].round(1),
        'Bandwidth (GB/s)': df['theoritical_GBps_med'].round(3),
        'Active Groups': df['num_groups'].astype(int)
    })

    summary.to_csv('Q1_summary_table.csv', index=False)
    print("Saved: Q1_summary_table.csv")

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

    return summary

def plot_all_q1(csv_file='q1_results.csv'):
    """
    Main function: Generate all plots for thesis RQs
    """
    print(f"\nLoading data from {csv_file}...")
    df = load_data(csv_file)

    print("\nGenerating plots...\n")

    RQ1_divergence_and_bandwidth(df)
    RQ2_end_to_end(df)
    RQ3_kernel_scaling_and_bandwidth(df)
    RQ4_strategy_zones(df)
    Q1_active_groups(df)

    summary = generate_summary_table(df)

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED")
    print("="*80)
    print("\nGenerated files:")
    print("  1. Q1_RQ1_divergence_bandwidth.png  - Predicate pass rate + absolute data waste")
    print("  2. Q1_RQ2_end_to_end.png            - Timing breakdown + overhead")
    print("  3. Q1_RQ3_scaling_bandwidth.png     - Kernel scaling + zoomed bandwidth")
    print("  4. Q1_rq4_strategy_zones.png        - Constant cost band + group cardinality")
    print("  5. Q1_active_groups.png             - Active GROUP BY groups")
    print("  6. Q1_summary_table.csv             - Numerical summary")

if __name__ == '__main__':
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'q1_results.csv'
    plot_all_q1(csv_file)