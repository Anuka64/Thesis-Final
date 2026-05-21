import pandas as pd
import matplotlib.pyplot as plt

IGPU_PEAK_BW = 51.0


def load_data(csv_file='q3_results.csv'):
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
        'wasted_threads_percentage': 'wasted_threads_pct',
        'data_efficiency_percentage': 'data_efficiency_pct',
        'bandwidth_GB_per_sec': 'theoretical_GBps_med'
    }
    df = df.rename(columns=column_mapping)
    
    df['sel_pct'] = df['achieved_s'] * 100
    
    # N handling
    if "N" in df.columns:
        df["n_rows"] = df["N"]
    else:
        df["n_rows"] = df["total_data_MB"] * 1e6 / 16.0
    
    df["matched_rows"] = df["achieved_s"] * df["n_rows"]
    df['err_per_matched_row'] = df['abs_err_cents'] / df['matched_rows']
    
    return df


def RQ1_divergence_and_bandwidth(df):
    """
    Left: Thread utilization 
    Right: Thread divergence waste.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df['sel_pct'], df['thread_utilization_pct'], 'o-',
             linewidth=2, markersize=8)
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Active Threads (%)')
    ax1.set_title('Thread Utilization\n'
                  '(% of threads passing the filter predicate)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['sel_pct'], df['wasted_threads_pct'], 'o-',
             linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Wasted Threads (%)')
    ax2.set_title('Thread Divergence Waste\n'
                  '(% of threads that exit early doing no work)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Q3_RQ1_divergence_bandwidth.png', dpi=300)
    print("Saved: Q3_RQ1_divergence_bandwidth.png")
    plt.close()


def RQ2_end_to_end(df):
    """
    Left:  Kernel vs total wall time shows the overhead gap.
    Right: Non-kernel overhead % vs selectivity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df['sel_pct'], df['kernel_ms_med'], 'o-', linewidth=2,
             label='Kernel Only', markersize=8)
    ax1.plot(df['sel_pct'], df['wall_ms_med'], 's-', linewidth=2,
             label='Total Time', markersize=8)
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Kernel vs End-to-End Time')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(df['sel_pct'], df['overhead_pct'], 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=10, color='r', linestyle='--', linewidth=2, label='10% threshold')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Overhead (%)')
    ax2.set_title('Non-Kernel Overhead')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('Q3_RQ2_end_to_end.png', dpi=300)
    print("Saved: Q3_RQ2_end_to_end.png")
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
    ax1.set_ylabel('Kernel Time (ms)')
    ax1.set_title('Kernel Performance Scaling')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(df['sel_pct'], df['theoretical_GBps_med'], 'o-', linewidth=2,
             color='green', markersize=8, label='Observed BW')
    ax2.axhline(y=IGPU_PEAK_BW, color='red', linestyle='--', linewidth=2,
                label=f'DRAM peak ({IGPU_PEAK_BW} GB/s)')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Memory Bandwidth Utilization')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    bw_max = df['theoretical_GBps_med'].max()
    bw_min = df['theoretical_GBps_med'].min()
    ax2.text(0.98, 0.95,
             f'Observed: {bw_min:.1f}–{bw_max:.1f} GB/s\n'
             f'DRAM peak: {IGPU_PEAK_BW} GB/s',
             transform=ax2.transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('Q3_RQ3_scaling_bandwidth.png', dpi=300)
    print("Saved: Q3_RQ3_scaling_bandwidth.png")
    plt.close()


def generate_summary_table(df):
    summary = pd.DataFrame({
        'Selectivity (%)':    df['sel_pct'].round(3),
        'Kernel (ms)':        df['kernel_ms_med'].round(3),
        'Total (ms)':         df['wall_ms_med'].round(3),
        'Overhead (%)':       df['overhead_pct'].round(1),
        'Thread Util (%)':    df['thread_utilization_pct'].round(1),
        'Wasted Threads (%)': df['wasted_threads_pct'].round(1),
        'Bandwidth (GB/s)':   df['theoretical_GBps_med'].round(2),
        'Err (cents)':        df['abs_err_cents'].astype(int),
        'Err/Row (cents)':    df['err_per_matched_row'].round(6)
    })
    
    summary.to_csv('Q3summary_table.csv', index=False)
    print("Saved: Q3summary_table.csv")
    
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)
    return summary


def plot_all_q3(csv_file='q3_results.csv'):
    print(f"\nLoading data from {csv_file}...")
    df = load_data(csv_file)
    print("\nGenerating plots...\n")
    
    RQ1_divergence_and_bandwidth(df)
    RQ2_end_to_end(df)
    RQ3_kernel_scaling_and_bandwidth(df)
    RQ4_strategy_zones(df)
    generate_summary_table(df)
    
    print("\n" + "=" * 70)
    print("ALL PLOTS GENERATED")


if __name__ == '__main__':
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'q3_results.csv'
    plot_all_q3(csv_file)