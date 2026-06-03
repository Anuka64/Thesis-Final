import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_file='q6_results.csv'):
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    column_mapping = {
        'achieved_selectivity': 'achieved_s',
        'kernel_ms_median': 'kernel_ms_med',
        'total_execution_time_median': 'wall_ms_med',
        'overhead_percentage': 'overhead_pct',
        'bandwidth_GB_per_sec': 'bandwidth_GBps'
    }

    df = df.rename(columns=column_mapping)
    df['sel_pct'] = df['achieved_s'] * 100  # convert to percentage
    return df

def RQ1_divergence_and_bandwidth(df):
    """
    Shows Thread utilization and Data efficiency (memory waste)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: predicate pass rate vs idle threads
    ax1.plot(df['sel_pct'], df['sel_pct'], 'o-', linewidth=2, markersize=8,
             label='Pred. pass rate')
    ax1.plot(df['sel_pct'], 100 - df['sel_pct'], 's--', linewidth=2, markersize=8,
             label='Aggregation-idle threads')
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Thread fraction (%)')
    ax1.set_title('Predicate Pass Rate vs Selectivity')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    #Right: Memory Bandwidth waste (what % of read data is actually used)

    total_mb = df['total_data_MB'].iloc[0]
    bandwidth_waste_pct = (1 - df['useful_data_MB'] / total_mb) * 100

    ax2.plot(df['sel_pct'], bandwidth_waste_pct, 'o-', linewidth=2,
             markersize=8, color='red', label='Bandwidth waste (%)')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Wasted Bandwidth (%)')
    ax2.set_title('Bandwidth Waste vs Selectivity\n'
                  '(% of scanned data that did not contribute to result)')
    ax2.set_xscale('log')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('RQ1_divergence_bandwidth.png', dpi=300)
    print ("Saved r1")
    plt.close()

def RQ2_end_to_end(df):
    """
    Shows kernel vs Total time + overhead breakdown
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #Left: Kernel time vs Total time
    ax1.plot(df['sel_pct'], df ['kernel_ms_med'], 'o-', linewidth=2, label='Kernel Only', markersize=8)
    ax1.plot(df['sel_pct'], df ['wall_ms_med'], 's-', linewidth=2, label='Total Time', markersize=8)
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('kernel vs End-to-End Time')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    #Right: Overhead breakdown
    ax2.plot(df['sel_pct'], df ['overhead_pct'], 'o-', linewidth=2, markersize=8)
    ax2.axhline(y=10, color='r', linestyle='--', linewidth=2, label='10% threshold')
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Overhead (%)')
    ax2.set_title('Non-kernel overhead\n(Low = (D2H transfer + CPU finalization)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('RQ2_end_to_end.png', dpi=300)
    print ("Saved r2")
    plt.close()

def RQ3_kernel_scaling_and_bandwidth(df):
    """
    Shows kernel time scaling and memory utilization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(df['sel_pct'], df ['kernel_ms_med'], 'o-', linewidth=2, markersize=8)
    ax1.fill_between(df['sel_pct'], df ['kernel_ms_min'], df['kernel_ms_max'], alpha=0.3, label= 'Min-Max range')
    ax1.set_xlabel('Selectivity (%)')
    ax1.set_ylabel('Kernel time (ms)')
    ax1.set_title('kernel perfoemance scaling\n(sub-linear = Good efficiency)')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    #Right: Memory Bandwidth 
    ax2.plot(df['sel_pct'], df ['bandwidth_GBps'], 'o-', linewidth=2, color='green', markersize=8)
    ax2.set_xlabel('Selectivity (%)')
    ax2.set_ylabel('Bandwidth (GB/s)')
    ax2.set_title('Memory Bandwidth Utilization)')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('RQ3_scaling_bandwidth.png', dpi=300)
    print ("Saved r3.png")
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
        'Bandwidth (GB/s)': df['bandwidth_GBps'].round(1)
    })
    
    summary.to_csv('summary_table.csv', index=False)
    print("Saved: summary_table.csv")
    
    # Print to console
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    
    return summary

def plot_all_q6(csv_file='q6_results.csv'):
    """
    Main function: Generate all plots for thesis RQs
    """
    print(f"\nLoading data from {csv_file}...")
    df = load_data(csv_file)
    
    print("\nGenerating essential plots...\n")
    
    # Generate plots for each RQ
    RQ1_divergence_and_bandwidth(df)
    RQ2_end_to_end(df)
    RQ3_kernel_scaling_and_bandwidth(df)
    
    # Generate summary table
    summary = generate_summary_table(df)
    
    print("\n" + "="*80)
    print("ALL ESSENTIAL PLOTS GENERATED")
    print("="*80)
    print("\nGenerated files:")
    print("  1. rq1_divergence_bandwidth.png  - Thread divergence + memory waste")
    print("  2. rq2_end_to_end.png            - Timing breakdown + overhead")
    print("  3. rq3_scaling_bandwidth.png     - Kernel scaling + bandwidth")
    print("  5. summary_table.csv             - Numerical summary")
    print("\n")

if __name__ == '__main__':
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'q6_results.csv'
    plot_all_q6(csv_file)

