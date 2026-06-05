from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE
OUT = BASE / "plots"
OUT.mkdir(exist_ok=True)

SF = [1, 5, 10]
COLORS = {"Q1": "C0", "Q3": "C1", "Q6": "C2"}
DRAM_PEAK_GBPS = 100.0

def read_result(query, sf):
    name = {"Q1": f"q1_result_sf{sf}.csv", "Q3": f"q3_results_sf{sf}.csv", "Q6": f"q6_results_sf{sf}.csv"}[query]
    return pd.read_csv(DATA / name, comment="#")

def pct(df):
    return df["achieved_selectivity"] * 100.0

def kernel_share(df):
    return 100.0 * df["kernel_ms_median"] / df["total_execution_time_median"]

def plot_kernel_share():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, sf in zip(axes, SF):
        for q, label in [("Q1", "Q1 — filter + GROUP BY"), ("Q3", "Q3 — join + filter + aggregate"), ("Q6", "Q6 — filter + reduction")]:
            df = read_result(q, sf)
            ax.plot(pct(df), kernel_share(df), marker={"Q1":"o","Q3":"s","Q6":"o"}[q], label=label, color=COLORS[q])
        ax.axhline(90, linestyle=":", linewidth=1, color="C0")
        ax.set_xscale("log")
        ax.set_title(f"SF{sf}")
        ax.set_xlabel("Achieved selectivity (%)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(30, 100)
    axes[0].set_ylabel("kernel time vs non-kernel time(%)")
    axes[2].legend(loc="lower right")
    fig.suptitle("RQ4 — Kernel vs non-kernel overhead of total GPU execution time across queries and selectivity", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT / "RQ4_kernel_gpu_time.png", dpi=300)
    plt.close(fig)

def plot_performance_limitations():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    for sf in SF:
        df = read_result("Q1", sf)
        ax.plot(pct(df), df["estimated_bandwidth_GB_per_sec"], marker="o", label=f"SF{sf}")
    ax.axhline(DRAM_PEAK_GBPS, linestyle="--", linewidth=1, color="C0", label="DRAM peak reference")
    ax.set_xscale("log")
    ax.set_title("Q1: low effective bandwidth")
    ax.set_xlabel("Achieved selectivity (%)")
    ax.set_ylabel("Estimated bandwidth / DRAM peak (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    labels, values = [], []
    for sf in SF:
        df = read_result("Q3", sf)
        labels.append(f"SF{sf}")
        values.append(df["cpu_join_preprocessing_ms"].iloc[0] / 1000.0)
    ax.bar(labels, values, label="CPU pre-join preprocessing")
    ax.set_title("Q3: CPU pre-join dominates full pipeline")
    ax.set_ylabel("CPU preprocessing time (seconds)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    for sf in SF:
        df = read_result("Q6", sf)
        ax.plot(pct(df), df["overhead_percentage"], marker="o", label=f"SF{sf}")
    ax.axhline(10, linestyle=":", linewidth=1, color="C0", label="10% reference")
    ax.set_xscale("log")
    ax.set_title("Q6: overhead decreases with scale factor")
    ax.set_xlabel("Achieved selectivity (%)")
    ax.set_ylabel("Non-kernel overhead share (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("RQ4 — Performance limitations by query type", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "The panels summarize three observed limitations: low effective bandwidth in Q1, CPU preprocessing in Q3, and non-kernel overhead in Q6.", ha="center", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT / "RQ4_performance_limitations_by_query.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    plot_kernel_share()
    plot_performance_limitations()
