from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE
OUT = BASE / "plots"
OUT.mkdir(exist_ok=True)

SF = [1, 5, 10]

def read_result(query, sf):
    name = {"Q1": f"q1_result_sf{sf}.csv", "Q3": f"q3_results_sf{sf}.csv", "Q6": f"q6_results_sf{sf}.csv"}[query]
    return pd.read_csv(DATA / name, comment="#")

def pct(df):
    return df["achieved_selectivity"] * 100.0

def nearest(df, target_pct):
    return df.iloc[(pct(df) - target_pct).abs().argmin()]

def plot_kernel_vs_total():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, q in zip(axes, ["Q1", "Q3", "Q6"]):
        for sf in SF:
            df = read_result(q, sf)
            x = pct(df)
            ax.plot(x, df["kernel_ms_median"], marker="o", label=f"SF{sf} kernel")
            ax.plot(x, df["total_execution_time_median"], marker="o", linestyle="--", label=f"SF{sf} total")
        ax.set_xscale("log")
        ax.set_xlabel("Achieved selectivity (%)")
        ax.set_title(q)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Time (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("RQ2 — Kernel time vs measured total GPU execution time", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "Q1 is included as contrast; RQ2 interpretation focuses mainly on Q3 and Q6.", ha="center", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT / "RQ2_kernel_vs_total.png", dpi=300)
    plt.close(fig)

def plot_overhead_percentage():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, q in zip(axes, ["Q1", "Q3", "Q6"]):
        for sf in SF:
            df = read_result(q, sf)
            ax.plot(pct(df), df["overhead_percentage"], marker="o", label=f"SF{sf}")
        ax.axhline(10, linestyle=":", linewidth=1, color="C0", label="10% reference" if q == "Q1" else None)
        ax.set_xscale("log")
        ax.set_title(q)
        ax.set_xlabel("Achieved selectivity (%)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Overhead share of total GPU execution time (%)")
    axes[0].legend()
    fig.suptitle("RQ2 — Non-kernel overhead percentage vs selectivity", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.savefig(OUT / "RQ2_overhead_percentage.png", dpi=300)
    plt.close(fig)

def plot_overhead_breakdown():
    cases = [("Q3", 5, [0.25, 1.5, 2.5]), ("Q6", 5, [0.014, 5.0, 25.0])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, (q, sf, targets) in zip(axes, cases):
        df = read_result(q, sf)
        rows = [nearest(df, t) for t in targets]
        labels = [f"{pct(pd.DataFrame([r])) .iloc[0]:.3g}%" for r in rows]
        d2h = np.array([r["gpu_to_cpu_transfer_in_ms"] for r in rows])
        cpu = np.array([r["cpu_reduction_time_in_ms"] for r in rows])
        overhead = np.array([r["overhead_ms"] for r in rows])
        other = np.maximum(overhead - d2h - cpu, 0)
        x = np.arange(len(rows))
        ax.bar(x, d2h, label="D2H transfer")
        ax.bar(x, cpu, bottom=d2h, label="CPU finalisation")
        ax.bar(x, other, bottom=d2h + cpu, label="Other overhead")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{q} SF{sf}")
        ax.set_xlabel("Selectivity case")
        ax.set_ylabel("Overhead (ms)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle("RQ2 — Simplified overhead breakdown for filter-dominated queries", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(OUT / "RQ2_overhead_breakdown_simple.png", dpi=300)
    plt.close(fig)

def plot_q3_preprocessing_limitation():
    fig, ax = plt.subplots(figsize=(10, 5.8))
    labels, gpu, prep = [], [], []
    for sf in SF:
        df = read_result("Q3", sf)
        labels.append(f"SF{sf}")
        gpu.append(df["total_execution_time_median"].mean())
        prep.append(df["cpu_join_preprocessing_ms"].iloc[0])
    x = np.arange(len(labels))
    ax.bar(x, gpu, label="Mean GPU phase")
    ax.bar(x, prep, bottom=gpu, label="CPU pre-join preprocessing")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Time (ms, log scale)")
    ax.set_title("Q3 full-pipeline limitation: CPU pre-join vs GPU phase", fontsize=16)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "RQ2_RQ4_preprocessing_limitation.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    plot_kernel_vs_total()
    plot_overhead_percentage()
    plot_overhead_breakdown()
    plot_q3_preprocessing_limitation()
