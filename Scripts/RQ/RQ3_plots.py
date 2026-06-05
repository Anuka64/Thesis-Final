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

def read_result(query, sf):
    name = {"Q1": f"q1_result_sf{sf}.csv", "Q3": f"q3_results_sf{sf}.csv", "Q6": f"q6_results_sf{sf}.csv"}[query]
    return pd.read_csv(DATA / name, comment="#")

def pct(df):
    return df["achieved_selectivity"] * 100.0

def vtune_row(path, task):
    df = pd.read_csv(path)
    return df[df["Computing Task"] == task].iloc[0]

def plot_normalized_kernel_sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, sf in zip(axes, SF):
        for q in ["Q1", "Q3", "Q6"]:
            df = read_result(q, sf)
            y = df["kernel_ms_median"] / df["kernel_ms_median"].min()
            ax.plot(pct(df), y, marker={"Q1":"o","Q3":"s","Q6":"^"}[q], label=q, color=COLORS[q], linestyle="--" if q == "Q6" else "-")
        ax.set_xscale("log")
        ax.set_title(f"SF{sf}")
        ax.set_xlabel("Achieved selectivity (%)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Kernel time normalized to lowest selectivity")
    axes[0].legend()
    fig.suptitle("RQ3 — Selectivity sensitivity of normalized Kernel time ", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "Q1 is the aggregation-heavy case; Q3 and Q6 are shown only as baselines.", ha="center", style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT / "RQ3_normalized_kernel_sensitivity.png", dpi=300)
    plt.close(fig)

def plot_q1_active_groups():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, sf in zip(axes, SF):
        df = read_result("Q1", sf)
        x = np.arange(len(df))
        ax.bar(x, df["num_groups"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.6g}" for v in pct(df)], rotation=45, ha="right")
        ax.set_title(f"SF{sf}")
        ax.set_xlabel("Achieved selectivity (%)")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Active GROUP BY groups")
    fig.suptitle("RQ3 — Q1 active GROUP BY groups vs selectivity", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "Group count is supporting evidence for aggregation complexity;", ha="center", style="italic")
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(OUT / "RQ3_Q1active_groups.png", dpi=300)
    plt.close(fig)

def plot_vtune_barrier_activity():
    files = {
        "Q1": [("Low", "vt_q1_low.csv"), ("~1%", "q1_1%selectivity.csv"), ("High", "vt_q1_high.csv")],
        "Q3": [("Low", "vt_q3_low.csv"), ("~1%", "q3_1%selectivity.csv"), ("High", "vt_q3_high.csv")],
        "Q6": [("Low", "vt_q6_low.csv"), ("~1%", "q6_1%selectivity.csv"), ("High", "vt_q6_high.csv")],
    }
    tasks = {"Q1":"q1_aggregate", "Q3":"q3_aggregate", "Q6":"q6_kernel_reduce"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))
    for col, q in enumerate(["Q1", "Q3", "Q6"]):
        labels, active, stalled, idle, barriers = [], [], [], [], []
        for label, fname in files[q]:
            r = vtune_row(DATA / fname, tasks[q])
            labels.append(label)
            active.append(r["Active"] * 100)
            stalled.append(r["Stalled"] * 100)
            idle.append(r["Idle"] * 100)
            barriers.append(r["GPU Barriers"] / 1000.0)
        x = np.arange(len(labels))
        axes[0, col].bar(x, active, label="Active")
        axes[0, col].bar(x, stalled, bottom=active, label="Stalled")
        axes[0, col].bar(x, idle, bottom=np.array(active) + np.array(stalled), label="Idle")
        axes[0, col].set_title(q)
        axes[0, col].set_xticks(x)
        axes[0, col].set_xticklabels(labels)
        axes[0, col].set_ylim(0, 100)
        axes[0, col].grid(True, axis="y", alpha=0.3)
        axes[1, col].bar(x, barriers)
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(labels)
        axes[1, col].grid(True, axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("EU state share (%)")
    axes[1, 0].set_ylabel("GPU barriers per 1k threads")
    axes[0, 0].legend()
    fig.suptitle("RQ3 — VTune EU state and barrier activity across different selectivity (SF5)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(OUT / "RQ3_vtune_eu_barrier_activity.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    plot_normalized_kernel_sensitivity()
    plot_q1_active_groups()
    plot_vtune_barrier_activity()
