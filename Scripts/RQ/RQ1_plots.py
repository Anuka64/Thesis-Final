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
DRAM_PEAK_GBPS = 51.0

def read_result(query, sf):
    name = {"Q1": f"q1_result_sf{sf}.csv", "Q3": f"q3_results_sf{sf}.csv", "Q6": f"q6_results_sf{sf}.csv"}[query]
    return pd.read_csv(DATA / name, comment="#")

def pct(df):
    return df["achieved_selectivity"] * 100.0

def vtune_row(path, task):
    df = pd.read_csv(path)
    return df[df["Computing Task"] == task].iloc[0]

def plot_effective_bandwidth():
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, sf in zip(axes, SF):
        for q in ["Q1", "Q3", "Q6"]:
            df = read_result(q, sf)
            ax.plot(pct(df), df["estimated_bandwidth_GB_per_sec"], marker={"Q1":"o","Q3":"s","Q6":"^"}[q], label=q, color=COLORS[q], linestyle="--" if q == "Q6" else "-")
        ax.axhline(DRAM_PEAK_GBPS, linestyle="--", linewidth=1, color="C0", label="DRAM peak reference" if sf == 1 else None)
        ax.set_xscale("log")
        ax.set_title(f"SF{sf}")
        ax.set_xlabel("Achieved selectivity (%)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Estimated effective bandwidth (GB/s)")
    axes[0].legend()
    fig.suptitle("RQ1 — Estimated effective bandwidth vs selectivity", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(OUT / "RQ1_bandwidth.png", dpi=300)
    plt.close(fig)

def plot_vtune_eu_state():
    files = {
        "Q1": [("Low", "q1_1%selectivity.csv"), ("~1%", "q1_1%selectivity.csv"), ("High", "vt_q1_high.csv")],
        "Q3": [("Low", "vt_q3_low.csv"), ("~1%", "q3_1%selectivity.csv"), ("High", "vt_q3_high.csv")],
        "Q6": [("Low", "vt_q6_low.csv"), ("~1%", "q6_1%selectivity.csv"), ("High", "vt_q6_high.csv")],
    }
    tasks = {"Q1":"q1_aggregate", "Q3":"q3_aggregate", "Q6":"q6_kernel_reduce"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for ax, q in zip(axes, ["Q1", "Q3", "Q6"]):
        labels, active, stalled, idle, occ = [], [], [], [], []
        for label, fname in files[q]:
            r = vtune_row(DATA / fname, tasks[q])
            labels.append(label)
            active.append(r["Active"] * 100)
            stalled.append(r["Stalled"] * 100)
            idle.append(r["Idle"] * 100)
            occ.append(r["EU Threads Occupancy"] * 100)
        x = np.arange(len(labels))
        ax.bar(x, active, label="Active", color="C0")
        ax.bar(x, stalled, bottom=active, label="Stalled", color="C1")
        ax.bar(x, idle, bottom=np.array(active) + np.array(stalled), label="Idle", color="C2")
        ax.plot(x, occ, marker="D", linestyle="--", color="black", linewidth=1.6, label="EU threads occupancy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Selectivity case")
        ax.set_title(q)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Percentage (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("RQ1 — VTune EU state and occupancy across different selectivity(SF5)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.17, 1, 0.92])
    fig.savefig(OUT / "RQ1_vtune_occupancy.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    plot_effective_bandwidth()
    plot_vtune_eu_state()
