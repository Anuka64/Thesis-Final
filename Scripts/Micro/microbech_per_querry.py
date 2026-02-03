import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("micro_results.csv")

Q6_OPS = ["scan_filter_count", "scalar_agg_sum"]
Q1_OPS = ["scan_filter_count", "groupby_sum_count"]



# Numeric conversions 
for c in ["selectivity_target", "selectivity_achieved", "kernel_ms", "rows_per_s", "eff_GBps", "N"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# valid selectivity 
df = df.dropna(subset=["selectivity_target", "selectivity_achieved"])
df = df[(df["selectivity_target"] > 0) & (df["selectivity_achieved"] > 0)]

# Derive ns/tuple 
df = df.dropna(subset=["kernel_ms", "N"])
df["ns_per_tuple"] = (df["kernel_ms"] * 1e6) / df["N"]

# Median metric per (operator, selectivity_target)
def med(metric):
    g = df.groupby(["operator", "selectivity_target"], as_index=False).agg(
        x=("selectivity_achieved", "median"),
        y=(metric, "median"),
    )
    return g.dropna(subset=["x", "y"])

time = med("kernel_ms")
thr  = med("rows_per_s")
bw   = med("eff_GBps")
npt  = med("ns_per_tuple")

def plot_panel(ax, data, ops, title, ylabel):
    for op in ops:
        s = data[data["operator"] == op].sort_values("x")
        if not s.empty:
            ax.plot(s["x"], s["y"], "o-", label=op)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Selectivity (achieved, log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

fig, axs = plt.subplots(3, 2, figsize=(14, 10))

# Row 1: Q6
plot_panel(axs[0, 0], time, Q6_OPS, "Q6 group: time vs selectivity", "Kernel time (ms)")
plot_panel(axs[0, 1], thr,  Q6_OPS, "Q6 group: throughput vs selectivity", "Throughput (rows/s)")

# Row 2: Q1
plot_panel(axs[1, 0], time, Q1_OPS, "Q1 group: time vs selectivity", "Kernel time (ms)")
plot_panel(axs[1, 1], thr,  Q1_OPS, "Q1 group: throughput vs selectivity", "Throughput (rows/s)")

# Row 3: “why”
plot_panel(axs[2, 0], bw,  ["scan_filter_count"], "Scan+filter: Estimated memory bandwidth (GB/s) vs selectivity", "Effective bandwidth (GB/s)")
plot_panel(axs[2, 1], npt, ["groupby_sum_count"], "Group-by: ns/tuple vs selectivity", "Time per tuple (ns/tuple)")

fig.suptitle("Microbenchmark (query-grouped): selectivity sensitivity and bottleneck indicators", y=0.995)
plt.tight_layout()
plt.show()

