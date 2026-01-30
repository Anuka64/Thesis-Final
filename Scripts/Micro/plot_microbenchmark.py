import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("results.csv")

# Keep only correct runs
df = df[df["correct"] == 1]

# Aggregate: median kernel time per operator per selectivity
agg = (
    df.groupby(["operator", "selectivity_achieved"])["kernel_ms"]
      .median()
      .reset_index()
)

# Plot: kernel time vs selectivity
plt.figure(figsize=(7, 5))

for op, sub in agg.groupby("operator"):
    sub = sub.sort_values("selectivity_achieved")
    plt.plot(
        sub["selectivity_achieved"],
        sub["kernel_ms"],
        marker="o",
        label=op
    )

plt.xlabel("Achieved Selectivity")
plt.ylabel("Kernel Time (ms)")
plt.title("Microbenchmark: Kernel Time vs Selectivity")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
