"""Publication-size paired bootstrap figure for XGBoost and DLinear comparisons."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
stats = pd.read_csv(ROOT / "results" / "paired_bootstrap" / "results.csv")
stats = stats[stats.block_hours == 24]
dest = ROOT / "results" / "figures"
dest.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.1), constrained_layout=True)
specs = [("hr005_gain", "(a) HR@0.05 gain", (0, .052), ".3f"),
         ("hr010_gain", "(b) HR@0.10 gain", (0, .052), ".3f"),
         ("mae_gain", "(c) MAE reduction", (0, .0075), ".4f")]
names = ["XGBoost residual", "DLinear residual"]
colors = ["#16669A", "#DB8C24"]
for ax, (metric, title, lim, fmt) in zip(axes, specs):
    subset = stats[stats.metric == metric].set_index("baseline").loc[names]
    for y, (name, color) in enumerate(zip(names, colors)):
        row = subset.loc[name]
        point = row.point_gain
        ax.errorbar(point, y, xerr=[[point - row.ci_low], [row.ci_high - point]],
                    fmt="D", ms=6.5, capsize=4, elinewidth=1.8, color=color, ecolor=color)
        offset = .0003 if metric == "mae_gain" else .002
        ax.text(row.ci_high + offset, y, format(point, fmt), ha="left", va="center", fontsize=9.5, color=color)
    ax.set_title(title, fontweight="bold", loc="left", pad=8)
    ax.set_xlim(*lim)
    ax.set_ylim(-.55, 1.4)
    ax.set_yticks([0, 1], names)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D8DEE4", linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_xlabel("Improvement over comparator", labelpad=3)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Paired improvement of LDR-CatBoost (24 h blocks)", fontsize=12, fontweight="bold")
fig.savefig(dest / "Figure6_Paired_Comparisons_600dpi.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
fig.savefig(dest / "Figure6_Paired_Comparisons_preview.png", dpi=170, bbox_inches="tight")
plt.close(fig)
