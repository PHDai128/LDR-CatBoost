"""Rebuild Figure 7 from the segment values reported in manuscript Table 8."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [
    "LDR-CatBoost",
    "iTransformer residual",
    "PatchTST residual",
    "Persistence",
    "XGBoost residual",
    "DLinear residual",
]
SEGMENTS = ["Easy persistence", "Medium correction", "Hard correction"]
METRICS = ["HR@0.05", "HR@0.10", "MAE"]

# Model order follows MODELS; values are transcribed from manuscript Table 8.
VALUES = np.array(
    [
        [[0.7367, 0.9625, 0.0362], [0.8436, 0.9843, 0.0277],
         [0.9560, 0.9964, 0.0195], [1.0000, 1.0000, 0.0144],
         [0.7117, 0.9173, 0.0399], [0.7080, 0.9362, 0.0401]],
        [[0.6218, 0.9278, 0.0457], [0.5564, 0.9503, 0.0486],
         [0.4956, 0.9827, 0.0513], [0.4783, 1.0000, 0.0522],
         [0.5657, 0.8978, 0.0513], [0.5710, 0.8925, 0.0517]],
        [[0.2667, 0.6159, 0.1002], [0.0821, 0.3953, 0.1331],
         [0.0128, 0.2831, 0.1496], [0.0000, 0.2673, 0.1543],
         [0.2538, 0.5730, 0.1056], [0.2502, 0.5673, 0.1049]],
    ], dtype=float
)


def ranks_for_segment(values):
    ranks = np.zeros_like(values, dtype=int)
    for col in range(3):
        for row in range(6):
            v = values[row, col]
            if col < 2:
                ranks[row, col] = 1 + np.count_nonzero(values[:, col] > v)
            else:
                ranks[row, col] = 1 + np.count_nonzero(values[:, col] < v)
    return ranks


RANKS = np.array([ranks_for_segment(x) for x in VALUES])
MEAN_RANK = RANKS.transpose(1, 0, 2).reshape(6, 9).mean(axis=1)
BEST_COUNT = (RANKS == 1).sum(axis=(0, 2))
assert np.allclose(MEAN_RANK, [2.3333333, 3.3333333, 3.4444444,
                               3.7777778, 3.7777778, 4.2222222])
assert np.array_equal(BEST_COUNT, [5, 0, 0, 4, 0, 0])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "savefig.facecolor": "white",
})

fig = plt.figure(figsize=(14.4, 5.7), facecolor="white")
gs = fig.add_gridspec(1, 4, left=0.18, right=0.985, top=0.82, bottom=0.24,
                      width_ratios=[3, 3, 3, 2.1], wspace=0.055)
cmap = plt.get_cmap("Blues_r")
norm = Normalize(vmin=1, vmax=6)

for s, title in enumerate(SEGMENTS):
    ax = fig.add_subplot(gs[0, s])
    ax.imshow(RANKS[s], cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, pad=31, fontweight="bold")
    ax.set_xticks(range(3), METRICS, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_yticks(range(6))
    ax.set_yticklabels(MODELS if s == 0 else [], fontsize=10.5)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(6):
        for col in range(3):
            value = VALUES[s, row, col]
            rank = RANKS[s, row, col]
            ax.text(col, row, f"{value:.4f}", ha="center", va="center",
                    fontsize=9.5, color="white" if rank <= 2 else "#1f2933",
                    fontweight="bold" if rank == 1 else "normal")
    if s == 0:
        labels = ax.get_yticklabels()
        labels[0].set_color("#0067a5")
        labels[0].set_fontweight("bold")

ax = fig.add_subplot(gs[0, 3])
ax.set_title("Overall summary", pad=31, fontweight="bold")
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(5.5, -0.5)
ax.set_xticks([0, 1], ["Mean rank", "Best count"], fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(axis="x", length=0, pad=9)
ax.set_yticks([])
for row in range(6):
    for col in range(2):
        highlight = row == 0
        rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                             facecolor="#e7f2f9" if highlight else "white",
                             edgecolor="#cbd2d8", linewidth=0.8)
        ax.add_patch(rect)
        label = f"{MEAN_RANK[row]:.2f}" if col == 0 else f"{BEST_COUNT[row]}/9"
        ax.text(col, row, label, ha="center", va="center", fontsize=10,
                color="#0067a5" if highlight else "#222222",
                fontweight="bold" if highlight else "normal")
for spine in ax.spines.values():
    spine.set_visible(False)

sm = ScalarMappable(norm=norm, cmap=cmap)
cax = fig.add_axes([0.26, 0.10, 0.57, 0.035])
cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[1, 2, 3, 4, 5, 6])
cb.ax.tick_params(labelsize=9, length=3)
cb.set_label("Within-segment rank (1 = best)", fontsize=10, labelpad=5)
cb.outline.set_linewidth(0.6)

preview = OUT / "Figure7_LDR_CatBoost_preview.png"
tiff = OUT / "Figure7_LDR_CatBoost_600dpi.tiff"
fig.savefig(preview, dpi=180)
fig.savefig(tiff, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print(preview)
print(tiff)
