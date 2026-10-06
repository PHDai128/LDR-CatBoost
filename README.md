# LDR-CatBoost research code

Code used for the quantitative experiments in **V2_1006**: next-hour hot-metal
silicon prediction under an assumed laboratory-result release delay. LDR means
**laboratory-delay-aware residual**. The deterministic availability rule is
`shift(delay).ffill()`; it is not presented as a new delay-estimation algorithm.

## Contents

| Manuscript result | Reproduction entry point | Main output |
| --- | --- | --- |
| Tables 4–5; Figures 4–5; Table 8/Figure 7 source metrics | `src/run_final_experiments.py` | `results/primary/` |
| Table 6; Figure 6 | `analysis/paired_baseline_bootstrap_20261005.py`, then `analysis/plot_primary_paired_bootstrap_20261005.py` | `results/paired_bootstrap/`, `results/figures/` |
| Table 7 controlled ablation | `analysis/controlled_component_study_20261005.py` | `results/controlled_components/` |
| Table 9; Figure 8 source metrics | `src/run_delay_sensitivity.py` | `results/delay_sensitivity/` |
| Table 1 target summaries and supporting fold/segment tables | `analysis/audit_hourly_si_sources_20261005.py`, `analysis/build_paper_support_tables_20261005.py` | `results/hourly_si_audit/`, `results/support_tables/` |
| Figure 7 heatmap | `analysis/plot_figure7_ldr.py` | `results/figures/` |

`src/` also contains the feature-building and neural-model helpers used by the
primary experiment. The extracted helpers retain the model-relevant calculations
while omitting unused intermediate outputs and use anonymous furnace identifiers. Unrelated anomaly-detection,
imputation, exploratory scaling, SHAP, and abandoned model-comparison scripts
are excluded. Figures 1–3 are method diagrams rather than computed results.

## Setup

Use Python 3.10 or later. Install a PyTorch build appropriate for your CUDA
version first if using a GPU, then install `requirements.txt`. CatBoost's GPU
mode is used in the recorded controlled component batch. The primary script
selects GPU when CUDA is available and otherwise uses CPU. Numerical equality
across hardware or library versions is not guaranteed. In a local three-fold
smoke run of the public delay-sensitivity entry point (Furnace 1, 2 h), all
2,703 test rows and HR@0.05 matched the archived run, while HR@0.10 and MAE
were 0.9253 and 0.0433 versus archived 0.9319 and 0.0428. Treat the archived
paper values as the original-run record, rather than promising bitwise
reproduction across GPU environments.

The three industrial CSVs are not distributed. Put authorized copies in `data/`
using the filenames in [data/README.md](data/README.md). Run all commands from
the repository root. A lightweight synthetic availability check is available:

```bash
python -m unittest discover -s tests -v
```

## Reproduce the experiments

The full primary run trains three furnaces × three rolling folds, with five
random seeds for each neural baseline. It can take substantial GPU time.

```bash
python src/run_final_experiments.py --data-dir data --output-dir results/primary --furnaces furnace_1 furnace_2 furnace_3 --window 24 --delay 2 --epochs 50 --patience 8 --batch-size 256 --deep-seed-count 5 --cat-iterations 500
python analysis/paired_baseline_bootstrap_20261005.py --predictions results/primary/predictions.csv --out results/paired_bootstrap --repeats 5000 --blocks 12 24 48
python analysis/controlled_component_study_20261005.py --data-dir data --out results/controlled_components --device GPU --iterations 500
python src/run_delay_sensitivity.py --data-dir data --output-dir results/delay_sensitivity --delays 0 1 2 4 --iterations 500
python analysis/audit_hourly_si_sources_20261005.py
python analysis/build_paper_support_tables_20261005.py
python analysis/plot_primary_paired_bootstrap_20261005.py
python analysis/plot_figure7_ldr.py
```

The Table 7 full-model reference was fitted **in the same controlled batch** as
its variants. It should not be numerically mixed with the earlier primary run's
full-model row. The original primary experiment internally calls the final
model `Ours: Standard CDR-CatBoost`; this is the LDR-CatBoost configuration
reported in the paper. The public copy removes an unused exploratory
static-scaling prediction from that script; it does not alter the reported
primary configurations.

The main comparison uses nine equal-weight furnace–fold results. DLinear has
five seed runs per fold; the paired analysis averages per-timestamp errors and
hit indicators across those seeds before comparison. Delay sensitivity fits a
separate model for each fixed delay. These experiments use offline rolling
evaluation and do not replay an online model-update schedule.

## Data and publication boundary

No production CSV, row-level prediction, trained model, manuscript, or figure
binary is part of this source package. The input-side release rule is applied
to a supplied hourly Si target series; actual laboratory reporting timestamps
are not reconstructed from the files. The repository contains no script for
the upstream creation of those hourly target columns. See the paper for the
scope of the reported evaluation. Before making this directory a public GitHub
repository, review the institution's code and data-release requirements.
