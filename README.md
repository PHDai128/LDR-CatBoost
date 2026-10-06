# LDR-CatBoost

This repository contains the experimental code for **“LDR-CatBoost: A Laboratory-Delay-Aware, Persistence-Anchored Residual Framework for Next-Hour Hot-Metal Silicon Prediction.”** It covers the proposed model, forecasting baselines, controlled ablations, paired comparisons, and release-delay sensitivity. Plotting code and industrial data are not included.

| Experiment | Entry point |
| --- | --- |
| LDR-CatBoost and forecasting baselines (persistence, XGBoost residual, DLinear residual, iTransformer residual, and PatchTST residual) | `src/run_final_experiments.py` |
| Controlled component, target-formulation, and loss ablations | `analysis/controlled_component_study_20261005.py` |
| Paired block-bootstrap comparison | `analysis/paired_baseline_bootstrap_20261005.py` |
| Release-delay sensitivity (0, 1, 2, and 4 h) | `src/run_delay_sensitivity.py` |

The remaining files in `src/` provide the preprocessing, feature construction, rolling splits, and model definitions required by these entry points. The Si availability rule is `shift(delay).ffill()` on the supplied hourly Si series. The evaluation is offline; release delays are assumed rather than reconstructed from actual laboratory reporting timestamps.

## Run

Use Python 3.10 or later and install the packages in `requirements.txt` (with a PyTorch build compatible with your hardware). From the repository root, place authorized hourly CSVs in `data/` as described in [data/README.md](data/README.md), then run:

```bash
python src/run_final_experiments.py --data-dir data --output-dir results/primary --furnaces furnace_1 furnace_2 furnace_3 --window 24 --delay 2 --epochs 50 --patience 8 --batch-size 256 --deep-seed-count 5 --cat-iterations 500
python analysis/paired_baseline_bootstrap_20261005.py --predictions results/primary/predictions.csv --out results/paired_bootstrap --repeats 5000 --blocks 12 24 48
python analysis/controlled_component_study_20261005.py --data-dir data --out results/controlled_components --device GPU --iterations 500
python src/run_delay_sensitivity.py --data-dir data --output-dir results/delay_sensitivity --delays 0 1 2 4 --iterations 500
```

The controlled ablation refits its full-model reference in the same batch as its variants; do not combine that reference with the earlier primary-run result. The primary script retains the historical internal label `Standard CDR-CatBoost` for the model named LDR-CatBoost in the paper. GPU and library differences can change fitted metrics slightly; the paper reports the original experimental run.

Industrial CSVs, row-level predictions, and trained models are excluded from Git by `.gitignore`. The repository does not contain the upstream code that created the supplied hourly Si labels.
