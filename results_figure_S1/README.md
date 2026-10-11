# Supplementary Figure S1 — minimal publication results

This directory contains the eight **selected-configuration** pooled confusion matrices used for Supplementary Figure S1, the published Figure S1 PNG and PDF, the class-wise summary `table_S1_updated.csv`, and `run_config.json` for provenance.

`confusion_matrices_raw/` intentionally excludes the other experimental configurations. The complete model comparison outputs are available in the full analysis archive.

The `run_config.json` records *requested* outer folds (5); the effective number of outer folds was 3 and inner folds 2, as determined by the capture-group constraints.

To regenerate Figure S1, use the original `scripts/plot_figure_S1.py`. It expects this folder at `~/results_figure_S1` by default; change `BASE` in that script to point to this directory if using another location.

This is a subset of previously generated outputs, not a new model evaluation.
