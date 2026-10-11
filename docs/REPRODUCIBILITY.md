# Reproducibility notes

## Inputs and outputs

Run `scripts/verify_dataset.py` first. To rerun the main saved five-feature grouped experiment:

```bash
python scripts/imbalance_comparison.py --data data/windows_grouped_final.csv --out results/reproduced --cv group --outer-splits 5 --feature-sets 5 --seed 42 --n-jobs 2
```

The original saved `run_config.json` uses an absolute Kali path `/root/windows_grouped_final.csv`; the command above replaces it with a repository-relative path without changing the supplied script. The script was supplied unchanged; check any optional flags and installed package versions before expecting exact numerical agreement.

## Grouped nested validation

The saved configs request five outer folds and three inner folds. The original script reduces fold counts according to the smallest number of distinct captures per activity. With this CSV, reconstructed partitions yield three outer folds and two inner folds for all eight devices. The reconstruction found at least two captures per class in every outer training partition. This does not independently verify every original run's execution trace.

The outer fold is held out for evaluating classifier hyperparameters tuned inside the training partition. Imbalance-treatment configurations were subsequently compared on observed outer-fold predictions; selected best results are descriptive comparisons, not fully nested unbiased imbalance-treatment selection estimates.

## Saved results

- `results_grouped_tuned_final/` contains saved metrics, fold summaries, class-wise outputs and figures.
- `results_figure_S1/` includes confusion-matrix CSVs and supplementary figure outputs.

The plotting script expects specific input paths; inspect its defaults or supply the appropriate paths before rerunning. The archive includes original outputs for comparison. No end-to-end rerun was executed in this packaging step.

## Known limitations

- Original PCAP-to-CSV extraction code not supplied.
- Original dependency version lockfile not supplied; `requirements.txt` specifies package names only.
- Exact match to manuscript Tables VI–VIII should be manually verified before release.
- Raspberry Pi feature-extraction timing is not reproduced here.
- The feature dataset excludes windows without observed traffic.
