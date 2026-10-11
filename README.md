# Behavioral IoT Dataset — reproducibility release v1.1.0 (draft)

Companion data and analysis materials for *Behavioral Characterization of Consumer IoT Devices through Encrypted Network Traffic Analysis for Smart-Home Security* (IEEE Access manuscript).

**Release status:** prepared for review; not independently reproduced end-to-end. This package is intended to be added to the existing GitHub repository `juliD-star/Behavioral_IoT_datasets` as a **new** `v1.1.0` release. Preserve the existing `v1.0` release and its archived DOI. Do not claim this version has been archived on Zenodo until the new release appears there.

## Dataset

`data/windows_grouped_final.csv` has **4,704 retained traffic-active one-second windows** across **eight devices** and **83 device-specific capture IDs**. It is a **feature-level** dataset, not a full set of raw packet captures. Inactive one-second intervals were excluded, so this dataset supports classification of retained traffic-active windows, not continuous detection over all time intervals.

| Device | Windows | Device-specific captures |
|---|---:|---:|
| Google Doorbell | 783 | 13 |
| Google Nest Hub | 729 | 6 |
| Google Nest Mini | 1092 | 9 |
| Ring Doorbell | 616 | 10 |
| Smart Bulb | 267 | 14 |
| Smart Plug | 69 | 12 |
| Temperature Sensor | 335 | 7 |
| Wyze Camera | 813 | 12 |

The initial longer idle characterization recordings are distinct from the three independent idle captures per device used for the final machine-learning dataset. The existing `v1.0` release contains representative PCAP captures; they do not constitute the full source-PCAP collection for this feature-level dataset.

## Files

- `data/windows_grouped_final.csv`: feature-level classification dataset.
- `scripts/imbalance_comparison.py`: original supplied model comparison and grouped cross-validation script (unchanged).
- `scripts/plot_figure_S1.py`: original supplied plotting script (unchanged).
- `scripts/verify_dataset.py`: structural checks for the published feature dataset.
- `results/results_grouped_tuned_final/`: original saved tuned evaluation outputs.
- `results/results_figure_S1/`: original saved results and supplementary figure source matrices.
- `docs/DATA_DICTIONARY.md`: CSV schema and caveats.
- `docs/REPRODUCIBILITY.md`: instructions, CV details, and limits.
- `docs/RELEASE_CHECKLIST.md`: mandatory checks before public publication.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/verify_dataset.py data/windows_grouped_final.csv
python scripts/imbalance_comparison.py --data data/windows_grouped_final.csv --out results/reproduced --cv group --outer-splits 5 --feature-sets 5 --seed 42 --n-jobs 2
```

**Important:** The supplied `run_config.json` files record *requested* `outer_splits=5`. Based on the current script and capture-group counts, the reconstructed effective evaluation uses **three outer** and **two inner** grouped folds per device. This was checked by reconstructing the folds, not by instrumenting every historical run. Imbalance-handling methods were compared descriptively on observed outer-fold predictions; best-observed configurations should not be interpreted as unbiased estimates from fully nested strategy selection.

The feature extraction pipeline that transforms PCAPs into this CSV was **not provided**. Thus, the release supports independent inspection and rerunning of the machine-learning evaluation from supplied features, not end-to-end reproduction from raw PCAPs. Raspberry Pi feature-extraction latency measurements reported in the manuscript are not reproduced by this package.

## Citation and archiving

Original release `v1.0`: https://doi.org/10.5281/zenodo.21607546 . When `v1.1.0` is archived, cite its new version-specific DOI in the manuscript. Confirm the DOI resolves to the intended updated archive.

## Privacy and license

The CSV contains traffic statistics and labels rather than packet payloads. A preliminary automated check does **not** replace a human privacy review. Inspect the full repository (including previously released PCAPs) for identifiers, secrets, third-party data, and permissions before publishing. No license is asserted by this draft; choose an appropriate license only after verifying ownership and rights.
