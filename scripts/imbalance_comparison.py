#!/usr/bin/env python3
"""
imbalance_comparison.py
=======================

Compare class-imbalance handling strategies for per-device IoT activity
classification from one-second encrypted-traffic metadata windows.

What this script does
---------------------
For every device (and every feature set) it evaluates

    classifiers : RF, SVM (RBF), LR            (+ Balanced RF, + Dummy baselines)
    imbalance   : none, class_weight, ROS, RUS, SMOTE, SMOTE-Tomek

using a leakage-safe protocol:

  * OUTER split  : grouped by capture / trial (StratifiedGroupKFold or
                   Leave-One-Group-Out), so windows from the same capture never
                   appear in both training and test data.
  * INNER split  : hyper-parameters are tuned with GridSearchCV on the training
                   part of each outer fold only (nested CV). The test fold is
                   never used for model or hyper-parameter selection.
  * Resampling   : done inside an imblearn Pipeline, therefore ONLY on the
                   training data of each (inner and outer) fold. Test folds keep
                   their original class distribution.
  * Scaling      : StandardScaler is fitted inside the pipeline on training data.

Reported for every configuration
--------------------------------
  * fold-wise mean +/- std : accuracy, balanced accuracy, macro-P/R/F1
  * pooled out-of-fold     : macro-F1, balanced accuracy, per-class P/R/F1,
                             confusion matrix, recall of the smallest class
  * cost                   : final-model training time, prediction latency
  * statistics             : Friedman test across devices + Wilcoxon signed-rank
                             test of every method against "none"

Input format
------------
One CSV row per one-second window, e.g.

    device,capture_id,activity,bytes_per_s,pkts_per_s,mean_pkt_size,std_pkt_size,mean_iat
    Ring Doorbell,ring_t01,Idle,512.0,4,128.0,20.1,0.25
    ...

`capture_id` must identify the capture / trial / session the window was cut
from. Column names are configurable on the command line.

Usage
-----
    # quick check on synthetic data
    python imbalance_comparison.py --demo --out demo_results

    # real data, grouped nested CV, all methods (recommended main experiment)
    python imbalance_comparison.py --data windows.csv --out results_group

    # the OLD protocol (window-level split), only to show the leakage effect
    python imbalance_comparison.py --data windows.csv --cv window --out results_window

Requirements: numpy, pandas, scikit-learn>=1.3, imbalanced-learn>=0.12,
              scipy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_FEATURES = [
    "bytes_per_s",   # B_t
    "pkts_per_s",    # P_t
    "mean_pkt_size",
    "std_pkt_size",
    "mean_iat",
]

ALL_METHODS = ["none", "class_weight", "ros", "rus", "smote", "smote_tomek"]
ALL_CLASSIFIERS = ["RF", "SVM", "LR"]

# Small grids on purpose: few samples per fold, and the paper's point is a
# lightweight baseline. The chosen values per fold are saved, which also gives
# the hyper-parameter sensitivity information Reviewer 1 asked for.
PARAM_GRIDS = {
    "RF":  {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 3]},
    "BRF": {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 3]},
    "SVM": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.1, 1]},
    "LR":  {"clf__C": [0.01, 0.1, 1, 10]},
}


# --------------------------------------------------------------------------- #
# Samplers and models
# --------------------------------------------------------------------------- #

class AdaptiveSMOTE(SMOTE):
    """SMOTE whose k_neighbors shrinks to fit the smallest class of the
    current training fold (plain SMOTE crashes when a minority class has
    fewer than k_neighbors + 1 samples, which happens with ~25 Doorbell
    Press windows split over several folds). If a class has a single sample
    no synthetic data can be generated and the fold is returned unchanged."""

    def _fit_resample(self, X, y):
        requested = self.k_neighbors
        min_count = min(Counter(y).values())
        if min_count < 2:
            return X, y
        self.k_neighbors = max(1, min(requested, min_count - 1))
        try:
            return super()._fit_resample(X, y)
        finally:
            self.k_neighbors = requested


def make_sampler(method: str, seed: int):
    if method in ("none", "class_weight"):
        return None
    if method == "ros":
        return RandomOverSampler(random_state=seed)
    if method == "rus":
        return RandomUnderSampler(random_state=seed)
    if method == "smote":
        return AdaptiveSMOTE(k_neighbors=5, random_state=seed)
    if method == "smote_tomek":
        return SMOTETomek(smote=AdaptiveSMOTE(k_neighbors=5, random_state=seed),
                          random_state=seed)
    raise ValueError(f"unknown method {method!r}")


def make_classifier(name: str, weighted: bool, seed: int, dummy_strategy=None):
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample" if weighted else None,
            random_state=seed, n_jobs=1)
    if name == "SVM":
        return SVC(kernel="rbf", class_weight="balanced" if weighted else None)
    if name == "LR":
        return LogisticRegression(max_iter=2000,
                                  class_weight="balanced" if weighted else None)
    if name == "BRF":
        # every tree is trained on a class-balanced bootstrap (under-sampling)
        return BalancedRandomForestClassifier(
            n_estimators=200, sampling_strategy="all", replacement=True,
            bootstrap=False, random_state=seed, n_jobs=1)
    if name == "Dummy":
        return DummyClassifier(strategy=dummy_strategy, random_state=seed)
    raise ValueError(f"unknown classifier {name!r}")


def build_pipeline(clf_name: str, method: str, seed: int) -> Pipeline:
    steps = [("scale", StandardScaler())]  # harmless for trees, needed for SVM/LR/SMOTE
    if clf_name == "Dummy":
        clf = make_classifier("Dummy", False, seed, dummy_strategy=method)
    elif clf_name == "BRF":
        clf = make_classifier("BRF", False, seed)
    else:
        sampler = make_sampler(method, seed)
        if sampler is not None:
            steps.append(("sampler", sampler))
        clf = make_classifier(clf_name, method == "class_weight", seed)
    steps.append(("clf", clf))
    return Pipeline(steps)


def configurations(classifiers, methods, include_brf=True, include_dummy=True):
    configs = [(c, m) for c in classifiers for m in methods]
    if include_brf:
        configs.append(("BRF", "balanced_bagging"))
    if include_dummy:
        configs += [("Dummy", "most_frequent"), ("Dummy", "stratified")]
    return configs


# --------------------------------------------------------------------------- #
# Cross-validation schemes
# --------------------------------------------------------------------------- #

def choose_cv(y, groups, scheme: str, n_splits: int, seed: int):
    """Return (cv_object, description). Falls back to a window-level split,
    with a clearly labelled description, when grouping is impossible because
    some class occurs in only one capture."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    min_class = min(Counter(y).values())

    if scheme == "window":
        k = max(2, min(n_splits, min_class))
        return (StratifiedKFold(k, shuffle=True, random_state=seed),
                f"window-level StratifiedKFold(k={k})")

    groups_per_class = pd.Series(groups).groupby(pd.Series(y)).nunique()
    gmin = int(groups_per_class.min())
    if scheme == "logo" and gmin >= 2:
        return LeaveOneGroupOut(), f"LeaveOneGroupOut({len(np.unique(groups))} groups)"
    if scheme == "group" and gmin >= 2:
        k = min(n_splits, gmin)
        return (StratifiedGroupKFold(k, shuffle=True, random_state=seed),
                f"StratifiedGroupKFold(k={k})")

    worst = groups_per_class.idxmin()
    k = max(2, min(n_splits, min_class))
    return (StratifiedKFold(k, shuffle=True, random_state=seed),
            f"FALLBACK window-level StratifiedKFold(k={k}): class '{worst}' "
            f"occurs in only {gmin} capture(s)")


# --------------------------------------------------------------------------- #
# Evaluation of one configuration on one device
# --------------------------------------------------------------------------- #

def evaluate(X, y, groups, classes, clf_name, method, outer_cv, scheme,
             tune, seed, n_jobs):
    oof = np.empty(len(y), dtype=object)
    fold_rows = []
    inner_scheme = "window" if scheme == "window" else "group"

    for fold, (tr, te) in enumerate(outer_cv.split(X, y, groups)):
        pipe = build_pipeline(clf_name, method, seed)
        grid = PARAM_GRIDS.get(clf_name) if tune else None

        t0 = time.perf_counter()
        if grid:
            inner_cv, _ = choose_cv(y[tr], groups[tr], inner_scheme, 3, seed)
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner_cv,
                              n_jobs=n_jobs, error_score=np.nan, refit=True)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            model, best = gs.best_estimator_, gs.best_params_
            train_time = gs.refit_time_          # time to train the final model
        else:
            model, best = pipe.fit(X[tr], y[tr]), {}
            train_time = time.perf_counter() - t0
        search_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred = model.predict(X[te])
        latency_us = (time.perf_counter() - t1) / len(te) * 1e6
        oof[te] = pred

        fold_rows.append({
            "fold": fold,
            "n_test": len(te),
            "accuracy": accuracy_score(y[te], pred),
            "balanced_accuracy": balanced_accuracy_score(y[te], pred),
            "macro_precision": precision_score(y[te], pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y[te], pred, average="macro", zero_division=0),
            "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
            "train_time_s": train_time,
            "search_time_s": search_time,
            "predict_us_per_window": latency_us,
            "best_params": json.dumps(best, default=str),
        })

    # pooled out-of-fold predictions: every window predicted exactly once by a
    # model that never saw its capture. More stable than fold means when the
    # minority class has only a handful of windows per fold.
    p, r, f, s = precision_recall_fscore_support(
        y, oof, labels=classes, zero_division=0)
    per_class = pd.DataFrame({"class": classes, "precision": p, "recall": r,
                              "f1": f, "support": s})
    cm = confusion_matrix(y, oof, labels=classes)
    pooled = {
        "pooled_accuracy": accuracy_score(y, oof),
        "pooled_balanced_accuracy": balanced_accuracy_score(y, oof),
        "pooled_macro_f1": f1_score(y, oof, labels=classes, average="macro",
                                    zero_division=0),
    }
    return pd.DataFrame(fold_rows), pooled, per_class, cm


# --------------------------------------------------------------------------- #
# Statistics across devices
# --------------------------------------------------------------------------- #

def statistical_tests(summary: pd.DataFrame, metric="pooled_macro_f1") -> pd.DataFrame:
    rows = []
    for (fs, clf), sub in summary[summary.classifier.isin(ALL_CLASSIFIERS)] \
            .groupby(["feature_set", "classifier"]):
        table = sub.pivot(index="device", columns="method", values=metric).dropna()
        if table.shape[0] < 3 or table.shape[1] < 3:
            continue
        ranks = table.rank(axis=1, ascending=False).mean()
        try:
            fr_stat, fr_p = friedmanchisquare(*[table[c] for c in table.columns])
        except ValueError:
            fr_stat, fr_p = np.nan, np.nan
        for m in table.columns:
            w_p = np.nan
            if m != "none" and "none" in table.columns:
                diff = table[m] - table["none"]
                if np.any(diff != 0):
                    w_p = wilcoxon(table[m], table["none"], zero_method="zsplit").pvalue
            rows.append({
                "feature_set": fs, "classifier": clf, "method": m,
                "n_devices": table.shape[0],
                f"mean_{metric}": table[m].mean(),
                "mean_rank": ranks[m],
                "wilcoxon_p_vs_none": w_p,
                "friedman_chi2": fr_stat, "friedman_p": fr_p,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_confusion_grid(cms, classes, device, fs, out_dir, classifiers, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(classifiers), len(methods),
                             figsize=(2.6 * len(methods), 2.6 * len(classifiers)),
                             squeeze=False)
    for i, c in enumerate(classifiers):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            if m == "balanced_bagging":          # Balanced RF: shown once, in row 0
                cm = cms.get(("BRF", m)) if i == 0 else None
                title = "BRF (balanced RF)"
            else:
                cm = cms.get((c, m))
                title = f"{c} / {m}"
            if cm is None:
                ax.axis("off")
                continue
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for a in range(len(classes)):
                for b in range(len(classes)):
                    ax.text(b, a, f"{norm[a, b]:.2f}\n({cm[a, b]})", ha="center",
                            va="center", fontsize=6,
                            color="white" if norm[a, b] > 0.6 else "black")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(classes if j == 0 else [], fontsize=6)
            ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel("True", fontsize=7)
            if i == len(classifiers) - 1:
                ax.set_xlabel("Predicted", fontsize=7)
    fig.suptitle(f"{device} ({fs}) - pooled out-of-fold confusion matrices",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"cm_{safe(device)}_{fs}.png", dpi=200)
    plt.close(fig)


def plot_heatmaps(summary, out_dir, metric="pooled_macro_f1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fs, sub_fs in summary.groupby("feature_set"):
        clfs = [c for c in ALL_CLASSIFIERS if c in set(sub_fs.classifier)]
        if not clfs:
            continue
        fig, axes = plt.subplots(1, len(clfs), figsize=(4.2 * len(clfs), 0.45 *
                                 sub_fs.device.nunique() + 1.8), squeeze=False)
        for ax, c in zip(axes[0], clfs):
            t = sub_fs[sub_fs.classifier == c].pivot(
                index="device", columns="method", values=metric)
            t = t[[m for m in ALL_METHODS if m in t.columns]]
            ax.imshow(t.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for a in range(t.shape[0]):
                for b in range(t.shape[1]):
                    ax.text(b, a, f"{t.values[a, b]:.2f}", ha="center",
                            va="center", fontsize=7,
                            color="black" if t.values[a, b] > 0.55 else "white")
            ax.set_xticks(range(t.shape[1]))
            ax.set_xticklabels(t.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(t.shape[0]))
            ax.set_yticklabels(t.index, fontsize=7)
            ax.set_title(c, fontsize=9)
        fig.suptitle(f"{metric} by imbalance method ({fs})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{metric}_{fs}.png", dpi=200)
        plt.close(fig)


def safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


# --------------------------------------------------------------------------- #
# Synthetic demo data (only for testing the pipeline)
# --------------------------------------------------------------------------- #

def make_demo_data(seed=0) -> pd.DataFrame:
    """Synthetic windows with (a) strong class imbalance, (b) a per-capture
    offset and (c) AR(1) autocorrelation inside each capture, so that the
    window-level split visibly over-estimates performance. NOT real data."""
    rng = np.random.default_rng(seed)
    devices = {
        "Ring Doorbell":   {"Idle": 32, "Live View": 495, "Doorbell Press": 25},
        "Google Doorbell": {"Idle": 51, "Live View": 514, "Doorbell Press": 30},
        "Smart Bulb":      {"Idle": 60, "BA": 40, "CC": 40, "PT": 35},
    }
    rows = []
    n_trials = 8
    for d_i, (dev, counts) in enumerate(devices.items()):
        centers = {c: rng.normal(0, 1.0, 5) for c in counts}
        for c, n in counts.items():
            per_trial = np.array_split(np.arange(n), n_trials)
            for t, idx in enumerate(per_trial):
                offset = rng.normal(0, 0.6, 5)          # capture effect
                prev = rng.normal(0, 1, 5)
                for _ in idx:
                    prev = 0.8 * prev + rng.normal(0, 0.6, 5)  # autocorrelation
                    z = centers[c] + offset + prev
                    rows.append({
                        "device": dev,
                        "capture_id": f"{safe(dev)}_trial{t:02d}",
                        "activity": c,
                        "bytes_per_s": float(np.exp(7 + 0.8 * z[0])),
                        "pkts_per_s": float(np.exp(2 + 0.5 * z[1])),
                        "mean_pkt_size": float(400 + 150 * z[2]),
                        "std_pkt_size": float(abs(200 + 60 * z[3])),
                        "mean_iat": float(np.exp(-3 + 0.4 * z[4])),
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="CSV with one row per window")
    src.add_argument("--demo", action="store_true", help="use synthetic demo data")
    ap.add_argument("--out", type=Path, default=Path("imbalance_results"))
    ap.add_argument("--device-col", default="device")
    ap.add_argument("--label-col", default="activity")
    ap.add_argument("--group-col", default="capture_id")
    ap.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                    help="comma-separated feature columns (full set)")
    ap.add_argument("--feature-sets", default="5",
                    help="'5' (all features), '2' (first two) or '2,5'")
    ap.add_argument("--cv", choices=["group", "logo", "window"], default="group",
                    help="outer split: group=StratifiedGroupKFold by capture, "
                         "logo=leave-one-capture-out, window=old window-level split")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--classifiers", default=",".join(ALL_CLASSIFIERS))
    ap.add_argument("--no-brf", action="store_true")
    ap.add_argument("--no-dummy", action="store_true")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip inner GridSearchCV (fast, for debugging only)")
    ap.add_argument("--devices", default=None, help="comma-separated subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = make_demo_data(args.seed) if args.demo else pd.read_csv(args.data)
    feats_all = [f.strip() for f in args.features.split(",")]
    missing = [c for c in [args.device_col, args.label_col, args.group_col, *feats_all]
               if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns in input: {missing}\nAvailable: {list(df.columns)}")

    feature_sets = {}
    for tok in args.feature_sets.split(","):
        tok = tok.strip()
        if tok == "2":
            feature_sets["2feat"] = feats_all[:2]
        elif tok == "5":
            feature_sets[f"{len(feats_all)}feat"] = feats_all
        else:
            sys.exit(f"--feature-sets: unknown token {tok!r}")

    methods = [m.strip() for m in args.methods.split(",")]
    classifiers = [c.strip() for c in args.classifiers.split(",")]
    configs = configurations(classifiers, methods, not args.no_brf, not args.no_dummy)

    devices = sorted(df[args.device_col].unique())
    if args.devices:
        wanted = [d.strip() for d in args.devices.split(",")]
        devices = [d for d in devices if d in wanted]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))

    summary_rows, fold_frames, class_frames, dataset_rows = [], [], [], []

    for dev in devices:
        d = df[df[args.device_col] == dev].dropna(subset=feats_all + [args.label_col])
        y = d[args.label_col].astype(str).to_numpy()
        groups = d[args.group_col].astype(str).to_numpy()
        classes = sorted(np.unique(y), key=lambda c: -Counter(y)[c])
        minority = classes[-1]

        outer_cv, cv_desc = choose_cv(y, groups, args.cv, args.outer_splits, args.seed)
        counts = Counter(y)
        caps = pd.Series(groups).groupby(pd.Series(y)).nunique()
        for c in classes:
            dataset_rows.append({"device": dev, "class": c, "windows": counts[c],
                                 "captures": int(caps[c])})
        print(f"\n=== {dev}: {len(y)} windows, {len(np.unique(groups))} captures, "
              f"classes={dict(counts)}\n    outer CV: {cv_desc}")

        for fs_name, cols in feature_sets.items():
            X = d[cols].to_numpy(dtype=float)
            cms = {}
            for clf_name, method in configs:
                folds, pooled, per_class, cm = evaluate(
                    X, y, groups, classes, clf_name, method, outer_cv, args.cv,
                    not args.no_tune, args.seed, args.n_jobs)
                cms[(clf_name, method)] = cm

               # Save raw pooled out-of-fold confusion matrix
                cm_dir = out / "confusion_matrices_raw"
                cm_dir.mkdir(parents=True, exist_ok=True)

                cm_df = pd.DataFrame(
                cm,
                index=classes,
                columns=classes
                 )

                cm_filename = (
                f"cm_{safe(dev)}_{fs_name}_{clf_name}_{method}.csv"
                )

                cm_df.to_csv(cm_dir / cm_filename)

                key = {"device": dev, "feature_set": fs_name,
                       "classifier": clf_name, "method": method}

                folds = folds.assign(**key)
                fold_frames.append(folds)
                class_frames.append(per_class.assign(**key))

                row = dict(key, cv_scheme=cv_desc, n_windows=len(y),
                           n_folds=len(folds), minority_class=minority)
                for m in ["accuracy", "balanced_accuracy", "macro_precision",
                          "macro_recall", "macro_f1"]:
                    row[f"{m}_mean"] = folds[m].mean()
                    row[f"{m}_std"] = folds[m].std(ddof=1) if len(folds) > 1 else 0.0
                row.update(pooled)
                row["minority_recall"] = float(
                    per_class.loc[per_class["class"] == minority, "recall"].iloc[0])
                row["min_class_recall"] = float(per_class["recall"].min())
                row["train_time_s_mean"] = folds["train_time_s"].mean()
                row["predict_us_per_window_mean"] = folds["predict_us_per_window"].mean()
                summary_rows.append(row)

                print(f"    [{fs_name}] {clf_name:5s} {method:16s} "
                      f"macroF1={row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f} "
                      f"pooled={pooled['pooled_macro_f1']:.3f} "
                      f"balAcc={pooled['pooled_balanced_accuracy']:.3f} "
                      f"recall[{minority}]={row['minority_recall']:.2f}")

            if not args.no_plots:
                plot_confusion_grid(cms, classes, dev, fs_name, out, classifiers,
                                    methods + ([] if args.no_brf else ["balanced_bagging"]))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(fold_frames).to_csv(out / "folds.csv", index=False)
    pd.concat(class_frames).to_csv(out / "per_class.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(out / "dataset_composition.csv", index=False)

    stats = statistical_tests(summary)
    if not stats.empty:
        stats.to_csv(out / "stats_across_devices.csv", index=False)

    # compact paper-style table: best method per device & classifier
    best = (summary[summary.classifier.isin(ALL_CLASSIFIERS + ["BRF"])]
            .sort_values("pooled_macro_f1", ascending=False)
            .groupby(["device", "feature_set", "classifier"]).head(1))
    none = summary[summary.method == "none"][
        ["device", "feature_set", "classifier", "pooled_macro_f1", "minority_recall"]] \
        .rename(columns={"pooled_macro_f1": "macro_f1_none",
                         "minority_recall": "minority_recall_none"})
    best = best.merge(none, on=["device", "feature_set", "classifier"], how="left")
    best[["device", "feature_set", "classifier", "method", "macro_f1_none",
          "pooled_macro_f1", "minority_recall_none", "minority_recall",
          "cv_scheme"]].sort_values(["device", "feature_set", "classifier"]) \
        .to_csv(out / "best_method_per_device.csv", index=False)

    if not args.no_plots:
        plot_heatmaps(summary, out)

    print(f"\nDone. Results written to {out.resolve()}")
    if not stats.empty:
        print("\nAcross-device comparison (pooled macro-F1):")
        print(stats[["feature_set", "classifier", "method", "mean_pooled_macro_f1",
                     "mean_rank", "wilcoxon_p_vs_none", "friedman_p"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
