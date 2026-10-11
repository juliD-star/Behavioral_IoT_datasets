
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Input and output directories
BASE = Path.home() / "results_figure_S1"
CM_DIR = BASE / "confusion_matrices_raw"
OUT_DIR = BASE / "publication_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Configurations selected in Table VII
configs = [
    ("Google Doorbell", "RF", "smote_tomek"),
    ("Google Nest Hub", "RF", "smote_tomek"),
    ("Google Nest Mini", "RF", "class_weight"),
    ("Ring Doorbell", "RF", "smote_tomek"),
    ("Smart Bulb", "RF", "none"),
    ("Smart Plug", "LR", "smote"),
    ("Temperature Sensor", "RF", "ros"),
    ("Wyze Camera", "RF", "none"),
]


# Publication-style labels for Figure S1
METHOD_LABELS = {
    "smote_tomek": "SMOTE–Tomek",
    "class_weight": "Class Weight",
    "smote": "SMOTE",
    "ros": "ROS",
    "rus": "RUS",
    "none": "None"
}


# Expected pooled macro-F1 from Table VII
expected_f1 = {
    "Google Doorbell": 0.629,
    "Google Nest Hub": 0.819,
    "Google Nest Mini": 0.580,
    "Ring Doorbell": 0.711,
    "Smart Bulb": 0.454,
    "Smart Plug": 0.394,
    "Temperature Sensor": 0.727,
    "Wyze Camera": 0.744,
}

fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
axes = axes.flatten()

for i, (device, clf, method) in enumerate(configs):

    filename = (
        f"cm_{device.replace(' ', '_')}_"
        f"5feat_{clf}_{method}.csv"
    )

    path = CM_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing matrix: {path}")

    df = pd.read_csv(path, index_col=0)

    if list(df.index) != list(df.columns):
        raise ValueError(f"Class labels do not match: {device}")

    cm = df.to_numpy(dtype=int)
    classes = df.index.tolist()

    # Row-normalized confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)

    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    )

    # Verify macro-F1 using pooled counts
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)

    denominator = actual + predicted
    class_f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp),
        where=denominator != 0
    )
    macro_f1 = class_f1.mean()

    if abs(macro_f1 - expected_f1[device]) > 0.001:
        raise ValueError(
            f"Macro-F1 mismatch for {device}: "
            f"calculated={macro_f1:.6f}, "
            f"Table VII={expected_f1[device]:.3f}"
        )

    print(
        f"{device:22s} "
        f"Macro-F1={macro_f1:.6f} "
        f"PASS"
    )

    ax = axes[i]

    im = ax.imshow(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="equal"
    )

    for r in range(len(classes)):
        for c in range(len(classes)):
            value = normalized[r, c]
            count = cm[r, c]

            ax.text(
                c, r,
                f"{value * 100:.1f}%\n({count})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.55 else "black"
            )

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(
        classes,
        rotation=40,
        ha="right",
        fontsize=8
    )
    ax.set_yticklabels(classes, fontsize=8)

    panel = chr(ord("a") + i)

    display_method = METHOD_LABELS.get(method, method)

    ax.set_title(
    f"({panel}) {device}\n"
    f"{clf} / {display_method}",
    fontsize=10,
    fontweight="bold",
    pad=9
)


    ax.set_xlabel("Predicted class", fontsize=9)
    ax.set_ylabel("True class", fontsize=9)

# Shared colorbar
fig.subplots_adjust(
    left=0.055,
    right=0.91,
    bottom=0.10,
    top=0.91,
    wspace=0.45,
    hspace=0.55
)

cbar_ax = fig.add_axes([0.93, 0.22, 0.012, 0.55])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Fraction of true class", fontsize=10)

# Save publication-quality outputs
png_path = OUT_DIR / "Figure_S1_Grouped_Confusion_Matrices.png"
pdf_path = OUT_DIR / "Figure_S1_Grouped_Confusion_Matrices.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.close(fig)

print("\nFigure S1 saved:")
print(png_path)
print(pdf_path)
