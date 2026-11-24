#!/usr/bin/env python3
"""
Plot training curves and evaluation results from the files written by the
other scripts (results/training_history_*.json, results/per_sample_*_test.jsonl,
results/all_test_metrics.json).

    losses             training and validation loss
    val_microf1        validation micro-F1 per epoch and per-epoch change
    metric_comparison  main metrics for all models
    score_distributions  per-sample Levenshtein accuracy and BLEU histograms
    per_sample_accuracy  per-sample accuracy on the visualisation subset
    pairwise_scatter   per-sample Levenshtein accuracy, model vs model
    cumulative_em      cumulative exact match on the visualisation subset
    smiles_length      predicted vs actual SMILES length
    length_vs_accuracy SMILES length vs Levenshtein accuracy
    regression_errors  MAE / MSE / RMSE of length and edit error
    calibration        reliability diagrams

Usage
-----
    python scripts/08_make_figures.py
    python scripts/08_make_figures.py --figures metric_comparison calibration
    python scripts/08_make_figures.py --format pdf --dpi 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CFG, DATA_DIR, FIGURES_DIR, RESULTS_DIR
from src.evaluate import read_per_sample
from src.metrics import PerSampleRecord

MODELS = ("transformer", "gru", "ensemble")
LABELS = {"transformer": "Transformer", "gru": "GRU Seq2Seq", "ensemble": "Ensemble"}
COLORS = {"transformer": "#4C72B0", "gru": "#DD8452", "ensemble": "#55A868"}

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_histories(results_dir: Path) -> Dict[str, dict]:
    out = {}
    for m in ("transformer", "gru"):
        p = results_dir / f"training_history_{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
        else:
            print(f"[fig] {p.name} missing -- training-curve plots skipped.")
    return out


def load_records(results_dir: Path) -> Dict[str, List[PerSampleRecord]]:
    out = {}
    for m in MODELS:
        p = results_dir / f"per_sample_{m}_test.jsonl"
        if p.exists():
            out[m] = read_per_sample(p)
        else:
            print(f"[fig] {p.name} missing -- most figures need it.")
    return out


def load_subset_ids(data_dir: Path) -> Optional[List[str]]:
    p = data_dir / "viz_subset_300.json"
    if not p.exists():
        print(f"[fig] {p.name} missing -- using the first 300 test records.")
        return None
    return [str(d["id"]) for d in json.loads(p.read_text())]


def subset_records(
    records: Dict[str, List[PerSampleRecord]],
    ids: Optional[List[str]],
    n: int = 300,
) -> Dict[str, List[PerSampleRecord]]:
    """Restrict to the visualisation subset, same reaction order for every model."""
    if ids is None:
        order = [r.rxn_id for r in next(iter(records.values()))[:n]]
    else:
        order = ids[:n]
    out = {}
    for m, recs in records.items():
        by_id = {r.rxn_id: r for r in recs}
        out[m] = [by_id[i] for i in order if i in by_id]
    return out


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Wrote -> {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_losses(hist: Dict[str, dict], out: Path, dpi: int) -> None:
    """Training and validation loss for both models."""
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ("transformer", "gru")):
        if m not in hist:
            ax.set_visible(False)
            continue
        h = hist[m]["history"]
        ax.plot(h["epoch"], h["train_loss"], "o-", ms=3, label="Training loss")
        ax.plot(h["epoch"], h["val_loss"], "s-", ms=3, label="Validation loss")
        sel = CFG.train.selected_epoch
        if sel <= max(h["epoch"]):
            ax.axvline(sel, color="grey", ls="--", lw=1)
            ax.text(sel, ax.get_ylim()[1], f" epoch {sel}", va="top", fontsize=8, color="grey")
        ax.set_title(LABELS[m])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
    fig.suptitle("Training and validation loss", y=1.02)
    _save(fig, out, dpi)


def plot_val_microf1(hist: Dict[str, dict], out: Path, dpi: int) -> None:
    """Validation micro-F1 per epoch and the per-epoch improvement."""
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m, h in hist.items():
        hh = h["history"]
        axes[0].plot(hh["epoch"], hh["val_micro_f1"], "o-", ms=3,
                     color=COLORS[m], label=LABELS[m])
        d = np.diff(hh["val_micro_f1"], prepend=hh["val_micro_f1"][0])
        axes[1].plot(hh["epoch"], d, "o-", ms=3, color=COLORS[m], label=LABELS[m])
    sel = CFG.train.selected_epoch
    axes[0].axvline(sel, color="grey", ls="--", lw=1)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation micro-F1")
    axes[0].set_title("Validation micro-F1 per epoch"); axes[0].legend()
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel(r"$\Delta$F1")
    axes[1].set_title("Per-epoch improvement"); axes[1].legend()
    fig.suptitle("Validation micro-F1 and per-epoch improvement", y=1.02)
    _save(fig, out, dpi)


def plot_metric_comparison(records: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """Grouped bar chart of the six main metrics."""
    if not records:
        return
    from src.metrics import micro_f1

    keys = [
        ("char_accuracy", "Character\nAccuracy"),
        ("prefix_match", "Prefix\nMatch"),
        ("levenshtein_accuracy", "Levenshtein\nAccuracy"),
        ("bleu2", "BLEU\n(bigram)"),
        ("micro_f1", "Micro\nF1"),
        ("exact_match", "Exact\nMatch"),
    ]
    present = [m for m in MODELS if m in records]
    values = {}
    for m in present:
        recs = records[m]
        mf1 = micro_f1([r.pred_smiles for r in recs], [r.ref_smiles for r in recs])["micro_f1"]
        values[m] = [
            float(np.mean([r.char_accuracy for r in recs])),
            float(np.mean([r.prefix_match for r in recs])),
            float(np.mean([r.levenshtein_accuracy for r in recs])),
            float(np.mean([r.bleu2 for r in recs])),
            mf1,
            float(np.mean([r.exact_match for r in recs])),
        ]

    x = np.arange(len(keys))
    w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(present):
        bars = ax.bar(x + i * w - 0.4 + w / 2, values[m], w,
                      label=LABELS[m], color=COLORS[m])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x, [lbl for _, lbl in keys])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Performance comparison on the test set "
                 f"(N = {len(next(iter(records.values()))):,})")
    ax.legend(ncol=len(present))
    _save(fig, out, dpi)


def plot_score_distributions(records: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """Per-sample Levenshtein accuracy and BLEU distributions."""
    if not records:
        return
    present = [m for m in MODELS if m in records]
    fig, axes = plt.subplots(2, len(present), figsize=(4 * len(present), 6), squeeze=False)
    for j, m in enumerate(present):
        recs = records[m]
        la = [r.levenshtein_accuracy for r in recs]
        bl = [r.bleu2 for r in recs]
        axes[0][j].hist(la, bins=50, color=COLORS[m], alpha=0.85)
        axes[0][j].set_title(f"{LABELS[m]}\nLevenshtein, mean {np.mean(la):.4f}", fontsize=9)
        axes[0][j].set_xlabel("Levenshtein accuracy"); axes[0][j].set_ylabel("Count")
        axes[1][j].hist(bl, bins=50, color=COLORS[m], alpha=0.85)
        axes[1][j].set_title(f"BLEU (bigram), mean {np.mean(bl):.4f}", fontsize=9)
        axes[1][j].set_xlabel("BLEU score"); axes[1][j].set_ylabel("Count")
    fig.suptitle("Per-sample score distributions, test set", y=1.01)
    fig.tight_layout()
    _save(fig, out, dpi)


def plot_per_sample_accuracy(sub: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """Per-sample character and Levenshtein accuracy on the visualisation subset."""
    if not sub:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for m, recs in sub.items():
        idx = np.arange(len(recs))
        axes[0].plot(idx, [r.char_accuracy for r in recs], lw=0.9,
                     color=COLORS[m], label=LABELS[m], alpha=0.85)
        axes[1].plot(idx, [r.levenshtein_accuracy for r in recs], lw=0.9,
                     color=COLORS[m], label=LABELS[m], alpha=0.85)
    axes[0].set_ylabel("Character accuracy"); axes[0].legend(ncol=3)
    axes[1].set_ylabel("Levenshtein accuracy"); axes[1].set_xlabel("Test sample index")
    fig.suptitle("Per-sample accuracy, visualisation subset", y=0.98)
    fig.tight_layout()
    _save(fig, out, dpi)


def plot_pairwise_scatter(sub: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """Pairwise scatter of per-sample Levenshtein accuracy, with win/tie counts."""
    if "transformer" not in sub:
        return
    pairs = [(o, "transformer") for o in ("gru", "ensemble") if o in sub]
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(5.5 * len(pairs), 5), squeeze=False)
    for ax, (other, base) in zip(axes[0], pairs):
        a = np.array([r.levenshtein_accuracy for r in sub[base]])
        b = np.array([r.levenshtein_accuracy for r in sub[other]])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

        ties = int(np.sum(np.isclose(a, b)))
        other_wins = int(np.sum(b > a))
        base_wins = int(np.sum(a > b))

        ax.scatter(a, b, s=12, alpha=0.5, color=COLORS[other], edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel(f"{LABELS[base]} Levenshtein accuracy")
        ax.set_ylabel(f"{LABELS[other]} Levenshtein accuracy")
        ax.set_title(
            f"{LABELS[other]} vs {LABELS[base]}\n"
            f"{other_wins} {other} wins, {base_wins} {base} wins, {ties} ties",
            fontsize=9,
        )
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.suptitle("Pairwise per-sample comparison, visualisation subset", y=1.02)
    fig.tight_layout()
    _save(fig, out, dpi)


def plot_cumulative_em(sub: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """Cumulative exact-match accuracy over the subset."""
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for m, recs in sub.items():
        em = np.array([r.exact_match for r in recs], dtype=float)
        cum = np.cumsum(em) / np.arange(1, len(em) + 1)
        ax.plot(np.arange(1, len(em) + 1), cum, lw=1.4,
                color=COLORS[m],
                label=f"{LABELS[m]} (final {cum[-1]:.3f}, "
                      f"{int(em.sum())}/{len(em)})")
    ax.set_xlabel("Number of samples evaluated")
    ax.set_ylabel("Cumulative exact-match accuracy")
    ax.set_title("Cumulative exact-match accuracy, visualisation subset")
    ax.legend()
    _save(fig, out, dpi)


def plot_smiles_length(sub: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """Predicted against actual SMILES length."""
    if not sub:
        return
    fig, axes = plt.subplots(len(sub), 1, figsize=(12, 3 * len(sub)),
                             sharex=True, squeeze=False)
    for ax, (m, recs) in zip(axes[:, 0], sub.items()):
        idx = np.arange(len(recs))
        actual = [len(r.ref_smiles) for r in recs]
        pred = [len(r.pred_smiles) for r in recs]
        mae = float(np.mean(np.abs(np.array(pred) - np.array(actual))))
        ax.plot(idx, actual, lw=1.0, color="black", label="Actual", alpha=0.75)
        ax.plot(idx, pred, lw=1.0, color=COLORS[m], label="Predicted", alpha=0.85)
        ax.set_ylabel("SMILES length")
        ax.set_title(f"{LABELS[m]} — subset length MAE {mae:.2f}", fontsize=9)
        ax.legend(ncol=2)
    axes[-1, 0].set_xlabel("Test sample index")
    fig.suptitle("Predicted and actual SMILES length, visualisation subset", y=1.0)
    fig.tight_layout()
    _save(fig, out, dpi)


def plot_length_vs_accuracy(sub: Dict[str, List[PerSampleRecord]], out: Path, dpi: int) -> None:
    """SMILES length against Levenshtein accuracy, with trend lines and r, p."""
    if not sub:
        return
    try:
        from scipy import stats as sps

        have_scipy = True
    except ImportError:
        have_scipy = False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    lines = []
    for m, recs in sub.items():
        x = np.array([len(r.ref_smiles) for r in recs], dtype=float)
        y = np.array([r.levenshtein_accuracy for r in recs], dtype=float)
        ax.scatter(x, y, s=12, alpha=0.4, color=COLORS[m], edgecolors="none")
        if len(x) > 2 and x.std() > 0:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, np.polyval(coef, xs), "--", lw=1.4, color=COLORS[m])
            if have_scipy:
                r, p = sps.pearsonr(x, y)
                lines.append(f"{LABELS[m]}: r = {r:+.3f}, p = {p:.3f}")
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                lines.append(f"{LABELS[m]}: r = {r:+.3f} (install scipy for p)")
    ax.set_xlabel("Reference SMILES length (characters)")
    ax.set_ylabel("Levenshtein accuracy")
    ax.set_title("SMILES length against Levenshtein accuracy, visualisation subset")
    ax.legend(lines, loc="lower left", fontsize=8)
    _save(fig, out, dpi)


def plot_regression_errors(results_dir: Path, out: Path, dpi: int) -> None:
    """MAE / MSE / RMSE for length and Levenshtein error."""
    p = results_dir / "all_test_metrics.json"
    if not p.exists():
        print(f"[fig] {p.name} missing -- regression-error plot skipped.")
        return
    metrics = json.loads(p.read_text())
    present = [m for m in MODELS if m in metrics]
    if not present:
        return

    groups = [
        ("Length MAE", "length_mae"), ("Length MSE", "length_mse"),
        ("Length RMSE", "length_rmse"),
        ("Lev. MAE", "lev_mae_ref"), ("Lev. MSE", "lev_mse_ref"),
        ("Lev. RMSE", "lev_rmse_ref"),
    ]
    # Separate panels: length and Levenshtein errors differ greatly in scale.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, sl, title in (
        (axes[0], groups[:3], "Length error"),
        (axes[1], groups[3:], "Levenshtein error (reference-normalised)"),
    ):
        x = np.arange(len(sl))
        w = 0.8 / len(present)
        for i, m in enumerate(present):
            vals = [metrics[m].get(k, 0.0) for _, k in sl]
            bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w,
                          label=LABELS[m], color=COLORS[m])
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xticks(x, [lbl for lbl, _ in sl])
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Regression error analysis, test set", y=1.02)
    fig.tight_layout()
    _save(fig, out, dpi)


def plot_calibration(results_dir: Path, out: Path, dpi: int) -> None:
    """Reliability diagrams."""
    p = results_dir / "all_test_metrics.json"
    if not p.exists():
        return
    metrics = json.loads(p.read_text())
    present = [m for m in MODELS if m in metrics and "calibration_bins" in metrics[m]]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 4.2), squeeze=False)
    for ax, m in zip(axes[0], present):
        bins = metrics[m]["calibration_bins"]
        conf = [b["confidence"] for b in bins if b["count"] > 0]
        acc = [b["accuracy"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(conf, acc, "o-", color=COLORS[m], ms=5, label="Observed")
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Exact-match accuracy")
        ax.set_title(
            f"{LABELS[m]}\nECE {metrics[m]['ece']:.3f}, MCE {metrics[m]['mce']:.3f}",
            fontsize=9,
        )
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams", y=1.02)
    fig.tight_layout()
    _save(fig, out, dpi)


# ---------------------------------------------------------------------------
FIGURES = {
    "losses": (plot_losses, "hist"),
    "val_microf1": (plot_val_microf1, "hist"),
    "metric_comparison": (plot_metric_comparison, "records"),
    "score_distributions": (plot_score_distributions, "records"),
    "per_sample_accuracy": (plot_per_sample_accuracy, "sub"),
    "pairwise_scatter": (plot_pairwise_scatter, "sub"),
    "cumulative_em": (plot_cumulative_em, "sub"),
    "smiles_length": (plot_smiles_length, "sub"),
    "length_vs_accuracy": (plot_length_vs_accuracy, "sub"),
    "regression_errors": (plot_regression_errors, "results_dir"),
    "calibration": (plot_calibration, "results_dir"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--out-dir", type=str, default=str(FIGURES_DIR))
    ap.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figures", type=str, nargs="*", default=None,
                    choices=list(FIGURES),
                    help="Only these figures, e.g. --figures losses calibration")
    args = ap.parse_args()

    results_dir, data_dir, out_dir = Path(args.results_dir), Path(args.data_dir), Path(args.out_dir)
    ext = args.format
    want = set(args.figures) if args.figures else set(FIGURES)

    hist = load_histories(results_dir)
    records = load_records(results_dir)
    sub = subset_records(records, load_subset_ids(data_dir)) if records else {}

    inputs = {"hist": hist, "records": records, "sub": sub, "results_dir": results_dir}
    for name in FIGURES:
        if name in want:
            fn, source = FIGURES[name]
            fn(inputs[source], out_dir / f"{name}.{ext}", args.dpi)

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
