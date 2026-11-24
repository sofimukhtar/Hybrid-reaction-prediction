#!/usr/bin/env python3
"""
Evaluate the Transformer, GRU and ensemble on the test split (or val), write
per-sample records, aggregate metrics, summary CSVs and the element-provenance
analysis.

Usage
-----
    python scripts/05_evaluate_test.py --allow-test
    python scripts/05_evaluate_test.py --allow-test --models ensemble
    python scripts/05_evaluate_test.py --split val --max-samples 500   # dry run

Outputs (results/)
------------------
    per_sample_<model>_test.jsonl
    metrics_<model>_test.json
    test_metrics.csv
    tanimoto.csv
    validity.csv
    topk.csv
    regression_errors.csv
    calibration.csv
    element_provenance_<model>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, CKPT_DIR, DATA_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluate import evaluate_split
from src.models import GRUSeq2Seq, TransformerSeq2Seq
from src.tokenizer import Vocabulary
from src.train import load_checkpoint, resolve_device, set_seed
from src.validity import element_provenance_analysis, validity_table

MODELS = ("transformer", "gru", "ensemble")


def _write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[tables] Wrote -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-test", action="store_true",
                    help="Required to evaluate on the test split.")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--models", type=str, nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--device", type=str, default=CFG.runtime.device)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--batch-size", type=int, default=CFG.runtime.eval_batch_size)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Evaluate only the first N reactions (dry runs).")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()

    set_seed(CFG.train.random_seed)
    device = resolve_device(args.device)
    data_dir, ckpt_dir, out_dir = Path(args.data_dir), Path(args.ckpt_dir), Path(args.out_dir)

    vocab = Vocabulary.load(data_dir / "vocab.json", cfg=CFG.tokenizer)
    reactions = load_split(args.split, data_dir)
    print(f"[eval] {args.split} split: {len(reactions):,} reactions")

    need_gru = any(m in ("gru", "ensemble") for m in args.models)
    need_tf = any(m in ("transformer", "ensemble") for m in args.models)

    gru = tf = None
    if need_gru:
        gru = load_checkpoint(
            GRUSeq2Seq(vocab.size, CFG.gru, CFG.tokenizer),
            ckpt_dir / "gru" / "gru_selected.pt", device,
        )
    if need_tf:
        tf = load_checkpoint(
            TransformerSeq2Seq(vocab.size, CFG.transformer, CFG.tokenizer),
            ckpt_dir / "transformer" / "transformer_selected.pt", device,
        )

    all_metrics = {}
    all_records = {}

    for mode in args.models:
        print()
        print("=" * 72)
        print(f"  {mode.upper()}  --  beam width B = {CFG.beam.beam_width}")
        print("=" * 72)
        res = evaluate_split(
            split_name=args.split,
            reactions=reactions,
            vocab=vocab,
            gru_model=gru,
            tf_model=tf,
            mode=mode,
            device=device,
            beam_cfg=CFG.beam,
            ens_cfg=CFG.ensemble,
            eval_cfg=CFG.eval,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            out_dir=out_dir,
            allow_test=args.allow_test,
            with_bootstrap=not args.no_bootstrap,
        )
        all_metrics[mode] = res["metrics"]
        all_records[mode] = res["records"]

        print(f"\n  Validity, {mode}")
        validity_table(res["records"], CFG.eval)

        print(f"\n  Element provenance, {mode}")
        prov = element_provenance_analysis(res["records"], restrict_to_failures=True)
        (out_dir / f"element_provenance_{mode}.json").write_text(json.dumps(prov, indent=2))

    if args.split != "test":
        print("\n[eval] Validation run complete; summary CSVs are written for the test split only.")
        return

    # ---------------- assemble the tables -----------------------------------
    present = [m for m in MODELS if m in all_metrics]
    print()
    print("=" * 72)
    print("  Assembling tables")
    print("=" * 72)

    _write_csv(
        out_dir / "test_metrics.csv",
        ["Metric"] + present,
        [
            [label]
            + [f"{all_metrics[m][key]:.4f}" for m in present]
            for key, label in [
                ("char_accuracy", "Character-Level Accuracy"),
                ("prefix_match", "Prefix Match"),
                ("levenshtein_accuracy", "Levenshtein Accuracy"),
                ("bleu2", "BLEU Score (bigram)"),
                ("micro_f1", "Micro F1 Score"),
                ("exact_match", "Exact Match Accuracy"),
            ]
        ],
    )

    _write_csv(
        out_dir / "tanimoto.csv",
        ["Model", "mean", "std"],
        [
            [
                m,
                f"{all_metrics[m]['tanimoto_mean']:.3f}",
                f"{all_metrics[m]['tanimoto_std']:.3f}",
            ]
            for m in present
        ],
    )

    _write_csv(
        out_dir / "validity.csv",
        ["Metric"] + present,
        [
            ["Syntactically Valid SMILES"]
            + [f"{all_metrics[m]['syntactically_valid']:.3f}" for m in present],
            ["Chemically Valid Molecules"]
            + [f"{all_metrics[m]['chemically_valid']:.3f}" for m in present],
            [f"Invalid with LA < {CFG.eval.severe_la_threshold}"]
            + [f"{all_metrics[m]['severe_invalid_fraction']:.3f}" for m in present],
        ],
    )

    _write_csv(
        out_dir / "topk.csv",
        ["K"] + present,
        [
            [k]
            + [f"{all_metrics[m].get(f'top{k}', 0):.4f}" for m in present]
            for k in CFG.beam.topk_values
        ],
    )

    _write_csv(
        out_dir / "regression_errors.csv",
        ["Metric"] + present,
        [
            [label]
            + [f"{all_metrics[m][key]:.4f}" for m in present]
            for key, label in [
                ("length_mae", "Length MAE"),
                ("length_mse", "Length MSE"),
                ("length_rmse", "Length RMSE"),
                ("lev_mae_ref", "Levenshtein MAE (ref-normalised)"),
                ("lev_mse_ref", "Levenshtein MSE"),
                ("lev_rmse_ref", "Levenshtein RMSE"),
                ("lev_mae_symmetric", "Levenshtein MAE (1 - LA)"),
            ]
        ],
    )

    _write_csv(
        out_dir / "calibration.csv",
        ["Metric"] + present,
        [
            [label]
            + [f"{all_metrics[m][key]:.4f}" for m in present]
            for key, label in [
                ("ece", "Expected Calibration Error (ECE)"),
                ("mce", "Maximum Calibration Error (MCE)"),
                ("conf_acc_r", "Confidence-Accuracy Correlation (r)"),
                ("brier", "Brier Score"),
            ]
        ],
    )

    (out_dir / "all_test_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"[tables] Wrote -> {out_dir / 'all_test_metrics.json'}")

    print()
    print("Done. Next:")
    print("  python scripts/06_ablation.py                # fusion ablation (validation)")
    print("  python scripts/07_scaffold_eval.py --build   # unseen-scaffold evaluation")
    print("  python scripts/09_reaction_type_breakdown.py # per-reaction-type accuracy")
    print("  python scripts/08_make_figures.py            # figures")


if __name__ == "__main__":
    main()
