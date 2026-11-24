#!/usr/bin/env python3
"""
Sweep the ensemble fusion weight w_GRU over [0, 1] on the validation split and
report the value with the highest exact-match accuracy.

Use --subset for a faster sweep on a random validation subsample.

Usage
-----
    python scripts/04_sweep_fusion_weights.py --subset 3000
    python scripts/04_sweep_fusion_weights.py                 # full val split
    python scripts/04_sweep_fusion_weights.py --only-weights 0.7 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, CKPT_DIR, DATA_DIR, RESULTS_DIR
from src.data import load_split
from src.decode import BranchAdapter
from src.evaluate import decode_and_score
from src.metrics import aggregate
from src.models import GRUSeq2Seq, TransformerSeq2Seq
from src.tokenizer import Vocabulary
from src.train import load_checkpoint, resolve_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default=CFG.runtime.device)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--batch-size", type=int, default=CFG.runtime.eval_batch_size)
    ap.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Sweep on a random subsample of the validation split of this size.",
    )
    ap.add_argument(
        "--only-weights",
        type=float,
        nargs="*",
        default=None,
        help="Evaluate only these w_GRU values instead of the full 0.0-1.0 grid.",
    )
    args = ap.parse_args()

    set_seed(CFG.train.random_seed)
    device = resolve_device(args.device)
    data_dir, ckpt_dir, out_dir = Path(args.data_dir), Path(args.ckpt_dir), Path(args.out_dir)

    vocab = Vocabulary.load(data_dir / "vocab.json", cfg=CFG.tokenizer)
    val_rxns = load_split("val", data_dir)

    if args.subset:
        rng = np.random.default_rng(CFG.train.random_seed)
        idx = rng.choice(len(val_rxns), size=min(args.subset, len(val_rxns)), replace=False)
        val_rxns = [val_rxns[i] for i in sorted(idx)]
        print(f"[sweep] Using a {len(val_rxns):,}-reaction validation subsample.")
    else:
        print(f"[sweep] Using the full validation split ({len(val_rxns):,} reactions).")

    gru = load_checkpoint(
        GRUSeq2Seq(vocab.size, CFG.gru, CFG.tokenizer),
        ckpt_dir / "gru" / "gru_selected.pt", device,
    )
    tf = load_checkpoint(
        TransformerSeq2Seq(vocab.size, CFG.transformer, CFG.tokenizer),
        ckpt_dir / "transformer" / "transformer_selected.pt", device,
    )

    if args.only_weights is not None:
        grid = list(args.only_weights)
    else:
        e = CFG.ensemble
        n = int(round((e.sweep_stop - e.sweep_start) / e.sweep_step)) + 1
        grid = [round(e.sweep_start + i * e.sweep_step, 2) for i in range(n)]

    print(f"[sweep] w_GRU grid: {grid}")
    print()

    rows = []
    for w_gru in grid:
        w_tf = round(1.0 - w_gru, 2)

        # Endpoints are single-model decoding.
        if w_gru >= 1.0:
            branches = [BranchAdapter(gru, "gru", 1.0)]
        elif w_gru <= 0.0:
            branches = [BranchAdapter(tf, "transformer", 1.0)]
        else:
            branches = [
                BranchAdapter(gru, "gru", w_gru),
                BranchAdapter(tf, "transformer", w_tf),
            ]

        records = decode_and_score(
            val_rxns, vocab, branches, device,
            beam_cfg=CFG.beam, eval_cfg=CFG.eval,
            batch_size=args.batch_size, verbose=False,
        )
        m = aggregate(records, CFG.eval, CFG.beam.topk_values, with_bootstrap=False)

        row = {
            "w_gru": w_gru,
            "w_tf": w_tf,
            "n": m["n"],
            "exact_match": m["exact_match"],
            "char_accuracy": m["char_accuracy"],
            "levenshtein_accuracy": m["levenshtein_accuracy"],
            "micro_f1": m["micro_f1"],
            "top5": m.get("top5", 0.0),
        }
        rows.append(row)
        print(
            f"[sweep] w_GRU={w_gru:.1f}/w_TF={w_tf:.1f}  "
            f"EM {row['exact_match']:.4f}  CA {row['char_accuracy']:.4f}  "
            f"LA {row['levenshtein_accuracy']:.4f}"
        )

    best = max(rows, key=lambda r: r["exact_match"])

    print()
    print("=" * 72)
    print(
        f"  Selected on validation: w_GRU = {best['w_gru']:.1f}, "
        f"w_TF = {best['w_tf']:.1f}  (EM = {best['exact_match']:.4f})"
    )
    print(
        f"  Configured weights:     w_GRU = {CFG.ensemble.w_gru}, "
        f"w_TF = {CFG.ensemble.w_tf}"
    )
    if abs(best["w_gru"] - CFG.ensemble.w_gru) > 1e-6:
        print("  To use the selected weights, update EnsembleConfig in src/config.py.")
    print("=" * 72)

    out = {
        "split": "val",
        "n_reactions": len(val_rxns),
        "subset": bool(args.subset),
        "beam_width": CFG.beam.beam_width,
        "grid": grid,
        "sweep": rows,
        "selected": best,
        "configured": {"w_gru": CFG.ensemble.w_gru, "w_tf": CFG.ensemble.w_tf},
    }
    p = out_dir / "fusion_weight_sweep.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[sweep] Wrote -> {p}")
    print("\nDone. Next: python scripts/05_evaluate_test.py --allow-test")


if __name__ == "__main__":
    main()
