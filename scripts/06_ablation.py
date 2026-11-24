#!/usr/bin/env python3
"""
Fusion-strategy ablation on the validation split.

    No fusion (GRU)                 single-model beam search
    No fusion (Transformer)         single-model beam search
    Equal-weight log-space          0.5/0.5, step-wise
    Probability-space               0.5/0.5, weighted mean of probabilities
    Learned per-step weights        gating MLP (src/gating.py)
    Weighted log-space (0.4/0.6)    step-wise
    Weighted log-space (configured) step-wise, EnsembleConfig weights
    Sequence-level rescoring        GRU beam search, then weighted rescoring

Usage
-----
    python scripts/06_ablation.py --subset 3000
    python scripts/06_ablation.py --train-gate --subset 3000
    python scripts/06_ablation.py                      # full validation split
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, CKPT_DIR, DATA_DIR, RESULTS_DIR
from src.data import ReactionDataset, load_split, make_dataloader
from src.decode import BranchAdapter, beam_search_final_step_fusion
from src.evaluate import decode_and_score
from src.gating import StepwiseGate, beam_search_gated, load_gate, train_gate
from src.metrics import aggregate, score_one
from src.models import GRUSeq2Seq, TransformerSeq2Seq
from src.tokenizer import Vocabulary
from src.train import load_checkpoint, resolve_device, set_seed


def _score_results(reactions, results, eval_cfg, topk_values) -> Dict[str, object]:
    """Score a list of DecodeResult objects against their reactions."""
    records = []
    for rxn, res in zip(reactions, results):
        cands = [h.smiles for h in res.hypotheses]
        top = res.top1
        records.append(
            score_one(
                rxn_id=rxn.rxn_id,
                input_smiles=rxn.input_smiles,
                ref_smiles=rxn.output_smiles,
                candidates=cands,
                confidence=top.confidence if top else 0.0,
                norm_score=top.norm_score if top else float("-inf"),
                eval_cfg=eval_cfg,
                topk_values=topk_values,
            )
        )
    return aggregate(records, eval_cfg, topk_values, with_bootstrap=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=str, default=CFG.runtime.device)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--batch-size", type=int, default=CFG.runtime.eval_batch_size)
    ap.add_argument("--subset", type=int, default=None,
                    help="Run on a random validation subsample of this size.")
    ap.add_argument("--train-gate", action="store_true",
                    help="Train the learned per-step gating network first.")
    ap.add_argument("--gate-epochs", type=int, default=3)
    ap.add_argument("--skip-gate", action="store_true",
                    help="Omit the learned-weights row (e.g. no gate trained).")
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
    print(f"[ablation] Validation reactions: {len(val_rxns):,}")

    gru = load_checkpoint(
        GRUSeq2Seq(vocab.size, CFG.gru, CFG.tokenizer),
        ckpt_dir / "gru" / "gru_selected.pt", device,
    )
    tf = load_checkpoint(
        TransformerSeq2Seq(vocab.size, CFG.transformer, CFG.tokenizer),
        ckpt_dir / "transformer" / "transformer_selected.pt", device,
    )

    # ---- optionally train the gate ----------------------------------------
    gate = None
    if not args.skip_gate:
        if args.train_gate:
            train_rxns = load_split("train", data_dir)
            if args.subset:
                train_rxns = train_rxns[: max(args.subset * 4, 2000)]
            loader = make_dataloader(
                ReactionDataset(train_rxns, vocab),
                batch_size=CFG.train.batch_size, shuffle=True, runtime=CFG.runtime,
            )
            gate = train_gate(
                StepwiseGate(CFG.gru.hidden_dim, CFG.transformer.d_model),
                gru, tf, loader, vocab, device, epochs=args.gate_epochs,
                out_path=ckpt_dir / "stepwise_gate.pt",
            )
        else:
            try:
                gate = load_gate(ckpt_dir / "stepwise_gate.pt", device)
            except FileNotFoundError:
                print(
                    "[ablation] No trained gate found; the learned-weights row "
                    "will be skipped. Pass --train-gate to produce it."
                )

    rows: List[Dict[str, object]] = []

    def run_static(label: str, w_gru: float, w_tf: float, space: str = "log") -> None:
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
            batch_size=args.batch_size, fusion_space=space, verbose=False,
        )
        m = aggregate(records, CFG.eval, CFG.beam.topk_values, with_bootstrap=False)
        rows.append({
            "strategy": label,
            "w": f"{w_gru:.1f}/{w_tf:.1f}",
            "em": m["exact_match"],
            "ca": m["char_accuracy"],
            "la": m["levenshtein_accuracy"],
        })
        print(f"[ablation] {label:<32} EM {m['exact_match']:.4f}  "
              f"CA {m['char_accuracy']:.4f}  LA {m['levenshtein_accuracy']:.4f}")

    print()
    run_static("No fusion (GRU)", 1.0, 0.0)
    run_static("No fusion (Transformer)", 0.0, 1.0)
    run_static("Equal-weight log-space", 0.5, 0.5, space="log")
    run_static("Probability-space", 0.5, 0.5, space="prob")

    # ---- learned per-step weights ------------------------------------------
    if gate is not None:
        all_results = []
        for start in range(0, len(val_rxns), args.batch_size):
            batch = val_rxns[start : start + args.batch_size]
            src = torch.stack([
                torch.tensor(vocab.encode(r.input_smiles), dtype=torch.long)
                for r in batch
            ]).to(device)
            all_results.extend(beam_search_gated(gate, gru, tf, src, vocab, CFG.beam))
        m = _score_results(val_rxns, all_results, CFG.eval, CFG.beam.topk_values)
        rows.append({
            "strategy": "Learned per-step weights",
            "w": "Dynamic",
            "em": m["exact_match"], "ca": m["char_accuracy"], "la": m["levenshtein_accuracy"],
        })
        print(f"[ablation] {'Learned per-step weights':<32} EM {m['exact_match']:.4f}  "
              f"CA {m['char_accuracy']:.4f}  LA {m['levenshtein_accuracy']:.4f}")

    run_static("Weighted log-space (0.4/0.6)", 0.4, 0.6)
    run_static("Weighted log-space (configured)", CFG.ensemble.w_gru, CFG.ensemble.w_tf)

    # ---- sequence-level rescoring ------------------------------------------
    all_results = []
    for start in range(0, len(val_rxns), args.batch_size):
        batch = val_rxns[start : start + args.batch_size]
        src = torch.stack([
            torch.tensor(vocab.encode(r.input_smiles), dtype=torch.long)
            for r in batch
        ]).to(device)
        all_results.extend(
            beam_search_final_step_fusion(
                BranchAdapter(gru, "gru", 1.0),
                BranchAdapter(tf, "transformer", 1.0),
                src, vocab, CFG.beam,
                w_gru=CFG.ensemble.w_gru, w_tf=CFG.ensemble.w_tf,
            )
        )
    m = _score_results(val_rxns, all_results, CFG.eval, CFG.beam.topk_values)
    rows.append({
        "strategy": "Sequence-level rescoring",
        "w": f"{CFG.ensemble.w_gru:.1f}/{CFG.ensemble.w_tf:.1f}",
        "em": m["exact_match"], "ca": m["char_accuracy"], "la": m["levenshtein_accuracy"],
    })
    print(f"[ablation] {'Sequence-level rescoring':<32} EM {m['exact_match']:.4f}  "
          f"CA {m['char_accuracy']:.4f}  LA {m['levenshtein_accuracy']:.4f}")

    # ---- output ------------------------------------------------------------
    print()
    print("  Fusion ablation (validation split)")
    print(f"  {'Fusion Strategy':<32} {'GRU/TF':>9} {'EM':>7} {'CA':>7} {'LA':>7}")
    print(f"  {'-'*32} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")
    for r in rows:
        print(
            f"  {r['strategy']:<32} {r['w']:>9} {r['em']:>7.3f} {r['ca']:>7.3f} "
            f"{r['la']:>7.3f}"
        )
    print()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Fusion Strategy", "GRU/TF", "EM", "CA", "LA"])
        for r in rows:
            w.writerow([
                r["strategy"], r["w"],
                f"{r['em']:.4f}", f"{r['ca']:.4f}", f"{r['la']:.4f}",
            ])
    print(f"[ablation] Wrote -> {out_dir / 'ablation.csv'}")

    (out_dir / "ablation.json").write_text(json.dumps({
        "split": "val",
        "n_reactions": len(val_rxns),
        "subset": bool(args.subset),
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
