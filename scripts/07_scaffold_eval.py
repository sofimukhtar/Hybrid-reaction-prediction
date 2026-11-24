#!/usr/bin/env python3
"""
Evaluate models on test reactions whose main-reactant Murcko scaffold does not
occur in training.

    --build   compute scaffolds for train and test, filter the test split to
              unseen scaffolds, and write the subset ids to results/.
    (default) restrict the per-sample test records from scripts/05 to that
              subset and report exact match / Top-5 with bootstrap CIs and
              paired bootstrap tests against the ensemble.

--ids accepts an id file from scripts/run_scaffold_split.py instead.
--t5chem-predictions adds a T5Chem column (see scripts/10_t5chem_baseline.py).

Usage
-----
    python scripts/07_scaffold_eval.py --build
    python scripts/07_scaffold_eval.py
    python scripts/07_scaffold_eval.py --t5chem-predictions results/t5chem_test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_DIR, RESULTS_DIR
from src.data import load_split
from src.evaluate import read_per_sample
from src.metrics import PerSampleRecord
from src.scaffold_eval import build_unseen_scaffold_subset, load_unseen_scaffold_ids, scaffold_table

MODELS = ("transformer", "gru", "ensemble")


def load_t5chem_records(path: Path) -> List[PerSampleRecord]:
    """
    Load T5Chem predictions. JSONL schema per line:
        {"id": ..., "input": ..., "ref": ..., "candidates": [...]}
    Candidates ranked best first.
    """
    from src.metrics import score_one

    out: List[PerSampleRecord] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cands = d.get("candidates") or ([d["pred"]] if "pred" in d else [])
            out.append(
                score_one(
                    rxn_id=str(d["id"]),
                    input_smiles=d.get("input", ""),
                    ref_smiles=d["ref"],
                    candidates=cands,
                    confidence=float(d.get("confidence", 0.0)),
                    norm_score=float(d.get("norm_score", 0.0)),
                    eval_cfg=CFG.eval,
                    topk_values=CFG.beam.topk_values,
                )
            )
    print(f"[t5chem] Loaded {len(out):,} baseline predictions <- {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true",
                    help="Compute the unseen-scaffold partition from scratch.")
    ap.add_argument("--ids", type=str, default=None,
                    help="Use an existing id file instead of building one.")
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--t5chem-predictions", type=str, default=None)
    args = ap.parse_args()

    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)

    if args.build:
        print("=" * 72)
        print("  Building the unseen-scaffold subset")
        print("=" * 72)
        train_rxns = load_split("train", data_dir)
        test_rxns = load_split("test", data_dir)
        build_unseen_scaffold_subset(
            train_rxns, test_rxns, eval_cfg=CFG.eval, out_dir=results_dir
        )
        print()

    retained = load_unseen_scaffold_ids(args.ids, results_dir)
    print(f"[scaffold] Partition size: {len(retained):,} reactions")

    records_by_model: Dict[str, List[PerSampleRecord]] = {}
    for m in MODELS:
        p = results_dir / f"per_sample_{m}_test.jsonl"
        if p.exists():
            records_by_model[m] = read_per_sample(p)
            print(f"[scaffold] {m}: {len(records_by_model[m]):,} test records")
        else:
            print(
                f"[scaffold] {m}: {p.name} missing -- run "
                f"`python scripts/05_evaluate_test.py --allow-test` first."
            )

    if args.t5chem_predictions:
        records_by_model["t5chem"] = load_t5chem_records(Path(args.t5chem_predictions))
    else:
        print("[scaffold] No T5Chem predictions supplied; T5Chem column omitted.")

    if not records_by_model:
        print("\nNothing to evaluate. Run scripts/05 first.")
        return

    table = scaffold_table(records_by_model, retained, eval_cfg=CFG.eval)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "scaffold_eval.json").write_text(json.dumps(table, indent=2))

    with open(results_dir / "scaffold_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "N", "Exact Match", "EM +/-", "Top-5", "Top-5 +/-"])
        for name, d in table["models"].items():
            w.writerow([
                name, d["n"],
                f"{d['exact_match']:.4f}", f"{d['exact_match_ci_half_width']:.4f}",
                f"{d['top5']:.4f}", f"{d['top5_ci_half_width']:.4f}",
            ])
    print(f"[scaffold] Wrote -> {results_dir / 'scaffold_eval.csv'}")


if __name__ == "__main__":
    main()
