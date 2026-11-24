#!/usr/bin/env python3
"""
Exact-match accuracy per reaction category, from the per-sample test records
written by scripts/05.

--cross-check compares the labels with the CSV produced by
scripts/classify_reaction_categories.py.

Usage
-----
    python scripts/09_reaction_type_breakdown.py
    python scripts/09_reaction_type_breakdown.py --cross-check reaction_categories.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, RESULTS_DIR
from src.evaluate import read_per_sample
from src.metrics import PerSampleRecord
from src.reaction_types import CATEGORY_LABELS, classify_records, reaction_type_breakdown

MODELS = ("transformer", "gru", "ensemble")


def cross_check(
    categories: Dict[str, str],
    csv_path: Path,
) -> None:
    """Compare labels with scripts/classify_reaction_categories.py output."""
    if not csv_path.exists():
        print(f"[xcheck] {csv_path} not found; skipping cross-check.")
        return

    external: Dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            external[str(row["id"])] = row["category"]

    shared = set(categories) & set(external)
    if not shared:
        print("[xcheck] No overlapping reaction ids with the external CSV.")
        return

    agree = sum(1 for i in shared if categories[i] == external[i])
    disagree = [(i, categories[i], external[i]) for i in shared if categories[i] != external[i]]

    print(f"[xcheck] {len(shared):,} shared reactions, "
          f"{agree:,} agree ({100*agree/len(shared):.2f}%)")
    if disagree:
        print(f"[xcheck] {len(disagree):,} disagreements. Most common patterns:")
        patt = Counter((a, b) for _, a, b in disagree)
        for (a, b), n in patt.most_common(5):
            print(f"[xcheck]   src says {a!r}, script says {b!r}: {n:,}")
    else:
        print("[xcheck] The two implementations agree on every shared reaction.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--cross-check", type=str, default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    records_by_model: Dict[str, List[PerSampleRecord]] = {}
    for m in MODELS:
        p = results_dir / f"per_sample_{m}_test.jsonl"
        if p.exists():
            records_by_model[m] = read_per_sample(p)
            print(f"[rxntype] {m}: {len(records_by_model[m]):,} records")
        else:
            print(f"[rxntype] {p.name} missing; run scripts/05 first.")

    if not records_by_model:
        print("\nNothing to break down. Run scripts/05_evaluate_test.py first.")
        return

    print()
    categories = classify_records(next(iter(records_by_model.values())))

    if args.cross_check:
        print()
        cross_check(categories, Path(args.cross_check))

    table = reaction_type_breakdown(
        records_by_model, categories=categories, out_dir=results_dir
    )

    present = [m for m in MODELS if m in table["models"]]
    with open(results_dir / "reaction_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Reaction Type", "N"] + present)
        for cat in ("functional_group_transformation", "ring_formation", "substitution"):
            n = table["models"][present[0]].get(cat, {}).get("n", 0)
            w.writerow(
                [CATEGORY_LABELS[cat], n]
                + [f"{table['models'][m].get(cat, {}).get('exact_match', 0):.4f}" for m in present]
            )
    print(f"[rxntype] Wrote -> {results_dir / 'reaction_types.csv'}")


if __name__ == "__main__":
    main()
