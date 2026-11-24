"""
Reaction-category assignment (ring-count rules) and per-category accuracy.

Categories: substitution, ring formation, functional group transformation.
The same rules are used in `scripts/classify_reaction_categories.py`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import CFG, RESULTS_DIR
from .metrics import PerSampleRecord, bootstrap_ci

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore


CATEGORY_LABELS = {
    "ring_formation": "Ring Formations",
    "substitution": "Substitutions",
    "functional_group_transformation": "Functional Group Transformations",
    "parse_error": "Parse error (excluded)",
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _parse_fragments(smiles: str) -> List:
    """Parse a '.'-separated SMILES string, skipping fragments that fail."""
    out = []
    for f in smiles.strip().split("."):
        if not f:
            continue
        m = Chem.MolFromSmiles(f)
        if m is not None:
            out.append(m)
    return out


def _ring_count(mol) -> int:
    """Number of rings in the SSSR."""
    return mol.GetRingInfo().NumRings()


def classify_reaction(input_smiles: str, output_smiles: str) -> str:
    """
    Assign one reaction to a category, by ring-count rules in priority order.

    1. Ring formation:  product_rings > max ring count over all reactant fragments
    2. Substitution:    main_reactant_rings > 0 and product_rings == main_reactant_rings
    3. Functional group transformation: everything else
    """
    if Chem is None:
        raise RuntimeError("RDKit is required. pip install rdkit")

    reactants = _parse_fragments(input_smiles)
    if not reactants:
        return "parse_error"

    products = _parse_fragments(output_smiles)
    if not products:
        return "parse_error"

    product_mol = max(products, key=lambda m: m.GetNumHeavyAtoms())
    main_reactant = max(reactants, key=lambda m: m.GetNumHeavyAtoms())

    max_reactant_rings = max(_ring_count(m) for m in reactants)
    product_rings = _ring_count(product_mol)
    main_rings = _ring_count(main_reactant)

    if product_rings > max_reactant_rings:
        return "ring_formation"
    if main_rings > 0 and product_rings == main_rings:
        return "substitution"
    return "functional_group_transformation"


def classify_records(
    records: Sequence[PerSampleRecord],
    verbose: bool = True,
) -> Dict[str, str]:
    """Classify each record by its reference (ground-truth) product."""
    out: Dict[str, str] = {}
    counts: Counter = Counter()
    for r in records:
        cat = classify_reaction(r.input_smiles, r.ref_smiles)
        out[r.rxn_id] = cat
        counts[cat] += 1

    if verbose:
        total = sum(v for k, v in counts.items() if k != "parse_error")
        print(f"[rxntype] Classified {len(records):,} reactions:")
        for key in ("substitution", "ring_formation", "functional_group_transformation"):
            n = counts.get(key, 0)
            pct = 100 * n / total if total else 0.0
            print(f"[rxntype]   {CATEGORY_LABELS[key]:<34} {n:>7,}  {pct:5.1f}%")
        if counts.get("parse_error"):
            print(f"[rxntype]   parse errors excluded: {counts['parse_error']:,}")
    return out


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------
def reaction_type_breakdown(
    records_by_model: Dict[str, Sequence[PerSampleRecord]],
    categories: Optional[Dict[str, str]] = None,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Exact-match accuracy per reaction category for each model, plus the
    frequency-weighted average and the aggregate exact match.
    """
    out_dir = Path(out_dir or RESULTS_DIR)

    if categories is None:
        any_records = next(iter(records_by_model.values()))
        categories = classify_records(any_records, verbose=verbose)

    result: Dict[str, object] = {"models": {}, "categories": {}}

    for name, records in records_by_model.items():
        by_cat: Dict[str, List[int]] = defaultdict(list)
        for r in records:
            cat = categories.get(r.rxn_id)
            if cat is None or cat == "parse_error":
                continue
            by_cat[cat].append(r.exact_match)

        model_out: Dict[str, object] = {}
        for cat, ems in by_cat.items():
            ci = bootstrap_ci(ems, CFG.eval.n_bootstrap, CFG.eval.bootstrap_ci, CFG.eval.bootstrap_seed)
            model_out[cat] = {
                "n": len(ems),
                "exact_match": ci["mean"],
                "ci_half_width": ci["half_width"],
            }

        # Frequency-weighted average using corpus-level category proportions.
        weights = CFG.data.reaction_type_proportions
        wavg = sum(
            weights.get(cat, 0.0) * d["exact_match"]
            for cat, d in model_out.items()
        )
        overall = float(np.mean([r.exact_match for r in records])) if records else 0.0
        model_out["_frequency_weighted_average"] = wavg
        model_out["_aggregate_exact_match"] = overall
        result["models"][name] = model_out

    counts = Counter(categories.values())
    total = sum(v for k, v in counts.items() if k != "parse_error")
    result["categories"] = {
        k: {"count": v, "fraction": v / total if total else 0.0}
        for k, v in counts.items()
    }

    if verbose:
        print()
        print("  Exact-match accuracy by reaction type")
        header = f"  {'Reaction Type':<34}"
        for name in records_by_model:
            header += f" {name[:11]:>12}"
        print(header)
        print(f"  {'-'*34}" + "".join(f" {'-'*12}" for _ in records_by_model))
        for cat in ("functional_group_transformation", "ring_formation", "substitution"):
            line = f"  {CATEGORY_LABELS[cat]:<34}"
            for name in records_by_model:
                d = result["models"][name].get(cat)
                line += f" {d['exact_match']:>12.3f}" if d else f" {'-':>12}"
            print(line)
        print()
        for name in records_by_model:
            m = result["models"][name]
            print(
                f"  {name:<12} weighted avg {m['_frequency_weighted_average']:.4f}  "
                f"aggregate EM {m['_aggregate_exact_match']:.4f}"
            )
        print()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reaction_type_breakdown.json").write_text(json.dumps(result, indent=2))
    print(f"[rxntype] Wrote -> {out_dir / 'reaction_type_breakdown.json'}")
    return result


__all__ = [
    "CATEGORY_LABELS",
    "classify_reaction",
    "classify_records",
    "reaction_type_breakdown",
]
