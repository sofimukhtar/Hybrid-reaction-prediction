"""
Chemical validity analysis and element-provenance check.

Element provenance: a violation is recorded when a predicted product contains
a heavy-atom element (hydrogen excluded) that is absent from all input
reactants and reagents. Elements missing from the product are not counted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

from .config import EvalConfig
from .metrics import PerSampleRecord

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore


# ---------------------------------------------------------------------------
# Elemental inventory
# ---------------------------------------------------------------------------
def heavy_atom_elements(smiles: str) -> Set[str]:
    """
    Heavy-atom element symbols in a SMILES string (hydrogen excluded).
    Falls back to unsanitized parsing so invalid predictions can be inspected.
    """
    if Chem is None:
        raise RuntimeError("RDKit is required. pip install rdkit")
    if not smiles or not smiles.strip():
        return set()

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
        except Exception:
            return set()
    if mol is None:
        return set()

    out: Set[str] = set()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != "H":
            out.add(sym)
    return out


@dataclass
class ProvenanceViolation:
    """One prediction that invented an element."""

    rxn_id: str
    input_smiles: str
    ref_smiles: str
    pred_smiles: str
    input_elements: List[str]
    predicted_elements: List[str]
    invented_elements: List[str]
    was_exact_match: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def check_element_provenance(
    input_smiles: str,
    pred_smiles: str,
) -> Set[str]:
    """Elements present in the prediction but absent from the input."""
    e_input = heavy_atom_elements(input_smiles)
    e_pred = heavy_atom_elements(pred_smiles)
    return e_pred - e_input


def element_provenance_analysis(
    records: Sequence[PerSampleRecord],
    restrict_to_failures: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Count element-provenance violations.

    restrict_to_failures : scan only predictions that are not exact matches.

    Returns counts, rates (of failures and of all predictions), a per-element
    tally and the list of violations.
    """
    n_all = len(records)
    failures = [r for r in records if not r.exact_match]
    scan = failures if restrict_to_failures else list(records)

    violations: List[ProvenanceViolation] = []
    invented_tally: Counter = Counter()

    for r in scan:
        invented = check_element_provenance(r.input_smiles, r.pred_smiles)
        if invented:
            violations.append(
                ProvenanceViolation(
                    rxn_id=r.rxn_id,
                    input_smiles=r.input_smiles,
                    ref_smiles=r.ref_smiles,
                    pred_smiles=r.pred_smiles,
                    input_elements=sorted(heavy_atom_elements(r.input_smiles)),
                    predicted_elements=sorted(heavy_atom_elements(r.pred_smiles)),
                    invented_elements=sorted(invented),
                    was_exact_match=bool(r.exact_match),
                )
            )
            invented_tally.update(invented)

    n_viol = len(violations)
    out: Dict[str, object] = {
        "n_all_predictions": n_all,
        "n_incorrect_predictions": len(failures),
        "n_scanned": len(scan),
        "n_violations": n_viol,
        "violation_rate_of_failures": (n_viol / len(failures)) if failures else 0.0,
        "violation_rate_of_all": (n_viol / n_all) if n_all else 0.0,
        "inventory_respected_rate": 1.0 - ((n_viol / n_all) if n_all else 0.0),
        "invented_element_counts": dict(invented_tally.most_common()),
        "violations": [v.to_dict() for v in violations],
    }

    if verbose:
        print(f"[provenance] predictions scanned      : {len(scan):,}")
        print(f"[provenance] incorrect predictions    : {len(failures):,}")
        print(f"[provenance] element violations       : {n_viol:,}")
        print(
            f"[provenance] as % of failures         : "
            f"{100 * out['violation_rate_of_failures']:.2f}%"
        )
        print(
            f"[provenance] as % of all reactions    : "
            f"{100 * out['violation_rate_of_all']:.2f}%"
        )
        print(
            f"[provenance] inventory respected in   : "
            f"{100 * out['inventory_respected_rate']:.2f}% of products"
        )
        if invented_tally:
            top = ", ".join(f"{el} x{c}" for el, c in invented_tally.most_common(8))
            print(f"[provenance] most-invented elements   : {top}")
    return out


# ---------------------------------------------------------------------------
# Validity summary
# ---------------------------------------------------------------------------
def validity_table(
    records: Sequence[PerSampleRecord],
    eval_cfg: Optional[EvalConfig] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    syntactically_valid : RDKit parses without sanitization
    chemically_valid    : passes full RDKit sanitization
    severe_invalid_fraction : among invalid predictions, the fraction with
                              Levenshtein accuracy below the threshold
    """
    eval_cfg = eval_cfg or EvalConfig()
    n = len(records)
    if n == 0:
        return {}

    n_syn = sum(r.syntactically_valid for r in records)
    n_chem = sum(r.chemically_valid for r in records)
    invalid = [r for r in records if not r.chemically_valid]
    severe = [
        r for r in invalid
        if r.levenshtein_accuracy < eval_cfg.severe_la_threshold
    ]

    out = {
        "n": n,
        "syntactically_valid": n_syn / n,
        "chemically_valid": n_chem / n,
        "n_invalid": len(invalid),
        "n_severe_invalid": len(severe),
        "severe_invalid_fraction": (len(severe) / len(invalid)) if invalid else 0.0,
        "near_miss_invalid_fraction": (
            1.0 - len(severe) / len(invalid)
        ) if invalid else 0.0,
    }

    if verbose:
        print(f"[validity] syntactically valid : {out['syntactically_valid']:.3f}")
        print(f"[validity] chemically valid    : {out['chemically_valid']:.3f}")
        print(
            f"[validity] invalid with LA<{eval_cfg.severe_la_threshold}  : "
            f"{out['severe_invalid_fraction']:.3f} "
            f"({len(severe):,} of {len(invalid):,} invalid)"
        )
    return out


__all__ = [
    "heavy_atom_elements",
    "ProvenanceViolation",
    "check_element_provenance",
    "element_provenance_analysis",
    "validity_table",
]
