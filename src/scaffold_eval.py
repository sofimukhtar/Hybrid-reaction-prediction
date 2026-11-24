"""
Unseen-scaffold evaluation on the existing train/test split.

Murcko scaffolds of the main reactant (the input fragment with the most heavy
atoms) are computed for train and test. The test set is filtered to reactions
whose non-empty scaffold does not occur in training; acyclic main reactants
are excluded. Already-trained models are scored on that subset.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .config import CFG, RESULTS_DIR, EvalConfig
from .data import Reaction
from .metrics import PerSampleRecord, bootstrap_ci, paired_bootstrap_test

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore


# ---------------------------------------------------------------------------
# Murcko scaffold extraction
# ---------------------------------------------------------------------------
def canonical_scaffold(mol) -> str:
    """Canonical Murcko scaffold SMILES, or "" if the molecule has no rings."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def main_reactant_scaffold(input_smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Main reactant = the '.'-separated input fragment with the most heavy atoms.

    Returns (scaffold_smiles_or_empty, main_fragment_smiles), or (None, None)
    if nothing in the input parses.
    """
    if Chem is None:
        raise RuntimeError("RDKit is required. pip install rdkit")

    parsed = []
    for frag in input_smiles.split("."):
        if not frag:
            continue
        m = Chem.MolFromSmiles(frag)
        if m is not None:
            parsed.append((frag, m, m.GetNumHeavyAtoms()))
    if not parsed:
        return None, None

    parsed.sort(key=lambda x: -x[2])
    main_smi, main_mol, _ = parsed[0]
    return canonical_scaffold(main_mol), main_smi


# ---------------------------------------------------------------------------
# Building the unseen-scaffold partition
# ---------------------------------------------------------------------------
def scaffold_sets(
    reactions: Sequence[Reaction],
    verbose: bool = True,
    label: str = "",
) -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """
    Compute scaffolds for a split.

    Returns
    -------
    by_id        : reaction id -> scaffold SMILES ("" for acyclic main reactant)
    scaffolds    : the set of non-empty scaffolds observed
    acyclic_ids  : ids whose main reactant has no ring system
    """
    by_id: Dict[str, str] = {}
    scaffolds: Set[str] = set()
    acyclic: Set[str] = set()
    n_fail = 0

    for i, r in enumerate(reactions, 1):
        sc, _ = main_reactant_scaffold(r.input_smiles)
        if sc is None:
            n_fail += 1
            continue
        by_id[r.rxn_id] = sc
        if sc:
            scaffolds.add(sc)
        else:
            acyclic.add(r.rxn_id)
        if verbose and i % 25_000 == 0:
            print(f"[scaffold] {label} {i:,}/{len(reactions):,} ...")

    if verbose:
        print(
            f"[scaffold] {label}: {len(by_id):,} reactions, "
            f"{len(scaffolds):,} unique non-empty scaffolds, "
            f"{len(acyclic):,} acyclic main reactants, "
            f"{n_fail:,} unparsable inputs excluded"
        )
    return by_id, scaffolds, acyclic


def build_unseen_scaffold_subset(
    train_reactions: Sequence[Reaction],
    test_reactions: Sequence[Reaction],
    eval_cfg: Optional[EvalConfig] = None,
    out_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Filter the test set to reactions whose main-reactant scaffold does not
    occur in training. Returns retained ids and summary statistics.
    """
    eval_cfg = eval_cfg or CFG.eval
    out_dir = Path(out_dir or RESULTS_DIR)

    train_by_id, train_scaffolds, train_acyclic = scaffold_sets(
        train_reactions, verbose=verbose, label="train"
    )
    test_by_id, test_scaffolds, test_acyclic = scaffold_sets(
        test_reactions, verbose=verbose, label="test"
    )

    overlap = train_scaffolds & test_scaffolds
    novel_scaffolds = test_scaffolds - train_scaffolds

    # Retain test reactions whose scaffold is non-empty and unseen in training.
    retained_ids = [
        rid for rid, sc in test_by_id.items()
        if sc and sc not in train_scaffolds
    ]

    group_sizes = Counter()
    per_scaffold = defaultdict(list)
    for rid in retained_ids:
        per_scaffold[test_by_id[rid]].append(rid)
    for sc, ids in per_scaffold.items():
        group_sizes[len(ids)] += 1

    stats: Dict[str, object] = {
        "n_train_reactions": len(train_by_id),
        "n_test_reactions": len(test_by_id),
        "n_unique_train_scaffolds": len(train_scaffolds),
        "n_unique_test_scaffolds": len(test_scaffolds),
        "n_overlapping_scaffolds": len(overlap),
        "n_novel_test_scaffolds": len(novel_scaffolds),
        "n_test_acyclic_excluded": len(test_acyclic),
        "n_train_acyclic": len(train_acyclic),
        "n_retained_test_reactions": len(retained_ids),
        "n_retained_scaffold_groups": len(per_scaffold),
        "retained_group_size_distribution": dict(sorted(group_sizes.items())),
        "singleton_scaffold_groups": group_sizes.get(1, 0),
    }

    # Leak check.
    retained_scaffolds = {test_by_id[rid] for rid in retained_ids}
    leak = retained_scaffolds & train_scaffolds
    assert not leak, (
        f"LEAK: {len(leak)} scaffolds appear in both the training set and the "
        f"retained unseen-scaffold subset. Refusing to proceed."
    )
    stats["leak_check"] = "CLEAN (retained scaffolds disjoint from train)"

    if verbose:
        print(
            f"[scaffold] Retained {len(retained_ids):,} test reactions with "
            f"unseen main-reactant scaffolds "
            f"({len(per_scaffold):,} distinct scaffolds)"
        )
        print(f"[scaffold] Leak check: {stats['leak_check']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unseen_scaffold_ids.txt").write_text("\n".join(sorted(retained_ids)))
    (out_dir / "unseen_scaffold_stats.json").write_text(json.dumps(stats, indent=2))
    with open(out_dir / "scaffold_assignments_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "main_reactant_scaffold", "retained_as_unseen"])
        for rid, sc in test_by_id.items():
            w.writerow([rid, sc, int(rid in set(retained_ids))])
    print(f"[scaffold] Wrote partition files -> {out_dir}")

    stats["retained_ids"] = retained_ids
    stats["test_scaffold_by_id"] = test_by_id
    return stats


def load_unseen_scaffold_ids(
    path: Optional[Path | str] = None,
    results_dir: Optional[Path] = None,
) -> List[str]:
    """
    Read retained ids from `build_unseen_scaffold_subset` (.txt) or from
    `scripts/run_scaffold_split.py` (.csv).
    """
    results_dir = Path(results_dir or RESULTS_DIR)
    candidates = [
        Path(path) if path else None,
        results_dir / "unseen_scaffold_ids.txt",
        results_dir / "scaffold_split_test_ids.csv",
        Path("scaffold_split_test_ids.csv"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            p = Path(p)
            if p.suffix == ".csv":
                with open(p, newline="") as f:
                    reader = csv.DictReader(f)
                    return [row["id"] for row in reader]
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    raise FileNotFoundError(
        "No unseen-scaffold id file found. Run "
        "`python scripts/07_scaffold_eval.py --build` first, or point --ids at "
        "the output of scripts/run_scaffold_split.py."
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def scaffold_table(
    records_by_model: Dict[str, Sequence[PerSampleRecord]],
    retained_ids: Sequence[str],
    eval_cfg: Optional[EvalConfig] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Exact match and Top-5 on the unseen-scaffold subset (mean +/- bootstrap
    CI half-width), computed from existing per-sample test records, plus a
    paired bootstrap of each model against the ensemble.
    """
    eval_cfg = eval_cfg or CFG.eval
    keep = set(retained_ids)

    out: Dict[str, object] = {"n_retained": len(keep), "models": {}}
    subsets: Dict[str, List[PerSampleRecord]] = {}

    for name, records in records_by_model.items():
        sub = [r for r in records if r.rxn_id in keep]
        subsets[name] = sub
        if not sub:
            print(f"[scaffold] WARNING: no records for {name} in the partition.")
            continue

        em = [r.exact_match for r in sub]
        t5 = [r.topk.get(5, 0) for r in sub]

        em_ci = bootstrap_ci(em, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed)
        t5_ci = bootstrap_ci(t5, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed)

        out["models"][name] = {
            "n": len(sub),
            "exact_match": em_ci["mean"],
            "exact_match_ci_half_width": em_ci["half_width"],
            "exact_match_ci": [em_ci["lo"], em_ci["hi"]],
            "top5": t5_ci["mean"],
            "top5_ci_half_width": t5_ci["half_width"],
            "top5_ci": [t5_ci["lo"], t5_ci["hi"]],
        }

    # Paired tests against the ensemble, aligned by reaction id.
    if "ensemble" in subsets:
        ens_by_id = {r.rxn_id: r for r in subsets["ensemble"]}
        paired: Dict[str, object] = {}
        for name, sub in subsets.items():
            if name == "ensemble":
                continue
            ids = [r.rxn_id for r in sub if r.rxn_id in ens_by_id]
            if not ids:
                continue
            a = [ens_by_id[i].exact_match for i in ids]
            b = [next(r for r in sub if r.rxn_id == i).exact_match for i in ids]
            paired[f"ensemble_vs_{name}"] = paired_bootstrap_test(
                a, b, eval_cfg.n_bootstrap, eval_cfg.bootstrap_seed
            )
        out["paired_bootstrap_exact_match"] = paired

    if verbose:
        print()
        print(f"  Unseen-scaffold subset (N = {len(keep):,})")
        print(f"  {'Model':<14} {'EM':>16} {'Top-5':>16}")
        print(f"  {'-'*14} {'-'*16} {'-'*16}")
        for name, d in out["models"].items():
            print(
                f"  {name:<14} "
                f"{d['exact_match']:.3f} +/- {d['exact_match_ci_half_width']:.3f}   "
                f"{d['top5']:.3f} +/- {d['top5_ci_half_width']:.3f}"
            )
        if "paired_bootstrap_exact_match" in out:
            print()
            print("  Paired bootstrap on exact match (ensemble minus other):")
            for k, v in out["paired_bootstrap_exact_match"].items():
                print(
                    f"    {k:<28} diff {v['mean_diff']:+.4f}  "
                    f"95% CI [{v['lo']:+.4f}, {v['hi']:+.4f}]  p = {v['p_value']:.3f}"
                )
        print()
    return out


__all__ = [
    "canonical_scaffold",
    "main_reactant_scaffold",
    "scaffold_sets",
    "build_unseen_scaffold_subset",
    "load_unseen_scaffold_ids",
    "scaffold_table",
]
