#!/usr/bin/env python3
"""
Standalone reaction classifier: downloads SLM4CRP_with_RTs and assigns each
unique forward-prediction reaction to one category using ring-count rules
(see classify_reaction). Installs rdkit, datasets and pandas if missing.

If scaffold_groups.csv (from run_scaffold_split.py) is in the same directory,
a per-split breakdown is also reported.

Usage
-----
    python classify_reaction_categories.py

Outputs
-------
    reaction_categories.csv
    reaction_category_stats.json
    reaction_category_by_split.json
"""

import sys
import subprocess
import importlib
import json
import csv
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Self-install dependencies
# ---------------------------------------------------------------------------
def ensure_installed(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
        return
    except ImportError:
        print(f"[setup] Installing {pip_name} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--break-system-packages", pip_name]
        )

for pip_name, import_name in [
    ("rdkit", "rdkit"),
    ("datasets", "datasets"),
    ("pandas", "pandas"),
]:
    ensure_installed(pip_name, import_name)

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_DATASET_ID  = "liupf/SLM4CRP_with_RTs"
TASK_FILTER    = "forward reaction prediction"

try:
    OUTPUT_DIR = Path(__file__).parent
except NameError:
    OUTPUT_DIR = Path.cwd()

# ---------------------------------------------------------------------------
# 1. Chemistry helpers
# ---------------------------------------------------------------------------

def ring_count(mol):
    """Number of rings in the SSSR."""
    return mol.GetRingInfo().NumRings()


def parse_smiles_list(smiles_str):
    """
    Parse a '.'-separated SMILES string into a list of (smiles, mol) pairs.
    Fragments that fail to parse are silently skipped.
    Returns [] if nothing parses.
    """
    frags = smiles_str.strip().split(".")
    parsed = []
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is not None:
            parsed.append((f, m))
    return parsed


def classify_reaction(input_smiles: str, output_smiles: str) -> str:
    """
    Returns 'ring_formation', 'substitution',
    'functional_group_transformation' or 'parse_error'.

    1. Ring formation:  product_rings > max ring count over all reactant fragments
    2. Substitution:    main_reactant_rings > 0 and product_rings == main_reactant_rings
    3. Functional group transformation: everything else

    The main reactant and main product are the heaviest fragments.
    """
    # --- Parse reactant side ---
    reactant_frags = parse_smiles_list(input_smiles)
    if not reactant_frags:
        return "parse_error"

    # --- Parse product side (take heaviest fragment as main product) ---
    product_frags = parse_smiles_list(output_smiles)
    if not product_frags:
        return "parse_error"
    _, product_mol = max(product_frags, key=lambda x: x[1].GetNumHeavyAtoms())

    # --- Main reactant = fragment with most heavy atoms ---
    _, main_react_mol = max(reactant_frags, key=lambda x: x[1].GetNumHeavyAtoms())

    max_reactant_rings = max(ring_count(m) for _, m in reactant_frags)
    product_rings      = ring_count(product_mol)
    main_react_rings   = ring_count(main_react_mol)

    if product_rings > max_reactant_rings:
        return "ring_formation"

    if main_react_rings > 0 and product_rings == main_react_rings:
        return "substitution"

    return "functional_group_transformation"


# ---------------------------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------------------------
def load_records():
    print(f"[load] Downloading '{HF_DATASET_ID}' from Hugging Face "
          f"(cached after first run)...")
    ds = load_dataset(HF_DATASET_ID, split="train")
    print(f"[load] Loaded {len(ds):,} total rows.")

    records = []
    skipped = 0
    for row in ds:
        meta = row.get("metadata")
        task = None
        if isinstance(meta, dict):
            task = meta.get("task")
        elif isinstance(meta, str):
            try:
                task = json.loads(meta).get("task")
            except Exception:
                pass

        if TASK_FILTER and task != TASK_FILTER:
            skipped += 1
            continue
        records.append({
            "id"    : row.get("id", ""),
            "input" : row.get("input", ""),
            "output": row.get("output", ""),
        })

    print(f"[load] {skipped:,} rows skipped (different task). "
          f"{len(records):,} rows kept for task='{TASK_FILTER}'.")

    # De-duplicate by id
    by_id = {}
    for r in records:
        if r["id"] not in by_id:
            by_id[r["id"]] = r
    unique_records = list(by_id.values())
    print(f"[load] {len(records):,} rows -> {len(unique_records):,} unique "
          f"reactions after de-duplicating by id.")
    return unique_records


# ---------------------------------------------------------------------------
# 3. Load scaffold split assignment (optional, if available)
# ---------------------------------------------------------------------------
def load_split_assignment():
    """
    Read scaffold_groups.csv (produced by run_scaffold_split.py) to get
    the train/test assignment per reaction id. Returns {} if file absent.
    """
    p = OUTPUT_DIR / "scaffold_groups.csv"
    if not p.exists():
        print("[split] scaffold_groups.csv not found -- split breakdown will be skipped.")
        return {}
    assignment = {}
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assignment[row["id"]] = row["split"]
    print(f"[split] Loaded split assignment for {len(assignment):,} reactions "
          f"from scaffold_groups.csv.")
    return assignment


# ---------------------------------------------------------------------------
# 4. Classify all reactions
# ---------------------------------------------------------------------------
def classify_all(unique_records):
    results = []  # list of {"id", "category"}
    counts  = Counter()
    n       = len(unique_records)

    print(f"[classify] Classifying {n:,} unique reactions ...")
    for i, r in enumerate(unique_records, 1):
        cat = classify_reaction(r["input"], r["output"])
        results.append({"id": r["id"], "category": cat})
        counts[cat] += 1
        if i % 50_000 == 0:
            print(f"[classify]   {i:,}/{n:,} done ...")

    print(f"[classify] Done. Counts: {dict(counts)}")
    return results, counts


# ---------------------------------------------------------------------------
# 5. Build stats dict
# ---------------------------------------------------------------------------
def build_stats(counts, split_assignment, results):
    total_valid = sum(v for k, v in counts.items() if k != "parse_error")
    parse_errors = counts.get("parse_error", 0)

    label_map = {
        "ring_formation"               : "Ring Formations",
        "substitution"                 : "Substitutions",
        "functional_group_transformation": "Functional Group Transformations",
    }

    category_stats = {}
    for key, label in label_map.items():
        n = counts.get(key, 0)
        category_stats[label] = {
            "count"      : n,
            "percentage" : round(100 * n / total_valid, 1) if total_valid else 0,
        }

    stats = {
        "dataset"                   : HF_DATASET_ID,
        "task"                      : TASK_FILTER,
        "n_unique_reactions_total"  : len(results),
        "n_parse_errors_excluded"   : parse_errors,
        "n_valid_reactions"         : total_valid,
        "classification_method"     : "ring_count_rules",
        "categories"                : category_stats,
    }

    # Per-split breakdown (only if scaffold_groups.csv was available)
    if split_assignment:
        split_cats = defaultdict(Counter)
        for r in results:
            sp = split_assignment.get(r["id"], "unknown")
            if r["category"] != "parse_error":
                split_cats[sp][r["category"]] += 1

        split_stats = {}
        for sp, cat_counts in split_cats.items():
            sp_total = sum(cat_counts.values())
            split_stats[sp] = {
                label_map.get(k, k): {
                    "count"     : v,
                    "percentage": round(100 * v / sp_total, 1) if sp_total else 0,
                }
                for k, v in cat_counts.items()
            }
        stats["breakdown_by_split"] = split_stats

    return stats


# ---------------------------------------------------------------------------
# 6. Write outputs
# ---------------------------------------------------------------------------
def write_outputs(results, stats, split_assignment):
    # reaction_categories.csv
    cats_path = OUTPUT_DIR / "reaction_categories.csv"
    with open(cats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "split"])
        for r in results:
            sp = split_assignment.get(r["id"], "")
            w.writerow([r["id"], r["category"], sp])
    print(f"[write] Wrote {len(results):,} rows -> {cats_path}")

    # reaction_category_stats.json
    stats_path = OUTPUT_DIR / "reaction_category_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[write] Wrote stats -> {stats_path}")

    # reaction_category_by_split.json (if split data present)
    if "breakdown_by_split" in stats:
        split_path = OUTPUT_DIR / "reaction_category_by_split.json"
        with open(split_path, "w") as f:
            json.dump(stats["breakdown_by_split"], f, indent=2)
        print(f"[write] Wrote per-split breakdown -> {split_path}")


# ---------------------------------------------------------------------------
# 7. Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(stats):
    print("\n" + "=" * 55)
    print("  REACTION CATEGORY SUMMARY")
    print("=" * 55)
    print(f"  Dataset      : {stats['dataset']}")
    print(f"  Task         : {stats['task']}")
    print(f"  Valid rxns   : {stats['n_valid_reactions']:,}")
    print(f"  Parse errors : {stats['n_parse_errors_excluded']:,} (excluded)")
    print()
    print(f"  {'Category':<35} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*6}")
    for label, d in stats["categories"].items():
        print(f"  {label:<35} {d['count']:>8,}  {d['percentage']:>5.1f}%")
    print()

    if "breakdown_by_split" in stats:
        print("  Per-split breakdown:")
        for sp, cat_map in stats["breakdown_by_split"].items():
            sp_total = sum(v["count"] for v in cat_map.values())
            print(f"\n    Split = {sp!r}  (n={sp_total:,})")
            for label, d in cat_map.items():
                print(f"      {label:<35} {d['count']:>8,}  {d['percentage']:>5.1f}%")
    print("=" * 55)
    print()
    print("Files written:")
    print("  reaction_categories.csv         -- per-reaction category + split")
    print("  reaction_category_stats.json    -- counts and percentages")
    if "breakdown_by_split" in stats:
        print("  reaction_category_by_split.json -- breakdown by train/test split")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    unique_records  = load_records()
    split_assign    = load_split_assignment()
    results, counts = classify_all(unique_records)
    stats           = build_stats(counts, split_assign, results)
    write_outputs(results, stats, split_assign)
    print_summary(stats)


if __name__ == "__main__":
    main()
