#!/usr/bin/env python3
"""
Standalone scaffold split: downloads SLM4CRP_with_RTs, extracts main-reactant
Murcko scaffolds and builds an 80/20 train/test split with no scaffold
overlap. Installs rdkit, datasets and pandas if missing.

Method
------
1. Keep rows with task == "forward reaction prediction".
2. De-duplicate to one row per reaction id (flags ids mapping to different
   inputs).
3. Main reactant = input fragment with the most heavy atoms; compute its
   canonical Murcko scaffold.
4. Group reactions by scaffold; acyclic main reactants go to train.
5. Assign scaffold groups smallest-first to test until ~20% of unique
   reactions are covered; the rest go to train.
6. Assert no scaffold appears in both splits.
7. Expand the split back to all original rows.

Usage
-----
    python run_scaffold_split.py

Outputs (next to this script, or the current directory in a notebook)
---------------------------------------------------------------------
    scaffold_split_stats.json
    scaffold_split_train_ids.csv / scaffold_split_test_ids.csv
    scaffold_split_train_rows.json / scaffold_split_test_rows.json
    scaffold_groups.csv
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
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_DATASET_URL = "https://huggingface.co/datasets/liupf/SLM4CRP_with_RTs"
HF_DATASET_ID = "liupf/SLM4CRP_with_RTs"
TASK_FILTER = "forward reaction prediction"   # set to None to use ALL rows/tasks
TEST_FRAC = 0.20
RANDOM_SEED = 42

try:
    OUTPUT_DIR = Path(__file__).parent
except NameError:
    OUTPUT_DIR = Path.cwd()


# ---------------------------------------------------------------------------
# 1. Scaffold logic
# ---------------------------------------------------------------------------
def canonical_scaffold(mol):
    """Return canonical Murcko scaffold SMILES, or '' if the molecule has no ring."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def main_reactant_scaffold(input_smiles):
    """
    'Main reactant' = the '.'-separated input fragment with the most heavy
    atoms. Returns (scaffold_smiles_or_empty, main_fragment_smiles) or
    (None, None) if nothing in the input parses.
    """
    frags = input_smiles.split(".")
    parsed = []
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is not None:
            parsed.append((f, m, m.GetNumHeavyAtoms()))
    if not parsed:
        return None, None
    parsed.sort(key=lambda x: -x[2])
    main_smi, main_mol, _ = parsed[0]
    return canonical_scaffold(main_mol), main_smi


# ---------------------------------------------------------------------------
# 2. Load the full dataset from Hugging Face
# ---------------------------------------------------------------------------
def load_records():
    print(f"[load] Dataset page : {HF_DATASET_URL}")
    print(f"[load] Downloading '{HF_DATASET_ID}' from Hugging Face ...")
    ds = load_dataset(HF_DATASET_ID, split="train")
    print(f"[load] Loaded {len(ds):,} total rows.")

    records = []
    skipped_task = 0
    for row in ds:
        meta = row.get("metadata")
        task = None
        if isinstance(meta, dict):
            task = meta.get("task")
        elif isinstance(meta, str):
            try:
                task = json.loads(meta).get("task")
            except Exception:
                task = None

        if TASK_FILTER is not None and task != TASK_FILTER:
            skipped_task += 1
            continue

        records.append({"id": row["id"], "input": row["input"], "output": row.get("output", "")})

    print(f"[load] Kept {len(records):,} rows matching task='{TASK_FILTER}' "
          f"(skipped {skipped_task:,} other-task rows).")

    # --- De-duplicate to one row per reaction id ---------------------------
    by_id = {}
    collision_ids = []
    for r in records:
        if r["id"] in by_id:
            if by_id[r["id"]]["input"] != r["input"]:
                collision_ids.append(r["id"])
            continue
        by_id[r["id"]] = r

    if collision_ids:
        print(f"[load] WARNING: {len(collision_ids)} ids map to DIFFERENT inputs "
              f"(true id collisions, not just paraphrase duplicates). "
              f"First few: {collision_ids[:5]}. First-seen input used for each id.")

    unique_records = list(by_id.values())
    print(f"[load] {len(records):,} rows -> {len(unique_records):,} unique reactions "
          f"after de-duplicating by id ({len(records) - len(unique_records):,} duplicate rows collapsed).")

    # unique_records: used for the split; records: all rows, for row-level output.
    return unique_records, records


# ---------------------------------------------------------------------------
# 3. Build the leakage-free scaffold split
# ---------------------------------------------------------------------------
def build_split(records, test_frac=TEST_FRAC):
    reaction_info = []
    parse_failures = 0

    print("[scaffold] Computing main-reactant Murcko scaffolds ...")
    for i, r in enumerate(records, 1):
        scaffold, main_frag = main_reactant_scaffold(r["input"])
        if scaffold is None:
            parse_failures += 1
            continue
        reaction_info.append({"id": r["id"], "scaffold": scaffold, "main_fragment": main_frag})
        if i % 50000 == 0:
            print(f"[scaffold]   processed {i:,}/{len(records):,} ...")

    print(f"[scaffold] Parse failures (excluded entirely): {parse_failures:,}")

    scaffold_to_ids = defaultdict(set)
    acyclic_ids = set()
    for x in reaction_info:
        if x["scaffold"]:
            scaffold_to_ids[x["scaffold"]].add(x["id"])
        else:
            acyclic_ids.add(x["id"])

    n_total = len(reaction_info)
    target_test = int(round(test_frac * n_total))

    print(f"[split] {len(scaffold_to_ids):,} unique main-reactant scaffolds found "
          f"across {n_total:,} usable reactions.")
    print(f"[split] {len(acyclic_ids):,} reactions have an acyclic main reactant "
          f"(no scaffold) -> forced into TRAIN.")

    # Greedy: smallest scaffold groups first, fill TEST up to target, rest -> TRAIN.
    groups_sorted = sorted(scaffold_to_ids.items(), key=lambda kv: len(kv[1]))
    test_ids, train_ids = set(), set()
    for sc, ids in groups_sorted:
        if len(test_ids) < target_test:
            test_ids |= ids
        else:
            train_ids |= ids
    train_ids |= (acyclic_ids - test_ids)

    # --- Leak check ---
    train_scaffolds = {x["scaffold"] for x in reaction_info if x["id"] in train_ids and x["scaffold"]}
    test_scaffolds = {x["scaffold"] for x in reaction_info if x["id"] in test_ids and x["scaffold"]}
    overlap = train_scaffolds & test_scaffolds
    assert not overlap, (
        f"LEAK DETECTED: {len(overlap)} scaffolds appear in both train and test."
    )

    stats = {
        "hf_dataset": HF_DATASET_ID,
        "task_filter": TASK_FILTER,
        "n_unique_reactions_used": n_total,
        "n_parse_failures_excluded": parse_failures,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_fraction": round(len(train_ids) / n_total, 4),
        "test_fraction": round(len(test_ids) / n_total, 4),
        "n_unique_scaffolds": len(scaffold_to_ids),
        "n_singleton_scaffolds": sum(1 for v in scaffold_to_ids.values() if len(v) == 1),
        "singleton_scaffold_rate": round(
            sum(1 for v in scaffold_to_ids.values() if len(v) == 1) / max(len(scaffold_to_ids), 1), 4
        ),
        "n_acyclic_main_reactant_forced_train": len(acyclic_ids),
        "group_size_distribution": dict(sorted(Counter(len(v) for v in scaffold_to_ids.values()).items())),
        "top_10_largest_scaffold_groups": [
            {"scaffold": sc, "n_reactions": len(ids)}
            for sc, ids in sorted(scaffold_to_ids.items(), key=lambda kv: -len(kv[1]))[:10]
        ],
        "scaffold_overlap_check": "CLEAN (S_train ∩ S_test = ∅)",
    }

    return reaction_info, train_ids, test_ids, stats


# ---------------------------------------------------------------------------
# 4. Write outputs
# ---------------------------------------------------------------------------
def write_outputs(reaction_info, all_records, train_ids, test_ids, stats):
    stats_path = OUTPUT_DIR / "scaffold_split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[write] Wrote stats -> {stats_path}")

    train_path = OUTPUT_DIR / "scaffold_split_train_ids.csv"
    with open(train_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        for rid in sorted(train_ids):
            w.writerow([rid])
    print(f"[write] Wrote {len(train_ids):,} unique train ids -> {train_path}")

    test_path = OUTPUT_DIR / "scaffold_split_test_ids.csv"
    with open(test_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        for rid in sorted(test_ids):
            w.writerow([rid])
    print(f"[write] Wrote {len(test_ids):,} unique test ids -> {test_path}")

    groups_path = OUTPUT_DIR / "scaffold_groups.csv"
    with open(groups_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "main_reactant_scaffold", "split"])
        for x in reaction_info:
            split = "train" if x["id"] in train_ids else ("test" if x["id"] in test_ids else "excluded")
            w.writerow([x["id"], x["scaffold"], split])
    print(f"[write] Wrote per-reaction scaffold assignments -> {groups_path}")

    # --- Row-level output (all rows whose id is in each split) ---
    train_rows = [r for r in all_records if r["id"] in train_ids]
    test_rows = [r for r in all_records if r["id"] in test_ids]

    train_rows_path = OUTPUT_DIR / "scaffold_split_train_rows.json"
    with open(train_rows_path, "w") as f:
        json.dump(train_rows, f)
    print(f"[write] Wrote {len(train_rows):,} full train ROWS (all instruction "
          f"paraphrases included) -> {train_rows_path}")

    test_rows_path = OUTPUT_DIR / "scaffold_split_test_rows.json"
    with open(test_rows_path, "w") as f:
        json.dump(test_rows, f)
    print(f"[write] Wrote {len(test_rows):,} full test ROWS (all instruction "
          f"paraphrases included) -> {test_rows_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    unique_records, all_records = load_records()
    reaction_info, train_ids, test_ids, stats = build_split(unique_records)
    write_outputs(reaction_info, all_records, train_ids, test_ids, stats)

    print("\n================ SUMMARY ================")
    print(json.dumps(stats, indent=2))
    print("===========================================")
    print("\nDone.")
    print("- scaffold_split_train_ids.csv / test_ids.csv    : unique reaction ids per split")
    print("- scaffold_split_train_rows.json / test_rows.json: full rows per split")
    print("- scaffold_groups.csv                            : per-reaction scaffold + split")


if __name__ == "__main__":
    main()
