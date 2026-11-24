"""
Dataset construction for SLM4CRP_with_RTs.

  1. Load the raw records and keep the forward-reaction-prediction task.
  2. De-duplicate by reaction ID (first representative record kept).
  3. Canonicalize reactants/reagents (fragments sorted) and products with RDKit.
  4. Remove duplicate canonical reactant-product pairs across IDs.
  5. Split 70/15/15 at the reaction level with a fixed seed.
  6. Build the vocabulary from the training split only.
  7. Draw a fixed visualisation subset from the test split.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import CFG, DATA_DIR, DataConfig, RuntimeConfig, TokenizerConfig
from .tokenizer import Vocabulary

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False
    Chem = None  # type: ignore


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
def canonicalize_smiles(
    smiles: str,
    isomeric: bool = True,
) -> Optional[str]:
    """RDKit canonical SMILES for one molecule, or None if it fails to parse."""
    if not _RDKIT:
        raise RuntimeError("RDKit is required. pip install rdkit")
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True)
    except Exception:
        return None


def canonicalize_precursor_set(
    smiles: str,
    isomeric: bool = True,
    sort_fragments: bool = True,
) -> Optional[str]:
    """
    Canonicalize each '.'-separated fragment, sort, and re-join. Fragments
    that fail to parse are dropped; returns None if none parse.
    """
    if not smiles or not smiles.strip():
        return None
    frags = [f for f in smiles.strip().split(".") if f]
    canon: List[str] = []
    for f in frags:
        c = canonicalize_smiles(f, isomeric=isomeric)
        if c is not None:
            canon.append(c)
    if not canon:
        return None
    if sort_fragments:
        canon = sorted(canon)
    return ".".join(canon)


def canonicalize_product(
    smiles: str,
    isomeric: bool = True,
) -> Optional[str]:
    """Canonicalize the target; with several fragments, keep the heaviest."""
    if not smiles or not smiles.strip():
        return None
    frags = [f for f in smiles.strip().split(".") if f]
    best, best_size = None, -1
    for f in frags:
        mol = Chem.MolFromSmiles(f)
        if mol is None:
            continue
        n = mol.GetNumHeavyAtoms()
        if n > best_size:
            best, best_size = mol, n
    if best is None:
        return None
    try:
        return Chem.MolToSmiles(best, isomericSmiles=isomeric, canonical=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loading and de-duplication
# ---------------------------------------------------------------------------
@dataclass
class Reaction:
    rxn_id: str
    input_smiles: str      # canonical, dot-delimited, sorted
    output_smiles: str     # canonical primary product

    def to_dict(self) -> dict:
        return {
            "id": self.rxn_id,
            "input": self.input_smiles,
            "output": self.output_smiles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reaction":
        return cls(str(d["id"]), d["input"], d["output"])


def _extract_task(row: dict) -> Optional[str]:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        return meta.get("task")
    if isinstance(meta, str):
        try:
            return json.loads(meta).get("task")
        except Exception:
            return None
    return row.get("task")


def load_unique_reactions(
    cfg: Optional[DataConfig] = None,
    hf_cache_dir: Optional[str] = None,
    local_json: Optional[Path | str] = None,
    verbose: bool = True,
) -> List[Reaction]:
    """
    Load, filter, de-duplicate and canonicalize the corpus.

    local_json : optional path to a downloaded copy of the raw corpus
                 (JSON list of records with id/input/output/metadata).
    """
    cfg = cfg or CFG.data

    if local_json is not None:
        local_json = Path(local_json)
        if verbose:
            print(f"[data] Reading raw corpus from local file: {local_json}")
        raw_rows = json.loads(local_json.read_text())
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The `datasets` package is required to download the corpus. "
                "pip install datasets -- or pass local_json=<path>."
            ) from exc
        if verbose:
            print(f"[data] Downloading '{cfg.hf_dataset_id}' ({cfg.hf_dataset_url})")
        raw_rows = load_dataset(cfg.hf_dataset_id, split="train", cache_dir=hf_cache_dir)

    if verbose:
        print(f"[data] Loaded {len(raw_rows):,} raw records.")

    # Filter by task.
    task_rows: List[dict] = []
    n_other_task = 0
    for row in raw_rows:
        task = _extract_task(row)
        if cfg.task_filter is not None and task is not None and task != cfg.task_filter:
            n_other_task += 1
            continue
        task_rows.append(
            {
                "id": str(row.get("id", "")),
                "input": row.get("input", "") or "",
                "output": row.get("output", "") or "",
            }
        )
    if verbose:
        print(
            f"[data] {len(task_rows):,} rows kept for task='{cfg.task_filter}' "
            f"({n_other_task:,} other-task rows skipped)."
        )

    # De-duplicate by reaction ID.
    first_by_id: Dict[str, dict] = {}
    id_collisions: List[str] = []
    for r in task_rows:
        rid = r["id"]
        if rid in first_by_id:
            if first_by_id[rid]["input"] != r["input"]:
                id_collisions.append(rid)
            continue
        first_by_id[rid] = r

    if verbose:
        print(f"[data] {len(task_rows):,} rows -> {len(first_by_id):,} distinct reaction IDs.")
        if id_collisions:
            print(
                f"[data] WARNING: {len(id_collisions)} reaction IDs map to "
                f"different inputs; first-seen record kept. First few: "
                f"{id_collisions[:5]}"
            )

    # Canonicalize.
    if verbose:
        print("[data] Canonicalizing with RDKit ...")
    canon: List[Reaction] = []
    n_input_fail = n_output_fail = 0
    for i, (rid, r) in enumerate(first_by_id.items(), 1):
        cin = canonicalize_precursor_set(
            r["input"],
            isomeric=cfg.isomeric_smiles,
            sort_fragments=cfg.sort_precursor_fragments,
        )
        if cin is None:
            n_input_fail += 1
            continue
        cout = canonicalize_product(r["output"], isomeric=cfg.isomeric_smiles)
        if cout is None:
            n_output_fail += 1
            continue
        canon.append(Reaction(rid, cin, cout))
        if verbose and i % 100_000 == 0:
            print(f"[data]   canonicalized {i:,}/{len(first_by_id):,} ...")

    if verbose:
        print(
            f"[data] Canonicalization: {len(canon):,} kept, "
            f"{n_input_fail:,} unparsable inputs, "
            f"{n_output_fail:,} unparsable products."
        )

    # Remove duplicate canonical (input, output) pairs across IDs.
    seen_pairs: Dict[Tuple[str, str], str] = {}
    unique: List[Reaction] = []
    n_cross_dupes = 0
    for rxn in canon:
        key = (rxn.input_smiles, rxn.output_smiles)
        if key in seen_pairs:
            n_cross_dupes += 1
            continue
        seen_pairs[key] = rxn.rxn_id
        unique.append(rxn)

    if verbose:
        print(f"[data] Removed {n_cross_dupes:,} cross-ID duplicate pairs.")
        print(f"[data] Unique reactions: {len(unique):,}")
    return unique


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_reactions(
    reactions: Sequence[Reaction],
    cfg: Optional[DataConfig] = None,
    verbose: bool = True,
) -> Dict[str, List[Reaction]]:
    """Random reaction-level train/val/test split with a fixed seed."""
    cfg = cfg or CFG.data
    n = len(reactions)
    idx = list(range(n))
    rng = random.Random(cfg.random_seed)
    rng.shuffle(idx)

    f_train, f_val, _ = cfg.split_fractions
    n_train = int(n * f_train)
    n_val = int(n * f_val)

    splits = {
        "train": [reactions[i] for i in idx[:n_train]],
        "val": [reactions[i] for i in idx[n_train : n_train + n_val]],
        "test": [reactions[i] for i in idx[n_train + n_val :]],
    }

    if verbose:
        print(
            f"[split] seed={cfg.random_seed}  "
            f"train={len(splits['train']):,}  "
            f"val={len(splits['val']):,}  "
            f"test={len(splits['test']):,}"
        )

    ids = {k: {r.rxn_id for r in v} for k, v in splits.items()}
    assert not (ids["train"] & ids["val"]), "train/val reaction ID overlap"
    assert not (ids["train"] & ids["test"]), "train/test reaction ID overlap"
    assert not (ids["val"] & ids["test"]), "val/test reaction ID overlap"

    return splits


def draw_viz_subset(
    test_reactions: Sequence[Reaction],
    cfg: Optional[DataConfig] = None,
) -> List[Reaction]:
    """Fixed random subset of the test split used for per-sample plots."""
    cfg = cfg or CFG.data
    rng = random.Random(cfg.viz_subset_seed)
    k = min(cfg.viz_subset_size, len(test_reactions))
    return rng.sample(list(test_reactions), k)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_splits(
    splits: Dict[str, List[Reaction]],
    out_dir: Path | str = DATA_DIR,
    viz_subset: Optional[Sequence[Reaction]] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rxns in splits.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps([r.to_dict() for r in rxns]))
        print(f"[data] Wrote {len(rxns):,} reactions -> {p}")

        pid = out_dir / f"{name}_ids.txt"
        pid.write_text("\n".join(r.rxn_id for r in rxns))

    if viz_subset is not None:
        p = out_dir / "viz_subset_300.json"
        p.write_text(json.dumps([r.to_dict() for r in viz_subset]))
        print(f"[data] Wrote {len(viz_subset):,} visualisation reactions -> {p}")


def load_split(name: str, data_dir: Path | str = DATA_DIR) -> List[Reaction]:
    p = Path(data_dir) / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python scripts/01_prepare_data.py` first."
        )
    return [Reaction.from_dict(d) for d in json.loads(p.read_text())]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
def build_vocabulary(
    train_reactions: Sequence[Reaction],
    tok_cfg: Optional[TokenizerConfig] = None,
    verbose: bool = True,
) -> Vocabulary:
    """Shared vocabulary built from training inputs and outputs."""
    tok_cfg = tok_cfg or CFG.tokenizer

    def _strings() -> Iterable[str]:
        for r in train_reactions:
            yield r.input_smiles
            yield r.output_smiles

    return Vocabulary.build(_strings(), cfg=tok_cfg, verbose=verbose)


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader
# ---------------------------------------------------------------------------
class ReactionDataset(Dataset):
    """Returns (encoded input, encoded target) tensor pairs."""

    def __init__(
        self,
        reactions: Sequence[Reaction],
        vocab: Vocabulary,
        max_len: Optional[int] = None,
    ):
        self.reactions = list(reactions)
        self.vocab = vocab
        self.max_len = max_len or vocab.cfg.max_len

    def __len__(self) -> int:
        return len(self.reactions)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.reactions[i]
        src = self.vocab.encode(r.input_smiles, max_len=self.max_len)
        tgt = self.vocab.encode(r.output_smiles, max_len=self.max_len)
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

    def raw(self, i: int) -> Reaction:
        return self.reactions[i]


def make_dataloader(
    dataset: ReactionDataset,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    runtime: Optional[RuntimeConfig] = None,
) -> DataLoader:
    runtime = runtime or CFG.runtime
    batch_size = batch_size or CFG.train.batch_size
    nw = runtime.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=runtime.pin_memory and torch.cuda.is_available(),
        persistent_workers=runtime.persistent_workers and nw > 0,
        drop_last=False,
    )


__all__ = [
    "Reaction",
    "canonicalize_smiles",
    "canonicalize_precursor_set",
    "canonicalize_product",
    "load_unique_reactions",
    "split_reactions",
    "draw_viz_subset",
    "save_splits",
    "load_split",
    "build_vocabulary",
    "ReactionDataset",
    "make_dataloader",
]
