#!/usr/bin/env python3
"""
Build the dataset: download, de-duplicate, canonicalize, split 70/15/15,
build the vocabulary from the training split, and draw the visualisation
subset.

Usage
-----
    python scripts/01_prepare_data.py
    python scripts/01_prepare_data.py --local-json /path/to/raw_corpus.json
    python scripts/01_prepare_data.py --hf-cache /mnt/big/hf_cache

Outputs (data/)
---------------
    train.json / val.json / test.json
    train_ids.txt / val_ids.txt / test_ids.txt
    viz_subset_300.json
    vocab.json
    dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_DIR
from src.data import (
    build_vocabulary,
    draw_viz_subset,
    load_unique_reactions,
    save_splits,
    split_reactions,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--local-json",
        type=str,
        default=None,
        help="Path to a locally downloaded copy of the raw corpus, for "
             "machines without Hugging Face access.",
    )
    ap.add_argument("--hf-cache", type=str, default=None, help="Hugging Face cache dir.")
    ap.add_argument("--out-dir", type=str, default=str(DATA_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    print("=" * 72)
    print("  Dataset construction")
    print("=" * 72)

    reactions = load_unique_reactions(
        cfg=CFG.data, hf_cache_dir=args.hf_cache, local_json=args.local_json
    )

    print()
    print("-" * 72)
    print("  Splitting")
    print("-" * 72)
    splits = split_reactions(reactions, cfg=CFG.data)
    viz = draw_viz_subset(splits["test"], cfg=CFG.data)
    save_splits(splits, out_dir=out_dir, viz_subset=viz)

    print()
    print("-" * 72)
    print("  Vocabulary (training split only)")
    print("-" * 72)
    vocab = build_vocabulary(splits["train"], tok_cfg=CFG.tokenizer)
    vocab.save(out_dir / "vocab.json")

    # Out-of-vocabulary rate on val/test.
    from src.tokenizer import tokenize_smiles

    for name in ("val", "test"):
        total = unk = 0
        for r in splits[name]:
            for s in (r.input_smiles, r.output_smiles):
                for t in tokenize_smiles(s):
                    total += 1
                    if t not in vocab.stoi:
                        unk += 1
        rate = 100 * unk / total if total else 0.0
        print(
            f"[vocab] {name}: {unk:,}/{total:,} tokens map to [UNK] ({rate:.4f}%)"
        )

    stats = {
        "n_unique_reactions": len(reactions),
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "n_viz_subset": len(viz),
        "vocab_size": vocab.size,
        "n_chemical_tokens": vocab.n_chemical_tokens,
        "random_seed": CFG.data.random_seed,
        "max_len": CFG.tokenizer.max_len,
    }
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\n[data] Wrote stats -> {out_dir / 'dataset_stats.json'}")
    print("\nDone. Next: python scripts/02_train_gru.py")


if __name__ == "__main__":
    main()
