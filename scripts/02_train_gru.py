#!/usr/bin/env python3
"""
Train the GRU Seq2Seq model. Saves a checkpoint every epoch to
checkpoints/gru/ and copies the selected epoch to gru_selected.pt.

Usage
-----
    python scripts/02_train_gru.py
    python scripts/02_train_gru.py --epochs 20 --batch-size 64 --device cuda
    python scripts/02_train_gru.py --smoke-test        # 200 reactions, 1 epoch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, CKPT_DIR, DATA_DIR
from src.data import ReactionDataset, load_split, make_dataloader
from src.models import GRUSeq2Seq
from src.tokenizer import Vocabulary
from src.train import resolve_device, set_seed, train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.train.learning_rate)
    ap.add_argument("--device", type=str, default=CFG.runtime.device)
    ap.add_argument("--num-workers", type=int, default=CFG.runtime.num_workers)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train on 200 reactions for 1 epoch.",
    )
    args = ap.parse_args()

    # Apply CLI overrides to the shared config.
    CFG.train.__dict__["epochs"] = 1 if args.smoke_test else args.epochs
    CFG.train.__dict__["batch_size"] = args.batch_size
    CFG.train.__dict__["learning_rate"] = args.lr
    CFG.runtime.num_workers = args.num_workers

    set_seed(CFG.train.random_seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    vocab = Vocabulary.load(data_dir / "vocab.json", cfg=CFG.tokenizer)

    train_rxns = load_split("train", data_dir)
    val_rxns = load_split("val", data_dir)
    if args.smoke_test:
        train_rxns, val_rxns = train_rxns[:200], val_rxns[:50]
        print("[smoke] 200 train / 50 val reactions, 1 epoch")

    train_loader = make_dataloader(
        ReactionDataset(train_rxns, vocab), batch_size=CFG.train.batch_size,
        shuffle=True, runtime=CFG.runtime,
    )
    val_loader = make_dataloader(
        ReactionDataset(val_rxns, vocab), batch_size=CFG.train.batch_size,
        shuffle=False, runtime=CFG.runtime,
    )

    print("=" * 72)
    print("  GRU Seq2Seq")
    print("=" * 72)
    model = GRUSeq2Seq(vocab.size, cfg=CFG.gru, tok_cfg=CFG.tokenizer)
    model.count_parameters()

    train_model(
        model=model,
        model_name="gru",
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        cfg=CFG.train,
        device=device,
        ckpt_dir=Path(args.ckpt_dir),
    )
    print("\nDone. Next: python scripts/03_train_transformer.py")


if __name__ == "__main__":
    main()
