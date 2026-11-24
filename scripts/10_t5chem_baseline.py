#!/usr/bin/env python3
"""
Fine-tune and decode a T5Chem baseline on the same splits and input strings,
with the same beam width. T5Chem's own tokenizer is used. A checkpoint must
be supplied via --model-name-or-path.

Output
------
    results/t5chem_test.jsonl   one line per test reaction:
        {"id", "input", "ref", "candidates": [...], "confidence"}
    (accepted by scripts/07 via --t5chem-predictions)

Usage
-----
    # fine-tune, then decode
    python scripts/10_t5chem_baseline.py --model-name-or-path <ckpt> --train --decode
    # decode only, from an already fine-tuned checkpoint
    python scripts/10_t5chem_baseline.py --model-name-or-path ./t5chem_ft --decode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_DIR, RESULTS_DIR
from src.data import load_split


def require_transformers():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script needs `transformers` and `torch`:\n"
            "    pip install transformers torch sentencepiece"
        ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name-or-path", type=str, required=True,
                    help="T5Chem checkpoint directory or hub id.")
    ap.add_argument("--train", action="store_true", help="Fine-tune on the train split.")
    ap.add_argument("--decode", action="store_true", help="Decode the test split.")
    ap.add_argument("--output-dir", type=str, default="./t5chem_ft")
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--batch-size", type=int, default=CFG.train.batch_size)
    ap.add_argument("--lr", type=float, default=CFG.train.learning_rate)
    ap.add_argument("--max-len", type=int, default=CFG.tokenizer.max_len)
    ap.add_argument("--beam-width", type=int, default=CFG.beam.beam_width)
    ap.add_argument("--device", type=str, default=CFG.runtime.device)
    args = ap.parse_args()

    if not (args.train or args.decode):
        raise SystemExit("Pass --train and/or --decode.")

    require_transformers()
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    data_dir, results_dir = Path(args.data_dir), Path(args.results_dir)

    print(f"[t5chem] Loading {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(device)

    # ---------------- fine-tune ---------------------------------------------
    if args.train:
        from torch.utils.data import DataLoader, Dataset

        class T5Data(Dataset):

            def __init__(self, reactions):
                self.r = list(reactions)

            def __len__(self):
                return len(self.r)

            def __getitem__(self, i):
                return self.r[i].input_smiles, self.r[i].output_smiles

        def collate(batch):
            srcs, tgts = zip(*batch)
            enc = tokenizer(
                list(srcs), padding=True, truncation=True,
                max_length=args.max_len, return_tensors="pt",
            )
            lab = tokenizer(
                list(tgts), padding=True, truncation=True,
                max_length=args.max_len, return_tensors="pt",
            ).input_ids
            # Ignore padding in the loss.
            lab[lab == tokenizer.pad_token_id] = -100
            enc["labels"] = lab
            return enc

        train_loader = DataLoader(
            T5Data(load_split("train", data_dir)),
            batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        )

        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=CFG.train.weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr,
            total_steps=len(train_loader) * args.epochs,
            pct_start=CFG.train.onecycle_pct_start,
            anneal_strategy=CFG.train.onecycle_anneal_strategy,
        )

        model.train()
        for epoch in range(1, args.epochs + 1):
            total, nb = 0.0, 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.train.grad_clip_norm)
                opt.step()
                sched.step()
                total += loss.item()
                nb += 1
            print(f"[t5chem] epoch {epoch}/{args.epochs}  loss {total/max(nb,1):.4f}")

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[t5chem] Saved fine-tuned model -> {args.output_dir}")

    # ---------------- decode ------------------------------------------------
    if args.decode:
        model.eval()
        test = load_split("test", data_dir)
        out_path = results_dir / "t5chem_test.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        B = args.beam_width
        written = 0
        with open(out_path, "w") as f, torch.no_grad():
            for start in range(0, len(test), 16):
                batch = test[start : start + 16]
                enc = tokenizer(
                    [r.input_smiles for r in batch], padding=True, truncation=True,
                    max_length=args.max_len, return_tensors="pt",
                ).to(device)

                gen = model.generate(
                    **enc,
                    num_beams=B,
                    num_return_sequences=B,
                    max_length=args.max_len,
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                    length_penalty=CFG.beam.length_penalty_beta,
                )
                seqs = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)

                # Length-normalized beam score; exp() gives the confidence.
                scores = (
                    gen.sequences_scores.detach().cpu().tolist()
                    if getattr(gen, "sequences_scores", None) is not None
                    else [0.0] * len(seqs)
                )

                for i, r in enumerate(batch):
                    cands = [s.strip() for s in seqs[i * B : (i + 1) * B]]
                    sc = scores[i * B : (i + 1) * B]
                    f.write(json.dumps({
                        "id": r.rxn_id,
                        "input": r.input_smiles,
                        "ref": r.output_smiles,
                        "candidates": cands,
                        "norm_score": sc[0] if sc else 0.0,
                        "confidence": float(torch.tensor(sc[0]).exp()) if sc else 0.0,
                    }) + "\n")
                    written += 1

                if (start // 16) % 50 == 0:
                    print(f"[t5chem] decoded {min(start+16, len(test)):,}/{len(test):,}")

        print(f"[t5chem] Wrote {written:,} predictions -> {out_path}")
        print(
            "\nNext:\n"
            f"  python scripts/07_scaffold_eval.py --t5chem-predictions {out_path}"
        )


if __name__ == "__main__":
    main()
