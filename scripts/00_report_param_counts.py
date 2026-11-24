#!/usr/bin/env python3
"""
Report trainable-parameter counts for both models, broken down by component.

Usage
-----
    python scripts/00_report_param_counts.py
    python scripts/00_report_param_counts.py --vocab-size 84
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_DIR, RESULTS_DIR
from src.models import GRUSeq2Seq, TransformerSeq2Seq


def breakdown(model, groups: dict) -> dict:
    """Sum trainable parameters per named prefix group."""
    out = {g: 0 for g in groups}
    out["_other"] = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        placed = False
        for g, prefixes in groups.items():
            if any(name.startswith(pre) for pre in prefixes):
                out[g] += p.numel()
                placed = True
                break
        if not placed:
            out["_other"] += p.numel()
    return {k: v for k, v in out.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="Defaults to data/vocab.json if present, else 84.")
    ap.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    args = ap.parse_args()

    if args.vocab_size:
        V = args.vocab_size
    else:
        p = DATA_DIR / "vocab.json"
        if p.exists():
            V = json.loads(p.read_text())["size"]
            print(f"[params] Using |V| = {V} from {p}")
        else:
            V = 84
            print(f"[params] data/vocab.json not found; using |V| = {V}")

    print()
    print("=" * 72)
    print(f"  Parameter counts at |V| = {V}")
    print("=" * 72)

    gru = GRUSeq2Seq(V, CFG.gru, CFG.tokenizer)
    tf = TransformerSeq2Seq(V, CFG.transformer, CFG.tokenizer)

    n_gru = sum(p.numel() for p in gru.parameters() if p.requires_grad)
    n_tf = sum(p.numel() for p in tf.parameters() if p.requires_grad)

    gru_parts = breakdown(gru, {
        "encoder embedding": ["encoder.embedding"],
        "encoder GRU": ["encoder.rnn"],
        "bridge": ["encoder.bridge"],
        "decoder embedding": ["decoder.embedding"],
        "attention": ["decoder.attention"],
        "decoder GRU": ["decoder.rnn"],
        "output head": ["decoder.fc_out"],
    })
    tf_parts = breakdown(tf, {
        "token embedding": ["embedding"],
        "encoder stack": ["transformer.encoder"],
        "decoder stack": ["transformer.decoder"],
        "output head": ["fc_out"],
    })

    print()
    print(f"  GRU Seq2Seq                  {n_gru:>12,}  ({n_gru/1e6:5.2f}M)")
    print(f"  {'-'*66}")
    for k, v in gru_parts.items():
        print(f"    {k:<28} {v:>12,}  ({100*v/n_gru:5.1f}%)")

    print()
    print(f"  Transformer                  {n_tf:>12,}  ({n_tf/1e6:5.2f}M)")
    print(f"  {'-'*66}")
    for k, v in tf_parts.items():
        print(f"    {k:<28} {v:>12,}  ({100*v/n_tf:5.1f}%)")

    print()
    print(f"  Combined ensemble            {n_gru + n_tf:>12,}  "
          f"({(n_gru+n_tf)/1e6:5.2f}M)")
    print("=" * 72)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parameter_counts.json").write_text(json.dumps({
        "vocab_size": V,
        "gru": {
            "total": n_gru, "components": gru_parts,
        },
        "transformer": {
            "total": n_tf, "components": tf_parts,
        },
        "combined": n_gru + n_tf,
    }, indent=2))
    print(f"\n[params] Wrote -> {out / 'parameter_counts.json'}")


if __name__ == "__main__":
    main()
