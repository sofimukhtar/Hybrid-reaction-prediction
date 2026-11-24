"""
Evaluation driver: decode a split with one system ("gru", "transformer" or
"ensemble"), score every reaction, and write per-sample records (JSONL) and
aggregate metrics (JSON) to results/.

Evaluating the test split requires allow_test=True.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CFG, RESULTS_DIR, BeamConfig, EnsembleConfig, EvalConfig
from .data import Reaction, ReactionDataset
from .decode import BranchAdapter, beam_search, make_branches
from .metrics import PerSampleRecord, aggregate, score_one
from .tokenizer import Vocabulary


# ---------------------------------------------------------------------------
# Core decode-and-score loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_and_score(
    reactions: Sequence[Reaction],
    vocab: Vocabulary,
    branches: Sequence[BranchAdapter],
    device: torch.device,
    beam_cfg: Optional[BeamConfig] = None,
    eval_cfg: Optional[EvalConfig] = None,
    batch_size: int = 32,
    fusion_space: str = "log",
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> List[PerSampleRecord]:
    """
    Decode every reaction with beam search and score the results.

    Returns one PerSampleRecord per reaction, in input order.
    """
    beam_cfg = beam_cfg or CFG.beam
    eval_cfg = eval_cfg or CFG.eval

    rxns = list(reactions)[: max_samples] if max_samples else list(reactions)
    n = len(rxns)
    records: List[PerSampleRecord] = []
    t0 = time.time()

    for start in range(0, n, batch_size):
        batch = rxns[start : start + batch_size]
        src = torch.stack(
            [
                torch.tensor(
                    vocab.encode(r.input_smiles, max_len=vocab.cfg.max_len),
                    dtype=torch.long,
                )
                for r in batch
            ]
        ).to(device)

        results = beam_search(
            branches, src, vocab, beam_cfg=beam_cfg, fusion_space=fusion_space
        )

        for rxn, res in zip(batch, results):
            cands = [h.smiles for h in res.hypotheses]
            top = res.top1
            records.append(
                score_one(
                    rxn_id=rxn.rxn_id,
                    input_smiles=rxn.input_smiles,
                    ref_smiles=rxn.output_smiles,
                    candidates=cands,
                    confidence=top.confidence if top else 0.0,
                    norm_score=top.norm_score if top else float("-inf"),
                    eval_cfg=eval_cfg,
                    topk_values=beam_cfg.topk_values,
                )
            )

        if verbose and (start // batch_size) % 20 == 0:
            done = min(start + batch_size, n)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(
                f"[eval] {done:,}/{n:,} decoded  "
                f"({rate:.1f} rxn/s, ETA {eta/60:.1f} min)"
            )

    if verbose:
        print(f"[eval] Decoded {n:,} reactions in {(time.time()-t0)/60:.1f} min")
    return records


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def write_per_sample(
    records: Sequence[PerSampleRecord],
    path: Path | str,
) -> None:
    """Write per-sample records as JSONL, one record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"[eval] Wrote {len(records):,} per-sample records -> {path}")


def read_per_sample(path: Path | str) -> List[PerSampleRecord]:
    """Read back a JSONL file written by `write_per_sample`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    out: List[PerSampleRecord] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            d["topk"] = {int(k): v for k, v in d.get("topk", {}).items()}
            out.append(PerSampleRecord(**d))
    return out


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(measured: Dict[str, object], model_name: str) -> None:
    """Print the main string and token metrics."""
    keys = [
        ("char_accuracy", "Character-Level Accuracy"),
        ("prefix_match", "Prefix Match"),
        ("levenshtein_accuracy", "Levenshtein Accuracy"),
        ("bleu2", "BLEU Score (bigram)"),
        ("micro_f1", "Micro F1 Score"),
        ("exact_match", "Exact Match Accuracy"),
    ]
    print()
    print(f"  {model_name.upper()}")
    print(f"  {'Metric':<28} {'value':>10}")
    print(f"  {'-'*28} {'-'*10}")
    for key, label in keys:
        got = measured.get(key)
        if got is None:
            continue
        print(f"  {label:<28} {got:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def evaluate_split(
    split_name: str,
    reactions: Sequence[Reaction],
    vocab: Vocabulary,
    gru_model=None,
    tf_model=None,
    mode: str = "ensemble",
    device: Optional[torch.device] = None,
    beam_cfg: Optional[BeamConfig] = None,
    ens_cfg: Optional[EnsembleConfig] = None,
    eval_cfg: Optional[EvalConfig] = None,
    fusion_space: str = "log",
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    out_dir: Optional[Path] = None,
    tag: Optional[str] = None,
    allow_test: bool = False,
    with_bootstrap: bool = True,
) -> Dict[str, object]:
    """
    Evaluate one system on one split and write results.

    Parameters
    ----------
    split_name : "train", "val" or "test". Only used for file naming and the
                 test-set guard.
    mode       : "gru", "transformer" or "ensemble".
    allow_test : must be True to evaluate the test split.
    tag        : optional suffix for the output filenames, e.g. "scaffold" or
                 "w0.7". Keeps sweep runs from overwriting each other.
    """
    if split_name == "test" and not allow_test:
        raise RuntimeError(
            "Refusing to evaluate on the test split without allow_test=True. "
            "Use split_name='val' for weight sweeps and ablations."
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beam_cfg = beam_cfg or CFG.beam
    ens_cfg = ens_cfg or CFG.ensemble
    eval_cfg = eval_cfg or CFG.eval
    batch_size = batch_size or CFG.runtime.eval_batch_size
    out_dir = Path(out_dir or RESULTS_DIR)

    branches = make_branches(gru_model, tf_model, ens_cfg=ens_cfg, mode=mode)

    print(
        f"[eval] split={split_name}  mode={mode}  B={beam_cfg.beam_width}  "
        f"weights={[b.weight for b in branches]}  fusion={fusion_space}"
    )

    records = decode_and_score(
        reactions, vocab, branches, device,
        beam_cfg=beam_cfg, eval_cfg=eval_cfg,
        batch_size=batch_size, fusion_space=fusion_space,
        max_samples=max_samples,
    )

    agg = aggregate(
        records, eval_cfg=eval_cfg,
        topk_values=beam_cfg.topk_values, with_bootstrap=with_bootstrap,
    )
    agg["split"] = split_name
    agg["mode"] = mode
    agg["beam_width"] = beam_cfg.beam_width
    agg["fusion_space"] = fusion_space
    agg["weights"] = {b.kind: b.weight for b in branches}

    suffix = f"_{tag}" if tag else ""
    stem = f"{mode}_{split_name}{suffix}"
    write_per_sample(records, out_dir / f"per_sample_{stem}.jsonl")
    (out_dir / f"metrics_{stem}.json").write_text(json.dumps(agg, indent=2))
    print(f"[eval] Wrote aggregate metrics -> {out_dir / f'metrics_{stem}.json'}")

    print_summary(agg, mode)
    return {"records": records, "metrics": agg}


__all__ = [
    "decode_and_score",
    "write_per_sample",
    "read_per_sample",
    "print_summary",
    "evaluate_split",
]
