"""
Step-wise log-probability fusion and beam search decoding.

Fusion (applied at every decoding step):
    log p_ens(y_t) = w_GRU log p_GRU(y_t) + w_TF log p_TF(y_t) - Z_t
    Z_t = log sum_v exp(w_GRU log p_GRU(v) + w_TF log p_TF(v))

Beam search ranking of completed hypotheses:
    score(Y) = (1 / |Y|^beta) * sum_t log p(y_t | y_<t, x)

Partial hypotheses are expanded by cumulative log-probability; length
normalization is applied when ranking completed hypotheses. If fewer than B
hypotheses complete, only those are returned (no padding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BeamConfig, EnsembleConfig, TokenizerConfig
from .tokenizer import Vocabulary


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def fuse_log_probs(
    log_prob_list: Sequence[torch.Tensor],
    weights: Sequence[float],
    space: str = "log",
) -> torch.Tensor:
    """
    Combine per-model log-probabilities into one normalized distribution.

    Parameters
    ----------
    log_prob_list : one (B, |V|) log-softmax tensor per model, all over the
                    same shared vocabulary.
    weights       : one scalar weight per model, in the same order.
    space         : "log"  -- normalized weighted geometric mean.
                    "prob" -- weighted arithmetic mean in probability space,
                             then log.

    Returns
    -------
    (B, |V|) log-probabilities that sum to 1 in probability space.
    """
    if len(log_prob_list) != len(weights):
        raise ValueError(
            f"Got {len(log_prob_list)} log-prob tensors but {len(weights)} weights."
        )
    if not log_prob_list:
        raise ValueError("No log-prob tensors given.")

    if len(log_prob_list) == 1:
        # Single model: return unchanged (weighting would act as a temperature).
        return log_prob_list[0]

    if space == "log":
        acc = torch.zeros_like(log_prob_list[0])
        for lp, w in zip(log_prob_list, weights):
            acc = acc + float(w) * lp
        return F.log_softmax(acc, dim=-1)

    if space == "prob":
        acc = torch.zeros_like(log_prob_list[0])
        for lp, w in zip(log_prob_list, weights):
            acc = acc + float(w) * lp.exp()
        acc = acc / acc.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return acc.clamp_min(1e-12).log()

    raise ValueError(f"Unknown fusion space {space!r}; expected 'log' or 'prob'.")


# ---------------------------------------------------------------------------
# Model wrapper so both branches look the same to the beam search
# ---------------------------------------------------------------------------
class BranchAdapter:
    """
    Uniform interface over both architectures: the GRU consumes the last
    token, the Transformer consumes the full prefix.
    """

    def __init__(self, model: nn.Module, kind: str, weight: float = 1.0):
        if kind not in ("gru", "transformer"):
            raise ValueError(f"kind must be 'gru' or 'transformer', got {kind!r}")
        self.model = model
        self.kind = kind
        self.weight = float(weight)
        self.state: Optional[dict] = None

    def encode(self, src: torch.Tensor) -> None:
        self.state = self.model.encode(src)

    def expand_for_beams(self, beam_width: int) -> None:
        """Repeat each batch entry beam_width times, interleaved."""
        n = next(iter(self.state.values())).size(0)
        idx = torch.arange(n, device=src_device(self.state)).repeat_interleave(beam_width)
        self.state = self.model.reorder_state(self.state, idx)

    def step(self, y_prefix: torch.Tensor) -> torch.Tensor:
        """
        y_prefix : (N*B, t) the generated prefix including the leading [START].

        Returns (N*B, |V|) log-probabilities for the next token.
        """
        if self.kind == "gru":
            log_probs, self.state, _ = self.model.decode_step(y_prefix[:, -1], self.state)
        else:
            log_probs, self.state, _ = self.model.decode_step(y_prefix, self.state)
        return log_probs

    def reorder(self, index: torch.Tensor) -> None:
        self.state = self.model.reorder_state(self.state, index)


def src_device(state: dict) -> torch.device:
    return next(iter(state.values())).device


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Hypothesis:
    """One completed (or truncated) candidate for one input reaction."""

    token_ids: List[int]          # generated ids, excluding [START], including [END]
    smiles: str                   # decoded string, specials stripped
    sum_log_prob: float           # sum_t log p_ens(y_t)
    length: int                   # |Y|, counting [END] as y_{|Y|}
    finished: bool                # True if it terminated with [END]

    @property
    def norm_score(self) -> float:
        """Length-normalized score (beta = 1)."""
        return self.sum_log_prob / max(self.length, 1)

    @property
    def confidence(self) -> float:
        """Geometric mean of the per-token probabilities."""
        import math

        return math.exp(self.norm_score)


@dataclass
class DecodeResult:
    """All candidates for one input reaction, ranked best first."""

    hypotheses: List[Hypothesis]

    @property
    def top1(self) -> Optional[Hypothesis]:
        return self.hypotheses[0] if self.hypotheses else None

    def top_k_smiles(self, k: int) -> List[str]:
        """Up to k candidate strings; never padded or duplicated to reach k."""
        return [h.smiles for h in self.hypotheses[:k]]

    def __len__(self) -> int:
        return len(self.hypotheses)


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_search(
    branches: Sequence[BranchAdapter],
    src: torch.Tensor,
    vocab: Vocabulary,
    beam_cfg: Optional[BeamConfig] = None,
    fusion_space: str = "log",
    return_all_beams: bool = True,
) -> List[DecodeResult]:
    """
    Batched beam search with step-wise fused scores.

    Parameters
    ----------
    branches : one BranchAdapter per model, each carrying its fusion weight.
               A single adapter with weight 1.0 gives single-model beam search.
    src      : (N, S) int64 source token ids.

    Returns
    -------
    One DecodeResult per input row, hypotheses sorted by length-normalized
    score descending.
    """
    beam_cfg = beam_cfg or BeamConfig()
    B = beam_cfg.beam_width
    max_len = beam_cfg.max_decode_len
    device = src.device
    N = src.size(0)
    V = vocab.size

    weights = [b.weight for b in branches]

    # --- encode once, then expand to N*B -----------------------------------
    for br in branches:
        br.encode(src)
        idx = torch.arange(N, device=device).repeat_interleave(B)
        br.reorder(idx)

    # --- beam state ---------------------------------------------------------
    # tokens: (N*B, t) generated prefix, starting with [START]
    tokens = torch.full((N * B, 1), vocab.start_id, dtype=torch.long, device=device)

    # Cumulative log-probability per beam. Only beam 0 of each example starts
    # alive; the other B-1 are -inf so the first expansion cannot produce B
    # copies of the same single best token.
    beam_scores = torch.full((N, B), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)                     # (N*B,)

    finished = torch.zeros(N * B, dtype=torch.bool, device=device)
    lengths = torch.zeros(N * B, dtype=torch.long, device=device)

    NEG = torch.finfo(torch.float32).min

    for _ in range(max_len):
        # --- per-model log-probs, then fuse ---------------------------------
        log_prob_list = [br.step(tokens) for br in branches]
        fused = fuse_log_probs(log_prob_list, weights, space=fusion_space)  # (N*B, V)

        # --- a finished beam must not grow ----------------------------------
        # Keep it at an unchanged score via a single zero-cost [PAD] continuation.
        if finished.any():
            fused = fused.masked_fill(finished.unsqueeze(1), NEG)
            fused[finished, vocab.pad_id] = 0.0

        # --- expand and select the top B per example ------------------------
        cand = beam_scores.unsqueeze(1) + fused                 # (N*B, V)
        cand = cand.view(N, B * V)
        top_scores, top_idx = cand.topk(B, dim=-1)              # (N, B)

        beam_idx = top_idx // V                                 # which beam
        token_idx = top_idx % V                                 # which token

        # Flat indices into the N*B dimension for state/prefix reordering.
        flat = (torch.arange(N, device=device).unsqueeze(1) * B + beam_idx).view(-1)

        tokens = torch.cat(
            [tokens.index_select(0, flat), token_idx.view(-1, 1)], dim=1
        )
        beam_scores = top_scores.view(-1)

        prev_finished = finished.index_select(0, flat)
        lengths = lengths.index_select(0, flat) + (~prev_finished).long()
        finished = prev_finished | token_idx.view(-1).eq(vocab.end_id)

        for br in branches:
            br.reorder(flat)

        if beam_cfg.early_stopping and bool(finished.all()):
            break

    # --- collect ------------------------------------------------------------
    results: List[DecodeResult] = []
    tokens_cpu = tokens[:, 1:].cpu()          # drop the leading [START]
    scores_cpu = beam_scores.detach().cpu()
    lengths_cpu = lengths.cpu()
    finished_cpu = finished.cpu()

    for i in range(N):
        hyps: List[Hypothesis] = []
        for b in range(B):
            r = i * B + b
            score = float(scores_cpu[r])
            if score == float("-inf"):
                continue
            length = int(lengths_cpu[r])
            if length == 0:
                continue

            ids = [int(t) for t in tokens_cpu[r].tolist()[:length]]
            hyps.append(
                Hypothesis(
                    token_ids=ids,
                    smiles=vocab.decode(ids),
                    sum_log_prob=score,
                    length=length,
                    finished=bool(finished_cpu[r]),
                )
            )

        # Rank by length-normalized score.
        hyps.sort(key=lambda h: h.norm_score, reverse=True)
        if not return_all_beams:
            hyps = hyps[:1]
        results.append(DecodeResult(hyps))

    return results


# ---------------------------------------------------------------------------
# Final-step (sequence-level) fusion
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_search_final_step_fusion(
    gru_branch: BranchAdapter,
    tf_branch: BranchAdapter,
    src: torch.Tensor,
    vocab: Vocabulary,
    beam_cfg: Optional[BeamConfig] = None,
    w_gru: float = 0.7,
    w_tf: float = 0.3,
) -> List[DecodeResult]:
    """
    Generate candidates with the GRU alone, then rescore the completed
    hypotheses with the weighted sum of both models' sequence log-probabilities.
    """
    beam_cfg = beam_cfg or BeamConfig()

    # 1. Generate with the GRU branch alone.
    gru_branch.weight = 1.0
    results = beam_search(
        [gru_branch], src, vocab, beam_cfg=beam_cfg, fusion_space="log"
    )

    # 2. Rescore each completed candidate with the weighted combination of
    #    both models' sequence log-probabilities.
    rescored: List[DecodeResult] = []
    for i, res in enumerate(results):
        src_i = src[i : i + 1]
        new_hyps: List[Hypothesis] = []
        for h in res.hypotheses:
            lp_gru = sequence_log_prob(gru_branch.model, "gru", src_i, h.token_ids, vocab)
            lp_tf = sequence_log_prob(tf_branch.model, "transformer", src_i, h.token_ids, vocab)
            fused_lp = w_gru * lp_gru + w_tf * lp_tf
            new_hyps.append(
                Hypothesis(
                    token_ids=h.token_ids,
                    smiles=h.smiles,
                    sum_log_prob=fused_lp,
                    length=h.length,
                    finished=h.finished,
                )
            )
        new_hyps.sort(key=lambda x: x.norm_score, reverse=True)
        rescored.append(DecodeResult(new_hyps))
    return rescored


@torch.no_grad()
def sequence_log_prob(
    model: nn.Module,
    kind: str,
    src: torch.Tensor,            # (1, S)
    token_ids: Sequence[int],
    vocab: Vocabulary,
) -> float:
    """Teacher-forced total log-probability of a token sequence under one model."""
    device = src.device
    tgt = torch.tensor(
        [[vocab.start_id] + list(token_ids)], dtype=torch.long, device=device
    )
    logits = model(src, tgt)                       # (1, T-1, |V|)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    gold = tgt[:, 1:]                              # (1, T-1)
    picked = log_probs.gather(2, gold.unsqueeze(2)).squeeze(2)   # (1, T-1)
    mask = gold.ne(vocab.pad_id)
    return float((picked * mask).sum())


# ---------------------------------------------------------------------------
# Convenience: build adapters from loaded models
# ---------------------------------------------------------------------------
def make_branches(
    gru_model: Optional[nn.Module] = None,
    tf_model: Optional[nn.Module] = None,
    ens_cfg: Optional[EnsembleConfig] = None,
    mode: str = "ensemble",
) -> List[BranchAdapter]:
    """
    Assemble the adapter list for a given evaluation mode.

    mode : "gru"         -- GRU only, weight 1.0
           "transformer"  -- Transformer only, weight 1.0
           "ensemble"     -- both, at the configured fusion weights
    """
    ens_cfg = ens_cfg or EnsembleConfig()

    if mode == "gru":
        if gru_model is None:
            raise ValueError("mode='gru' requires gru_model")
        return [BranchAdapter(gru_model, "gru", 1.0)]

    if mode == "transformer":
        if tf_model is None:
            raise ValueError("mode='transformer' requires tf_model")
        return [BranchAdapter(tf_model, "transformer", 1.0)]

    if mode == "ensemble":
        if gru_model is None or tf_model is None:
            raise ValueError("mode='ensemble' requires both models")
        return [
            BranchAdapter(gru_model, "gru", ens_cfg.w_gru),
            BranchAdapter(tf_model, "transformer", ens_cfg.w_tf),
        ]

    raise ValueError(f"Unknown mode {mode!r}")


__all__ = [
    "fuse_log_probs",
    "BranchAdapter",
    "Hypothesis",
    "DecodeResult",
    "beam_search",
    "beam_search_final_step_fusion",
    "sequence_log_prob",
    "make_branches",
]
