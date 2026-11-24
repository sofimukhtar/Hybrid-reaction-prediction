"""
Learned per-step fusion weights.

A two-layer perceptron with sigmoid output maps the concatenated GRU and
Transformer decoder hidden states to alpha_t in [0, 1]:

    log p_ens(y_t) = log_softmax(alpha_t log p_GRU(y_t) + (1 - alpha_t) log p_TF(y_t))

Both branches stay frozen; only the gate is trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, CKPT_DIR, BeamConfig, GRUConfig, TransformerConfig
from .decode import DecodeResult, Hypothesis
from .tokenizer import Vocabulary


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
class StepwiseGate(nn.Module):
    """
    Two-layer perceptron mapping concatenated decoder hidden states to a
    scalar mixing weight in [0, 1].
    """

    def __init__(
        self,
        gru_hidden: int = 256,
        tf_hidden: int = 512,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = gru_hidden + tf_hidden
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, mlp_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, h_gru: torch.Tensor, h_tf: torch.Tensor) -> torch.Tensor:
        """
        h_gru : (B, 256)
        h_tf  : (B, 512)

        Returns alpha of shape (B, 1), the GRU weight at this step.
        """
        return torch.sigmoid(self.net(torch.cat([h_gru, h_tf], dim=-1)))


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------
def gru_step_with_hidden(
    gru_model: nn.Module,
    y_prev: torch.Tensor,
    state: dict,
) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    """One GRU decoding step, also returning the decoder hidden state."""
    logits, h_new, _ = gru_model.decoder(
        y_prev, state["h"], state["enc_out"], state["src_mask"]
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    new_state = {
        "enc_out": state["enc_out"],
        "h": h_new,
        "src_mask": state["src_mask"],
    }
    return log_probs, new_state, h_new


def transformer_step_with_hidden(
    tf_model: nn.Module,
    y_prefix: torch.Tensor,
    state: dict,
) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    """One Transformer decoding step, also returning the last-position hidden state."""
    tgt_emb = tf_model._embed(y_prefix)
    out = tf_model.transformer.decoder(
        tgt=tgt_emb,
        memory=state["memory"],
        tgt_mask=tf_model._causal_mask(y_prefix.size(1), y_prefix.device),
        tgt_key_padding_mask=tf_model._pad_mask(y_prefix),
        memory_key_padding_mask=state["src_kpm"],
    )
    h_last = out[:, -1]                               # (B, 512)
    log_probs = F.log_softmax(tf_model.fc_out(h_last).float(), dim=-1)
    return log_probs, state, h_last


# ---------------------------------------------------------------------------
# Training the gate (branches frozen)
# ---------------------------------------------------------------------------
def train_gate(
    gate: StepwiseGate,
    gru_model: nn.Module,
    tf_model: nn.Module,
    loader,
    vocab: Vocabulary,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> StepwiseGate:
    """
    Train only the gate (branches frozen) with teacher forcing and
    label-smoothed cross-entropy.
    """
    from .train import LabelSmoothingCrossEntropy

    gru_model.eval()
    tf_model.eval()
    for p in gru_model.parameters():
        p.requires_grad_(False)
    for p in tf_model.parameters():
        p.requires_grad_(False)

    gate = gate.to(device).train()
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=1e-4)
    criterion = LabelSmoothingCrossEntropy(
        vocab.size, epsilon=CFG.train.label_smoothing, pad_id=vocab.pad_id
    )

    for epoch in range(1, epochs + 1):
        total, nb = 0.0, 0
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            with torch.no_grad():
                gru_state = gru_model.encode(src)
                tf_state = tf_model.encode(src)

            T = tgt.size(1)
            fused_steps: List[torch.Tensor] = []

            for t in range(T - 1):
                with torch.no_grad():
                    lp_g, gru_state, h_g = gru_step_with_hidden(
                        gru_model, tgt[:, t], gru_state
                    )
                    lp_t, tf_state, h_t = transformer_step_with_hidden(
                        tf_model, tgt[:, : t + 1], tf_state
                    )

                alpha = gate(h_g, h_t)                        # (B, 1)
                fused = alpha * lp_g + (1.0 - alpha) * lp_t
                fused_steps.append(F.log_softmax(fused, dim=-1))

            logits = torch.stack(fused_steps, dim=1)          # (B, T-1, |V|)

            # Fused tensor is already log-probabilities.
            gold = tgt[:, 1:]
            mask = gold.ne(vocab.pad_id)
            nll = -logits.gather(2, gold.unsqueeze(2)).squeeze(2)
            smooth = -logits.mean(dim=-1)
            eps = CFG.train.label_smoothing
            loss_map = (1 - eps) * nll + eps * smooth
            loss = loss_map[mask].mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), CFG.train.grad_clip_norm)
            opt.step()

            total += loss.item()
            nb += 1

        if verbose:
            print(f"[gate] epoch {epoch}/{epochs}  loss {total/max(nb,1):.4f}")

    out_path = Path(out_path or (CKPT_DIR / "stepwise_gate.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gate_state_dict": gate.state_dict(), "in_dim": gate.in_dim}, out_path)
    print(f"[gate] Saved -> {out_path}")
    return gate.eval()


# ---------------------------------------------------------------------------
# Beam search with the learned gate
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_search_gated(
    gate: StepwiseGate,
    gru_model: nn.Module,
    tf_model: nn.Module,
    src: torch.Tensor,
    vocab: Vocabulary,
    beam_cfg: Optional[BeamConfig] = None,
) -> List[DecodeResult]:
    """Beam search with the mixing weight predicted per step by the gate."""
    beam_cfg = beam_cfg or CFG.beam
    B = beam_cfg.beam_width
    device = src.device
    N = src.size(0)
    V = vocab.size
    NEG = torch.finfo(torch.float32).min

    gru_state = gru_model.encode(src)
    tf_state = tf_model.encode(src)
    idx = torch.arange(N, device=device).repeat_interleave(B)
    gru_state = gru_model.reorder_state(gru_state, idx)
    tf_state = tf_model.reorder_state(tf_state, idx)

    tokens = torch.full((N * B, 1), vocab.start_id, dtype=torch.long, device=device)
    beam_scores = torch.full((N, B), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)
    finished = torch.zeros(N * B, dtype=torch.bool, device=device)
    lengths = torch.zeros(N * B, dtype=torch.long, device=device)

    for _ in range(beam_cfg.max_decode_len):
        lp_g, gru_state, h_g = gru_step_with_hidden(gru_model, tokens[:, -1], gru_state)
        lp_t, tf_state, h_t = transformer_step_with_hidden(tf_model, tokens, tf_state)

        alpha = gate(h_g, h_t)
        fused = F.log_softmax(alpha * lp_g + (1.0 - alpha) * lp_t, dim=-1)

        if finished.any():
            fused = fused.masked_fill(finished.unsqueeze(1), NEG)
            fused[finished, vocab.pad_id] = 0.0

        cand = (beam_scores.unsqueeze(1) + fused).view(N, B * V)
        top_scores, top_idx = cand.topk(B, dim=-1)
        beam_idx = top_idx // V
        token_idx = top_idx % V
        flat = (torch.arange(N, device=device).unsqueeze(1) * B + beam_idx).view(-1)

        tokens = torch.cat([tokens.index_select(0, flat), token_idx.view(-1, 1)], dim=1)
        beam_scores = top_scores.view(-1)
        prev_finished = finished.index_select(0, flat)
        lengths = lengths.index_select(0, flat) + (~prev_finished).long()
        finished = prev_finished | token_idx.view(-1).eq(vocab.end_id)

        gru_state = gru_model.reorder_state(gru_state, flat)
        tf_state = tf_model.reorder_state(tf_state, flat)

        if beam_cfg.early_stopping and bool(finished.all()):
            break

    results: List[DecodeResult] = []
    tokens_cpu = tokens[:, 1:].cpu()
    scores_cpu = beam_scores.detach().cpu()
    lengths_cpu = lengths.cpu()
    finished_cpu = finished.cpu()

    for i in range(N):
        hyps: List[Hypothesis] = []
        for b in range(B):
            r = i * B + b
            score = float(scores_cpu[r])
            length = int(lengths_cpu[r])
            if score == float("-inf") or length == 0:
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
        hyps.sort(key=lambda h: h.norm_score, reverse=True)
        results.append(DecodeResult(hyps))
    return results


def load_gate(
    path: Optional[Path | str] = None,
    device: Optional[torch.device] = None,
) -> StepwiseGate:
    path = Path(path or (CKPT_DIR / "stepwise_gate.pt"))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train the gate first with "
            f"`python scripts/06_ablation.py --train-gate`."
        )
    payload = torch.load(path, map_location=device or "cpu")
    gate = StepwiseGate(
        gru_hidden=GRUConfig().hidden_dim,
        tf_hidden=TransformerConfig().d_model,
    )
    gate.load_state_dict(payload["gate_state_dict"])
    gate.eval()
    if device:
        gate.to(device)
    print(f"[gate] Loaded <- {path}")
    return gate


__all__ = [
    "StepwiseGate",
    "gru_step_with_hidden",
    "transformer_step_with_hidden",
    "train_gate",
    "beam_search_gated",
    "load_gate",
]
