"""
Attention-enhanced GRU sequence-to-sequence model.

  GRUEncoder         embedding -> bidirectional multi-layer GRU; the top-layer
                     forward/backward states are concatenated and projected
                     with tanh to form the initial decoder state.
  BahdanauAttention  additive attention with [PAD] positions masked.
  GRUDecoder         one autoregressive step; the output head reads
                     [decoder output ; context ; previous-token embedding].

Training (`forward`) and inference (`decode_step`) share the same decoder
step implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GRUConfig, TokenizerConfig


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: GRUConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(cfg.dropout)

        self.rnn = nn.GRU(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )

        self.bridge = nn.Linear(cfg.enc_out_dim, cfg.hidden_dim)

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        src : (B, S) token ids, right-padded.

        Returns
          enc_out  : (B, S, enc_out_dim)
          h0_dec   : (B, hidden)
          src_mask : (B, S) bool, True at real tokens
        """
        src_mask = src.ne(self.pad_id)

        emb = self.dropout(self.embedding(src))

        # Pack so padded positions do not advance the recurrence.
        lengths = src_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1)
        )

        if self.cfg.bidirectional:
            h_n = h_n.view(self.cfg.n_layers, 2, src.size(0), self.cfg.hidden_dim)
            h_cat = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=-1)
        else:
            h_cat = h_n[-1]

        h0_dec = torch.tanh(self.bridge(h_cat))
        return enc_out, h0_dec, src_mask


class BahdanauAttention(nn.Module):
    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.W_enc = nn.Linear(cfg.enc_out_dim, cfg.attn_dim, bias=False)
        self.W_dec = nn.Linear(cfg.hidden_dim, cfg.attn_dim, bias=False)
        self.v = nn.Linear(cfg.attn_dim, 1, bias=False)

    def forward(
        self,
        h_dec_prev: torch.Tensor,     # (B, hidden)
        enc_out: torch.Tensor,        # (B, S, enc_out_dim)
        src_mask: torch.Tensor,       # (B, S) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns context (B, enc_out_dim) and attention weights (B, S)."""
        proj = self.W_enc(enc_out) + self.W_dec(h_dec_prev).unsqueeze(1)
        e = self.v(torch.tanh(proj)).squeeze(-1)
        e = e.masked_fill(~src_mask, torch.finfo(e.dtype).min)

        alpha = F.softmax(e, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)
        return context, alpha


class GRUDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: GRUConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(cfg.dropout)
        self.attention = BahdanauAttention(cfg)

        self.rnn = nn.GRU(
            input_size=cfg.emb_dim + cfg.enc_out_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc_out = nn.Linear(cfg.fc_in_dim, vocab_size)

    def forward(
        self,
        y_prev: torch.Tensor,         # (B,)
        h_prev: torch.Tensor,         # (B, hidden)
        enc_out: torch.Tensor,        # (B, S, enc_out_dim)
        src_mask: torch.Tensor,       # (B, S) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step. Returns logits (B, |V|), new hidden (B, hidden), alpha (B, S)."""
        emb = self.dropout(self.embedding(y_prev))
        context, alpha = self.attention(h_prev, enc_out, src_mask)

        rnn_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
        rnn_out, h_new = self.rnn(rnn_in, h_prev.unsqueeze(0))
        h_tilde = rnn_out.squeeze(1)

        logits = self.fc_out(torch.cat([h_tilde, context, emb], dim=-1))
        return logits, h_new.squeeze(0), alpha


class GRUSeq2Seq(nn.Module):
    """
    forward(src, tgt)          teacher-forced training pass
    encode(src)                run the encoder once
    decode_step(y_prev, state) one inference step
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: Optional[GRUConfig] = None,
        tok_cfg: Optional[TokenizerConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or GRUConfig()
        self.tok = tok_cfg or TokenizerConfig()
        self.vocab_size = vocab_size

        self.encoder = GRUEncoder(vocab_size, self.cfg, pad_id=self.tok.pad_id)
        self.decoder = GRUDecoder(vocab_size, self.cfg, pad_id=self.tok.pad_id)

    def forward(
        self,
        src: torch.Tensor,            # (B, S)
        tgt: torch.Tensor,            # (B, T) beginning with [START]
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits (B, T-1, |V|); position i predicts tgt[:, i+1]."""
        enc_out, h, src_mask = self.encoder(src)
        T = tgt.size(1)

        logits_all, attn_all = [], []
        for t in range(T - 1):
            logits, h, alpha = self.decoder(tgt[:, t], h, enc_out, src_mask)
            logits_all.append(logits)
            if return_attention:
                attn_all.append(alpha)

        logits = torch.stack(logits_all, dim=1)
        if return_attention:
            return logits, torch.stack(attn_all, dim=1)
        return logits

    @torch.no_grad()
    def encode(self, src: torch.Tensor) -> dict:
        enc_out, h0, src_mask = self.encoder(src)
        return {"enc_out": enc_out, "h": h0, "src_mask": src_mask}

    @torch.no_grad()
    def decode_step(
        self,
        y_prev: torch.Tensor,         # (B,)
        state: dict,
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Returns log_probs (B, |V|), new state, attention weights (B, S)."""
        logits, h_new, alpha = self.decoder(
            y_prev, state["h"], state["enc_out"], state["src_mask"]
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)
        new_state = {
            "enc_out": state["enc_out"],
            "h": h_new,
            "src_mask": state["src_mask"],
        }
        return log_probs, new_state, alpha

    @staticmethod
    def reorder_state(state: dict, index: torch.Tensor) -> dict:
        return {
            "enc_out": state["enc_out"].index_select(0, index),
            "h": state["h"].index_select(0, index),
            "src_mask": state["src_mask"].index_select(0, index),
        }

    def count_parameters(self, verbose: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print(f"[GRU] trainable parameters: {n:,} ({n/1e6:.2f}M)")
        return n


__all__ = ["GRUEncoder", "BahdanauAttention", "GRUDecoder", "GRUSeq2Seq"]
