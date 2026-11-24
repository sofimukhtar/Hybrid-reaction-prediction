"""Transformer encoder-decoder built on torch.nn.Transformer."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import TokenizerConfig, TransformerConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(1):
            raise ValueError(
                f"Sequence length {L} exceeds positional-encoding max_len {self.pe.size(1)}."
            )
        return self.dropout(x + self.pe[:, :L])


class TransformerSeq2Seq(nn.Module):
    """
    forward(src, tgt)             teacher-forced training pass
    encode(src)                   run the encoder once
    decode_step(y_prefix, state)  next-token log-probs given the full prefix
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: Optional[TransformerConfig] = None,
        tok_cfg: Optional[TokenizerConfig] = None,
    ):
        super().__init__()
        self.cfg = cfg or TransformerConfig()
        self.tok = tok_cfg or TokenizerConfig()
        self.vocab_size = vocab_size
        self.d_model = self.cfg.d_model

        self.embedding = nn.Embedding(
            vocab_size, self.cfg.d_model, padding_idx=self.tok.pad_id
        )
        self.pos_encoding = SinusoidalPositionalEncoding(
            self.cfg.d_model, max_len=self.cfg.max_len, dropout=self.cfg.dropout
        )

        self.transformer = nn.Transformer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.nhead,
            num_encoder_layers=self.cfg.num_encoder_layers,
            num_decoder_layers=self.cfg.num_decoder_layers,
            dim_feedforward=self.cfg.dim_feedforward,
            dropout=self.cfg.dropout,
            activation=self.cfg.activation,
            batch_first=True,
            norm_first=self.cfg.norm_first,
        )

        self.fc_out = nn.Linear(self.cfg.d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pos_encoding(self.embedding(tokens) * math.sqrt(self.d_model))

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """True at positions to ignore."""
        return tokens.eq(self.tok.pad_id)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        src: torch.Tensor,            # (B, S)
        tgt: torch.Tensor,            # (B, T) beginning with [START]
    ) -> torch.Tensor:
        """Returns logits (B, T-1, |V|) aligned with tgt[:, 1:]."""
        tgt_in = tgt[:, :-1]

        out = self.transformer(
            src=self._embed(src),
            tgt=self._embed(tgt_in),
            tgt_mask=self._causal_mask(tgt_in.size(1), tgt_in.device),
            src_key_padding_mask=self._pad_mask(src),
            tgt_key_padding_mask=self._pad_mask(tgt_in),
            memory_key_padding_mask=self._pad_mask(src),
        )
        return self.fc_out(out)

    @torch.no_grad()
    def encode(self, src: torch.Tensor) -> dict:
        src_kpm = self._pad_mask(src)
        memory = self.transformer.encoder(self._embed(src), src_key_padding_mask=src_kpm)
        return {"memory": memory, "src_kpm": src_kpm}

    @torch.no_grad()
    def decode_step(
        self,
        y_prefix: torch.Tensor,       # (B, t)
        state: dict,
    ) -> Tuple[torch.Tensor, dict, None]:
        """
        Run the decoder over the full prefix and return log-probs (B, |V|) for
        the last position. The third return value keeps the signature aligned
        with the GRU model.
        """
        out = self.transformer.decoder(
            tgt=self._embed(y_prefix),
            memory=state["memory"],
            tgt_mask=self._causal_mask(y_prefix.size(1), y_prefix.device),
            tgt_key_padding_mask=self._pad_mask(y_prefix),
            memory_key_padding_mask=state["src_kpm"],
        )
        logits = self.fc_out(out[:, -1])
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs, state, None

    @staticmethod
    def reorder_state(state: dict, index: torch.Tensor) -> dict:
        return {
            "memory": state["memory"].index_select(0, index),
            "src_kpm": state["src_kpm"].index_select(0, index),
        }

    def count_parameters(self, verbose: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print(f"[Transformer] trainable parameters: {n:,} ({n/1e6:.2f}M)")
        return n


__all__ = ["SinusoidalPositionalEncoding", "TransformerSeq2Seq"]
