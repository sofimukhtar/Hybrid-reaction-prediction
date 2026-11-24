"""Regex SMILES tokenizer and the vocabulary shared by both model branches."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import TokenizerConfig

# Alternatives are tried left to right, so longer patterns come first.
SMILES_TOKEN_PATTERN = (
    r"(\[[^\]]+\]"                   # bracketed atoms, incl. stereo and charges
    r"|Br|Cl|Si|Se|se|As|as|Te|te"   # multi-character element symbols
    r"|B|C|N|O|P|S|F|I|b|c|n|o|p|s"  # single-character organic subset
    r"|%[0-9]{2}"                    # two-digit ring-closure labels
    r"|[0-9]"                        # single-digit ring-closure labels
    r"|\(|\)"                        # branches
    r"|\.|=|#|\$|:|~|-|\+"           # bonds, dot separator, charges
    r"|/|\\"                         # directional bonds
    r"|@@|@"                         # chirality markers
    r"|\*"                           # wildcard atom
    r"|>>|>"                         # reaction arrow
    r")"
)

SMILES_REGEX = re.compile(SMILES_TOKEN_PATTERN)


def tokenize_smiles(smiles: str) -> List[str]:
    """
    Split a SMILES string into tokens. Characters the regex cannot match are
    emitted as single-character tokens.

    >>> tokenize_smiles("Brc1ccccc1")
    ['Br', 'c', '1', 'c', 'c', 'c', 'c', 'c', '1']
    >>> tokenize_smiles("C[C@@H](Cl)N")
    ['C', '[C@@H]', '(', 'Cl', ')', 'N']
    >>> tokenize_smiles("C1CC%10CC%10C1")
    ['C', '1', 'C', 'C', '%10', 'C', 'C', '%10', 'C', '1']
    """
    if not smiles:
        return []

    tokens: List[str] = []
    pos = 0
    for match in SMILES_REGEX.finditer(smiles):
        start, end = match.span()
        if start > pos:
            tokens.extend(list(smiles[pos:start]))
        tokens.append(match.group(0))
        pos = end
    if pos < len(smiles):
        tokens.extend(list(smiles[pos:]))
    return tokens


def detokenize_smiles(tokens: Sequence[str]) -> str:
    return "".join(tokens)


class Vocabulary:
    """
    Token <-> index mapping.

        0 -> [PAD], 1 -> [START], 2 -> [END], 3 -> [UNK]
        4.. -> chemical tokens, by descending training frequency then
               lexicographically.
    """

    def __init__(self, itos: List[str], cfg: Optional[TokenizerConfig] = None):
        self.cfg = cfg or TokenizerConfig()
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

        expected_prefix = self.cfg.special_tokens
        if self.itos[: len(expected_prefix)] != expected_prefix:
            raise ValueError(
                f"Vocabulary must begin with {expected_prefix}, "
                f"got {self.itos[:len(expected_prefix)]}"
            )

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def n_chemical_tokens(self) -> int:
        return len(self.itos) - self.cfg.n_special_tokens

    @property
    def pad_id(self) -> int:
        return self.cfg.pad_id

    @property
    def start_id(self) -> int:
        return self.cfg.start_id

    @property
    def end_id(self) -> int:
        return self.cfg.end_id

    @property
    def unk_id(self) -> int:
        return self.cfg.unk_id

    @classmethod
    def build(
        cls,
        train_strings: Iterable[str],
        cfg: Optional[TokenizerConfig] = None,
        min_freq: int = 1,
        verbose: bool = True,
    ) -> "Vocabulary":
        """Build the vocabulary from training input and output strings."""
        cfg = cfg or TokenizerConfig()
        counter: Counter = Counter()
        n_strings = 0
        for s in train_strings:
            counter.update(tokenize_smiles(s))
            n_strings += 1

        chemical = sorted(
            (t for t, c in counter.items() if c >= min_freq),
            key=lambda t: (-counter[t], t),
        )
        itos = list(cfg.special_tokens) + chemical
        vocab = cls(itos, cfg)

        if verbose:
            print(
                f"[vocab] Built from {n_strings:,} training strings: "
                f"|V| = {vocab.size} "
                f"({cfg.n_special_tokens} special + {vocab.n_chemical_tokens} chemical)"
            )
            dropped = [t for t, c in counter.items() if c < min_freq]
            if dropped:
                print(f"[vocab] {len(dropped)} tokens dropped below min_freq={min_freq}")
        return vocab

    def encode(
        self,
        smiles: str,
        max_len: Optional[int] = None,
        add_special: bool = True,
        pad: bool = True,
    ) -> List[int]:
        """
        SMILES -> ids: [START] t1 ... tn [END] [PAD] ...
        Truncation keeps a trailing [END].
        """
        max_len = max_len or self.cfg.max_len
        tokens = tokenize_smiles(smiles)
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]

        if add_special:
            budget = max_len - 2
            if budget < 0:
                raise ValueError(f"max_len={max_len} is too small for special tokens")
            ids = [self.start_id] + ids[:budget] + [self.end_id]
        else:
            ids = ids[:max_len]

        if pad and len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def decode(
        self,
        ids: Sequence[int],
        strip_special: bool = True,
    ) -> str:
        """Ids -> SMILES. Stops at [END]; [PAD], [START] and [UNK] are dropped."""
        out: List[str] = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if strip_special and i in (self.pad_id, self.start_id):
                continue
            if i == self.unk_id:
                continue
            if 0 <= i < self.size:
                out.append(self.itos[i])
        return "".join(out)

    def decode_tokens(self, ids: Sequence[int]) -> List[str]:
        """Ids -> token list (for token-level metrics)."""
        out: List[str] = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if i in (self.pad_id, self.start_id):
                continue
            if 0 <= i < self.size:
                out.append(self.itos[i])
        return out

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "itos": self.itos,
            "size": self.size,
            "n_special": self.cfg.n_special_tokens,
            "max_len": self.cfg.max_len,
            "regex": SMILES_TOKEN_PATTERN,
        }
        path.write_text(json.dumps(payload, indent=2))
        print(f"[vocab] Saved |V| = {self.size} -> {path}")

    @classmethod
    def load(cls, path: Path | str, cfg: Optional[TokenizerConfig] = None) -> "Vocabulary":
        path = Path(path)
        payload = json.loads(path.read_text())
        vocab = cls(payload["itos"], cfg)
        print(f"[vocab] Loaded |V| = {vocab.size} <- {path}")
        return vocab


__all__ = [
    "SMILES_TOKEN_PATTERN",
    "SMILES_REGEX",
    "tokenize_smiles",
    "detokenize_smiles",
    "Vocabulary",
]
