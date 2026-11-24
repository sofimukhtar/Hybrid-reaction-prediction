"""Hyperparameters, constants and repository paths."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"

for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    hf_dataset_id: str = "liupf/SLM4CRP_with_RTs"
    hf_dataset_url: str = "https://huggingface.co/datasets/liupf/SLM4CRP_with_RTs"

    # Only the forward-reaction-prediction task is used.
    task_filter: str = "forward reaction prediction"

    # 70 / 15 / 15 partition at the reaction level.
    split_fractions: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    random_seed: int = 42

    # Precursor canonicalization: RDKit canonical SMILES per fragment,
    # dot-delimited and sorted alphabetically.
    sort_precursor_fragments: bool = True
    isomeric_smiles: bool = True
    canonicalize_targets: bool = True

    # Fixed subset used only for plots.
    viz_subset_size: int = 300
    viz_subset_seed: int = 42

    # Corpus-level reaction-category proportions (used as weights).
    reaction_type_proportions: Dict[str, float] = field(
        default_factory=lambda: {
            "substitution": 0.606,
            "ring_formation": 0.288,
            "functional_group_transformation": 0.106,
        }
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TokenizerConfig:
    pad_token: str = "[PAD]"
    start_token: str = "[START]"
    end_token: str = "[END]"
    unk_token: str = "[UNK]"

    # [PAD] = 0 so padded positions double as the loss ignore_index and the
    # key-padding-mask value.
    pad_id: int = 0
    start_id: int = 1
    end_id: int = 2
    unk_id: int = 3

    n_special_tokens: int = 4

    # Vocabulary is built from the training split only.
    build_vocab_from: str = "train"

    # Maximum sequence length including [START] and [END].
    max_len: int = 128

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.start_token, self.end_token, self.unk_token]


# ---------------------------------------------------------------------------
# GRU Seq2Seq
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GRUConfig:
    emb_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1

    # Bahdanau attention projection size.
    attn_dim: int = 128

    # Encoder-to-decoder bridge activation.
    bridge_activation: str = "tanh"

    @property
    def enc_out_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    @property
    def fc_in_dim(self) -> int:
        # [decoder hidden ; context ; previous-token embedding]
        return self.hidden_dim + self.enc_out_dim + self.emb_dim


# ---------------------------------------------------------------------------
# Transformer encoder-decoder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    positional_encoding: str = "sinusoidal"
    max_len: int = 128
    norm_first: bool = False
    activation: str = "relu"

    @property
    def d_k(self) -> int:
        return self.d_model // self.nhead


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "OneCycleLR"
    onecycle_pct_start: float = 0.05
    onecycle_anneal_strategy: str = "cos"
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    label_smoothing: float = 0.1
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    teacher_forcing: bool = True
    random_seed: int = 42

    # Epoch whose checkpoint is used for evaluation.
    selected_epoch: int = 16


# ---------------------------------------------------------------------------
# Ensemble fusion and beam search
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnsembleConfig:
    w_gru: float = 0.7
    w_tf: float = 0.3

    # Validation sweep grid for w_GRU.
    sweep_start: float = 0.0
    sweep_stop: float = 1.0
    sweep_step: float = 0.1

    # Fusion is applied at every decoding step in log space and renormalized.
    fusion_space: str = "log"
    normalize_per_step: bool = True


@dataclass(frozen=True)
class BeamConfig:
    beam_width: int = 5
    length_penalty_beta: float = 1.0
    max_decode_len: int = 128
    early_stopping: bool = True
    topk_values: Tuple[int, ...] = (1, 2, 3, 5)

    constrained_decoding: bool = False

    # False: re-encode the full Transformer prefix at every step.
    # True:  use an incremental cache (faster, numerically equivalent up to
    #        floating-point associativity).
    use_kv_cache: bool = False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    morgan_radius: int = 2
    morgan_nbits: int = 2048

    # Sentence-level bigram BLEU with add-1 smoothing on the raw token sequence.
    bleu_max_n: int = 2
    bleu_smoothing: str = "add1"

    n_calibration_bins: int = 10

    n_bootstrap: int = 1_000
    bootstrap_ci: float = 0.95
    bootstrap_seed: int = 42

    # Levenshtein-accuracy threshold below which a prediction counts as
    # severely degraded.
    severe_la_threshold: float = 0.8


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@dataclass
class RuntimeConfig:
    device: str = "cuda"          # falls back to cpu automatically
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    log_every_n_steps: int = 100
    eval_batch_size: int = 32


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    beam: BeamConfig = field(default_factory=BeamConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict:
        return asdict(self)


CFG = Config()

__all__ = [
    "CFG", "Config", "DataConfig", "TokenizerConfig", "GRUConfig",
    "TransformerConfig", "TrainConfig", "EnsembleConfig", "BeamConfig",
    "EvalConfig", "RuntimeConfig",
    "REPO_ROOT", "DATA_DIR", "CKPT_DIR", "RESULTS_DIR", "FIGURES_DIR",
]
