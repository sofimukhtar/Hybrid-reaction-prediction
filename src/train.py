"""Training loop shared by both model branches."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CFG, CKPT_DIR, RESULTS_DIR, TrainConfig
from .tokenizer import Vocabulary


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    """
    L = -(1 - eps) log p(y*) - (eps / |V|) * sum_v log p(v)

    Padded target positions are excluded from the loss.
    """

    def __init__(self, vocab_size: int, epsilon: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.epsilon = epsilon
        self.pad_id = pad_id

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, T, |V|)
        target : (B, T)
        """
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

        mask = target.ne(self.pad_id)
        if mask.sum() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits.float(), dim=-1)

        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)

        loss = (1.0 - self.epsilon) * nll + self.epsilon * smooth
        return loss[mask].mean()


# ---------------------------------------------------------------------------
# Per-epoch validation micro-F1 (teacher-forced argmax)
# ---------------------------------------------------------------------------
def teacher_forced_micro_f1(
    pred_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    pad_id: int = 0,
) -> Tuple[int, int, int]:
    """
    Token-multiset TP / FP / FN for one batch, excluding [PAD]:

        TP_v = min(count_pred(v), count_gold(v))
        FP_v = max(0, count_pred(v) - count_gold(v))
        FN_v = max(0, count_gold(v) - count_pred(v))
    """
    tp = fp = fn = 0
    pred_ids = pred_ids.detach().cpu()
    gold_ids = gold_ids.detach().cpu()

    for p_row, g_row in zip(pred_ids, gold_ids):
        p_counts = Counter(int(t) for t in p_row.tolist() if int(t) != pad_id)
        g_counts = Counter(int(t) for t in g_row.tolist() if int(t) != pad_id)
        for v in set(p_counts) | set(g_counts):
            pc, gc = p_counts.get(v, 0), g_counts.get(v, 0)
            tp += min(pc, gc)
            fp += max(0, pc - gc)
            fn += max(0, gc - pc)
    return tp, fp, fn


def micro_f1_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": p, "recall": r, "micro_f1": f1}


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    log_every: int = 100,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    t0 = time.time()
    use_amp = scaler is not None and device.type == "cuda"

    for step, (src, tgt) in enumerate(loader, 1):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(src, tgt)          # (B, T-1, |V|)
            loss = criterion(logits, tgt[:, 1:])

        if use_amp:
            scaler.scale(loss).backward()
            # Unscale before clipping.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch {epoch:02d} step {step:5d}/{len(loader)} "
                f"loss {total_loss/n_batches:.4f} lr {lr:.2e} "
                f"({time.time()-t0:.0f}s)"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int = 0,
) -> Dict[str, float]:
    """Validation loss plus teacher-forced token-level micro-F1."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    tp = fp = fn = 0

    for src, tgt in loader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        logits = model(src, tgt)
        gold = tgt[:, 1:]
        total_loss += criterion(logits, gold).item()
        n_batches += 1

        pred = logits.argmax(dim=-1)
        # Ignore positions whose gold token is [PAD].
        pred = pred.masked_fill(gold.eq(pad_id), pad_id)

        b_tp, b_fp, b_fn = teacher_forced_micro_f1(pred, gold, pad_id=pad_id)
        tp += b_tp
        fp += b_fp
        fn += b_fn

    out = {"val_loss": total_loss / max(n_batches, 1)}
    out.update(micro_f1_from_counts(tp, fp, fn))
    return out


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab: Vocabulary,
    cfg: Optional[TrainConfig] = None,
    device: Optional[torch.device] = None,
    ckpt_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Train one model, saving a checkpoint every epoch and a history JSON."""
    cfg = cfg or CFG.train
    device = device or resolve_device(CFG.runtime.device)
    ckpt_dir = Path(ckpt_dir or CKPT_DIR) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir or RESULTS_DIR)

    set_seed(cfg.random_seed)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {model_name}: {n_params:,} trainable parameters ({n_params/1e6:.2f}M)")
    print(f"[train] device={device}  epochs={cfg.epochs}  batch_size={cfg.batch_size}")

    criterion = LabelSmoothingCrossEntropy(
        vocab_size=vocab.size, epsilon=cfg.label_smoothing, pad_id=vocab.pad_id
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.onecycle_pct_start,
        anneal_strategy=cfg.onecycle_anneal_strategy,
        div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div_factor,
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if (cfg.mixed_precision and device.type == "cuda")
        else None
    )
    if scaler is None and cfg.mixed_precision:
        print("[train] FP16 requested but device is not CUDA; training in FP32.")

    history: Dict[str, List[float]] = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_micro_f1": [], "val_precision": [], "val_recall": [], "lr": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, cfg, epoch, log_every=CFG.runtime.log_every_n_steps,
        )
        val = validate_one_epoch(
            model, val_loader, criterion, device, pad_id=vocab.pad_id
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val["val_loss"])
        history["val_micro_f1"].append(val["micro_f1"])
        history["val_precision"].append(val["precision"])
        history["val_recall"].append(val["recall"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"[train] epoch {epoch:02d}  train_loss {tr_loss:.4f}  "
            f"val_loss {val['val_loss']:.4f}  val_micro_f1 {val['micro_f1']:.4f}"
        )

        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": tr_loss,
                "val_loss": val["val_loss"],
                "val_micro_f1": val["micro_f1"],
                "vocab_size": vocab.size,
                "train_config": asdict(cfg),
            },
            ckpt_path,
        )

    # ---- Checkpoint selection ----------------------------------------------
    sel = cfg.selected_epoch
    sel_path = ckpt_dir / f"epoch_{sel:03d}.pt"
    final_path = ckpt_dir / f"{model_name}_selected.pt"

    if sel_path.exists():
        torch.save(torch.load(sel_path, map_location="cpu"), final_path)
        print(f"[train] Selected epoch {sel} -> {final_path}")
    else:
        print(f"[train] WARNING: epoch {sel} checkpoint missing; cannot select it.")

    best_loss_epoch = int(np.argmin(history["val_loss"])) + 1
    best_f1_epoch = int(np.argmax(history["val_micro_f1"])) + 1
    print(
        f"[train] min val_loss at epoch {best_loss_epoch} "
        f"({min(history['val_loss']):.4f}), "
        f"peak val_micro_f1 at epoch {best_f1_epoch} "
        f"({max(history['val_micro_f1']):.4f})."
    )
    if best_loss_epoch != sel or best_f1_epoch != sel:
        print(
            f"[train] NOTE: best epoch differs from selected_epoch={sel}; "
            f"epoch {sel} was still used for {final_path.name}."
        )

    hist_path = results_dir / f"training_history_{model_name}.json"
    hist_path.write_text(json.dumps(
        {
            "model_name": model_name,
            "n_parameters": n_params,
            "selected_epoch": sel,
            "best_loss_epoch": best_loss_epoch,
            "best_f1_epoch": best_f1_epoch,
            "history": history,
        },
        indent=2,
    ))
    print(f"[train] Wrote history -> {hist_path}")

    return {"history": history, "n_parameters": n_params, "checkpoint": str(final_path)}


def load_checkpoint(
    model: nn.Module,
    path: Path | str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """Load weights into an already-constructed model and set eval mode."""
    device = device or torch.device("cpu")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint {path} not found. Train the model first or pass --ckpt."
        )
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    ep = payload.get("epoch", "?")
    print(f"[train] Loaded {path.name} (epoch {ep}) onto {device}")
    return model


__all__ = [
    "set_seed",
    "resolve_device",
    "LabelSmoothingCrossEntropy",
    "teacher_forced_micro_f1",
    "micro_f1_from_counts",
    "train_one_epoch",
    "validate_one_epoch",
    "train_model",
    "load_checkpoint",
]
