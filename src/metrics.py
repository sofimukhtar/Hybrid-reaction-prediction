"""
Evaluation metrics.

Exact match, character accuracy, prefix match, Levenshtein accuracy, bigram
BLEU, micro-F1 over chemical tokens, Top-K accuracy, length/edit regression
errors, calibration (ECE, MCE, Brier, confidence-accuracy correlation),
Morgan-fingerprint Tanimoto similarity and bootstrap confidence intervals.

For a prediction RDKit cannot parse/sanitize: exact match = 0 and Tanimoto =
0.0; string metrics are still computed on the raw string.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import EvalConfig
from .tokenizer import tokenize_smiles

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:  # pragma: no cover
    _RDKIT = False


# ===========================================================================
# RDKit helpers
# ===========================================================================
def canon(smiles: str, isomeric: bool = True) -> Optional[str]:
    """RDKit canonical SMILES with full sanitization, or None."""
    if not _RDKIT:
        raise RuntimeError("RDKit is required. pip install rdkit")
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())      # sanitizes by default
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric, canonical=True)
    except Exception:
        return None


def parses_syntactically(smiles: str) -> bool:
    """True if RDKit can build a molecule graph without sanitization."""
    if not smiles or not smiles.strip():
        return False
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
        return mol is not None
    except Exception:
        return False


def is_chemically_valid(smiles: str) -> bool:
    """True if the string passes full RDKit sanitization."""
    return canon(smiles) is not None


def morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    nbits: int = 2048,
):
    """Morgan fingerprint bit vector, or None."""
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto_similarity(
    pred: str,
    ref: str,
    radius: int = 2,
    nbits: int = 2048,
) -> float:
    """Tanimoto similarity of Morgan fingerprints; 0.0 if either is invalid."""
    fp_pred = morgan_fingerprint(pred, radius, nbits)
    fp_ref = morgan_fingerprint(ref, radius, nbits)
    if fp_pred is None or fp_ref is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_pred, fp_ref))


# ===========================================================================
# String distance primitives
# ===========================================================================
def levenshtein_distance(a: str, b: str) -> int:
    """Levenshtein edit distance (unit costs)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,                      # deletion
                cur[j - 1] + 1,                   # insertion
                prev[j - 1] + (ca != cb),         # substitution
            )
        prev = cur
    return prev[-1]


def longest_common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix."""
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


# ===========================================================================
# Per-sample metrics
# ===========================================================================
def exact_match(pred: str, ref: str) -> int:
    """1 if the prediction is valid and canonically identical to the reference."""
    cp = canon(pred)
    if cp is None:
        return 0
    cr = canon(ref)
    if cr is None:
        return 0
    return int(cp == cr)


def char_accuracy(pred: str, ref: str) -> float:
    """Position-wise character agreement, normalized by max(|pred|, |ref|)."""
    if not pred and not ref:
        return 1.0
    denom = max(len(pred), len(ref))
    if denom == 0:
        return 1.0
    n = min(len(pred), len(ref))
    matches = sum(1 for j in range(n) if pred[j] == ref[j])
    return matches / denom


def prefix_match(pred: str, ref: str) -> float:
    """Longest common prefix length divided by reference length."""
    if len(ref) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    return longest_common_prefix_len(pred, ref) / len(ref)


def levenshtein_accuracy(pred: str, ref: str) -> float:
    """
    1 - d_lev / max(|pred|, |ref|). Both empty -> 1.0; one empty -> 0.0.
    """
    if len(pred) == 0 and len(ref) == 0:
        return 1.0
    if len(pred) == 0 or len(ref) == 0:
        return 0.0
    denom = max(len(pred), len(ref))
    return 1.0 - levenshtein_distance(pred, ref) / denom


def levenshtein_error_ref_normalized(pred: str, ref: str) -> float:
    """Edit distance normalized by reference length (d_lev / |ref|)."""
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein_distance(pred, ref) / len(ref)


def bleu2(pred: str, ref: str, smoothing: str = "add1") -> float:
    """
    Sentence-level BLEU over SMILES tokens with n = 1, 2, brevity penalty
    min(1, exp(1 - |ref|/|pred|)) and optional add-1 smoothing.
    """
    p_tokens = tokenize_smiles(pred)
    r_tokens = tokenize_smiles(ref)

    if not p_tokens or not r_tokens:
        return 0.0

    log_precisions: List[float] = []
    for n in (1, 2):
        p_ngrams = Counter(
            tuple(p_tokens[i : i + n]) for i in range(len(p_tokens) - n + 1)
        )
        r_ngrams = Counter(
            tuple(r_tokens[i : i + n]) for i in range(len(r_tokens) - n + 1)
        )
        overlap = sum(min(c, r_ngrams.get(g, 0)) for g, c in p_ngrams.items())
        total = max(sum(p_ngrams.values()), 0)

        if smoothing == "add1":
            prec = (overlap + 1) / (total + 1)
        else:
            if total == 0 or overlap == 0:
                return 0.0
            prec = overlap / total
        log_precisions.append(np.log(max(prec, 1e-12)))

    ratio = len(r_tokens) / max(len(p_tokens), 1)
    bp = min(1.0, float(np.exp(1.0 - ratio)))

    return float(bp * np.exp(np.mean(log_precisions)))


# ===========================================================================
# Micro-averaged F1
# ===========================================================================
def token_counts_for_f1(
    pred: str, ref: str
) -> Tuple[int, int, int]:
    """
    Per-sample TP / FP / FN over SMILES token multisets.

        TP_v = min(count_pred(v), count_ref(v))
        FP_v = max(0, count_pred(v) - count_ref(v))
        FN_v = max(0, count_ref(v) - count_pred(v))
    """
    pc = Counter(tokenize_smiles(pred))
    rc = Counter(tokenize_smiles(ref))
    tp = fp = fn = 0
    for v in set(pc) | set(rc):
        a, b = pc.get(v, 0), rc.get(v, 0)
        tp += min(a, b)
        fp += max(0, a - b)
        fn += max(0, b - a)
    return tp, fp, fn


def micro_f1(preds: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    """Micro-averaged precision / recall / F1 with globally summed counts."""
    TP = FP = FN = 0
    for p, r in zip(preds, refs):
        tp, fp, fn = token_counts_for_f1(p, r)
        TP += tp
        FP += fp
        FN += fn
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F = (2 * P * R / (P + R)) if (P + R) else 0.0
    return {"precision_micro": P, "recall_micro": R, "micro_f1": F,
            "tp": TP, "fp": FP, "fn": FN}


# ===========================================================================
# Top-K accuracy
# ===========================================================================
def topk_candidate_set(candidates: Sequence[str], k: int) -> List[str]:
    """
    Up to K valid, canonicalized, distinct candidates in rank order.
    Duplicates are removed before truncating to K.
    """
    seen: List[str] = []
    for c in candidates:
        cc = canon(c)
        if cc is None:
            continue
        if cc not in seen:
            seen.append(cc)
        if len(seen) >= k:
            break
    return seen[:k]


def topk_hit(candidates: Sequence[str], ref: str, k: int) -> int:
    """1 if the canonical reference is among the top-K distinct candidates."""
    cr = canon(ref)
    if cr is None:
        return 0
    return int(cr in topk_candidate_set(candidates, k))


# ===========================================================================
# Regression errors
# ===========================================================================
def regression_errors(errors: Sequence[float]) -> Dict[str, float]:
    """MAE, MSE and RMSE for an error series."""
    e = np.asarray(errors, dtype=float)
    if e.size == 0:
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}
    mae = float(np.mean(np.abs(e)))
    mse = float(np.mean(e ** 2))
    return {"mae": mae, "mse": mse, "rmse": float(np.sqrt(mse))}


# ===========================================================================
# Calibration
# ===========================================================================
def calibration_bins(
    confidences: Sequence[float],
    correct: Sequence[int],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """
    Equal-width confidence bins ((m-1)/M, m/M] (first bin includes 0) with
    per-bin accuracy and mean confidence.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = len(conf)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []

    for m in range(n_bins):
        lo, hi = edges[m], edges[m + 1]
        if m == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        bins.append(
            {
                "bin": m + 1,
                "lower": float(lo),
                "upper": float(hi),
                "count": cnt,
                "fraction": cnt / n if n else 0.0,
                "accuracy": float(corr[mask].mean()) if cnt else 0.0,
                "confidence": float(conf[mask].mean()) if cnt else 0.0,
            }
        )
    return bins


def expected_calibration_error(bins: Sequence[Dict[str, float]]) -> float:
    """Sum over bins of (|B_m|/N) * |acc(B_m) - conf(B_m)|."""
    return float(
        sum(b["fraction"] * abs(b["accuracy"] - b["confidence"]) for b in bins)
    )


def maximum_calibration_error(bins: Sequence[Dict[str, float]]) -> float:
    """Largest |accuracy - confidence| gap over non-empty bins."""
    gaps = [abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"] > 0]
    return float(max(gaps)) if gaps else 0.0


def brier_score(confidences: Sequence[float], correct: Sequence[int]) -> float:
    """Mean squared difference between confidence and binary correctness."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.size == 0:
        return 0.0
    return float(np.mean((c - y) ** 2))


def confidence_accuracy_correlation(
    confidences: Sequence[float], correct: Sequence[int]
) -> float:
    """Pearson r between two series; 0.0 if either has no variance."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.size < 2 or c.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(c, y)[0, 1])


# ===========================================================================
# Bootstrap confidence intervals
# ===========================================================================
def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile bootstrap CI for the mean: point estimate, bounds, half-width."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "half_width": 0.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = v[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    point = float(v.mean())
    return {
        "mean": point,
        "lo": lo,
        "hi": hi,
        "half_width": (hi - lo) / 2.0,
    }


# ===========================================================================
# Aggregate container
# ===========================================================================
@dataclass
class PerSampleRecord:
    """All per-sample measurements for one reaction."""

    rxn_id: str
    input_smiles: str
    ref_smiles: str
    pred_smiles: str
    candidates: List[str] = field(default_factory=list)

    exact_match: int = 0
    char_accuracy: float = 0.0
    prefix_match: float = 0.0
    levenshtein_accuracy: float = 0.0
    bleu2: float = 0.0
    tanimoto: float = 0.0

    length_error: float = 0.0          # |L_hat - L|
    lev_error_ref: float = 0.0         # d_lev / |y|
    lev_error_symmetric: float = 0.0   # 1 - LA

    syntactically_valid: bool = False
    chemically_valid: bool = False

    confidence: float = 0.0
    norm_score: float = 0.0
    n_candidates: int = 0
    topk: Dict[int, int] = field(default_factory=dict)

    reaction_type: Optional[str] = None
    unseen_scaffold: Optional[bool] = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["topk"] = {str(k): v for k, v in self.topk.items()}
        return d


def score_one(
    rxn_id: str,
    input_smiles: str,
    ref_smiles: str,
    candidates: Sequence[str],
    confidence: float,
    norm_score: float,
    eval_cfg: Optional[EvalConfig] = None,
    topk_values: Sequence[int] = (1, 2, 3, 5),
) -> PerSampleRecord:
    """Score one reaction. `candidates` must be ranked best first."""
    eval_cfg = eval_cfg or EvalConfig()
    pred = candidates[0] if candidates else ""

    rec = PerSampleRecord(
        rxn_id=rxn_id,
        input_smiles=input_smiles,
        ref_smiles=ref_smiles,
        pred_smiles=pred,
        candidates=list(candidates),
        confidence=confidence,
        norm_score=norm_score,
        n_candidates=len(candidates),
    )

    rec.syntactically_valid = parses_syntactically(pred)
    rec.chemically_valid = is_chemically_valid(pred)

    rec.exact_match = exact_match(pred, ref_smiles)
    rec.char_accuracy = char_accuracy(pred, ref_smiles)
    rec.prefix_match = prefix_match(pred, ref_smiles)
    rec.levenshtein_accuracy = levenshtein_accuracy(pred, ref_smiles)
    rec.bleu2 = bleu2(pred, ref_smiles, smoothing=eval_cfg.bleu_smoothing)
    rec.tanimoto = tanimoto_similarity(
        pred, ref_smiles, eval_cfg.morgan_radius, eval_cfg.morgan_nbits
    )

    rec.length_error = abs(len(pred) - len(ref_smiles))
    rec.lev_error_ref = levenshtein_error_ref_normalized(pred, ref_smiles)
    rec.lev_error_symmetric = 1.0 - rec.levenshtein_accuracy

    rec.topk = {k: topk_hit(candidates, ref_smiles, k) for k in topk_values}
    return rec


def aggregate(
    records: Sequence[PerSampleRecord],
    eval_cfg: Optional[EvalConfig] = None,
    topk_values: Sequence[int] = (1, 2, 3, 5),
    with_bootstrap: bool = True,
) -> Dict[str, object]:
    """Aggregate per-sample records, with bootstrap CIs on the main rates."""
    eval_cfg = eval_cfg or EvalConfig()
    n = len(records)
    if n == 0:
        return {"n": 0}

    preds = [r.pred_smiles for r in records]
    refs = [r.ref_smiles for r in records]

    em = [r.exact_match for r in records]
    ca = [r.char_accuracy for r in records]
    pm = [r.prefix_match for r in records]
    la = [r.levenshtein_accuracy for r in records]
    bl = [r.bleu2 for r in records]
    tn = [r.tanimoto for r in records]

    out: Dict[str, object] = {
        "n": n,
        "exact_match": float(np.mean(em)),
        "char_accuracy": float(np.mean(ca)),
        "prefix_match": float(np.mean(pm)),
        "levenshtein_accuracy": float(np.mean(la)),
        "bleu2": float(np.mean(bl)),
        "tanimoto_mean": float(np.mean(tn)),
        "tanimoto_std": float(np.std(tn)),
    }
    out.update({f"micro_{k}": v for k, v in micro_f1(preds, refs).items()})
    out["micro_f1"] = micro_f1(preds, refs)["micro_f1"]

    # --- Top-K ---
    for k in topk_values:
        out[f"top{k}"] = float(np.mean([r.topk.get(k, 0) for r in records]))

    # --- validity ---
    n_syn = sum(r.syntactically_valid for r in records)
    n_chem = sum(r.chemically_valid for r in records)
    invalid = [r for r in records if not r.chemically_valid]
    severe = [
        r for r in invalid if r.levenshtein_accuracy < eval_cfg.severe_la_threshold
    ]
    out["syntactically_valid"] = n_syn / n
    out["chemically_valid"] = n_chem / n
    out["n_invalid"] = len(invalid)
    out["severe_invalid_fraction"] = (len(severe) / len(invalid)) if invalid else 0.0

    # --- regression errors ---
    len_err = regression_errors([r.length_error for r in records])
    lev_err_ref = regression_errors([r.lev_error_ref for r in records])
    lev_err_sym = regression_errors([r.lev_error_symmetric for r in records])
    out["length_mae"] = len_err["mae"]
    out["length_mse"] = len_err["mse"]
    out["length_rmse"] = len_err["rmse"]
    out["lev_mae_ref"] = lev_err_ref["mae"]
    out["lev_mse_ref"] = lev_err_ref["mse"]
    out["lev_rmse_ref"] = lev_err_ref["rmse"]
    out["lev_mae_symmetric"] = lev_err_sym["mae"]

    # --- calibration ---
    conf = [r.confidence for r in records]
    bins = calibration_bins(conf, em, n_bins=eval_cfg.n_calibration_bins)
    out["calibration_bins"] = bins
    out["ece"] = expected_calibration_error(bins)
    out["mce"] = maximum_calibration_error(bins)
    out["brier"] = brier_score(conf, em)
    out["conf_acc_r"] = confidence_accuracy_correlation(conf, em)

    # --- correlation of invalidity with low EM / LA ---
    invalid_flag = [0 if r.chemically_valid else 1 for r in records]
    out["r_invalid_vs_em"] = confidence_accuracy_correlation(
        invalid_flag, [1 - e for e in em]
    )
    out["r_invalid_vs_la"] = float(
        np.corrcoef(invalid_flag, [1.0 - x for x in la])[0, 1]
    ) if n > 1 and np.std(invalid_flag) > 0 else 0.0

    # --- bootstrap CIs ------------------------------------------------------
    if with_bootstrap:
        out["ci"] = {
            "exact_match": bootstrap_ci(em, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed),
            "char_accuracy": bootstrap_ci(ca, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed),
            "levenshtein_accuracy": bootstrap_ci(la, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed),
            "bleu2": bootstrap_ci(bl, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed),
            "tanimoto": bootstrap_ci(tn, eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed),
        }
        for k in topk_values:
            out["ci"][f"top{k}"] = bootstrap_ci(
                [r.topk.get(k, 0) for r in records],
                eval_cfg.n_bootstrap, eval_cfg.bootstrap_ci, eval_cfg.bootstrap_seed,
            )

    return out


def paired_bootstrap_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap on the per-sample difference between two systems.
    Returns mean difference, 95% CI and a two-sided p-value.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Paired test needs equal lengths, got {a.shape} and {b.shape}")
    d = a - b
    n = len(d)
    if n == 0:
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = d[idx].mean(axis=1)

    point = float(d.mean())
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    if point >= 0:
        p = 2.0 * float(np.mean(means <= 0.0))
    else:
        p = 2.0 * float(np.mean(means >= 0.0))
    return {"mean_diff": point, "lo": lo, "hi": hi, "p_value": min(1.0, p)}


__all__ = [
    "canon", "parses_syntactically", "is_chemically_valid",
    "morgan_fingerprint", "tanimoto_similarity",
    "levenshtein_distance", "longest_common_prefix_len",
    "exact_match", "char_accuracy", "prefix_match", "levenshtein_accuracy",
    "levenshtein_error_ref_normalized", "bleu2",
    "token_counts_for_f1", "micro_f1",
    "topk_candidate_set", "topk_hit",
    "regression_errors",
    "calibration_bins", "expected_calibration_error",
    "maximum_calibration_error", "brier_score",
    "confidence_accuracy_correlation",
    "bootstrap_ci", "paired_bootstrap_test",
    "PerSampleRecord", "score_one", "aggregate",
]
