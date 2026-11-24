#!/usr/bin/env python3
"""
Gradio web interface with twelve tabs:

    Manual input        single reaction prediction with full outputs:
                        molecules, ranked candidates, metrics, diff, SMARTS
    Test set browser    step through built-in test reactions by index
    Batch evaluate      run N samples from the test set, with a summary and a
                        per-reaction-type breakdown
    Upload CSV          drag-and-drop user CSV files for batch prediction
    Model comparison    run all three models at once and compare
    History log         session prediction log with a star filter and CSV export
    SMILES toolkit      standalone draw, validate and property analysis
    Atom mapping        MCS-based atom correspondence visualisation
    3D Viewer           interactive 3Dmol.js conformer viewer
    Reagent browser     searchable library of common reagents and solvents
    Calibration plot    confidence against accuracy
    Error analysis      sorted failure analysis from batch results

The control panel at the top selects the model (Transformer / GRU /
Ensemble), the beam width, the number of Top-K candidates to display, and a
confidence threshold for filtering predictions.

Decoding and scoring use `src/decode.py` and `src/metrics.py`.

Usage
-----
    python app/gradio_app.py
    python app/gradio_app.py --share --port 7860
    python app/gradio_app.py --device cpu
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.config import CFG, CKPT_DIR, DATA_DIR, RESULTS_DIR, BeamConfig
from src.data import Reaction, canonicalize_precursor_set, load_split
from src.decode import BranchAdapter, beam_search
from src.metrics import (
    bleu2, canon, char_accuracy, exact_match, is_chemically_valid,
    levenshtein_accuracy, prefix_match, tanimoto_similarity,
)
from src.models import GRUSeq2Seq, TransformerSeq2Seq
from src.reaction_types import CATEGORY_LABELS, classify_reaction
from src.tokenizer import Vocabulary, tokenize_smiles
from src.train import load_checkpoint, resolve_device

try:
    import gradio as gr
except ImportError:
    raise SystemExit(
        "Gradio is required for the web server:\n"
        "    pip install gradio\n"
        "The rest of the repository does not need it."
    )

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, rdFMCS

RDLogger.DisableLog("rdApp.*")


# ===========================================================================
# Global state, initialised in main()
# ===========================================================================
class AppState:
    def __init__(self):
        self.vocab: Optional[Vocabulary] = None
        self.gru = None
        self.tf = None
        self.device = torch.device("cpu")
        self.test_reactions: List[Reaction] = []
        self.history: List[dict] = []
        self.last_batch: List[dict] = []


STATE = AppState()


# ===========================================================================
# Prediction core
# ===========================================================================
def predict(
    reactant_smiles: str,
    model_choice: str,
    beam_width: int,
    top_k: int,
) -> Tuple[List[Tuple[str, float]], float]:
    """
    Run one prediction and return (candidates with confidences, elapsed ms).

    Candidates are ranked by length-normalized beam score.
    """
    if not reactant_smiles or not reactant_smiles.strip():
        return [], 0.0

    mode = {"Transformer": "transformer", "GRU": "gru", "Ensemble": "ensemble"}[model_choice]
    if mode in ("gru", "ensemble") and STATE.gru is None:
        raise gr.Error("GRU checkpoint not loaded. See --ckpt-dir.")
    if mode in ("transformer", "ensemble") and STATE.tf is None:
        raise gr.Error("Transformer checkpoint not loaded. See --ckpt-dir.")

    if mode == "gru":
        branches = [BranchAdapter(STATE.gru, "gru", 1.0)]
    elif mode == "transformer":
        branches = [BranchAdapter(STATE.tf, "transformer", 1.0)]
    else:
        branches = [
            BranchAdapter(STATE.gru, "gru", CFG.ensemble.w_gru),
            BranchAdapter(STATE.tf, "transformer", CFG.ensemble.w_tf),
        ]

    beam_cfg = BeamConfig(
        beam_width=int(beam_width),
        length_penalty_beta=CFG.beam.length_penalty_beta,
        max_decode_len=CFG.beam.max_decode_len,
        early_stopping=True,
        topk_values=CFG.beam.topk_values,
        use_kv_cache=True,
    )

    src = torch.tensor(
        [STATE.vocab.encode(reactant_smiles.strip())], dtype=torch.long
    ).to(STATE.device)

    t0 = time.time()
    results = beam_search(branches, src, STATE.vocab, beam_cfg=beam_cfg)
    elapsed = (time.time() - t0) * 1000

    hyps = results[0].hypotheses[: int(top_k)]
    return [(h.smiles, h.confidence) for h in hyps], elapsed


# ===========================================================================
# Drawing helpers
# ===========================================================================
def draw_molecule(smiles: str, size: Tuple[int, int] = (380, 300)):
    """Render a molecule to a PIL image, or None if it cannot be parsed."""
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


def string_diff(pred: str, ref: str) -> str:
    """
    Character-level diff as HTML, so a near-miss is visible at a glance
    rather than requiring the user to compare two long SMILES by eye.
    """
    import difflib

    sm = difflib.SequenceMatcher(None, ref, pred)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(f"<span>{ref[i1:i2]}</span>")
        elif tag == "replace":
            out.append(
                f"<span style='background:#ffe0e0;text-decoration:line-through'>{ref[i1:i2]}</span>"
                f"<span style='background:#e0ffe0'>{pred[j1:j2]}</span>"
            )
        elif tag == "delete":
            out.append(f"<span style='background:#ffe0e0;text-decoration:line-through'>{ref[i1:i2]}</span>")
        elif tag == "insert":
            out.append(f"<span style='background:#e0ffe0'>{pred[j1:j2]}</span>")
    return (
        "<div style='font-family:monospace;font-size:13px;line-height:1.7;"
        "word-break:break-all'>" + "".join(out) + "</div>"
        "<p style='font-size:11px;color:#666'>Struck red: in the reference "
        "but not predicted. Green: predicted but not in the reference.</p>"
    )


def reaction_smarts(reactants: str, product: str) -> str:
    """The reaction in SMARTS-style 'reactants>>product' form."""
    if not reactants or not product:
        return ""
    return f"{reactants.strip()}>>{product.strip()}"


# ===========================================================================
# TAB 1 -- Manual input
# ===========================================================================
def tab_manual_fn(
    reactant: str, actual: str, model_choice: str,
    beam_width: int, top_k: int, conf_threshold: float,
):
    if not reactant or not reactant.strip():
        return None, None, "", "", "", "", ""

    cands, elapsed = predict(reactant, model_choice, beam_width, top_k)
    if not cands:
        return None, None, "No candidate produced.", "", "", "", ""

    kept = [(s, c) for s, c in cands if c >= conf_threshold]
    shown = kept if kept else cands
    top_smiles, top_conf = shown[0]

    img_r = draw_molecule(reactant)
    img_p = draw_molecule(top_smiles)

    rows = []
    for i, (s, c) in enumerate(shown, 1):
        valid = "valid" if is_chemically_valid(s) else "INVALID"
        mark = ""
        if actual and actual.strip():
            mark = " (exact match)" if exact_match(s, actual) else ""
        rows.append(f"{i}. [{c:.4f}] [{valid}] {s}{mark}")
    cand_text = "\n".join(rows)
    if kept and len(kept) < len(cands):
        cand_text += (
            f"\n\n{len(cands) - len(kept)} candidate(s) hidden below the "
            f"confidence threshold {conf_threshold:.2f}."
        )

    if actual and actual.strip():
        a = actual.strip()
        metrics = (
            f"Exact match (canonical)   : {exact_match(top_smiles, a)}\n"
            f"Character accuracy        : {char_accuracy(top_smiles, a):.4f}\n"
            f"Prefix match              : {prefix_match(top_smiles, a):.4f}\n"
            f"Levenshtein accuracy      : {levenshtein_accuracy(top_smiles, a):.4f}\n"
            f"BLEU (bigram)             : {bleu2(top_smiles, a):.4f}\n"
            f"Tanimoto (Morgan r=2)     : {tanimoto_similarity(top_smiles, a):.4f}\n"
            f"Chemically valid          : {is_chemically_valid(top_smiles)}\n"
            f"Confidence                : {top_conf:.4f}\n"
            f"Decode time               : {elapsed:.0f} ms"
        )
        diff_html = string_diff(top_smiles, a)
        rtype = classify_reaction(reactant, a)
        rtype_text = CATEGORY_LABELS.get(rtype, rtype)
    else:
        metrics = (
            f"No reference product supplied, so only model-side quantities "
            f"are available.\n\n"
            f"Chemically valid          : {is_chemically_valid(top_smiles)}\n"
            f"Confidence                : {top_conf:.4f}\n"
            f"Decode time               : {elapsed:.0f} ms"
        )
        diff_html = ""
        rtype_text = "unknown (needs the reference product)"

    STATE.history.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_choice,
        "reactant": reactant.strip(),
        "predicted": top_smiles,
        "actual": actual.strip() if actual else "",
        "confidence": round(top_conf, 4),
        "exact_match": exact_match(top_smiles, actual) if actual and actual.strip() else "",
        "starred": False,
    })

    return (
        img_r, img_p, top_smiles, cand_text, metrics,
        reaction_smarts(reactant, top_smiles), diff_html,
    )


# ===========================================================================
# TAB 2 -- Test set browser
# ===========================================================================
def tab_browser_fn(index: int, model_choice: str, beam_width: int, top_k: int, conf: float):
    if not STATE.test_reactions:
        return None, None, "", "", "No test split loaded (data/test.json).", "", ""
    i = int(index) % len(STATE.test_reactions)
    r = STATE.test_reactions[i]
    return tab_manual_fn(r.input_smiles, r.output_smiles, model_choice, beam_width, top_k, conf)


# ===========================================================================
# TAB 3 -- Batch evaluate
# ===========================================================================
def tab_batch_fn(n_samples: int, model_choice: str, beam_width: int, progress=gr.Progress()):
    if not STATE.test_reactions:
        return "No test split loaded (data/test.json).", None
    n = int(min(n_samples, len(STATE.test_reactions)))
    rows: List[dict] = []

    for i in progress.tqdm(range(n), desc="Evaluating"):
        r = STATE.test_reactions[i]
        cands, _ = predict(r.input_smiles, model_choice, beam_width, beam_width)
        pred = cands[0][0] if cands else ""
        conf = cands[0][1] if cands else 0.0
        rows.append({
            "id": r.rxn_id,
            "input": r.input_smiles,
            "ref": r.output_smiles,
            "pred": pred,
            "confidence": conf,
            "exact_match": exact_match(pred, r.output_smiles),
            "char_accuracy": char_accuracy(pred, r.output_smiles),
            "levenshtein_accuracy": levenshtein_accuracy(pred, r.output_smiles),
            "bleu2": bleu2(pred, r.output_smiles),
            "tanimoto": tanimoto_similarity(pred, r.output_smiles),
            "valid": is_chemically_valid(pred),
            "reaction_type": classify_reaction(r.input_smiles, r.output_smiles),
        })

    STATE.last_batch = rows

    summary = [
        f"Model                : {model_choice}",
        f"Beam width           : {beam_width}",
        f"Reactions evaluated  : {n:,}",
        "",
        f"Exact match          : {np.mean([x['exact_match'] for x in rows]):.4f}",
        f"Character accuracy   : {np.mean([x['char_accuracy'] for x in rows]):.4f}",
        f"Levenshtein accuracy : {np.mean([x['levenshtein_accuracy'] for x in rows]):.4f}",
        f"BLEU (bigram)        : {np.mean([x['bleu2'] for x in rows]):.4f}",
        f"Tanimoto             : {np.mean([x['tanimoto'] for x in rows]):.4f}",
        f"Chemically valid     : {np.mean([x['valid'] for x in rows]):.4f}",
        "",
        "Per reaction type (exact match):",
    ]
    by_type: Dict[str, List[int]] = {}
    for x in rows:
        by_type.setdefault(x["reaction_type"], []).append(x["exact_match"])
    for t, ems in sorted(by_type.items()):
        label = CATEGORY_LABELS.get(t, t)
        summary.append(f"  {label:<34} {np.mean(ems):.4f}  (n = {len(ems):,})")

    summary.append("")
    summary.append(f"Results on the first {n:,} test reactions.")

    table = [
        [x["id"], x["input"][:46], x["ref"][:34], x["pred"][:34],
         round(x["confidence"], 3), x["exact_match"], round(x["levenshtein_accuracy"], 3)]
        for x in rows
    ]
    return "\n".join(summary), table


# ===========================================================================
# TAB 4 -- Upload CSV
# ===========================================================================
def tab_upload_fn(file_obj, model_choice: str, beam_width: int, progress=gr.Progress()):
    """
    Accepts a CSV with a reactant column (named input/reactant/reactants/
    smiles) and optionally a reference column (output/product/actual).
    """
    if file_obj is None:
        return "Upload a CSV first.", None

    path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows_in = list(reader)
            fields = [c.strip().lower() for c in (reader.fieldnames or [])]
    except Exception as exc:
        return f"Could not read the CSV: {exc}", None

    in_col = next((c for c in ("input", "reactant", "reactants", "smiles") if c in fields), None)
    ref_col = next((c for c in ("output", "product", "actual", "ref") if c in fields), None)
    if in_col is None:
        return (
            f"No reactant column found. Columns present: {fields}. "
            f"Expected one of: input, reactant, reactants, smiles.", None
        )

    def get(row, col):
        for k, v in row.items():
            if k and k.strip().lower() == col:
                return (v or "").strip()
        return ""

    out_rows = []
    for row in progress.tqdm(rows_in, desc="Predicting"):
        src = get(row, in_col)
        if not src:
            continue
        ref = get(row, ref_col) if ref_col else ""
        cands, _ = predict(src, model_choice, beam_width, beam_width)
        pred = cands[0][0] if cands else ""
        conf = cands[0][1] if cands else 0.0
        out_rows.append([
            src[:50], ref[:34], pred[:34], round(conf, 3),
            exact_match(pred, ref) if ref else "",
            round(levenshtein_accuracy(pred, ref), 3) if ref else "",
            is_chemically_valid(pred),
        ])

    msg = f"Predicted {len(out_rows):,} reactions with {model_choice}."
    if ref_col:
        ems = [r[4] for r in out_rows if r[4] != ""]
        if ems:
            msg += f" Exact match on the {len(ems):,} rows with a reference: {np.mean(ems):.4f}."
    else:
        msg += " No reference column found, so accuracy columns are blank."
    return msg, out_rows


# ===========================================================================
# TAB 5 -- Model comparison
# ===========================================================================
def tab_compare_fn(reactant: str, actual: str, beam_width: int, top_k: int):
    if not reactant or not reactant.strip():
        return None, "Enter a reactant SMILES."

    rows = []
    for name in ("Transformer", "GRU", "Ensemble"):
        try:
            cands, ms = predict(reactant, name, beam_width, top_k)
        except gr.Error as exc:
            rows.append([name, "unavailable", "", "", "", "", str(exc)])
            continue
        if not cands:
            rows.append([name, "no candidate", "", "", "", "", ""])
            continue
        pred, conf = cands[0]
        a = actual.strip() if actual else ""
        rows.append([
            name, pred[:40], round(conf, 4),
            exact_match(pred, a) if a else "",
            round(levenshtein_accuracy(pred, a), 4) if a else "",
            round(tanimoto_similarity(pred, a), 4) if a else "",
            f"{ms:.0f} ms",
        ])

    note = (
        f"Ensemble weights: w_GRU = {CFG.ensemble.w_gru}, "
        f"w_TF = {CFG.ensemble.w_tf}, applied at every decoding step, "
        f"beam width {beam_width}."
    )
    return rows, note


# ===========================================================================
# TAB 6 -- History log
# ===========================================================================
def tab_history_fn(starred_only: bool):
    rows = STATE.history
    if starred_only:
        rows = [r for r in rows if r.get("starred")]
    if not rows:
        return None, "No predictions logged yet in this session."
    table = [
        [r["timestamp"], r["model"], r["reactant"][:40], r["predicted"][:34],
         r["actual"][:34], r["confidence"], r["exact_match"], r["starred"]]
        for r in rows
    ]
    return table, f"{len(rows):,} entries."


def tab_history_star_fn(index: int):
    if not STATE.history:
        return None, "Nothing logged yet."
    i = int(index)
    if not (0 <= i < len(STATE.history)):
        return None, f"Index out of range (0 to {len(STATE.history)-1})."
    STATE.history[i]["starred"] = not STATE.history[i]["starred"]
    return tab_history_fn(False)


def tab_history_export_fn():
    if not STATE.history:
        return None
    out = RESULTS_DIR / "session_history.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(STATE.history[0].keys()))
        w.writeheader()
        w.writerows(STATE.history)
    return str(out)


def tab_history_clear_fn():
    STATE.history.clear()
    return None, "History cleared."


# ===========================================================================
# TAB 7 -- SMILES toolkit
# ===========================================================================
FUNCTIONAL_GROUPS = {
    "Carboxylic acid": "[CX3](=O)[OX2H1]",
    "Ester": "[CX3](=O)[OX2H0][#6]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Primary amine": "[NX3;H2;!$(NC=O)]",
    "Secondary amine": "[NX3;H1;!$(NC=O)]",
    "Tertiary amine": "[NX3;H0;!$(NC=O);!$(N=*)]",
    "Alcohol": "[OX2H][CX4]",
    "Phenol": "[OX2H][c]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Nitrile": "[NX1]#[CX2]",
    "Nitro": "[$([NX3](=O)=O)]",
    "Sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "Halide": "[F,Cl,Br,I]",
    "Aromatic ring": "c1ccccc1",
    "Ether": "[OD2]([#6])[#6]",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#[CX2]",
    "Boronic acid/ester": "[#5]",
    "Azide": "[NX1]~[NX2]~[NX2]",
}


def tab_toolkit_fn(smiles: str, compare_smiles: str):
    if not smiles or not smiles.strip():
        return None, "Enter a SMILES string.", ""

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None, f"RDKit could not parse: {smiles}", ""

    img = draw_molecule(smiles, size=(420, 340))

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rings = mol.GetRingInfo().NumRings()
    rot = Lipinski.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

    # Lipinski's rule of five.
    violations = [
        name for name, bad in (
            ("MW > 500", mw > 500),
            ("LogP > 5", logp > 5),
            ("HBD > 5", hbd > 5),
            ("HBA > 10", hba > 10),
        ) if bad
    ]

    props = [
        f"Canonical SMILES    : {Chem.MolToSmiles(mol)}",
        f"Molecular formula   : {formula}",
        f"Molecular weight    : {mw:.2f}",
        f"LogP (Crippen)      : {logp:.2f}",
        f"TPSA                : {tpsa:.2f}",
        f"H-bond donors       : {hbd}",
        f"H-bond acceptors    : {hba}",
        f"Rings               : {rings}",
        f"Rotatable bonds     : {rot}",
        f"Heavy atoms         : {heavy}",
        "",
        f"Lipinski rule of five: "
        + ("passes all four criteria" if not violations
           else f"{len(violations)} violation(s) -- {', '.join(violations)}"),
    ]

    found = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            n = len(mol.GetSubstructMatches(patt))
            found.append(f"{name} (x{n})" if n > 1 else name)
    props += ["", "Detected functional groups:",
              "  " + (", ".join(found) if found else "none of the catalogued patterns")]

    if compare_smiles and compare_smiles.strip():
        sim = tanimoto_similarity(smiles, compare_smiles)
        props += [
            "",
            f"Tanimoto similarity to the comparison molecule "
            f"(Morgan r=2, 2048 bits): {sim:.4f}",
        ]

    return img, "\n".join(props), Chem.MolToSmiles(mol)


# ===========================================================================
# TAB 8 -- Atom mapping
# ===========================================================================
def tab_atommap_fn(reactant: str, product: str):
    """
    Highlight the maximum common substructure between a reactant and a
    product, so the user can trace which atoms are conserved.

    MCS-based correspondence, not a mechanistic atom mapping.
    """
    if not reactant or not product:
        return None, None, "Enter both a reactant and a product SMILES."

    m1 = Chem.MolFromSmiles(reactant.strip())
    m2 = Chem.MolFromSmiles(product.strip())
    if m1 is None or m2 is None:
        return None, None, "One of the two SMILES could not be parsed."

    try:
        res = rdFMCS.FindMCS(
            [m1, m2], timeout=10,
            ringMatchesRingOnly=True, completeRingsOnly=False,
        )
    except Exception as exc:
        return None, None, f"MCS search failed: {exc}"

    if not res or not res.smartsString:
        return draw_molecule(reactant), draw_molecule(product), "No common substructure found."

    patt = Chem.MolFromSmarts(res.smartsString)
    hit1 = list(m1.GetSubstructMatch(patt)) if patt else []
    hit2 = list(m2.GetSubstructMatch(patt)) if patt else []

    AllChem.Compute2DCoords(m1)
    AllChem.Compute2DCoords(m2)
    img1 = Draw.MolToImage(m1, size=(420, 340), highlightAtoms=hit1)
    img2 = Draw.MolToImage(m2, size=(420, 340), highlightAtoms=hit2)

    info = (
        f"MCS SMARTS           : {res.smartsString}\n"
        f"Shared atoms         : {res.numAtoms}\n"
        f"Shared bonds         : {res.numBonds}\n"
        f"Reactant heavy atoms : {m1.GetNumHeavyAtoms()}\n"
        f"Product heavy atoms  : {m2.GetNumHeavyAtoms()}\n"
        f"Conserved fraction   : {res.numAtoms / max(m2.GetNumHeavyAtoms(), 1):.3f} "
        f"of the product\n\n"
        f"Highlighted atoms are the maximum common substructure. This is a "
        f"structural correspondence, not a mechanistic atom map."
    )
    return img1, img2, info


# ===========================================================================
# TAB 9 -- 3D viewer
# ===========================================================================
def tab_3d_fn(smiles: str, style: str):
    """
    Generate a 3D conformer with ETKDGv3 embedding and UFF minimisation, and
    render it with 3Dmol.js.
    """
    if not smiles or not smiles.strip():
        return "<p>Enter a SMILES string.</p>"

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return f"<p>Could not parse: {smiles}</p>"

    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return "<p>Conformer embedding failed for this molecule.</p>"
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        molblock = Chem.MolToMolBlock(mol)
    except Exception as exc:
        return f"<p>3D generation failed: {exc}</p>"

    style_spec = {
        "stick": "{stick:{radius:0.15}}",
        "sphere": "{sphere:{scale:0.3}}",
        "ball and stick": "{stick:{radius:0.12},sphere:{scale:0.25}}",
        "wireframe": "{line:{linewidth:2.0}}",
    }.get(style, "{stick:{radius:0.15}}")

    payload = base64.b64encode(molblock.encode()).decode()
    return f"""
<div style="width:100%;height:460px;position:relative" id="viewer3d"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
(function() {{
  var el = document.getElementById('viewer3d');
  if (!el || typeof $3Dmol === 'undefined') return;
  var viewer = $3Dmol.createViewer(el, {{backgroundColor: 'white'}});
  var data = atob("{payload}");
  viewer.addModel(data, "mol");
  viewer.setStyle({{}}, {style_spec});
  viewer.zoomTo();
  viewer.render();
}})();
</script>
<p style="font-size:11px;color:#666">
ETKDGv3 embedding (seed 42) with UFF minimisation, 500 iterations.
Drag to rotate, scroll to zoom.</p>
"""


# ===========================================================================
# TAB 10 -- Reagent browser
# ===========================================================================
REAGENTS: List[Tuple[str, str, str]] = [
    ("Water", "O", "solvent"),
    ("Methanol", "CO", "solvent"),
    ("Ethanol", "CCO", "solvent"),
    ("Isopropanol", "CC(C)O", "solvent"),
    ("tert-Butanol", "CC(C)(C)O", "solvent"),
    ("Dichloromethane (DCM)", "ClCCl", "solvent"),
    ("Chloroform", "ClC(Cl)Cl", "solvent"),
    ("Tetrahydrofuran (THF)", "C1CCOC1", "solvent"),
    ("1,4-Dioxane", "C1COCCO1", "solvent"),
    ("Diethyl ether", "CCOCC", "solvent"),
    ("Ethyl acetate", "CCOC(C)=O", "solvent"),
    ("Acetone", "CC(C)=O", "solvent"),
    ("Acetonitrile", "CC#N", "solvent"),
    ("Dimethylformamide (DMF)", "CN(C)C=O", "solvent"),
    ("Dimethyl sulfoxide (DMSO)", "CS(C)=O", "solvent"),
    ("Toluene", "Cc1ccccc1", "solvent"),
    ("Benzene", "c1ccccc1", "solvent"),
    ("Hexane", "CCCCCC", "solvent"),
    ("Pyridine", "c1ccncc1", "solvent / base"),
    ("Triethylamine", "CCN(CC)CC", "base"),
    ("Diisopropylethylamine (DIPEA)", "CCN(C(C)C)C(C)C", "base"),
    ("DBU", "C1CCC2=NCCCN2CC1", "base"),
    ("Sodium hydroxide", "[Na+].[OH-]", "base"),
    ("Potassium hydroxide", "[K+].[OH-]", "base"),
    ("Potassium carbonate", "[K+].[K+].[O-]C([O-])=O", "base"),
    ("Cesium carbonate", "[Cs+].[Cs+].[O-]C([O-])=O", "base"),
    ("Sodium hydride", "[Na+].[H-]", "base"),
    ("Sodium tert-butoxide", "CC(C)(C)[O-].[Na+]", "base"),
    ("LDA", "CC(C)[N-]C(C)C.[Li+]", "base"),
    ("n-Butyllithium", "CCCC[Li]", "base / nucleophile"),
    ("Hydrochloric acid", "Cl", "acid"),
    ("Sulfuric acid", "OS(=O)(=O)O", "acid"),
    ("Trifluoroacetic acid (TFA)", "OC(=O)C(F)(F)F", "acid"),
    ("p-Toluenesulfonic acid", "Cc1ccc(S(=O)(=O)O)cc1", "acid"),
    ("Acetic acid", "CC(=O)O", "acid"),
    ("Sodium borohydride", "[Na+].[BH4-]", "reducing agent"),
    ("Lithium aluminium hydride", "[Li+].[AlH4-]", "reducing agent"),
    ("DIBAL-H", "CC(C)C[AlH]CC(C)C", "reducing agent"),
    ("Sodium triacetoxyborohydride", "CC(=O)O[BH-](OC(C)=O)OC(C)=O.[Na+]", "reducing agent"),
    ("Hydrogen", "[H][H]", "reducing agent"),
    ("Palladium on carbon", "[Pd]", "catalyst"),
    ("Pd(PPh3)4", "[Pd]", "catalyst"),
    ("Pd(dppf)Cl2", "[Pd]", "catalyst"),
    ("Copper(I) iodide", "[Cu]I", "catalyst"),
    ("Triphenylphosphine", "c1ccc(P(c2ccccc2)c2ccccc2)cc1", "ligand / reagent"),
    ("XPhos", "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1", "ligand"),
    ("mCPBA", "OC(=O)c1cccc(Cl)c1", "oxidant"),
    ("Hydrogen peroxide", "OO", "oxidant"),
    ("Dess-Martin periodinane", "CC(=O)OI1(OC(C)=O)(OC(C)=O)OC(=O)c2ccccc21", "oxidant"),
    ("Manganese dioxide", "O=[Mn]=O", "oxidant"),
    ("PCC", "[O-][Cr](=O)(=O)Cl", "oxidant"),
    ("Thionyl chloride", "O=S(Cl)Cl", "activating agent"),
    ("Oxalyl chloride", "ClC(=O)C(Cl)=O", "activating agent"),
    ("EDC", "CCN=C=NCCCN(C)C", "coupling agent"),
    ("DCC", "C1CCC(CC1)N=C=NC1CCCCC1", "coupling agent"),
    ("HATU", "CN(C)C(On1nnc2ccccc21)=[N+](C)C", "coupling agent"),
    ("HOBt", "On1nnc2ccccc21", "coupling additive"),
    ("Di-tert-butyl dicarbonate (Boc2O)", "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "protecting group"),
    ("Benzyl chloroformate (Cbz-Cl)", "ClC(=O)OCc1ccccc1", "protecting group"),
    ("Acetic anhydride", "CC(=O)OC(C)=O", "acylating agent"),
    ("Methyl iodide", "CI", "alkylating agent"),
    ("Benzyl bromide", "BrCc1ccccc1", "alkylating agent"),
    ("NBS", "BrN1C(=O)CCC1=O", "halogenating agent"),
    ("NCS", "ClN1C(=O)CCC1=O", "halogenating agent"),
    ("Sodium azide", "[Na+].[N-]=[N+]=[N-]", "nucleophile"),
    ("Sodium cyanide", "[Na+].[C-]#N", "nucleophile"),
    ("Magnesium sulfate", "[Mg+2].[O-]S([O-])(=O)=O", "drying agent"),
    ("Sodium chloride", "[Na+].[Cl-]", "salt"),
]


def tab_reagents_fn(query: str):
    q = (query or "").strip().lower()
    if not q:
        rows = REAGENTS
    else:
        rows = [
            (n, s, c) for n, s, c in REAGENTS
            if q in n.lower() or q in s.lower() or q in c.lower()
        ]
    if not rows:
        return None, f"No reagent matches {query!r}. Search by name, category, or partial SMILES."

    table = []
    for name, smi, cat in rows:
        mol = Chem.MolFromSmiles(smi)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else "unparsable"
        mw = f"{Descriptors.MolWt(mol):.2f}" if mol else ""
        table.append([name, cat, smi, formula, mw])
    return table, f"{len(rows)} of {len(REAGENTS)} reagents shown."


def tab_reagent_draw_fn(smiles: str):
    return draw_molecule(smiles, size=(380, 300))


# ===========================================================================
# TAB 11 -- Calibration plot
# ===========================================================================
def tab_calibration_fn():
    """
    Reliability curve from the current session's batch results, or from the
    saved full-test metrics if no batch has been run.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.metrics import calibration_bins, expected_calibration_error, maximum_calibration_error

    if STATE.last_batch:
        conf = [x["confidence"] for x in STATE.last_batch]
        corr = [x["exact_match"] for x in STATE.last_batch]
        source = f"the current session's batch of {len(conf):,} reactions"
    else:
        p = RESULTS_DIR / "all_test_metrics.json"
        if not p.exists():
            return None, (
                "No data yet. Run the Batch evaluate tab, or run "
                "scripts/05_evaluate_test.py to produce "
                "results/all_test_metrics.json."
            )
        metrics = json.loads(p.read_text())
        key = "ensemble" if "ensemble" in metrics else next(iter(metrics))
        bins = metrics[key].get("calibration_bins", [])
        if not bins:
            return None, "Saved metrics contain no calibration bins."
        fig, ax = plt.subplots(figsize=(6, 5))
        xs = [b["confidence"] for b in bins if b["count"] > 0]
        ys = [b["accuracy"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(xs, ys, "o-", lw=1.5, label=key)
        ax.set_xlabel("Mean confidence"); ax.set_ylabel("Exact-match accuracy")
        ax.set_title(f"Reliability -- {key}, full test set")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        info = (
            f"From results/all_test_metrics.json ({key}, full test set).\n"
            f"ECE = {metrics[key]['ece']:.4f}   MCE = {metrics[key]['mce']:.4f}   "
            f"Brier = {metrics[key]['brier']:.4f}"
        )
        return fig, info

    bins = calibration_bins(conf, corr, n_bins=CFG.eval.n_calibration_bins)
    ece = expected_calibration_error(bins)
    mce = maximum_calibration_error(bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [b["confidence"] for b in bins if b["count"] > 0]
    ys = [b["accuracy"] for b in bins if b["count"] > 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(xs, ys, "o-", lw=1.5, label="Observed")
    ax.scatter(conf, corr, s=8, alpha=0.2, label="Individual predictions")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Exact match")
    ax.set_title("Confidence against accuracy")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)

    info = (
        f"From {source}.\n"
        f"ECE = {ece:.4f}   MCE = {mce:.4f}"
    )
    return fig, info


# ===========================================================================
# TAB 12 -- Error analysis
# ===========================================================================
def tab_errors_fn(sort_by: str, max_rows: int):
    if not STATE.last_batch:
        return None, "Run the Batch evaluate tab first; this reads its results."

    failures = [x for x in STATE.last_batch if not x["exact_match"]]
    if not failures:
        return None, (
            f"No failures among the {len(STATE.last_batch):,} evaluated "
            f"reactions. Nothing to analyse."
        )

    keymap = {
        "severity (lowest Levenshtein first)": lambda x: x["levenshtein_accuracy"],
        "severity (lowest Tanimoto first)": lambda x: x["tanimoto"],
        "confidence (most overconfident first)": lambda x: -x["confidence"],
        "length error (largest first)": lambda x: -abs(len(x["pred"]) - len(x["ref"])),
    }
    failures.sort(key=keymap.get(sort_by, keymap["severity (lowest Levenshtein first)"]))

    table = [
        [x["id"], x["input"][:40], x["ref"][:30], x["pred"][:30],
         round(x["levenshtein_accuracy"], 3), round(x["tanimoto"], 3),
         round(x["confidence"], 3), x["valid"],
         CATEGORY_LABELS.get(x["reaction_type"], x["reaction_type"])[:18]]
        for x in failures[: int(max_rows)]
    ]

    n_invalid = sum(1 for x in failures if not x["valid"])
    n_severe = sum(1 for x in failures if x["levenshtein_accuracy"] < CFG.eval.severe_la_threshold)
    by_type: Dict[str, int] = {}
    for x in failures:
        by_type[x["reaction_type"]] = by_type.get(x["reaction_type"], 0) + 1

    summary = [
        f"Failures            : {len(failures):,} of {len(STATE.last_batch):,} "
        f"({100*len(failures)/len(STATE.last_batch):.1f}%)",
        f"Chemically invalid  : {n_invalid:,} ({100*n_invalid/len(failures):.1f}% of failures)",
        f"Severe (LA < {CFG.eval.severe_la_threshold})  : {n_severe:,} "
        f"({100*n_severe/len(failures):.1f}% of failures)",
        f"Near-miss failures  : {len(failures)-n_severe:,} "
        f"({100*(len(failures)-n_severe)/len(failures):.1f}% of failures)",
        "",
        "Failures by reaction type:",
    ]
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        summary.append(f"  {CATEGORY_LABELS.get(t, t):<34} {n:,}")
    summary += [
        "",
        f"Mean confidence on failures: "
        f"{np.mean([x['confidence'] for x in failures]):.4f}",
        f"Mean confidence on successes: "
        f"{np.mean([x['confidence'] for x in STATE.last_batch if x['exact_match']]):.4f}"
        if any(x["exact_match"] for x in STATE.last_batch) else "",
    ]
    return table, "\n".join(s for s in summary if s != "")


# ===========================================================================
# UI
# ===========================================================================
def build_ui() -> "gr.Blocks":
    with gr.Blocks(title="Hybrid GRU-Transformer Reaction Prediction") as demo:
        gr.Markdown(
            "# Hybrid GRU-Transformer Ensemble for Forward Reaction Prediction\n"
            "Weighted log-probability fusion within beam search. "
            f"Ensemble weights w_GRU = {CFG.ensemble.w_gru}, w_TF = {CFG.ensemble.w_tf}, "
            "applied at every decoding step."
        )

        # ---- control panel -------------------------------------------------
        with gr.Row():
            model_choice = gr.Dropdown(
                ["Ensemble", "GRU", "Transformer"], value="Ensemble", label="Model"
            )
            beam_width = gr.Slider(
                1, 10, value=CFG.beam.beam_width, step=1, label="Beam width",
                info="Higher is more accurate and slower.",
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K candidates")
            conf_threshold = gr.Slider(
                0.0, 1.0, value=0.0, step=0.01, label="Confidence threshold",
                info="Hide candidates below this confidence.",
            )

        with gr.Tabs():
            # ---- 1. Manual input -------------------------------------------
            with gr.Tab("Manual input"):
                with gr.Row():
                    with gr.Column():
                        m_react = gr.Textbox(
                            label="Reactant / reagent SMILES",
                            placeholder="Brc1ccccc1.NCc1ccccc1",
                            lines=2,
                        )
                        m_actual = gr.Textbox(
                            label="Actual product SMILES (optional)", lines=2
                        )
                        m_btn = gr.Button("Predict", variant="primary")
                        gr.Examples(
                            [
                                ["Brc1ccccc1.NCc1ccccc1", ""],
                                ["CC(=O)Oc1ccccc1C(=O)O.CO", ""],
                                ["O=C(O)c1ccccc1.CCN(CC)CC.ClC(=O)OCC", ""],
                            ],
                            inputs=[m_react, m_actual],
                        )
                    with gr.Column():
                        m_img_r = gr.Image(label="Reactant", height=280)
                        m_img_p = gr.Image(label="Predicted product", height=280)
                m_top = gr.Textbox(label="Top-1 predicted product", lines=2)
                with gr.Row():
                    m_cands = gr.Textbox(label="Ranked candidates", lines=8)
                    m_metrics = gr.Textbox(label="Metrics", lines=8)
                m_smarts = gr.Textbox(label="Reaction (SMARTS form)", lines=2)
                m_diff = gr.HTML(label="Character diff against the reference")

                m_btn.click(
                    tab_manual_fn,
                    [m_react, m_actual, model_choice, beam_width, top_k, conf_threshold],
                    [m_img_r, m_img_p, m_top, m_cands, m_metrics, m_smarts, m_diff],
                )

            # ---- 2. Test set browser ---------------------------------------
            with gr.Tab("Test set browser"):
                gr.Markdown(
                    "Step through the held-out test split by index. "
                    "The reference product is shown, so all metrics are available."
                )
                b_idx = gr.Number(value=0, label="Test reaction index", precision=0)
                b_btn = gr.Button("Load and predict", variant="primary")
                with gr.Row():
                    b_img_r = gr.Image(label="Reactant", height=280)
                    b_img_p = gr.Image(label="Predicted product", height=280)
                b_top = gr.Textbox(label="Top-1 predicted product", lines=2)
                with gr.Row():
                    b_cands = gr.Textbox(label="Ranked candidates", lines=8)
                    b_metrics = gr.Textbox(label="Metrics", lines=8)
                b_smarts = gr.Textbox(label="Reaction (SMARTS form)", lines=2)
                b_diff = gr.HTML()

                b_btn.click(
                    tab_browser_fn,
                    [b_idx, model_choice, beam_width, top_k, conf_threshold],
                    [b_img_r, b_img_p, b_top, b_cands, b_metrics, b_smarts, b_diff],
                )

            # ---- 3. Batch evaluate -----------------------------------------
            with gr.Tab("Batch evaluate"):
                gr.Markdown(
                    "Evaluate the first N reactions of the test split. "
                    "Results feed the Calibration plot and Error analysis tabs."
                )
                ba_n = gr.Slider(10, 2000, value=100, step=10, label="Reactions to evaluate")
                ba_btn = gr.Button("Run batch", variant="primary")
                ba_summary = gr.Textbox(label="Summary", lines=18)
                ba_table = gr.Dataframe(
                    headers=["id", "input", "reference", "predicted", "conf", "EM", "LA"],
                    label="Per-reaction results", wrap=True,
                )
                ba_btn.click(tab_batch_fn, [ba_n, model_choice, beam_width],
                             [ba_summary, ba_table])

            # ---- 4. Upload CSV ---------------------------------------------
            with gr.Tab("Upload CSV"):
                gr.Markdown(
                    "Upload a CSV with a reactant column named one of "
                    "`input`, `reactant`, `reactants` or `smiles`, and "
                    "optionally a reference column named `output`, `product`, "
                    "`actual` or `ref`."
                )
                up_file = gr.File(label="CSV file", file_types=[".csv"])
                up_btn = gr.Button("Predict all rows", variant="primary")
                up_msg = gr.Textbox(label="Status", lines=3)
                up_table = gr.Dataframe(
                    headers=["input", "reference", "predicted", "conf", "EM", "LA", "valid"],
                    label="Predictions", wrap=True,
                )
                up_btn.click(tab_upload_fn, [up_file, model_choice, beam_width],
                             [up_msg, up_table])

            # ---- 5. Model comparison ---------------------------------------
            with gr.Tab("Model comparison"):
                gr.Markdown("Run all three systems on one reaction and compare.")
                c_react = gr.Textbox(label="Reactant SMILES", lines=2)
                c_actual = gr.Textbox(label="Actual product SMILES (optional)", lines=2)
                c_btn = gr.Button("Compare", variant="primary")
                c_table = gr.Dataframe(
                    headers=["Model", "Prediction", "Confidence", "EM", "LA", "Tanimoto", "Time"],
                    label="Comparison", wrap=True,
                )
                c_note = gr.Textbox(label="Configuration", lines=2)
                c_btn.click(tab_compare_fn, [c_react, c_actual, beam_width, top_k],
                            [c_table, c_note])

            # ---- 6. History log --------------------------------------------
            with gr.Tab("History log"):
                with gr.Row():
                    h_starred = gr.Checkbox(label="Starred only", value=False)
                    h_refresh = gr.Button("Refresh")
                    h_star_idx = gr.Number(value=0, label="Row to star/unstar", precision=0)
                    h_star_btn = gr.Button("Toggle star")
                    h_export = gr.Button("Export CSV")
                    h_clear = gr.Button("Clear")
                h_table = gr.Dataframe(
                    headers=["time", "model", "reactant", "predicted", "actual",
                             "conf", "EM", "starred"],
                    label="Session log", wrap=True,
                )
                h_msg = gr.Textbox(label="Status", lines=1)
                h_file = gr.File(label="Exported CSV")

                h_refresh.click(tab_history_fn, [h_starred], [h_table, h_msg])
                h_starred.change(tab_history_fn, [h_starred], [h_table, h_msg])
                h_star_btn.click(tab_history_star_fn, [h_star_idx], [h_table, h_msg])
                h_export.click(tab_history_export_fn, None, [h_file])
                h_clear.click(tab_history_clear_fn, None, [h_table, h_msg])

            # ---- 7. SMILES toolkit -----------------------------------------
            with gr.Tab("SMILES toolkit"):
                with gr.Row():
                    with gr.Column():
                        t_smiles = gr.Textbox(label="SMILES", lines=2,
                                              placeholder="CC(=O)Oc1ccccc1C(=O)O")
                        t_compare = gr.Textbox(
                            label="Comparison SMILES for Tanimoto (optional)", lines=2
                        )
                        t_btn = gr.Button("Analyse", variant="primary")
                        t_canon = gr.Textbox(label="Canonical SMILES", lines=2)
                    t_img = gr.Image(label="Structure", height=340)
                t_props = gr.Textbox(label="Properties, drug-likeness and functional groups",
                                     lines=24)
                t_btn.click(tab_toolkit_fn, [t_smiles, t_compare], [t_img, t_props, t_canon])

            # ---- 8. Atom mapping -------------------------------------------
            with gr.Tab("Atom mapping"):
                gr.Markdown(
                    "Maximum common substructure between a reactant and a "
                    "product. Highlighted atoms are conserved."
                )
                with gr.Row():
                    am_react = gr.Textbox(label="Reactant SMILES", lines=2)
                    am_prod = gr.Textbox(label="Product SMILES", lines=2)
                am_btn = gr.Button("Map atoms", variant="primary")
                with gr.Row():
                    am_img1 = gr.Image(label="Reactant", height=340)
                    am_img2 = gr.Image(label="Product", height=340)
                am_info = gr.Textbox(label="MCS details", lines=10)
                am_btn.click(tab_atommap_fn, [am_react, am_prod],
                             [am_img1, am_img2, am_info])

            # ---- 9. 3D viewer ----------------------------------------------
            with gr.Tab("3D Viewer"):
                with gr.Row():
                    v_smiles = gr.Textbox(label="SMILES", lines=2)
                    v_style = gr.Dropdown(
                        ["stick", "ball and stick", "sphere", "wireframe"],
                        value="stick", label="Display style",
                    )
                v_btn = gr.Button("Generate 3D conformer", variant="primary")
                v_html = gr.HTML()
                v_btn.click(tab_3d_fn, [v_smiles, v_style], [v_html])

            # ---- 10. Reagent browser ---------------------------------------
            with gr.Tab("Reagent browser"):
                gr.Markdown("Search common reagents and solvents by name, category or SMILES.")
                with gr.Row():
                    rg_query = gr.Textbox(label="Search", placeholder="base, Pd, THF ...")
                    rg_btn = gr.Button("Search", variant="primary")
                rg_msg = gr.Textbox(label="Status", lines=1)
                rg_table = gr.Dataframe(
                    headers=["Name", "Category", "SMILES", "Formula", "MW"],
                    label="Reagents", wrap=True,
                )
                with gr.Row():
                    rg_draw_smiles = gr.Textbox(label="SMILES to draw", lines=1)
                    rg_draw_btn = gr.Button("Draw")
                rg_img = gr.Image(label="Structure", height=300)

                rg_btn.click(tab_reagents_fn, [rg_query], [rg_table, rg_msg])
                rg_draw_btn.click(tab_reagent_draw_fn, [rg_draw_smiles], [rg_img])
                demo.load(tab_reagents_fn, [rg_query], [rg_table, rg_msg])

            # ---- 11. Calibration plot --------------------------------------
            with gr.Tab("Calibration plot"):
                gr.Markdown(
                    "Confidence against accuracy. Uses the current session's "
                    "batch results if any, otherwise the saved full-test metrics."
                )
                cal_btn = gr.Button("Plot calibration", variant="primary")
                cal_plot = gr.Plot(label="Reliability")
                cal_info = gr.Textbox(label="Calibration metrics", lines=6)
                cal_btn.click(tab_calibration_fn, None, [cal_plot, cal_info])

            # ---- 12. Error analysis ----------------------------------------
            with gr.Tab("Error analysis"):
                gr.Markdown("Failed predictions from the last batch, ranked by severity.")
                with gr.Row():
                    e_sort = gr.Dropdown(
                        [
                            "severity (lowest Levenshtein first)",
                            "severity (lowest Tanimoto first)",
                            "confidence (most overconfident first)",
                            "length error (largest first)",
                        ],
                        value="severity (lowest Levenshtein first)", label="Sort by",
                    )
                    e_max = gr.Slider(10, 500, value=50, step=10, label="Rows to show")
                    e_btn = gr.Button("Analyse failures", variant="primary")
                e_summary = gr.Textbox(label="Failure summary", lines=14)
                e_table = gr.Dataframe(
                    headers=["id", "input", "reference", "predicted", "LA",
                             "Tanimoto", "conf", "valid", "type"],
                    label="Failures", wrap=True,
                )
                e_btn.click(tab_errors_fn, [e_sort, e_max], [e_table, e_summary])

    return demo


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    ap.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    ap.add_argument("--device", type=str, default=CFG.runtime.device)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    data_dir, ckpt_dir = Path(args.data_dir), Path(args.ckpt_dir)
    STATE.device = resolve_device(args.device)

    vocab_path = data_dir / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(
            f"{vocab_path} not found. Run `python scripts/01_prepare_data.py` first."
        )
    STATE.vocab = Vocabulary.load(vocab_path, cfg=CFG.tokenizer)

    # Load whichever checkpoints exist; non-model tabs work without them.
    gru_ckpt = ckpt_dir / "gru" / "gru_selected.pt"
    tf_ckpt = ckpt_dir / "transformer" / "transformer_selected.pt"

    if gru_ckpt.exists():
        STATE.gru = load_checkpoint(
            GRUSeq2Seq(STATE.vocab.size, CFG.gru, CFG.tokenizer), gru_ckpt, STATE.device
        )
    else:
        print(f"[app] {gru_ckpt} not found; GRU and Ensemble modes will error.")

    if tf_ckpt.exists():
        STATE.tf = load_checkpoint(
            TransformerSeq2Seq(STATE.vocab.size, CFG.transformer, CFG.tokenizer),
            tf_ckpt, STATE.device,
        )
    else:
        print(f"[app] {tf_ckpt} not found; Transformer and Ensemble modes will error.")

    try:
        STATE.test_reactions = load_split("test", data_dir)
        print(f"[app] Loaded {len(STATE.test_reactions):,} test reactions.")
    except FileNotFoundError:
        print("[app] data/test.json not found; the browser and batch tabs will be empty.")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
