#!/usr/bin/env bash
#
# Full pipeline. Run from the repository root:
#
#     bash scripts/run_all.sh
#
# SMOKE=1 runs a quick end-to-end check on tiny subsets:
#
#     SMOKE=1 bash scripts/run_all.sh

set -euo pipefail

SMOKE="${SMOKE:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ "$SMOKE" == "1" ]]; then
  TRAIN_FLAGS="--smoke-test"
  EVAL_FLAGS="--max-samples 200"
  SWEEP_FLAGS="--subset 200"
  ABLATION_FLAGS="--subset 200 --skip-gate"
  echo "=== SMOKE MODE: tiny subsets ==="
else
  TRAIN_FLAGS=""
  EVAL_FLAGS=""
  SWEEP_FLAGS="--subset 3000"
  ABLATION_FLAGS="--subset 3000 --train-gate"
fi

echo
echo "############  0. Parameter counts  ############"
python scripts/00_report_param_counts.py

echo
echo "############  1. Dataset  ############"
python scripts/01_prepare_data.py

echo
echo "############  2. Train GRU  ############"
python scripts/02_train_gru.py --device "$DEVICE" $TRAIN_FLAGS

echo
echo "############  3. Train Transformer  ############"
python scripts/03_train_transformer.py --device "$DEVICE" $TRAIN_FLAGS

echo
echo "############  4. Fusion weight sweep (validation only)  ############"
python scripts/04_sweep_fusion_weights.py --device "$DEVICE" $SWEEP_FLAGS

echo
echo "############  5. Test evaluation  ############"
python scripts/05_evaluate_test.py --allow-test --device "$DEVICE" $EVAL_FLAGS

echo
echo "############  6. Ablation (validation only)  ############"
python scripts/06_ablation.py --device "$DEVICE" $ABLATION_FLAGS

echo
echo "############  7. Unseen-scaffold evaluation  ############"
python scripts/07_scaffold_eval.py --build
python scripts/07_scaffold_eval.py

echo
echo "############  8. Reaction-type breakdown  ############"
python scripts/09_reaction_type_breakdown.py

echo
echo "############  9. Figures  ############"
python scripts/08_make_figures.py

echo
echo "############  Done  ############"
echo "Results and per-sample predictions: results/"
echo "Figures                           : figures/"
echo "Checkpoints                       : checkpoints/"
echo
echo "T5Chem baseline (requires a T5Chem checkpoint):"
echo "    python scripts/10_t5chem_baseline.py --model-name-or-path <ckpt> --train --decode"
echo "    python scripts/07_scaffold_eval.py --t5chem-predictions results/t5chem_test.jsonl"
echo
echo "Web server:"
echo "    python app/gradio_app.py"
