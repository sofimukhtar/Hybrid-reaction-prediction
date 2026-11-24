# Hybrid GRU–Transformer Ensemble for Forward Reaction Prediction

An attention-enhanced GRU sequence-to-sequence model and a Transformer encoder–decoder are trained independently on reactant SMILES and combined at inference by step-wise log-probability fusion inside beam search (default w_GRU = 0.7, w_TF = 0.3, beam width 5). Both models share one regex SMILES tokenizer and vocabulary.

## Install

```bash
git clone https://github.com/mukhtar-sofi/Hybrid-Reaction-Prediction.git
cd Hybrid-Reaction-Prediction

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional
pip install gradio                          # web interface
pip install transformers sentencepiece      # T5Chem baseline
```

Python 3.9+. A CUDA GPU is recommended; training falls back to CPU automatically.

## Pipeline

```bash
bash scripts/run_all.sh                 # everything, in order
SMOKE=1 bash scripts/run_all.sh         # quick end-to-end check
```

Or step by step:



| 1 | `prepare_data.py` | splits, split IDs, `vocab.json` |
| 2 | `train_gru.py` | `checkpoints/gru/` |
| 3 | `ptrain_transformer.py` | `checkpoints/transformer/` |
| 4 | `sweep_fusion_weights.py` | fusion-weight sweep (validation) |
| 5 | `evaluate_test.py --allow-test` | test metrics, per-sample predictions |
| 6 | `ablation.py --train-gate` | fusion ablation (validation) |
| 7 | `scaffold_eval.py --build`, then without `--build` | unseen-scaffold evaluation |
| 8 | `reaction_type_breakdown.py` | accuracy per reaction type |
| 9 | `make_figures.py` | figures |
| — | `python app/gradio_app.py` | web interface |

`--allow-test` is required to evaluate the test split; use `--split val` for anything involving selection or tuning.




## Per-sample predictions

`scripts/05_evaluate_test.py` writes `results/per_sample_<model>_test.jsonl`, one JSON object per reaction with the prediction, all beam candidates, every metric, the confidence and validity flags.

```python
from src.evaluate import read_per_sample
from src.metrics import paired_bootstrap_test

ens = read_per_sample("results/per_sample_ensemble_test.jsonl")
gru = read_per_sample("results/per_sample_gru_test.jsonl")

by_id = {r.rxn_id: r for r in gru}
ids = [r.rxn_id for r in ens if r.rxn_id in by_id]

print(paired_bootstrap_test(
    [next(r for r in ens if r.rxn_id == i).exact_match for i in ids],
    [by_id[i].exact_match for i in ids],
))
```

## Dataset

[`liupf/SLM4CRP_with_RTs`](https://huggingface.co/datasets/liupf/SLM4CRP_with_RTs), processed by:

1. keeping only `task == "forward reaction prediction"`;
2. de-duplicating by reaction ID;
3. RDKit canonicalization (isomeric SMILES), precursors dot-delimited and sorted alphabetically;
4. removing duplicate canonical reactant–product pairs across IDs;
5. a 70/15/15 split at seed 42, with the vocabulary built from the training split only.

If Hugging Face is unreachable, pass `--local-json` with a downloaded copy to `scripts/01_prepare_data.py`.

## Citation

## License

MIT — see `LICENSE`.
