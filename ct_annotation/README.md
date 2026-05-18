# CT Annotation Benchmark

Cell-type annotation benchmark across 5 single-cell foundation/representation models on the Multiple Sclerosis (MS) dataset. Parallels `embd_clustering/`.

## Layout

- `<model>-annotation-wrapper/annotation.py` — canonical source (one `# %%` cell per logical step)
- `<model>-annotation-wrapper/annotation.ipynb` — generated from `annotation.py` via `_py2nb.py`
- `<model>-annotation-wrapper/ms_annotation.h5ad` — output predictions (uniform schema)
- `benchmark.ipynb` — loads all 5 outputs, computes metrics, renders plots
- `metrics.csv`, `confusion_matrices.png`, `umap_comparison.png` — benchmark artifacts

## Output schema (all wrappers)

`ms_annotation.h5ad` is aligned to `filtered_ms_adata.h5ad`'s `obs_names`. Required columns in `.obs`:
- `celltype` — ground-truth label as string
- `predictions` — predicted label as string (from the celltype categories of the training file)

## Per-model recipe

Every wrapper follows its **original paper / official tutorial** recipe verbatim — preprocessing, hyperparameters, freezing pattern, optimizer settings. Deviations are documented in the wrapper's header markdown cell. See `docs/superpowers/specs/2026-05-10-ct-annotation-benchmark-design.md` for sources and rationale.

**Note on Geneformer.** The original spec sourced hyperparameters from `examples/cell_classification.ipynb`, which targets cardiomyopathy disease classification. Those settings collapse to ~random accuracy on celltype (0.9 epoch + heavy warmup + binary-tuned LR schedule). The wrapper instead uses the manual hyperparameters from `examples/multitask_cell_classification.ipynb` (`cell_type` task): `num_train_epochs=10, learning_rate=1e-3, lr_scheduler_type="cosine", warmup_ratio=0.01, weight_decay=0.1, per_device_train_batch_size=32, freeze_layers=2`. Still upstream-sourced — just from a celltype-relevant example rather than a disease example. The wrapper also propagates `obs_name` through Geneformer's tokenizer (which permutes row order) to align predictions back to `filtered_ms_adata.h5ad`.

| Wrapper folder | Model | Conda env | Method |
|---|---|---|---|
| `scGPT-annotation-wrapper/` | scGPT | `scgpt` | End-to-end fine-tune w/ CLS head (`Tutorial_Annotation.ipynb`) |
| `cellPLM-annotation-wrapper/` | cellPLM | `cellplm` | `CellTypeAnnotationPipeline` (`cell_type_annotation.ipynb`) |
| `Geneformer-annotation-wrapper/` | Geneformer | `geneformer` | `Classifier(classifier="cell")` via HF Trainer; `training_args` re-tuned for celltype (from `multitask_cell_classification.ipynb`'s manual hyperparameters) |
| `scFoundation-annotation-wrapper/` | scFoundation | `scfoundation` | `LinearProbingClassifier` from `finetune_model.py` |
| `scVI-annotation-wrapper/` | **scANVI** (folder named for parity) | `scvi` | scANVI semi-supervised w/ `"Unknown"` query labels |

## Current results

From `metrics.csv` (sorted by macro-F1):

| Model | accuracy | macro-F1 |
|---|---|---|
| cellPLM | 0.888 | 0.777 |
| scANVI | 0.867 | 0.743 |
| scFoundation | 0.834 | 0.702 |
| Geneformer | 0.831 | 0.685 |
| scGPT | 0.846 | 0.663 |

scGPT's macro-F1 sits below its accuracy because rare classes are under-predicted (class-imbalance effect on macro averaging). All other models are within a tight 0.66–0.78 macro-F1 band on this dataset.

## Running

Each wrapper is self-contained. To re-run one:

```bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/<env>/bin/python _py2nb.py \
    <model>-annotation-wrapper/annotation.py \
    <model>-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/<env>/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    <model>-annotation-wrapper/annotation.ipynb
```

After all 5 are run, re-run `benchmark.ipynb` (any env with scanpy + sklearn works).
