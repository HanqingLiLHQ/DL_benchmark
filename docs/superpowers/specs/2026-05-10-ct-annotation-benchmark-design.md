# Cell-Type Annotation Benchmark — Design

**Date:** 2026-05-10
**Scope:** Build a parallel set of cell-type (CT) annotation wrappers — one per model (cellPLM, Geneformer, scFoundation, scGPT, scVI/scANVI) — plus a unified benchmark notebook that compares them. Mirrors the existing `embd_clustering/` layout, but with fine-tuning instead of just embedding extraction.

## Goals

1. One notebook per model that fine-tunes / adapts the model on the Multiple Sclerosis (MS) dataset (`c_data.h5ad` train + `filtered_ms_adata.h5ad` test) and writes a predictions file with a uniform schema.
2. A single benchmark notebook that loads all 5 prediction files and produces side-by-side metrics + plots.
3. **Strict paper-faithfulness.** Each model uses the exact recipe (preprocessing, hyperparameters, freezing pattern, optimizer settings, splits) from its original paper / official tutorial. Adaptations are limited to what's required to point the recipe at the MS files; every deviation is called out in the wrapper header.

## Non-goals

- Cross-dataset generalization (only MS for now).
- Hyperparameter sweeps. We use each tutorial's default hyperparameters.
- A unified evaluation harness as a Python package. Notebooks are sufficient and match the existing pattern.
- wandb logging. Console logging only.

## Folder layout

Mirrors `embd_clustering/`:

```
ct_annotation/
  README.md
  benchmark.ipynb                                  ← NEW: compares all 5 models
  cellPLM-annotation-wrapper/
    annotation.ipynb
    ms_annotation.h5ad                             ← produced
  Geneformer-annotation-wrapper/
    annotation.ipynb
    ms_annotation.h5ad                             ← produced
  scFoundation-annotation-wrapper/
    annotation.ipynb
    ms_annotation.h5ad                             ← produced
  scGPT-annotation-wrapper/
    annotation.ipynb                               ← rewritten, replaces Tutorial_Annotation.ipynb
    ms_annotation.h5ad                             ← produced
  scVI-annotation-wrapper/                         ← actually uses scANVI; folder name kept for parallel naming
    annotation.ipynb
    ms_annotation.h5ad                             ← produced
```

The existing `Tutorial_Annotation.ipynb` at the top of `ct_annotation/` is discarded (user-approved).

## Wrapper contract (uniform across all 5)

**Inputs (from `/data/benchmark/data/cellPLM/data/`):**

- `c_data.h5ad` — training data (reference)
- `filtered_ms_adata.h5ad` — query data (test)
- Label column on both: `Factor Value[inferred cell type - authors labels]` → normalize to `obs["celltype"]`
- Batch column on both: derived; `c_data` → batch `"0"`, `filtered_ms_adata` → batch `"1"` (same convention as the original scGPT tutorial)

**Output: `ms_annotation.h5ad` in the wrapper folder.** Its schema is fixed so `benchmark.ipynb` can load all 5 the same way:

- Shape: `(n_test_cells, *)` — one row per cell in `filtered_ms_adata.h5ad`
- `obs_names`: identical to `filtered_ms_adata.h5ad`'s `obs_names` (so alignment is trivial)
- `obs["celltype"]`: ground-truth label as string
- `obs["predictions"]`: predicted label as string (drawn from the celltype categories present in the training file `c_data.h5ad` — if a model predicts a class not seen in `filtered_ms_adata.h5ad`, the benchmark will still count it as wrong, which is the desired behavior)

`.X` and `.var` can be whatever each model finds convenient — `benchmark.ipynb` only reads `obs`.

**Kernel:** each notebook runs in its model-specific conda env (`cellplm`, `geneformer`, `scfoundation`, `scgpt`, `scvi`). All envs already exist on this machine.

## Per-model recipes

**Rule:** each wrapper copies the official tutorial's hyperparameter block verbatim — same epochs, lr, batch size, mask ratio, freezing pattern, etc. Faithfulness to the paper recipe is the whole point of the comparison, and any silent substitution biases the result. Adaptations are limited to what's strictly required to point the recipe at the MS files; any deviation is called out in the wrapper notebook header.

### scGPT
- **Source:** `models/scGPT/tutorials/Tutorial_Annotation.ipynb`
- **Model:** `scgpt.model.TransformerModel` with `CLS=True` (end-to-end fine-tune w/ classification head)
- **Pretrained checkpoint:** `/data/benchmark/models/scGPT/save/scGPT_human`
- **Hyperparameters (verbatim from tutorial):** `seed=0`, `epochs=10`, `lr=1e-4`, `batch_size=32`, `mask_ratio=0.0`, `n_bins=51`, `max_seq_len=3001`, `layer_size=128`, `nlayers=4`, `nhead=4`, `dropout=0.2`, `schedule_ratio=0.9`, `amp=True`, `freeze=False`, `DSBN=False`, `include_zero_gene=False`, `input_style="binned"`, `cell_emb_style="cls"`
- **Preprocessing:** scGPT `Preprocessor` with `normalize_total=1e4`, `log1p=False` (MS data already log-normed), `subset_hvg=False`, `binning=51`
- **Adaptations vs tutorial:** strip wandb; write `ms_annotation.h5ad` instead of the tutorial's plotting / save_dict logic

### cellPLM
- **Source:** `models/cellPLM/tutorials/cell_type_annotation.ipynb`
- **Model:** `CellPLM.pipeline.cell_type_annotation.CellTypeAnnotationPipeline`
- **Pretrained version:** `'20230926_85M'` (checkpoint in `models/cellPLM/ckpt/`)
- **Configs:** `CellTypeAnnotationDefaultPipelineConfig` + `CellTypeAnnotationDefaultModelConfig` (verbatim from tutorial — only override `model_config['out_dim'] = data.obs['celltype'].nunique()`)
- **Hyperparameters:** `set_seed(42)`; data prep follows the tutorial — concatenate train+test into one AnnData, set `obs['split']` to `'test'` for the query rows, then 90/10 train/valid permutation over the training rows
- **Var setup:** `var = var.set_index('index_column')` on both train and test (exact tutorial step for MS)
- **Adaptations vs tutorial:** at the end, slice the predicted AnnData to rows where `split == 'test'` and write `ms_annotation.h5ad`

### Geneformer
- **Source:** `models/Geneformer/examples/cell_classification.ipynb` (the tutorial covers disease classification; same `Classifier(classifier="cell", ...)` API is used for cell-type classification per the Geneformer docs)
- **Model:** `geneformer.Classifier` (CellClassifier) via HF Trainer
- **Pretrained checkpoint:** Geneformer V1-10M (default; will load from `models/Geneformer` weights)
- **Tokenization:** `TranscriptomeTokenizer` converts h5ad → HF dataset using ensembl IDs as the gene vocab
- **Hyperparameters (verbatim from tutorial, except where noted):** `freeze_layers=2`, `forward_batch_size=200`, `nproc=16`, `seed=73`; `training_args = {num_train_epochs: 0.9, learning_rate: 8.04e-4, lr_scheduler_type: "polynomial", warmup_steps: 1812, weight_decay: 0.258828, per_device_train_batch_size: 12}`. **Note:** the tutorial explicitly says these are tuned for cardiomyopathy disease and recommends re-tuning per task via `n_hyperopt_trials=N`. We will start with the tutorial's values verbatim, document this in the wrapper header, and revisit only if the smoke run shows obvious failure.
- **State key:** `{"state_key": "celltype", "states": "all"}` (instead of `"disease"` from the tutorial)
- **Adaptations vs tutorial:** point `prepare_data` at MS-derived HF datasets; no `filter_data_dict` (we use all celltypes); split by source file (train rows from `c_data`, test rows from `filtered_ms_adata`) rather than by individual ID

### scFoundation
- **Source:** `models/scFoundation/model/finetune_model.py` (class `LinearProbingClassifier`) — this is the official fine-tuning script. The `annotation/celltype-plot.ipynb` is a *plotting* notebook for results produced by this script (it expects pre-computed `*-emb.pkl` logit files).
- **Model:** `LinearProbingClassifier` with `frozenmore=True`
- **Freezing pattern:** `token_emb` frozen, `pos_emb` frozen, all of `encoder` frozen, **then** `encoder.transformer_encoder[-2]` (the second-to-last layer of the transformer stack) is selectively unfrozen — i.e. only this one layer of the backbone is trainable, in addition to the MLP head and the `BatchNorm1d` running stats. (This matches `finetune_model.py:96-106` exactly.)
- **Head:** max-pool over seq dim → `BatchNorm1d(768, affine=False, eps=1e-6)` → `Linear(768, 256)` → `ReLU` → `Linear(256, n_class)`
- **Pretrained checkpoint:** `models/scFoundation/model/models/models.ckpt`
- **Hyperparameters:** seed=0; lr, batch size, epochs are not pinned in the public scFoundation repo (the `__main__` block only verifies the forward pass — it does not train). We use commonly-chosen linear-probing defaults: `lr=1e-4`, `AdamW`, `weight_decay=0`, `batch_size=8`, `epochs=10`, and explicitly flag this as a deviation in the wrapper header: "scFoundation paper does not publish the exact CT-annotation fine-tuning hyperparameters; defaults chosen to match a standard linear-probing setup."
- **Gene padding:** input matrix is padded to 19264 genes using `OS_scRNA_gene_index.19264.tsv` (`main_gene_selection` helper from `get_embedding.py`)
- **Adaptations vs script:** the script is a class definition with a `__main__` smoke test only — we wrap it in a proper training loop matching the demo signature

### scVI / scANVI
- **Source:** scvi-tools scANVI tutorial (`https://docs.scvi-tools.org/en/stable/tutorials/notebooks/scrna/harmonization.html` style; we will use the canonical scANVI recipe documented in the scvi-tools API)
- **Model:** `scvi.model.SCANVI` (semi-supervised; built-in label-prediction head). The published paper recipe for CT annotation in the scvi-tools ecosystem is scANVI, not raw scVI + KNN.
- **Workflow:**
  1. Concatenate `c_data` + `filtered_ms_adata` into one AnnData; set `obs["labels_scanvi"] = celltype` for train rows and `= "Unknown"` for test rows
  2. `scvi.model.SCVI.setup_anndata(adata, batch_key="str_batch")`; train scVI for `max_epochs=20`
  3. `scvi.model.SCANVI.from_scvi_model(scvi_model, unlabeled_category="Unknown", labels_key="labels_scanvi")`; train scANVI for `max_epochs=20`, `n_samples_per_label=100`
  4. `predictions = scanvi_model.predict(adata_test)`; write `ms_annotation.h5ad`
- **Hyperparameters:** scvi-tools defaults (`n_latent=10`, `n_layers=2`, `n_hidden=128`, `dropout_rate=0.1`); seed=0
- **Note:** folder name is `scVI-annotation-wrapper` (parallel to `embd_clustering/scVI-embedding-wrapper`); the model used inside is scANVI — the wrapper README states this clearly.

## benchmark.ipynb

Parallels `embd_clustering/benchmark.ipynb`. Structure:

1. **Load ground truth.** Read `filtered_ms_adata.h5ad`, normalize the label column to `obs["celltype"]`.
2. **Load each wrapper's predictions.** For each model, read `{model}-annotation-wrapper/ms_annotation.h5ad`, assert `obs_names` align with the ground-truth file, copy `obs["predictions"]` onto the ground-truth adata as `obs[f"{model}_pred"]`.
3. **Compute metrics.** For each model: accuracy, macro-precision, macro-recall, macro-F1 (`sklearn.metrics`). Collect into a pandas DataFrame (one row per model).
4. **Confusion matrices.** Render a 1×5 grid of confusion-matrix heatmaps (one per model), normalized per-row.
5. **UMAP comparison.** Compute one UMAP on the ground-truth adata (standard `sc.pp.normalize_total` → `log1p` → HVG → PCA → `sc.pp.neighbors` → `sc.tl.umap`, so all panels share coordinates), then plot six panels: ground truth + each model's predictions.

## Shared concerns

- **Gene-id handling.** `c_data.h5ad` / `filtered_ms_adata.h5ad` are keyed by ensembl IDs with a `gene_name` symbol column. Some model vocabularies are symbol-based (scGPT) and others are ensembl-based (Geneformer, scFoundation); each wrapper handles its own mapping, reusing `utils/gene_converter.py` when applicable.
- **Train/test split.** Use the provided files as the split — do not re-split. `c_data` is reference / `filtered_ms_adata` is query, matching the scGPT tutorial.
- **Reproducibility.** Each notebook sets `seed=0` at the top. Bitwise reproducibility is not required (some models use non-deterministic CUDA kernels), but runs should be qualitatively stable.
- **No wandb.** The original scGPT tutorial wraps everything in wandb; the wrappers strip that out and rely on plain print/logger output, matching the `embd_clustering/` notebooks.

## Components and their boundaries

Each of the 6 notebooks is an isolated unit:

- **5 wrapper notebooks** — each owns data loading, preprocessing, model setup, training, prediction, and saving `ms_annotation.h5ad`. Inputs are two fixed paths; output is one fixed-schema file. No notebook depends on any other.
- **benchmark.ipynb** — consumes the 5 output files. Knows nothing about how predictions were produced.

This means: any wrapper can be re-run, swapped, or replaced without touching others, and the benchmark notebook works as long as the schema contract holds.

## Testing approach

Per wrapper:
- Smoke check: train for 1–2 epochs (low budget) end-to-end before increasing to the tutorial's recommended epoch count, to surface env / data-loading issues fast.
- Sanity check: `ms_annotation.h5ad` exists, has the right `n_obs`, contains `obs["celltype"]` and `obs["predictions"]`, and `obs_names` match `filtered_ms_adata.h5ad`.
- Numerical sanity: macro-F1 > 0.5 on MS (the scGPT tutorial achieves ~0.87; foundation models should land in that ballpark; scFoundation w/ frozen backbone may be lower).

Benchmark:
- Verify it loads all 5 files, produces the metrics DataFrame, the confusion-matrix grid, and the UMAP-comparison grid.

## Risks / open questions

- **Geneformer hyperparameters are not paper-tuned for celltype.** The tutorial values (`num_train_epochs=0.9`, `learning_rate=8.04e-4`, `warmup_steps=1812`, `weight_decay=0.258828`, `per_device_train_batch_size=12`, `seed=73`) were optimized for cardiomyopathy disease classification. The tutorial recommends re-tuning. **Decision for this benchmark:** start verbatim with the tutorial's values for faithfulness; if the smoke run shows clearly poor results we revisit, but any tuning is documented.
- **scFoundation hyperparameters not pinned in the repo.** `finetune_model.py` is a class definition; the paper appendix doesn't expose exact lr/epochs for the CT annotation experiment. We use defaults from the script's `__main__` demo block and document this as a deviation.
- **scFoundation env quirks.** The repo expects a specific gene-index file (`OS_scRNA_gene_index.19264.tsv`); the wrapper must align MS genes to that index using the `main_gene_selection` helper from `get_embedding.py`.
- **scANVI label leakage caveat.** scANVI uses semi-supervised learning over unlabeled query cells; we mark the query cells as `"Unknown"` before training so labels can't leak into the model.
- **cellPLM 90/10 train/valid split is stochastic.** The tutorial uses `np.random.permutation(train_num)` with `set_seed(42)`; we reproduce this exactly so the split is deterministic.
- **Geneformer fine-tuning runtime.** HF Trainer on MS-scale data is typically minutes, not hours, but worth confirming with the smoke run.
- **Memory.** RTX PRO 6000 Blackwell has 97 GB — none of the per-model fine-tunes on MS should come close to that limit.
