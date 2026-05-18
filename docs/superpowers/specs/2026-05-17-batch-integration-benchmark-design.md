# Batch-Integration Benchmark — Design

**Date:** 2026-05-17
**Scope:** Add a third benchmark task — single-cell **data integration / batch-effect removal** — parallel to `ct_annotation/` and `embd_clustering/`. Run all our models with their *paper-faithful* integration recipe, score with the field-standard scIB panel, and add one results slide to the presentation reporting every model **except cellPLM**.

## Dataset

`/data/benchmark/data/cellPLM/data/gse155468.h5ad` — 48,082 cells × 12,382 genes (genome-wide; the only shipped genome-wide dataset, so the foundation-model comparison stays fair).

- **Batch key:** `obs['orig.ident']` — 11 samples (Con4, ATAA1–8, Con6/9, …).
- **Bio label:** `obs['celltype']` — 11 types (SMC1, Fibroblast, EC, Tcell, …).
- Rationale for not using GSE131907/GSE151530: both are ~400-gene panels that starve genome-wide foundation models and bias the comparison (violates paper-faithfulness). GSE155468's per-sample variation (`orig.ident`) is the batch signal; cell types are shared across samples.

## Per-model integration recipe (paper-faithful)

| Model | Faithful recipe | Source of embedding |
|---|---|---|
| **scVI** | scVI with `batch_key='orig.ident'` (flagship integration use) | **Reuse** `embd_clustering/scVI-embedding-wrapper/gse155468_embedding.h5ad` — that wrapper already sets `batch_key='orig.ident'`, so it is already a faithful integrated embedding (n_latent=30). |
| **scGPT (zero-shot)** | Pretrained `scGPT_human`, no training, `[CLS]` embedding — the same recipe the old Results-1 clustering slide used. The batch-naive reference point. | **Reuse** `embd_clustering/scGPT-embedding-wrapper`. |
| **scGPT (fine-tuned)** | Mirror `models/scGPT/tutorials/Tutorial_Integration.ipynb` verbatim (batch-aware fine-tune: MLM + zero-prob + GEPC/MVC + ECS + DAB + DSBN, `batch_key`→`orig.ident`), pretrained `scGPT_human`. | **NEW run** — the only heavy new compute. |
| **Geneformer** | No native batch correction; the paper's integration claim is that the pretrained rank-value embedding is batch-robust as-is → zero-shot embedding. | **Reuse** `embd_clustering/Geneformer-embedding-wrapper` (V2-104M). |
| **scFoundation** | No native batch correction → zero-shot embedding (POOL_TYPE='all', 3072-d). | **Reuse** `embd_clustering/scFoundation-embedding-wrapper`. |
| **cellPLM** | Zero-shot `CellEmbeddingPipeline` (its paper's integration approach). | **Reuse** `embd_clustering/cellPLM-embedding-wrapper`. Run-but-**not reported**. |
| **PCA** | Standard scanpy PCA = unintegrated reference baseline. | Compute in `benchmark`. |

The **scGPT zero-shot vs fine-tuned pair is the headline comparison** of this slide: same backbone, with vs without the integration fine-tune (domain adaptation + DAB/DSBN batch removal).

Reuse rule: a reused embedding is loaded and re-emitted under the integration output schema; the wrapper header documents *why zero-shot/batch-conditioned is this model's faithful integration method*. Only `scGPT (fine-tuned)` is re-run; everything else (incl. scGPT zero-shot) is reused from `embd_clustering/`.

## Folder layout (mirrors existing benchmarks)

```
batch_integration/
  README.md
  _py2nb.py                                       # copied from ct_annotation/_py2nb.py
  scGPT-finetuned-integration-wrapper/
    integration.py  integration.ipynb             # NEW; mirrors Tutorial_Integration.ipynb
    gse155468_integration.h5ad                    # produced
  scGPT-zeroshot-integration-wrapper/
  scVI-integration-wrapper/
  Geneformer-integration-wrapper/
  scFoundation-integration-wrapper/
  cellPLM-integration-wrapper/
    integration.py  integration.ipynb             # thin: load embd_clustering emb, re-emit integration schema
    gse155468_integration.h5ad                    # produced
  benchmark.py  benchmark.ipynb                   # scib-metrics + UMAPs + metrics.csv
  metrics.csv  umap_batch_celltype.png            # artifacts
```

Six wrappers (scGPT split into zero-shot + fine-tuned); all but `scGPT-finetuned` are thin reuse shims.

## Output schema (uniform across all 6 wrappers)

`gse155468_integration.h5ad`:
- `obs_names` identical to `gse155468.h5ad` (alignment by ID, not order)
- `obsm['X_emb']` — (n_cells, d) float32 integrated/zero-shot embedding
- `obs['celltype']` — bio label (string)
- `obs['batch']` — `orig.ident` value (string)
- `.X`/`.var` unconstrained; `benchmark` reads only `obsm`/`obs`.

## Metrics

`scib_metrics.benchmark.Benchmarker`:
- **Bio conservation:** NMI, ARI, cLISI, silhouette-label, isolated-labels.
- **Batch correction:** silhouette-batch, iLISI, kBET, graph connectivity, PCR.
- **Aggregate:** Total = 0.6·Bio + 0.4·Batch (scIB default weighting).
- Embeddings scored: scVI, **scGPT-zeroshot, scGPT-finetuned**, Geneformer, scFoundation, cellPLM, PCA (X_pca). `metrics.csv` = full Benchmarker results table.

## Evaluation validity & label usage

This benchmark follows the scIB (Luecken et al., *Nature Methods* 2022) convention for atlas-level integration:

- **Label firewall.** `obs['batch']` (`orig.ident`) is *method input / metric grouping only*. `obs['celltype']` is *bio scoring only*. Never the reverse. No model is fed `celltype`. `benchmark` enforces this — it passes `batch_key`/`label_key` to scib-metrics for scoring; cell-type labels never reach any wrapper.
- **No cell-type leakage in scGPT fine-tune.** scGPT's `Tutorial_Integration` objectives (MLM, zero-prob, GEPC/MVC, ECS) are self-supervised on expression; DAB/DSBN consume only **batch** labels. `celltype` is unseen → bio metrics remain a fair held-out evaluation despite transductive (same-cell) scoring.
- **Transductive evaluation is correct here.** Unlike supervised annotation (`ct_annotation/`, which needs a `c_data`→`filtered_ms_adata` split and scANVI `"Unknown"` masking), integration is evaluated on the dataset being integrated — standard for scIB/scVI/Harmony. Validity comes from bio-label blindness, not a cell holdout.
- **Batch-aware vs zero-shot asymmetry (stated, not hidden).** scGPT-finetuned and scVI consume `orig.ident`; scGPT-zeroshot/Geneformer/scFoundation/cellPLM emit zero-shot embeddings with no batch input. This is intended (the benchmark measures whether a model's batch mechanism works), not leakage — but it is an asymmetry. Each wrapper header records which labels it consumed (scGPT-finetuned/scVI: batch; others: none). The Results-1 slide speaker notes carry this caveat explicitly, paralleling the existing scANVI semi-supervised caveat in `ct_annotation/README.md`.
- **scGPT zero-shot vs fine-tuned is not a clean single-variable ablation.** The fine-tune differs from the zero-shot embedding on *three* axes at once: (a) DAB/DSBN batch removal, (b) in-domain self-supervised adaptation (MLM/GEPC/ECS retrained on GSE155468), and (c) gene set (`Tutorial_Integration` uses `n_hvg=1200` vs the zero-shot wrapper's full-gene `[CLS]`). Any batch-axis gain therefore reflects the integration *recipe as a whole*, not DAB/DSBN in isolation. A clean attribution would need a fine-tune-without-DAB/DSBN ablation — **out of scope**; the slide notes state this so the comparison isn't over-interpreted.

## Environment (hardware-aware)

- GPU: RTX PRO 6000 Blackwell, 97 GB, driver 580 — ample for scGPT fine-tune.
- `scib-metrics` is absent from every conda env. The Blackwell + driver-580 stack makes jax-CUDA fragile, so create a dedicated env running **`scib-metrics` on CPU jax** (48k cells → minutes; robust, no version chase). Env name `scib-metrics`; exact build pinned in `batch_integration/README.md`.
- scGPT wrapper runs in the existing `scgpt` env (same as `ct_annotation/scGPT-annotation-wrapper`).

## Deck integration (replace Results 1, in place)

`presentation/build_deck.py` is stale vs the live `presentation/sc_foundation_models.pptx` (user pasted paper architecture figures + edited titles; ~4 MB). **Do not rerun `build_deck.py`** — it would destroy pasted figures.

- **Results 1 is swapped, not added.** The current "Results 1 – zero shot embedding clustering" (GSE155468 ARI/NMI) slide is *replaced in place* by the batch-integration result on the same dataset. Deck slide count is unchanged; Results 2 (annotation) and all model slides are untouched.
- New script `presentation/swap_results1_slide.py`: opens the existing `.pptx`, locates the Results-1 slide, replaces its title → **"Results 1 – batch integration (GSE155468)"**, swaps its chart picture for a new matplotlib bar chart (scIB Total / Bio / Batch for scVI, **scGPT zero-shot, scGPT fine-tuned**, Geneformer, scFoundation, PCA — **cellPLM excluded**), and rewrites its speaker notes (headline = zero-shot vs fine-tuned scGPT; plus the batch-aware/zero-shot asymmetry and weak-batch-signal caveats). All other slides verified untouched by pic/notes counts.
- Rationale for swap (vs add): the old clustering slide's ARI/NMI are *subsumed* by scIB's Bio axis (which includes NMI/ARI); the integration slide is a strict superset that also shows batch removal. The PCA baseline carries over. No clustering information is lost.
- Also patch `build_deck.py`'s Results-1 builder so a from-scratch rebuild produces the integration slide (consistency), but the live deck is modified only via the swap script.

## Testing

- **Per wrapper:** `gse155468_integration.h5ad` exists; `n_obs == 48082`; `obsm['X_emb']` present & finite; `obs_names` match `gse155468.h5ad`; `obs['batch']`/`obs['celltype']` non-null.
- **scGPT:** 1-epoch smoke run before full fine-tune; confirm loss decreases and embedding shape.
- **Benchmark:** Benchmarker returns a table for all 7 scored embeddings (scVI, scGPT-zeroshot, scGPT-finetuned, Geneformer, scFoundation, cellPLM, PCA); sanity — PCA scores lowest on Batch correction, scVI high on Batch (it conditions on batch), scGPT-finetuned ≥ scGPT-zeroshot on Batch axis (expected direction); `metrics.csv` written.
- **Deck:** slide **count unchanged**; the Results-1 slide's title/chart/notes are replaced (new chart picture present, notes mention zero-shot vs fine-tuned scGPT); Results 2, Takeaways, and every model slide's picture/notes counts byte-for-byte unchanged (pasted figures preserved).

## Risks

- scGPT integration fine-tune runtime/VRAM on 48k cells — mitigated by smoke run + 97 GB GPU.
- `scib-metrics` CPU runtime — acceptable at 48k; documented.
- GSE155468 batch signal is within-study (samples), weaker than a cross-protocol benchmark — acknowledged in the slide's speaker notes; this is the only genome-wide shipped option.
- Reused embeddings assume `embd_clustering` outputs are current; benchmark asserts `obs_names` alignment and fails loudly otherwise.

## Out of scope

cellPLM in the slide; touching the Results-2 (annotation) or model slides; keeping a standalone clustering slide (its ARI/NMI live on inside scIB's Bio axis); a DAB/DSBN ablation; jax-GPU; any non-GSE155468 dataset; hyperparameter tuning (scGPT uses Tutorial_Integration defaults verbatim).
