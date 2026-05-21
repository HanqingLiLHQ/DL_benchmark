# Batch-integration benchmark (GSE155468)

Third benchmark task in this repo, parallel to `ct_annotation/` and
`embd_clustering/`: single-cell **data integration / batch-effect removal**.
Every model is run with its **paper-faithful** integration recipe, scored with
the field-standard **scIB** panel (`scib-metrics`), and reported on one deck
slide (every model **except cellPLM**).

## Dataset

`data/cellPLM/data/gse155468.h5ad` — 48,082 cells × 12,382 genes (human
ascending thoracic aortic aneurysm; the only genome-wide shipped dataset, kept
so the foundation-model comparison stays fair — ~400-gene panels would starve
the FMs and bias the comparison).

- **Batch key:** `obs['orig.ident']` — 11 samples. Method input / metric grouping only.
- **Bio label:** `obs['celltype']` — 11 types. Scoring only (see label firewall).
- Raw integer counts; 234 duplicate `obs_names` → every wrapper applies
  `obs_names_make_unique()` and **all** embeddings are aligned by that exact
  ID order (`benchmark.py` does the same before attaching them).

## Per-model integration recipe (paper-faithful)

| Model | Faithful recipe | Embedding source |
|---|---|---|
| **scVI** | scVI, `batch_key='orig.ident'` (flagship integration use), n_latent=30 | Reuse `embd_clustering/scVI-embedding-wrapper` (already batch-conditioned ⇒ already a faithful integrated embedding) |
| **scGPT (zero-shot)** | Pretrained `scGPT_human`, no training, `[CLS]` embedding — batch-naive reference | Reuse `embd_clustering/scGPT-embedding-wrapper` |
| **scGPT (fine-tuned)** | `Tutorial_Integration.ipynb` verbatim: MLM + zero-prob + GEPC/MVC + ECS + **DAB + DSBN**, `batch_key→orig.ident`, pretrained `scGPT_human` | **NEW run** (only heavy new compute; 15 epochs) |
| **Geneformer** | No native batch correction; paper claims the pretrained rank-value embedding is batch-robust as-is → zero-shot | Reuse `embd_clustering/Geneformer-embedding-wrapper` (V2-104M) |
| **scFoundation** | No native batch correction → zero-shot embedding (`POOL_TYPE='all'`, 3072-d) | Reuse `embd_clustering/scFoundation-embedding-wrapper` |
| **cellPLM** | Zero-shot `CellEmbeddingPipeline` (its paper's integration approach) | Reuse `embd_clustering/cellPLM-embedding-wrapper`. **Run-but-NOT-reported** (excluded from `metrics_reported.csv` and the slide) |
| **PCA** | Standard scanpy PCA (4500 HVG, 50 PCs) = unintegrated reference | Computed in `benchmark.py` |

Only `scGPT (fine-tuned)` is re-trained; every other embedding (including
scGPT zero-shot) is reused from `embd_clustering/` and re-emitted under the
common integration output schema (`obsm['X_emb']`, `obs['celltype']`,
`obs['batch']`, aligned `obs_names`) — each wrapper is self-contained and
performs that re-emit inline. The **scGPT zero-shot vs fine-tuned pair is the
headline comparison**: same backbone, with vs without the integration fine-tune.

## Evaluation validity — label firewall

`batch` (`orig.ident`) is consumed **only** as a method input (scVI,
scGPT-finetuned) and as the scIB metric grouping. `celltype` is attached
**only** for scoring — no model ever sees it. Integration is evaluated
transductively on the dataset being integrated (standard for scIB/scVI/
Harmony); validity comes from bio-label blindness, not a cell holdout.

**Batch-aware vs zero-shot asymmetry (stated, not hidden):** scVI and
scGPT-finetuned consume `orig.ident` by design; scGPT-zeroshot / Geneformer /
scFoundation / cellPLM emit zero-shot embeddings with no batch input. The
benchmark measures whether a model's batch mechanism works — this asymmetry is
intended, not leakage.

## scib-metrics cdist fix (memory + speed) — REQUIRED, paper-faithful

`scib_metrics==0.5.9`'s jax `cdist` (`scib_metrics/utils/_dist.py`, used by
`silhouette_samples` → `isolated_labels`, `silhouette_label`, `bras`) computes
euclidean distance as `vmap(vmap(sqrt(sum((x-y)**2))))`, materializing a dense
`(chunk_size, N, d)` tensor before reducing over `d`. With `chunk_size=256`,
N=48082 and **scFoundation's d=3072** that intermediate is ~156 GB → OOM
(machine: 62 GB RAM / 96 GB VRAM). d ≤ 768 embeddings fit; only scFoundation
failed (at the first silhouette metric, `isolated_labels`).

`benchmark.py` defines `cdist_lowmem` — the algebraically identical
`‖a‖² + ‖b‖² − 2·a·b` form (O(chunk·N) memory, single BLAS matmul) — and
rebinds it into both `scib_metrics.utils._dist` and `scib_metrics.utils._silhouette`
before `Benchmarker.benchmark()`. **The scIB protocol, metric set, and the full
3072-d scFoundation embedding are unchanged** — only the distance kernel's
memory layout differs, and silhouette per-sample values are independent of
chunking (per-chunk results are concatenated, never cross-reduced).

Historical equivalence checks (against stock `scib-metrics` and
`scipy.spatial.distance.cdist`):
- `cdist_lowmem` vs `scipy.spatial.distance.cdist`: max |Δ| ≈ 1e-6 (euclidean), 1e-7 (cosine).
- stock vs patched `silhouette_samples` on a real 2500-cell scFoundation subsample: ASW |Δ| ≈ 2e-6, per-sample max |Δ| ≈ 1e-4.
- full 48082×3072 scFoundation silhouette: completes, finite, ASW **0.175762 identical on CPU and GPU**.

## Hardware / env

Benchmark + model-free steps run in the `scib-metrics` conda env. **GPU is
used** (RTX PRO 6000 Blackwell, sm_120): `jax[cuda12]==0.10.0` (CUDA 12.9
plugin). Run **without** `JAX_PLATFORMS=cpu` and **with**
`JAX_DEFAULT_MATMUL_PRECISION=highest` — full FP32 matmul avoids lossy TF32 so
results stay faithful. Residual GPU floating-point non-determinism is accepted
(metric values reproducible to scIB's reported precision; the benign jax log
line `Could not get kernel mode driver version` is harmless). scGPT
fine-tuning runs in the separate `scgpt` env.

Env pin (`scib-metrics`):

```
anndata        0.12.14
jax            0.10.0
jax-cuda12-pjrt   0.10.0
jax-cuda12-plugin 0.10.0
jaxlib         0.10.0
numpy          2.4.5
pandas         2.3.3
scanpy         1.11.5
scib-metrics   0.5.9
scikit-learn   1.8.0
```

## Re-run

```bash
cd /data/benchmark/batch_integration
PY=/home/hanqing-li/anaconda3/envs/scib-metrics/bin/python

# 1. wrappers (only scGPT-finetuned is heavy; the other five just re-emit
#    their embd_clustering/ counterpart under the integration schema)
for d in scVI scGPT-zeroshot Geneformer scFoundation cellPLM; do
  ( cd "${d}-integration-wrapper" && $PY integration.py )
done
# scGPT-finetuned: run scGPT-finetuned-integration-wrapper/integration.py in the `scgpt` env

# 2. scIB benchmark (GPU; cdist patch is inlined in benchmark.py)
$PY _py2nb.py benchmark.py benchmark.ipynb
JAX_DEFAULT_MATMUL_PRECISION=highest \
  /home/hanqing-li/anaconda3/envs/scib-metrics/bin/jupyter nbconvert \
  --to notebook --execute --inplace benchmark.ipynb
# -> metrics.csv, metrics_reported.csv (cellPLM dropped), umap_batch_celltype.png

# 3. deck: ADD a non-destructive slide (does NOT rerun build_deck.py,
#    which would wipe the manually pasted paper figures)
cp /data/benchmark/presentation/sc_foundation_models.pptx \
   /data/benchmark/presentation/sc_foundation_models.bak.pptx
/home/hanqing-li/anaconda3/bin/python /data/benchmark/presentation/add_integration_slide.py
```

## Results (scIB, higher = better; `min_max_scale=False`, absolute scores)

| Model | Bio conservation | Batch correction | Total (0.6·Bio + 0.4·Batch) |
|---|---|---|---|
| PCA (unintegrated ref) | 0.708 | 0.373 | 0.574 |
| scGPT (fine-tuned) | 0.670 | 0.427 | 0.573 |
| scVI | 0.635 | 0.458 | 0.565 |
| cellPLM\* | 0.666 | 0.399 | 0.559 |
| scGPT (zero-shot) | 0.653 | 0.405 | 0.554 |
| scFoundation | 0.647 | 0.373 | 0.538 |
| Geneformer | 0.618 | 0.400 | 0.531 |

\*computed, excluded from `metrics_reported.csv` and the slide.

- **Headline:** scGPT zero-shot → fine-tuned improves **Batch** 0.405→0.427
  *and* **Bio** 0.653→0.670 (no trade-off); Total 0.554→0.573.
- **Sanity:** scVI Batch (0.458) ≥ PCA Batch (0.373) — a batch-correcting
  method beats unintegrated, as it must.

## Caveats

- **Within-study batch.** `orig.ident` is 11 within-study samples — a *milder*
  batch effect than a cross-protocol atlas. Batch-correction scores look high
  / compressed across the board; the signal is *relative* movement (esp.
  scGPT zs↔ft and vs the PCA baseline), not absolute Totals. The scIB-paper
  "strong" band (0.6–0.75 Total) applies to hard atlases, not this dataset.
- **Not a clean ablation.** scGPT fine-tune also adapts in-domain and uses a
  1200-HVG set, so attribute gains to the recipe as a whole, not DAB/DSBN alone.
- **PCR comparison clipped to 0 for all models** (`scib-metrics` zeroes a
  negative PCR). On this mild-batch dataset PCR is uninformative and uniformly
  depresses every model's Batch axis equally — it does not affect ranking.
- **cASW/clustering subsumption.** ARI/NMI from the old clustering slide live
  inside scIB's Bio-conservation axis; the integration slide is a strict
  superset (it also reports batch removal).
- cellPLM is run-but-not-reported by request.
