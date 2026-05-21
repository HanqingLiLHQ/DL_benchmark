# %% [markdown]
# # Batch-Integration Benchmark
#
# Consumes `gse155468_integration.h5ad` from each of the 6 wrappers, aligns by
# `obs_names`, scores the field-standard **scIB** panel via `scib-metrics`, and
# produces:
# 1. `metrics.csv` — all models.
# 2. `metrics_reported.csv` — deck slice (drops cellPLM, kept run-but-not-reported).
# 3. `umap_batch_celltype.png` — scGPT zero-shot vs fine-tuned, by batch & celltype.
#
# Label firewall: `batch` (= `orig.ident`) is method input / scIB grouping only;
# `celltype` is attached for scoring only — no model ever sees it.

# %%
import warnings
warnings.filterwarnings("ignore")

from functools import partial
from pathlib import Path
from typing import Literal

import hdf5plugin  # noqa: F401  (registers codecs used by the .h5ad files)
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scib_metrics.benchmark import Benchmarker

MODELS = {  # display name -> wrapper dir
    "scVI":               "scVI-integration-wrapper",
    "scGPT (zero-shot)":  "scGPT-zeroshot-integration-wrapper",
    "scGPT (fine-tuned)": "scGPT-finetuned-integration-wrapper",
    "Geneformer":         "Geneformer-integration-wrapper",
    "scFoundation":       "scFoundation-integration-wrapper",
    "cellPLM":            "cellPLM-integration-wrapper",
}
ROOT = Path("/data/benchmark/batch_integration")
SRC = "/data/benchmark/data/cellPLM/data/gse155468.h5ad"

# %%
# --- Base AnnData: labels + PCA baseline embedding (unintegrated reference) ---
base = sc.read_h5ad(SRC)
base.obs_names = base.obs_names.astype(str)
base.obs_names_make_unique()  # 234 duplicate ids in raw; all wrappers applied the same transform
base.obs["celltype"] = base.obs["celltype"].astype(str)
base.obs["batch"]    = base.obs["orig.ident"].astype(str)

pp = base.copy(); pp.X = pp.X.astype(np.float32)
sc.pp.normalize_total(pp, target_sum=1e4); sc.pp.log1p(pp)
sc.pp.highly_variable_genes(pp, n_top_genes=4500); pp = pp[:, pp.var.highly_variable].copy()
sc.pp.scale(pp, max_value=10); sc.tl.pca(pp, n_comps=50, random_state=137)
base.obsm["PCA"] = pp.obsm["X_pca"].astype(np.float32)

# %%
# --- Attach each wrapper's embedding, aligned by obs_names ---
for disp, wd in MODELS.items():
    a = ad.read_h5ad(ROOT / wd / "gse155468_integration.h5ad")
    a = a[base.obs_names].copy()
    base.obsm[disp] = np.asarray(a.obsm["X_emb"], dtype=np.float32)

emb_keys = ["PCA"] + list(MODELS.keys())
print("scoring:", emb_keys)

# %% [markdown]
# ## scib-metrics cdist patch (REQUIRED for scFoundation's 3072-d)
#
# `scib_metrics==0.5.9`'s jax `cdist` builds a dense `(chunk, N, d)` tensor
# before reducing over `d`. With `chunk_size=256`, N=48082 and scFoundation's
# d=3072 that intermediate is ~156 GB → OOM. We swap in the algebraically
# identical `‖a‖² + ‖b‖² − 2·a·b` form: a single BLAS matmul, O(chunk·N)
# memory. Silhouette per-sample values do not depend on `chunk_size` (per-chunk
# results are concatenated, never cross-reduced), so the scIB protocol and
# metric set are unchanged within float tolerance.

# %%
@partial(jax.jit, static_argnames=["metric"])
def cdist_lowmem(x: jnp.ndarray, y: jnp.ndarray,
                 metric: Literal["euclidean", "cosine"] = "euclidean") -> jnp.ndarray:
    if metric == "cosine":
        xx = jnp.sqrt(jnp.sum(x * x, axis=1))
        yy = jnp.sqrt(jnp.sum(y * y, axis=1))
        dist = 1.0 - (x @ y.T) / (xx[:, None] * yy[None, :])
        return jnp.clip(dist, 0.0, 2.0)
    if metric != "euclidean":
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    xx = jnp.sum(x * x, axis=1)
    yy = jnp.sum(y * y, axis=1)
    sq = xx[:, None] + yy[None, :] - 2.0 * (x @ y.T)
    return jnp.sqrt(jnp.maximum(sq, 0.0))

# scib_metrics.utils._silhouette does `from ._dist import cdist`, so the name
# must be patched in *that* module (not only in _dist); both are rebound.
import scib_metrics.utils._dist as _dist_mod
import scib_metrics.utils._silhouette as _sil_mod
_dist_mod.cdist = cdist_lowmem
_sil_mod.cdist  = cdist_lowmem
print("jax backend:", jax.default_backend(), "| devices:", jax.devices())

# %%
# --- scIB panel ---
bm = Benchmarker(base, batch_key="batch", label_key="celltype",
                 embedding_obsm_keys=emb_keys, n_jobs=-1)
bm.benchmark()
res = bm.get_results(min_max_scale=False)
res.to_csv(ROOT / "metrics.csv")
print(res)

# %%
# --- Reported slice (exclude cellPLM) for the deck ---
reported = res.drop(index=[i for i in ["cellPLM"] if i in res.index], errors="ignore")
reported.to_csv(ROOT / "metrics_reported.csv")

# %%
# --- UMAP grid: scGPT zero-shot vs fine-tuned, colored by batch & celltype ---
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for col, key in enumerate(["scGPT (zero-shot)", "scGPT (fine-tuned)"]):
    sc.pp.neighbors(base, use_rep=key, random_state=137)
    sc.tl.umap(base, random_state=137)
    for row, color in enumerate(["batch", "celltype"]):
        sc.pl.umap(base, color=color, ax=axes[row, col], show=False,
                   title=f"{key} - {color}", frameon=False)
fig.tight_layout(); fig.savefig(ROOT / "umap_batch_celltype.png", dpi=150, facecolor="white")
print("wrote metrics.csv, metrics_reported.csv, umap_batch_celltype.png")
