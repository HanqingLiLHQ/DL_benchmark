# %% [markdown]
# Batch-integration benchmark - scib-metrics on GSE155468 (batch=orig.ident).
# Label firewall: batch_key only groups/inputs; label_key (celltype) only scores.
# cellPLM is computed but EXCLUDED from the reported slice (run-not-reported).

# %%
import sys
sys.path.insert(0, "/data/benchmark/batch_integration")
import hdf5plugin, numpy as np, scanpy as sc, anndata as ad, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import jax
from scib_metrics.benchmark import Benchmarker
import _fix_silhouette_cdist  # memory-efficient jax cdist (see module docstring)

SRC = "/data/benchmark/data/cellPLM/data/gse155468.h5ad"
BI = "/data/benchmark/batch_integration"
WRAPPERS = {  # display name -> wrapper dir
    "scVI": "scVI", "scGPT (zero-shot)": "scGPT-zeroshot",
    "scGPT (fine-tuned)": "scGPT-finetuned", "Geneformer": "Geneformer",
    "scFoundation": "scFoundation", "cellPLM": "cellPLM",
}

# %% Base AnnData: labels + a PCA baseline embedding (unintegrated reference)
base = sc.read_h5ad(SRC)
base.obs_names = base.obs_names.astype(str)
base.obs_names_make_unique()
base.obs["celltype"] = base.obs["celltype"].astype(str)
base.obs["batch"] = base.obs["orig.ident"].astype(str)
pp = base.copy(); pp.X = pp.X.astype(np.float32)
sc.pp.normalize_total(pp, target_sum=1e4); sc.pp.log1p(pp)
sc.pp.highly_variable_genes(pp, n_top_genes=4500); pp = pp[:, pp.var.highly_variable].copy()
sc.pp.scale(pp, max_value=10); sc.tl.pca(pp, n_comps=50, random_state=137)
base.obsm["PCA"] = pp.obsm["X_pca"].astype(np.float32)

# %% Attach every model embedding, aligned by obs_names
order = base.obs_names
for disp, wd in WRAPPERS.items():
    a = ad.read_h5ad(f"{BI}/{wd}-integration-wrapper/gse155468_integration.h5ad")
    a = a[order].copy()
    base.obsm[disp] = np.asarray(a.obsm["X_emb"], dtype=np.float32)
emb_keys = ["PCA"] + list(WRAPPERS.keys())
print("scoring:", emb_keys)

# %% scIB panel
# Replace scib-metrics' O(chunk*N*d) jax cdist with the algebraically
# identical O(chunk*N) form so scFoundation's 3072-d embedding fits in
# memory (stock OOMs at ~156 GB); values unchanged within FP tolerance.
_fix_silhouette_cdist.apply()
print("jax backend:", jax.default_backend(), "| devices:", jax.devices())
bm = Benchmarker(base, batch_key="batch", label_key="celltype",
                 embedding_obsm_keys=emb_keys, n_jobs=-1)
bm.benchmark()
res = bm.get_results(min_max_scale=False)
res.to_csv(f"{BI}/metrics.csv")
print(res)

# %% Reported slice (exclude cellPLM) for the deck
reported = res.drop(index=[i for i in ["cellPLM"] if i in res.index], errors="ignore")
reported.to_csv(f"{BI}/metrics_reported.csv")

# %% UMAP grid: scGPT zero-shot vs fine-tuned, colored by batch & celltype
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for col, key in enumerate(["scGPT (zero-shot)", "scGPT (fine-tuned)"]):
    sc.pp.neighbors(base, use_rep=key, random_state=137)
    sc.tl.umap(base, random_state=137)
    for row, color in enumerate(["batch", "celltype"]):
        sc.pl.umap(base, color=color, ax=axes[row, col], show=False,
                   title=f"{key} - {color}", frameon=False)
fig.tight_layout(); fig.savefig(f"{BI}/umap_batch_celltype.png", dpi=150, facecolor="white")
print("wrote metrics.csv, metrics_reported.csv, umap_batch_celltype.png")
