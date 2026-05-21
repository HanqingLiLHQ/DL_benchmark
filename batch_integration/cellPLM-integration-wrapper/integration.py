# %% [markdown]
# # cellPLM — Batch-Integration Wrapper  (RUN BUT NOT REPORTED)
#
# Faithful recipe: zero-shot `CellEmbeddingPipeline` — the paper's own
# integration approach. We re-emit `embd_clustering/cellPLM-embedding-wrapper`
# under the integration output schema. Excluded from `metrics_reported.csv` and
# the deck slide by request, but kept for parity / reproducibility.
#
# Output schema (every wrapper):
#   obs_names           : aligned with gse155468.h5ad after obs_names_make_unique()
#   obs['celltype'] str : bio label, scoring only
#   obs['batch']    str : = orig.ident, method input / metric grouping only
#   obsm['X_emb']       : (n, d) float32

# %%
import hdf5plugin  # noqa: F401 — registers .h5ad codecs
import numpy as np
import anndata as ad

SRC_EMBED = "/data/benchmark/embd_clustering/cellPLM-embedding-wrapper/gse155468_embedding.h5ad"
GROUND_TRUTH = "/data/benchmark/data/cellPLM/data/gse155468.h5ad"
OUT = "gse155468_integration.h5ad"

src = ad.read_h5ad(SRC_EMBED)
emb = np.asarray(src.X, dtype=np.float32)
obs_names = [str(x) for x in src.obs_names]

gt = ad.read_h5ad(GROUND_TRUTH)
gt.obs_names = gt.obs_names.astype(str)
gt.obs_names_make_unique()
labels = gt.obs.loc[obs_names]

out = ad.AnnData(
    X=np.zeros((emb.shape[0], 1), dtype=np.float32),
    obs={"celltype": labels["celltype"].astype(str).values,
         "batch":    labels["orig.ident"].astype(str).values},
)
out.obs_names = obs_names
out.obsm["X_emb"] = emb
assert set(out.obs_names) == set(gt.obs_names), "obs_names diverge from gse155468.h5ad"
out.write_h5ad(OUT)
print(f"wrote {OUT}  emb={emb.shape}  batches={out.obs['batch'].nunique()}")
