# %% [markdown]
# # scANVI (scvi-tools) — Cell-Type Annotation Wrapper (MS dataset)
#
# Folder is named `scVI-annotation-wrapper` for parallel naming with `embd_clustering/`,
# but the model used here is **scANVI** — the canonical scvi-tools recipe for supervised
# CT annotation.
#
# Pipeline:
#   1. Concat train + test into one AnnData; set `labels_scanvi` to known celltype for
#      train rows and `"Unknown"` for test rows.
#   2. Reconstruct pseudo-counts from log-normed `.X` via `expm1 + round` (see deviation
#      note inline); register them as the `"counts"` layer for scvi-tools.
#   3. `scvi.model.SCVI.setup_anndata(..., batch_key="str_batch", layer="counts")`;
#      train scVI 20 epochs.
#   4. `scvi.model.SCANVI.from_scvi_model(..., unlabeled_category="Unknown",
#      labels_key="labels_scanvi")`; train scANVI 20 epochs, `n_samples_per_label=100`.
#   5. Predict labels for test rows; write `ms_annotation.h5ad`.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvi
import torch
from sklearn.metrics import accuracy_score, f1_score

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
scvi.settings.seed = SEED

# %%
# --- Load MS data and tag train/test by batch ---
train = ad.read_h5ad("/data/benchmark/data/cellPLM/data/c_data.h5ad")
test = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")

# c_data uses gene symbols as var_names; filtered_ms_adata uses ENSG ids.
# Both carry an `index_column` var entry (ENSG ids). Reindex both to it so
# `ad.concat(join="inner")` actually finds shared genes — without this, the
# inner join is empty and scANVI collapses to one class.
train.var = train.var.set_index("index_column")
test.var = test.var.set_index("index_column")

LABEL_COL = "Factor Value[inferred cell type - authors labels]"
train.obs["celltype"] = train.obs[LABEL_COL].astype(str)
test.obs["celltype"] = test.obs[LABEL_COL].astype(str)
train.obs["str_batch"] = "0"
test.obs["str_batch"] = "1"

# scANVI requires "Unknown" sentinel for unlabeled rows.
train.obs["labels_scanvi"] = train.obs["celltype"].astype(str)
test.obs["labels_scanvi"] = "Unknown"

# Preserve original test obs_names for the output file.
test_obs_names = test.obs_names.tolist()
test_celltype_strs = test.obs["celltype"].astype(str).values.copy()

# Concatenate. Make obs_names unique (concat appends batch suffix automatically).
adata = ad.concat([train, test], join="inner", label="orig", keys=["train", "test"])
adata.obs_names_make_unique()

# scvi-tools expects raw counts (NB/ZINB likelihood). c_data + filtered_ms_adata are
# already log-normed (log1p(normalize_total(X, 1e4))). Feeding that directly causes
# scANVI to collapse (all cells predicted as one class). **Deviation from the upstream
# scvi-tools tutorial:** reconstruct pseudo-counts by inverting the log1p transform
# (expm1) and rounding to integers. This is an information-lossy reverse — we get
# normalize_total-scaled integer "counts" (sum ≈ 1e4 per cell) rather than the true
# per-droplet raw counts — but they are count-distributed and scANVI's likelihood
# model works on them. The Poisson assumption is exact at the normalize_total scale.
from scipy.sparse import issparse, csr_matrix
X = adata.X.toarray() if issparse(adata.X) else adata.X
counts = np.rint(np.expm1(np.asarray(X, dtype=np.float64))).astype(np.int32)
counts[counts < 0] = 0
adata.layers["counts"] = csr_matrix(counts)
print("adata:", adata.shape, "labels:", adata.obs["labels_scanvi"].value_counts().head())

# %%
# --- Train scVI ---
scvi.model.SCVI.setup_anndata(adata, batch_key="str_batch", layer="counts")
vae = scvi.model.SCVI(adata, n_latent=10, n_layers=2, n_hidden=128, dropout_rate=0.1)
vae.train(max_epochs=20, check_val_every_n_epoch=5, early_stopping=False)

# %%
# --- Build scANVI from scVI; train ---
lvae = scvi.model.SCANVI.from_scvi_model(
    vae,
    adata=adata,
    unlabeled_category="Unknown",
    labels_key="labels_scanvi",
)
lvae.train(max_epochs=20, n_samples_per_label=100, check_val_every_n_epoch=5)

# %%
# --- Predict + write ---
pred_full = lvae.predict(adata)  # array of label strings over the concatenated adata

# Pull out the test slice. adata.obs['orig'] tags which file each row came from.
test_mask = adata.obs["orig"].values == "test"
test_preds = np.asarray(pred_full)[test_mask]

assert len(test_preds) == len(test_obs_names), f"{len(test_preds)} vs {len(test_obs_names)}"

out = ad.AnnData(np.zeros((len(test_obs_names), 1), dtype=np.float32))
out.obs_names = test_obs_names
out.obs["celltype"] = test_celltype_strs
out.obs["predictions"] = pd.Categorical(test_preds)

OUT_PATH = "/data/benchmark/ct_annotation/scVI-annotation-wrapper/ms_annotation.h5ad"
out.write_h5ad(OUT_PATH)

acc = accuracy_score(out.obs["celltype"], out.obs["predictions"])
macro_f1 = f1_score(out.obs["celltype"], out.obs["predictions"], average="macro")
print(f"scANVI  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  wrote {OUT_PATH}")
