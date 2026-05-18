# %% [markdown]
# # cellPLM — Cell-Type Annotation Wrapper (MS dataset)
#
# Verbatim recipe from `models/cellPLM/tutorials/cell_type_annotation.ipynb`:
# `CellTypeAnnotationPipeline` with `PRETRAIN_VERSION='20230926_85M'`, `set_seed(42)`,
# default pipeline + model configs. Only override: `model_config['out_dim'] = n_classes`.
#
# Adaptations: at the end, slice predicted AnnData to `split == 'test'` and write
# `ms_annotation.h5ad` with `obs_names` aligned to `filtered_ms_adata.h5ad`.
#
# **Deviation from tutorial:** the upstream tutorial declares `DEVICE = 'cuda:3'` at
# module level but never passes it to `pipeline.fit()` / `pipeline.predict()`, both of
# which default to `device='cpu'`. We pass `device=DEVICE` explicitly so training runs
# on GPU as the paper intends; running on CPU is infeasible at our patience budget.

# %%
import sys, os
sys.path.insert(0, "/data/benchmark/models/cellPLM")

import warnings
warnings.filterwarnings("ignore")

import hdf5plugin
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_type_annotation import (
    CellTypeAnnotationPipeline,
    CellTypeAnnotationDefaultPipelineConfig,
    CellTypeAnnotationDefaultModelConfig,
)

# %%
PRETRAIN_VERSION = "20230926_85M"   # exact version used in the cellPLM CT tutorial
DEVICE = "cuda:0"
set_seed(42)                         # exact seed used in the tutorial

# %%
# --- Load MS dataset (mirrors the cellPLM tutorial's MS branch) ---
data_train = ad.read_h5ad("/data/benchmark/data/cellPLM/data/c_data.h5ad")
data_test = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")

# Both files carry an `index_column` var entry per the cellPLM tutorial.
data_train.var = data_train.var.set_index("index_column")
data_test.var = data_test.var.set_index("index_column")

# Save the test obs_names before concatenation (we'll need these for the output file).
test_obs_names = data_test.obs_names.tolist()

train_num = data_train.shape[0]
data = ad.concat([data_train, data_test])
data.var_names_make_unique()

# The MS files store the label under `Factor Value[inferred cell type - authors labels]`.
# CellPLM expects `obs['celltype']` — normalize.
if "celltype" not in data.obs.columns:
    data.obs["celltype"] = data.obs["Factor Value[inferred cell type - authors labels]"].astype("category")

# Split: 90% of training cells = train, 10% = valid, all test cells = test.
# (Tutorial uses chained-indexing assignment; we use array assignment to avoid
# SettingWithCopyWarning / pandas 2.x silent-no-op behavior.)
split_array = np.array(["test"] * len(data), dtype=object)
tr = np.random.permutation(train_num)
split_array[tr[:int(train_num * 0.9)]] = "train"
split_array[tr[int(train_num * 0.9):train_num]] = "valid"
data.obs["split"] = split_array

# %%
# --- Configs (verbatim; only override out_dim) ---
pipeline_config = CellTypeAnnotationDefaultPipelineConfig.copy()
model_config = CellTypeAnnotationDefaultModelConfig.copy()
model_config["out_dim"] = data.obs["celltype"].nunique()
print("pipeline_config:", pipeline_config)
print("model_config:", model_config)

# %%
# --- Fit + predict (canonical pipeline API) ---
pipeline = CellTypeAnnotationPipeline(
    pretrain_prefix=PRETRAIN_VERSION,
    overwrite_config=model_config,
    pretrain_directory="/data/benchmark/data/cellPLM/ckpt",
)

pipeline.fit(
    data,
    pipeline_config,
    split_field="split",
    train_split="train",
    valid_split="valid",
    label_fields=["celltype"],
    device=DEVICE,
)

predictions = pipeline.predict(data, pipeline_config, device=DEVICE)
# `predictions` is a 1-D array of label indices over the full `data` (train+valid+test).

# %%
# --- Build ms_annotation.h5ad aligned to filtered_ms_adata.h5ad's obs_names ---
# Map prediction indices back to label strings using the celltype category ordering.
celltype_cats = data.obs["celltype"].astype("category").cat.categories.tolist()

import torch as _torch
if isinstance(predictions, _torch.Tensor):
    predictions = predictions.detach().cpu().numpy()

if hasattr(predictions, "argmax"):
    # Predictions may be returned as logits / probabilities.
    if predictions.ndim == 2:
        pred_idx = predictions.argmax(axis=1)
    else:
        pred_idx = predictions
else:
    pred_idx = np.asarray(predictions)

pred_idx = np.asarray(pred_idx).astype(int)
pred_strs = np.array([celltype_cats[i] for i in pred_idx])

# Pick the test rows.
test_mask = data.obs["split"].values == "test"
test_pred_strs = pred_strs[test_mask]
test_celltype_strs = data.obs["celltype"].astype(str).values[test_mask]

# Reload the original test file to get canonical obs_names ordering.
data_test_raw = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
assert len(test_pred_strs) == data_test_raw.n_obs, (
    f"len(predictions on test split)={len(test_pred_strs)} != n_obs={data_test_raw.n_obs}"
)

out = ad.AnnData(np.zeros((data_test_raw.n_obs, 1), dtype=np.float32))
out.obs_names = data_test_raw.obs_names
out.obs["celltype"] = test_celltype_strs
out.obs["predictions"] = pd.Categorical(test_pred_strs)

OUT_PATH = "/data/benchmark/ct_annotation/cellPLM-annotation-wrapper/ms_annotation.h5ad"
out.write_h5ad(OUT_PATH)

acc = accuracy_score(out.obs["celltype"], out.obs["predictions"])
macro_f1 = f1_score(out.obs["celltype"], out.obs["predictions"], average="macro")
print(f"cellPLM  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  wrote {OUT_PATH}")
