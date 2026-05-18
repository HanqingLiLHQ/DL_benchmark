# %% [markdown]
# # Geneformer — Cell-Type Annotation Wrapper (MS dataset)
#
# Recipe from `models/Geneformer/examples/cell_classification.ipynb`, with `training_args`
# **re-tuned for cell-type classification** using the manual hyperparameters from
# `models/Geneformer/examples/multitask_cell_classification.ipynb` (which explicitly
# targets a `cell_type` task). The Classifier API is unchanged.
#
# **Original cardiomyopathy hyperparameters (kept for provenance):**
#   `training_args = {num_train_epochs: 0.9, learning_rate: 8.04e-4,
#                     lr_scheduler_type: "polynomial", warmup_steps: 1812,
#                     weight_decay: 0.258828, per_device_train_batch_size: 12}`
# These collapse to ~random accuracy on celltype (0.9 epoch insufficient + warmup
# eats most of training + LR schedule designed for binary task).
#
# **Re-tuned hyperparameters (used here; sourced from multitask_cell_classification.ipynb
# `manual_hyperparameters`):**
#   `num_train_epochs: 10, learning_rate: 1e-3, lr_scheduler_type: "cosine",
#    warmup_ratio: 0.01, weight_decay: 0.1, per_device_train_batch_size: 32,
#    freeze_layers: 2 (== "max_layers_to_freeze": 2 in the MTL tutorial)`

# %%
import os, sys, shutil, tempfile, datetime, pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "/data/benchmark/models/Geneformer")
from geneformer import Classifier, TranscriptomeTokenizer

# %%
SEED = 73  # Geneformer tutorial seed
WORK = Path("/tmp/geneformer_ms")
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "h5ad").mkdir(exist_ok=True)

LABEL_COL = "Factor Value[inferred cell type - authors labels]"

train = ad.read_h5ad("/data/benchmark/data/cellPLM/data/c_data.h5ad")
test = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
train.obs["celltype"] = train.obs[LABEL_COL].astype(str)
test.obs["celltype"] = test.obs[LABEL_COL].astype(str)
# Geneformer expects an "ensembl_id" var column. Source-of-truth: `index_column`
# holds ENSG ids in both files (train.var_names are gene symbols, test.var_names are
# ENSG ids — only the index_column column is consistent across the two).
train.var["ensembl_id"] = train.var["index_column"].astype(str).tolist()
test.var["ensembl_id"] = test.var["index_column"].astype(str).tolist()
# Geneformer tokenizer requires `n_counts` in obs.
import scipy.sparse as _sp
def _row_sums(a):
    return np.asarray(a.X.sum(axis=1)).flatten() if _sp.issparse(a.X) else a.X.sum(axis=1)
train.obs["n_counts"] = _row_sums(train).astype(np.float32)
test.obs["n_counts"] = _row_sums(test).astype(np.float32)
# Tag rows with origin so we can split train/test later.
train.obs["source"] = "train"
test.obs["source"] = "test"
# Within the training source, mark a 90/10 train/eval fold for Classifier.validate().
# (The tutorial uses individual-ID-based splits; here we use a deterministic 90/10
# since MS has no individual-ID structure exposed.)
n_train = train.n_obs
rng = np.random.default_rng(SEED)
perm = rng.permutation(n_train)
fold_arr = np.array(["train"] * n_train, dtype=object)
fold_arr[perm[int(0.9 * n_train):]] = "eval"
train.obs["fold"] = fold_arr
test.obs["fold"] = "test"  # placeholder for the test rows; not used by validate()
# Tokenization permutes row order, so we propagate obs_names as a custom attr to
# realign predictions back to filtered_ms_adata.h5ad's order after evaluation.
train.obs["obs_name"] = train.obs_names.astype(str).tolist()
test.obs["obs_name"] = test.obs_names.astype(str).tolist()
test_obs_names = test.obs_names.tolist()
test_celltype_strs = test.obs["celltype"].astype(str).values.copy()

# Geneformer tokenizer requires raw counts as `n_counts` in obs (it computes if absent).
# Save the two files in a single directory; the tokenizer scans the directory.
train.write_h5ad(WORK / "h5ad" / "train.h5ad")
test.write_h5ad(WORK / "h5ad" / "test.h5ad")

# %%
# --- Tokenize both files to HF dataset format ---
tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict={"celltype": "celltype", "source": "source", "fold": "fold", "obs_name": "obs_name"},
    nproc=16,
    model_version="V1",  # matches the V1-10M checkpoint
)
tokenizer.tokenize_data(
    data_directory=str(WORK / "h5ad"),
    output_directory=str(WORK),
    output_prefix="ms_tokenized",
    file_format="h5ad",
)
TOKENIZED = WORK / "ms_tokenized.dataset"
print("tokenized dataset:", TOKENIZED)

# %%
# --- Train via Classifier.validate (verbatim training_args from tutorial) ---
training_args = {
    "num_train_epochs": 10,
    "learning_rate": 1e-3,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.01,
    "weight_decay": 0.1,
    "per_device_train_batch_size": 32,
    "seed": SEED,
}

cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "celltype", "states": "all"},
    training_args=training_args,
    max_ncells=None,
    freeze_layers=2,
    num_crossval_splits=1,
    forward_batch_size=200,
    model_version="V1",
    nproc=16,
)

# Split by source attribute: train rows → train, test rows → test.
split_dict = {"attr_key": "source", "train": ["train"], "test": ["test"]}
output_prefix = "ms_classifier"
cc.prepare_data(
    input_data_file=str(TOKENIZED),
    output_directory=str(WORK),
    output_prefix=output_prefix,
    split_id_dict=split_dict,
)

# For validate(), split the training source rows into disjoint train + eval folds
# (set up on each train row's `fold` attribute above).
train_eval_split = {"attr_key": "fold", "train": ["train"], "eval": ["eval"]}

cc.validate(
    model_directory="/data/benchmark/models/Geneformer/Geneformer-V1-10M",
    prepared_input_data_file=f"{WORK}/{output_prefix}_labeled_train.dataset",
    id_class_dict_file=f"{WORK}/{output_prefix}_id_class_dict.pkl",
    output_directory=str(WORK),
    output_prefix=output_prefix,
    split_id_dict=train_eval_split,
)

# %%
# --- Evaluate on test split, save predictions ---
# Find the fine-tuned model directory (the validate() call writes it under
# `{output_dir}/{datestamp_min}_geneformer_cellClassifier_{output_prefix}/ksplit1/`).
candidates = sorted(WORK.glob(f"*_geneformer_cellClassifier_{output_prefix}/ksplit1"))
assert candidates, f"could not find fine-tuned model under {WORK}"
fine_tuned = candidates[-1]
print("fine_tuned model:", fine_tuned)

eval_cc = Classifier(
    classifier="cell",
    cell_state_dict={"state_key": "celltype", "states": "all"},
    forward_batch_size=200,
    nproc=16,
)
all_metrics_test = eval_cc.evaluate_saved_model(
    model_directory=str(fine_tuned),
    id_class_dict_file=f"{WORK}/{output_prefix}_id_class_dict.pkl",
    test_data_file=f"{WORK}/{output_prefix}_labeled_test.dataset",
    output_directory=str(WORK),
    output_prefix=output_prefix,
)

# Load the prediction pickle.
with open(f"{WORK}/{output_prefix}_pred_dict.pkl", "rb") as f:
    pred_dict = pickle.load(f)
with open(f"{WORK}/{output_prefix}_id_class_dict.pkl", "rb") as f:
    id_class = pickle.load(f)

# Geneformer's pred_dict has three keys in this version:
#   'pred_ids' (1D int class index per cell), 'label_ids' (1D true class index),
#   'predictions' (2D raw logits — NOT the class index).
# Use 'pred_ids'. id_class is {int_id: label_str}.
pred_idx = np.asarray(pred_dict["pred_ids"])
pred_strs = np.array([id_class[int(i)] for i in pred_idx])

# %%
# --- Align predictions back to filtered_ms_adata obs_names ---
# Geneformer's tokenizer + prepare_data permute the row order: pred_strs[i]
# corresponds to whatever cell is at row i of the prepared test dataset, NOT row i
# of filtered_ms_adata.h5ad. We tracked obs_name as a custom_attr so the prepared
# test dataset carries the original cell IDs. Use them to reorder pred_strs back.
from datasets import load_from_disk
test_ds = load_from_disk(f"{WORK}/{output_prefix}_labeled_test.dataset")
geneformer_obs_names = list(test_ds["obs_name"])
assert len(geneformer_obs_names) == len(pred_strs) == len(test_obs_names), (
    f"length mismatch: ds={len(geneformer_obs_names)}, preds={len(pred_strs)}, "
    f"expected={len(test_obs_names)}"
)
pred_map = dict(zip(geneformer_obs_names, pred_strs))
missing = [n for n in test_obs_names if n not in pred_map]
assert not missing, f"{len(missing)} obs_names missing from Geneformer output (first: {missing[:3]})"
pred_strs_aligned = np.array([pred_map[n] for n in test_obs_names])

out = ad.AnnData(np.zeros((len(test_obs_names), 1), dtype=np.float32))
out.obs_names = test_obs_names
out.obs["celltype"] = test_celltype_strs
out.obs["predictions"] = pd.Categorical(pred_strs_aligned)

OUT_PATH = "/data/benchmark/ct_annotation/Geneformer-annotation-wrapper/ms_annotation.h5ad"
out.write_h5ad(OUT_PATH)

acc = accuracy_score(out.obs["celltype"], out.obs["predictions"])
macro_f1 = f1_score(out.obs["celltype"], out.obs["predictions"], average="macro")
print(f"Geneformer  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  wrote {OUT_PATH}")
