# %% [markdown]
# # scFoundation — Cell-Type Annotation Wrapper (MS dataset)
#
# Faithful re-implementation of the `LinearProbingClassifier` in
# `models/scFoundation/model/finetune_model.py` (lines 88-148).
#
# **Freezing pattern (verbatim from the script):**
# - `token_emb` frozen, `pos_emb` frozen
# - All `encoder` frozen, **then** `encoder.transformer_encoder[-2]` selectively unfrozen
#   (only this one transformer layer of the backbone is trainable)
# - Head: max-pool over seq dim → `BatchNorm1d(768, affine=False, eps=1e-6)` →
#   `Linear(768, 256)` → `ReLU` → `Linear(256, n_class)`
# - Input padded to 19264 genes via `OS_scRNA_gene_index.19264.tsv`
#
# **Hyperparameters (NOT pinned in the public scFoundation repo — flagged as deviation):**
#   `lr=1e-4, AdamW, weight_decay=0, batch_size=8, epochs=10, seed=0`.

# %%
import os, sys, random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, f1_score

# scFoundation requires its own module path on PYTHONPATH (load.py + finetune_model.py).
SCF_ROOT = "/data/benchmark/models/scFoundation/model"
sys.path.insert(0, SCF_ROOT)
from load import load_model_frommmf, gatherData  # type: ignore
from finetune_model import LinearProbingClassifier  # type: ignore

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
# --- Gene padding to 19264 (scFoundation's fixed input dimension) ---
GENE_INDEX_TSV = Path(SCF_ROOT) / "OS_scRNA_gene_index.19264.tsv"
gene_list_df = pd.read_csv(GENE_INDEX_TSV, header=0, delimiter="\t")
TARGET_GENES = gene_list_df["gene_name"].tolist()
print(f"target gene panel: {len(TARGET_GENES)} genes")

def main_gene_selection(X_df: pd.DataFrame, gene_list: list[str]) -> pd.DataFrame:
    # Same logic as `get_embedding.py::main_gene_selection`: pad missing genes with zeros,
    # then reindex.
    missing = list(set(gene_list) - set(X_df.columns))
    pad = pd.DataFrame(np.zeros((X_df.shape[0], len(missing))), columns=missing, index=X_df.index)
    X_df = pd.concat([X_df, pad], axis=1)
    return X_df[gene_list]

LABEL_COL = "Factor Value[inferred cell type - authors labels]"
train_ad = sc.read("/data/benchmark/data/cellPLM/data/c_data.h5ad")
test_ad = sc.read("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
train_ad.obs["celltype"] = train_ad.obs[LABEL_COL].astype(str)
test_ad.obs["celltype"] = test_ad.obs[LABEL_COL].astype(str)
test_obs_names = test_ad.obs_names.tolist()

def to_df(a):
    # scFoundation expects gene names as columns. `c_data` / `filtered_ms_adata` are
    # keyed by ensembl IDs but carry a `gene_name` var column.
    if "gene_name" in a.var.columns:
        cols = a.var["gene_name"].astype(str).tolist()
    else:
        cols = a.var_names.tolist()
    X = a.X.toarray() if issparse(a.X) else a.X
    return pd.DataFrame(X, index=a.obs_names, columns=cols)

train_df = main_gene_selection(to_df(train_ad), TARGET_GENES)
test_df = main_gene_selection(to_df(test_ad), TARGET_GENES)
print("train:", train_df.shape, "test:", test_df.shape)

# %%
# --- Label encoding ---
celltype_cats = sorted(set(train_ad.obs["celltype"].astype(str).unique())
                      | set(test_ad.obs["celltype"].astype(str).unique()))
label_to_int = {c: i for i, c in enumerate(celltype_cats)}
int_to_label = {i: c for c, i in label_to_int.items()}
y_train = np.array([label_to_int[c] for c in train_ad.obs["celltype"].astype(str)])
y_test = np.array([label_to_int[c] for c in test_ad.obs["celltype"].astype(str)])
n_class = len(celltype_cats)
print(f"n_class = {n_class}")

# %%
# --- Build LinearProbingClassifier with n_class output, override the head ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = Path(SCF_ROOT) / "models" / "models.ckpt"

clf = LinearProbingClassifier(ckpt_path=str(CKPT), frozenmore=True)
clf.build()
# The script's __init__ hard-codes n_class=10; replace fc1 with the correct n_class.
hidden_dim = clf.model_config["encoder"]["hidden_dim"]  # 768
clf.fc1 = nn.Sequential(
    nn.Linear(hidden_dim, 256),
    nn.ReLU(),
    nn.Linear(256, n_class),
)
clf = clf.to(device)

# Confirm the freezing pattern matches the script (token_emb frozen, pos_emb frozen,
# encoder frozen except transformer_encoder[-2]).
trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
print(f"trainable params: {len(trainable)} groups; first 3: {trainable[:3]}")

# %%
# --- DataLoaders (scFoundation forward expects sample_list dict with key 'x') ---
X_train_t = torch.from_numpy(train_df.values.astype(np.float32))
X_test_t = torch.from_numpy(test_df.values.astype(np.float32))
y_train_t = torch.from_numpy(y_train).long()
y_test_t = torch.from_numpy(y_test).long()

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

# %%
# --- Train ---
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, clf.parameters()), lr=LR, weight_decay=0)
ce = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    clf.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = clf({"x": xb, "targets": yb})
        loss = ce(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
    print(f"epoch {epoch:2d}  train_loss={total_loss/total_n:.4f}")

# %%
# --- Predict on test ---
clf.eval()
preds = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = clf({"x": xb, "targets": yb.to(device)})
        preds.append(logits.argmax(1).cpu().numpy())
pred_idx = np.concatenate(preds)
pred_strs = np.array([int_to_label[int(i)] for i in pred_idx])
assert len(pred_strs) == len(test_obs_names)

out = ad.AnnData(np.zeros((len(test_obs_names), 1), dtype=np.float32))
out.obs_names = test_obs_names
out.obs["celltype"] = test_ad.obs["celltype"].astype(str).values
out.obs["predictions"] = pd.Categorical(pred_strs)

OUT_PATH = "/data/benchmark/ct_annotation/scFoundation-annotation-wrapper/ms_annotation.h5ad"
out.write_h5ad(OUT_PATH)

acc = accuracy_score(out.obs["celltype"], out.obs["predictions"])
macro_f1 = f1_score(out.obs["celltype"], out.obs["predictions"], average="macro")
print(f"scFoundation  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  wrote {OUT_PATH}")
