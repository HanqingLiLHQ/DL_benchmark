# %% [markdown]
# # scGPT — Cell-Type Annotation Wrapper (MS dataset)
#
# Faithful re-implementation of `models/scGPT/tutorials/Tutorial_Annotation.ipynb`.
# Strips wandb + plotting; writes a uniform `ms_annotation.h5ad` for the benchmark.
#
# **Hyperparameters (verbatim from tutorial):** `epochs=10`, `lr=1e-4`, `batch_size=32`,
# `mask_ratio=0.0`, `n_bins=51`, `max_seq_len=3001`, `freeze=False`, `DSBN=False`,
# `include_zero_gene=False`, `input_style="binned"`, `cell_emb_style="cls"`, `amp=True`.
# Seed = 0.

# %%
import copy
import gc
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/data/benchmark/models/scGPT")
import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"
sc.set_figure_params(figsize=(6, 6))

# %%
# --- Hyperparameters (verbatim from Tutorial_Annotation.ipynb) ---
SEED = 0
DATASET_NAME = "ms"
LOAD_MODEL = "/data/benchmark/models/scGPT/save/scGPT_human"
MASK_RATIO = 0.0
EPOCHS = 10
N_BINS = 51
MVC = False
ECS_THRES = 0.0
DAB_WEIGHT = 0.0
LR = 1e-4
BATCH_SIZE = 32
LAYER_SIZE = 128
NLAYERS = 4
NHEAD = 4
DROPOUT = 0.2
SCHEDULE_RATIO = 0.9
FAST_TRANSFORMER = True
PRE_NORM = False
AMP = True
INCLUDE_ZERO_GENE = False
FREEZE = False
DSBN = False

# Derived (from tutorial)
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
MASK_VALUE = -1
PAD_VALUE = -2
N_INPUT_BINS = N_BINS
MAX_SEQ_LEN = 3001
INPUT_STYLE = "binned"
CLS = True
MLM = False
CCE = False
ECS = ECS_THRES > 0
DAB = False
INPUT_BATCH_LABELS = False
INPUT_EMB_STYLE = "continuous"
CELL_EMB_STYLE = "cls"
MVC_DECODER_STYLE = "inner product"
EXPLICIT_ZERO_PROB = MLM and INCLUDE_ZERO_GENE
DO_SAMPLE_IN_TRAIN = False and EXPLICIT_ZERO_PROB
PER_SEQ_BATCH_SAMPLE = False
LR_ADV = 1e-3
SCHEDULE_INTERVAL = 1
LOG_INTERVAL = 100

set_seed(SEED)

# %%
# --- Load MS dataset (same logic as tutorial) ---
DATA_DIR = Path("/data/benchmark/data/cellPLM/data")
adata = sc.read(DATA_DIR / "c_data.h5ad")
adata_test = sc.read(DATA_DIR / "filtered_ms_adata.h5ad")
adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"
adata.var.set_index(adata.var["gene_name"], inplace=True)
adata_test.var.set_index(adata.var["gene_name"], inplace=True)

# Remember the test-cell obs_names BEFORE concatenation (concatenation appends a batch suffix).
test_obs_names = adata_test.obs_names.tolist()
adata_test_raw = adata_test.copy()

adata = adata.concatenate(adata_test, batch_key="str_batch")

batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# %%
# --- Load pretrained scGPT model config + vocab (same as tutorial) ---
model_dir = Path(LOAD_MODEL)
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in SPECIAL_TOKENS:
    if s not in vocab:
        vocab.append_token(s)

adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
adata = adata[:, adata.var["id_in_vocab"] >= 0]

with open(model_config_file) as f:
    model_configs = json.load(f)
EMBSIZE = model_configs["embsize"]
NHEAD_M = model_configs["nheads"]
D_HID = model_configs["d_hid"]
NLAYERS_M = model_configs["nlayers"]
N_LAYERS_CLS = model_configs["n_layers_cls"]

vocab.set_default_index(vocab["<pad>"])

# %%
# --- Preprocess (tutorial settings) ---
preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=False,           # MS data is already log-normed (data_is_raw=False in tutorial)
    result_log1p_key="X_log1p",
    subset_hvg=False,
    hvg_flavor="cell_ranger",
    binning=N_BINS,
    result_binned_key="X_binned",
)

adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]
preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

INPUT_LAYER_KEY = "X_binned"
all_counts = (
    adata.layers[INPUT_LAYER_KEY].toarray()
    if issparse(adata.layers[INPUT_LAYER_KEY])
    else adata.layers[INPUT_LAYER_KEY]
)
genes = adata.var["gene_name"].tolist()
celltypes_labels = np.array(adata.obs["celltype_id"].tolist())
batch_ids = np.array(adata.obs["batch_id"].tolist())
num_batch_types = len(set(batch_ids.tolist()))

(
    train_data, valid_data,
    train_celltype_labels, valid_celltype_labels,
    train_batch_labels, valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data, gene_ids, max_len=MAX_SEQ_LEN, vocab=vocab,
    pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True,
    include_zero_gene=INCLUDE_ZERO_GENE,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data, gene_ids, max_len=MAX_SEQ_LEN, vocab=vocab,
    pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True,
    include_zero_gene=INCLUDE_ZERO_GENE,
)
print(f"train samples: {tokenized_train['genes'].shape}, valid: {tokenized_valid['genes'].shape}")

# %%
# --- Build + load model (same as tutorial) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)
model = TransformerModel(
    ntokens, EMBSIZE, NHEAD_M, D_HID, NLAYERS_M,
    nlayers_cls=N_LAYERS_CLS,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=DROPOUT,
    pad_token=PAD_TOKEN,
    pad_value=PAD_VALUE,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    input_emb_style=INPUT_EMB_STYLE,
    n_input_bins=N_INPUT_BINS,
    cell_emb_style=CELL_EMB_STYLE,
    mvc_decoder_style=MVC_DECODER_STYLE,
    ecs_threshold=ECS_THRES,
    explicit_zero_prob=EXPLICIT_ZERO_PROB,
    use_fast_transformer=FAST_TRANSFORMER,
    fast_transformer_backend="flash",
    pre_norm=PRE_NORM,
)
try:
    model.load_state_dict(torch.load(model_file))
except Exception:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
model.to(device)

criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-4 if AMP else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHEDULE_INTERVAL, gamma=SCHEDULE_RATIO)
scaler = torch.cuda.amp.GradScaler(enabled=AMP)

# %%
# --- Train / eval (verbatim simplified from tutorial; no wandb) ---
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    def __len__(self): return self.data["gene_ids"].shape[0]
    def __getitem__(self, idx): return {k: v[idx] for k, v in self.data.items()}

def prepare_loader(tokenized, batch_labels, celltype_labels, shuffle):
    masked_values = random_mask_value(tokenized["values"], mask_ratio=MASK_RATIO,
                                      mask_value=MASK_VALUE, pad_value=PAD_VALUE)
    data_pt = {
        "gene_ids": tokenized["genes"],
        "values": masked_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.from_numpy(batch_labels).long(),
        "celltype_labels": torch.from_numpy(celltype_labels).long(),
    }
    return DataLoader(
        SeqDataset(data_pt), batch_size=BATCH_SIZE, shuffle=shuffle,
        drop_last=False, num_workers=0, pin_memory=True,
    )

def train_epoch(model, loader):
    model.train()
    for batch in loader:
        input_gene_ids = batch["gene_ids"].to(device)
        input_values = batch["values"].to(device)
        celltype_labels = batch["celltype_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])
        with torch.cuda.amp.autocast(enabled=AMP):
            output = model(input_gene_ids, input_values,
                          src_key_padding_mask=src_key_padding_mask,
                          CLS=CLS, CCE=False, MVC=False, ECS=False, do_sample=False)
            loss = criterion_cls(output["cls_output"], celltype_labels)
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()

def evaluate(model, loader, return_raw=False):
    model.eval()
    total_loss, total_acc, total_num = 0.0, 0.0, 0
    preds = []
    with torch.no_grad():
        for batch in loader:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            celltype_labels = batch["celltype_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[PAD_TOKEN])
            with torch.cuda.amp.autocast(enabled=AMP):
                output = model(input_gene_ids, input_values,
                              src_key_padding_mask=src_key_padding_mask,
                              CLS=CLS, CCE=False, MVC=False, ECS=False, do_sample=False)
                output_values = output["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)
            total_loss += loss.item() * len(input_gene_ids)
            total_acc += (output_values.argmax(1) == celltype_labels).sum().item()
            total_num += len(input_gene_ids)
            preds.append(output_values.argmax(1).cpu().numpy())
    if return_raw:
        return np.concatenate(preds)
    return total_loss / total_num, 1 - total_acc / total_num

# %%
# --- Training loop ---
best_val_loss = float("inf")
best_model = None
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loader = prepare_loader(tokenized_train, train_batch_labels, train_celltype_labels, shuffle=True)
    valid_loader = prepare_loader(tokenized_valid, valid_batch_labels, valid_celltype_labels, shuffle=False)
    train_epoch(model, train_loader)
    val_loss, val_err = evaluate(model, valid_loader)
    print(f"epoch {epoch:3d} | time {time.time()-t0:5.1f}s | val_loss {val_loss:.4f} | val_err {val_err:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
    scheduler.step()

# %%
# --- Inference on test set + write ms_annotation.h5ad ---
test_counts = (
    adata_test.layers[INPUT_LAYER_KEY].toarray()
    if issparse(adata_test.layers[INPUT_LAYER_KEY])
    else adata_test.layers[INPUT_LAYER_KEY]
)
test_celltype_labels = np.array(adata_test.obs["celltype_id"].tolist())
test_batch_ids = np.array(adata_test.obs["batch_id"].tolist())
tokenized_test = tokenize_and_pad_batch(
    test_counts, gene_ids, max_len=MAX_SEQ_LEN, vocab=vocab,
    pad_token=PAD_TOKEN, pad_value=PAD_VALUE, append_cls=True,
    include_zero_gene=INCLUDE_ZERO_GENE,
)
test_loader = prepare_loader(tokenized_test, test_batch_ids, test_celltype_labels, shuffle=False)
predictions_int = evaluate(best_model, test_loader, return_raw=True)
pred_celltypes = [id2type[p] for p in predictions_int]

# Build output AnnData aligned to filtered_ms_adata.h5ad obs_names.
out = AnnData(np.zeros((adata_test_raw.n_obs, 1), dtype=np.float32))
out.obs_names = adata_test_raw.obs_names
out.obs["celltype"] = adata_test_raw.obs["Factor Value[inferred cell type - authors labels]"].astype(str).values
# adata_test (after concat + subset) may have had genes filtered; predictions length must equal raw test n_obs.
assert len(pred_celltypes) == adata_test_raw.n_obs, (
    f"prediction length {len(pred_celltypes)} != test n_obs {adata_test_raw.n_obs}"
)
out.obs["predictions"] = pd.Categorical(pred_celltypes)

OUT_PATH = "/data/benchmark/ct_annotation/scGPT-annotation-wrapper/ms_annotation.h5ad"
out.write_h5ad(OUT_PATH)

acc = accuracy_score(out.obs["celltype"], out.obs["predictions"])
macro_f1 = f1_score(out.obs["celltype"], out.obs["predictions"], average="macro")
print(f"scGPT  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  wrote {OUT_PATH}")
