# CT Annotation Benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 5 paper-faithful CT-annotation wrappers (one per model: scGPT, cellPLM, Geneformer, scFoundation, scANVI) plus a unified `benchmark.ipynb` that compares them — all under `/data/benchmark/ct_annotation/`, mirroring the existing `/data/benchmark/embd_clustering/` layout.

**Architecture:** Each wrapper is a Jupyter notebook in its own sub-folder that fine-tunes / adapts its model on the Multiple Sclerosis dataset (`c_data.h5ad` train + `filtered_ms_adata.h5ad` test) and writes a `ms_annotation.h5ad` file with a uniform schema (`obs["celltype"]` ground truth + `obs["predictions"]`). `benchmark.ipynb` reads all 5 files and produces metrics, confusion matrices, and a UMAP comparison.

**Tech Stack:** scanpy / anndata, PyTorch, model-specific conda envs (`scgpt`, `cellplm`, `geneformer`, `scfoundation`, `scvi`), `nbformat` for notebook generation.

**Spec:** `docs/superpowers/specs/2026-05-10-ct-annotation-benchmark-design.md` — consult for the per-model recipe rationale and hyperparameter sources.

**Paper-faithfulness mandate:** Every hyperparameter, freezing pattern, and preprocessing step in this plan is sourced verbatim from each model's official tutorial / code. Do **not** silently substitute "sensible defaults" — that biases the comparison and defeats the purpose. If you find a hyperparameter not in the spec, either pull it from the cited tutorial file or add it to the wrapper notebook header as an explicit deviation.

---

## File Structure

```
ct_annotation/
  README.md                                      ← Task 8
  _py2nb.py                                      ← Task 1 (helper: .py → .ipynb)
  benchmark.ipynb                                ← Task 7
  cellPLM-annotation-wrapper/
    annotation.py                                ← canonical source (Task 3)
    annotation.ipynb                             ← generated from annotation.py
    ms_annotation.h5ad                           ← produced by running the notebook
  Geneformer-annotation-wrapper/
    annotation.py                                ← Task 5
    annotation.ipynb
    ms_annotation.h5ad
  scFoundation-annotation-wrapper/
    annotation.py                                ← Task 6
    annotation.ipynb
    ms_annotation.h5ad
  scGPT-annotation-wrapper/
    annotation.py                                ← Task 2
    annotation.ipynb
    ms_annotation.h5ad
  scVI-annotation-wrapper/                       ← uses scANVI inside
    annotation.py                                ← Task 4
    annotation.ipynb
    ms_annotation.h5ad
```

**Source-of-truth convention:** each wrapper's canonical edit target is `annotation.py` (a Python file with `# %%` cell markers — same format used by VSCode interactive mode and `jupytext`). The `.ipynb` is generated from the `.py` via `_py2nb.py`. This keeps notebooks diff-friendly in git while preserving notebook-format parity with `embd_clustering/`. The `.py` files are committed; the `.ipynb` files are committed too (so the user can open them directly without re-running the converter).

**Per-wrapper conda envs (verified to exist on this machine):**

| Wrapper | Conda env | Python path |
|---|---|---|
| scGPT | `scgpt` | `/home/hanqing-li/anaconda3/envs/scgpt/bin/python` |
| cellPLM | `cellplm` | `/home/hanqing-li/anaconda3/envs/cellplm/bin/python` |
| Geneformer | `geneformer` | `/home/hanqing-li/anaconda3/envs/geneformer/bin/python` |
| scFoundation | `scfoundation` | `/home/hanqing-li/anaconda3/envs/scfoundation/bin/python` |
| scVI (scANVI) | `scvi` | `/home/hanqing-li/anaconda3/envs/scvi/bin/python` |

**Verified paths (do not re-verify):**

- Data: `/data/benchmark/data/cellPLM/data/c_data.h5ad`, `/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad`
- scGPT ckpt: `/data/benchmark/models/scGPT/save/scGPT_human/` (contains `args.json`, `best_model.pt`, `vocab.json`)
- cellPLM ckpt: `/data/benchmark/data/cellPLM/ckpt/` (contains both `20230926_85M.best.ckpt` and `20231027_85M.best.ckpt` — CT tutorial uses `20230926_85M`)
- Geneformer V1-10M: `/data/benchmark/models/Geneformer/Geneformer-V1-10M/` (contains `config.json`, `model.safetensors`, `pytorch_model.bin`)
- scFoundation ckpt: `/data/benchmark/models/scFoundation/model/models/models.ckpt`
- scFoundation gene index: `/data/benchmark/models/scFoundation/model/OS_scRNA_gene_index.19264.tsv`
- scvi-tools version: 1.4.2 (in `scvi` env)

---

## Task 1: Scaffold folder structure + helper

**Files:**
- Create: `/data/benchmark/ct_annotation/_py2nb.py`
- Create: `/data/benchmark/ct_annotation/cellPLM-annotation-wrapper/` (empty dir)
- Create: `/data/benchmark/ct_annotation/Geneformer-annotation-wrapper/` (empty dir)
- Create: `/data/benchmark/ct_annotation/scFoundation-annotation-wrapper/` (empty dir)
- Create: `/data/benchmark/ct_annotation/scGPT-annotation-wrapper/` (empty dir)
- Create: `/data/benchmark/ct_annotation/scVI-annotation-wrapper/` (empty dir)
- Delete: `/data/benchmark/ct_annotation/Tutorial_Annotation.ipynb` (replaced by scGPT-annotation-wrapper)
- Delete: `/data/benchmark/ct_annotation/save/` (output of old tutorial runs; user-approved discard)
- Delete: `/data/benchmark/ct_annotation/wandb/` (wandb logs from old runs; user-approved discard)

- [ ] **Step 1.1: Create the converter helper `_py2nb.py`**

Write this exact content to `/data/benchmark/ct_annotation/_py2nb.py`:

```python
"""Convert a .py file with `# %%` cell markers to a .ipynb notebook.

Usage: python _py2nb.py <input.py> <output.ipynb>

Cell markers (one per line, starting in column 0):
    # %%              -> start a new code cell
    # %% [markdown]   -> start a new markdown cell

All lines between markers form the cell body. The first cell starts at
the top of the file (an implicit `# %%` is assumed).
"""

import sys
from pathlib import Path
import nbformat as nbf


def py_to_notebook(py_path: Path) -> nbf.NotebookNode:
    text = py_path.read_text()
    nb = nbf.v4.new_notebook()
    cells = []
    cell_lines: list[str] = []
    cell_type = "code"  # default first cell is code

    def flush():
        if not cell_lines:
            return
        # Strip trailing blank lines but keep internal structure.
        body = "\n".join(cell_lines).rstrip()
        if not body:
            return
        if cell_type == "markdown":
            # In markdown cells the body is prefixed `# ` — strip one leading `# ` per line.
            stripped = "\n".join(
                ln[2:] if ln.startswith("# ") else (ln[1:] if ln == "#" else ln)
                for ln in body.split("\n")
            )
            cells.append(nbf.v4.new_markdown_cell(stripped))
        else:
            cells.append(nbf.v4.new_code_cell(body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "# %%":
            flush()
            cell_lines = []
            cell_type = "code"
        elif stripped == "# %% [markdown]":
            flush()
            cell_lines = []
            cell_type = "markdown"
        else:
            cell_lines.append(line)
    flush()

    nb.cells = cells
    return nb


def main():
    if len(sys.argv) != 3:
        print("Usage: python _py2nb.py <input.py> <output.ipynb>", file=sys.stderr)
        sys.exit(2)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    nb = py_to_notebook(src)
    nbf.write(nb, dst)
    print(f"wrote {dst} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Create the empty wrapper directories**

Run:

```bash
mkdir -p /data/benchmark/ct_annotation/cellPLM-annotation-wrapper \
         /data/benchmark/ct_annotation/Geneformer-annotation-wrapper \
         /data/benchmark/ct_annotation/scFoundation-annotation-wrapper \
         /data/benchmark/ct_annotation/scGPT-annotation-wrapper \
         /data/benchmark/ct_annotation/scVI-annotation-wrapper
```

- [ ] **Step 1.3: Remove the old tutorial notebook and its artifacts**

Run (all three are explicitly approved for discard in the spec):

```bash
rm /data/benchmark/ct_annotation/Tutorial_Annotation.ipynb
rm -rf /data/benchmark/ct_annotation/save
rm -rf /data/benchmark/ct_annotation/wandb
```

- [ ] **Step 1.4: Sanity-check the converter against a trivial example**

Run:

```bash
cd /data/benchmark/ct_annotation
cat > /tmp/_py2nb_smoke.py <<'EOF'
# %% [markdown]
# # Hello
# This is markdown.

# %%
print("hello")

# %%
x = 1 + 1
EOF
/home/hanqing-li/anaconda3/bin/python _py2nb.py /tmp/_py2nb_smoke.py /tmp/_py2nb_smoke.ipynb
/home/hanqing-li/anaconda3/bin/python -c "import nbformat; nb=nbformat.read('/tmp/_py2nb_smoke.ipynb', as_version=4); assert len(nb.cells)==3, len(nb.cells); assert nb.cells[0].cell_type=='markdown'; assert nb.cells[1].cell_type=='code'; print('OK')"
rm /tmp/_py2nb_smoke.py /tmp/_py2nb_smoke.ipynb
```

Expected output: `wrote /tmp/_py2nb_smoke.ipynb with 3 cells` then `OK`.

If the assertion fails: the converter is buggy — re-read `_py2nb.py` against the smoke file and fix before proceeding.

---

## Task 2: scGPT wrapper

**Files:**
- Create: `/data/benchmark/ct_annotation/scGPT-annotation-wrapper/annotation.py`
- Create: `/data/benchmark/ct_annotation/scGPT-annotation-wrapper/annotation.ipynb` (generated)
- Output: `/data/benchmark/ct_annotation/scGPT-annotation-wrapper/ms_annotation.h5ad` (produced when notebook runs)

**Source recipe:** `/data/benchmark/models/scGPT/tutorials/Tutorial_Annotation.ipynb` — copy hyperparameters verbatim; strip wandb; replace plotting+save_dict logic with a single `ms_annotation.h5ad` write.

- [ ] **Step 2.1: Write `annotation.py`**

Write this exact content to `/data/benchmark/ct_annotation/scGPT-annotation-wrapper/annotation.py`. The hyperparameter block, preprocessor settings, train/eval/test functions are taken verbatim from `Tutorial_Annotation.ipynb` — only wandb and plotting are stripped, and the output goes to `ms_annotation.h5ad`.

```python
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
```

- [ ] **Step 2.2: Convert to .ipynb**

Run:

```bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/scgpt/bin/python _py2nb.py \
    scGPT-annotation-wrapper/annotation.py \
    scGPT-annotation-wrapper/annotation.ipynb
```

Expected: `wrote scGPT-annotation-wrapper/annotation.ipynb with N cells` (N around 10-12).

- [ ] **Step 2.3: Smoke run (1 epoch)**

Temporarily set `EPOCHS = 1` in `annotation.py` (use Edit tool to change `EPOCHS = 10` → `EPOCHS = 1`), regenerate the notebook, run it, then revert.

Run:

```bash
cd /data/benchmark/ct_annotation
sed -i 's/^EPOCHS = 10$/EPOCHS = 1  # SMOKE/' scGPT-annotation-wrapper/annotation.py
/home/hanqing-li/anaconda3/envs/scgpt/bin/python _py2nb.py \
    scGPT-annotation-wrapper/annotation.py \
    scGPT-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/scgpt/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    scGPT-annotation-wrapper/annotation.ipynb
```

Expected: notebook executes without error; last cell prints `scGPT accuracy=... macro-F1=... wrote ...ms_annotation.h5ad`. If it errors, fix the error in `annotation.py`, regenerate, re-run.

- [ ] **Step 2.4: Verify smoke output schema**

Run:

```bash
/home/hanqing-li/anaconda3/envs/scgpt/bin/python <<'EOF'
import anndata as ad
out = ad.read_h5ad("/data/benchmark/ct_annotation/scGPT-annotation-wrapper/ms_annotation.h5ad")
ref = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
assert out.n_obs == ref.n_obs, f"{out.n_obs} != {ref.n_obs}"
assert list(out.obs_names) == list(ref.obs_names), "obs_names misaligned"
assert "celltype" in out.obs.columns and "predictions" in out.obs.columns
assert out.obs["predictions"].notna().all()
print("OK schema:", out)
EOF
```

Expected: `OK schema: AnnData object with n_obs × n_vars = <N> × 1 ...` where N matches `filtered_ms_adata.h5ad`. Any assertion failure means the wrapper output is malformed — debug before proceeding.

- [ ] **Step 2.5: Revert to full epochs and run full**

```bash
cd /data/benchmark/ct_annotation
sed -i 's/^EPOCHS = 1  # SMOKE$/EPOCHS = 10/' scGPT-annotation-wrapper/annotation.py
/home/hanqing-li/anaconda3/envs/scgpt/bin/python _py2nb.py \
    scGPT-annotation-wrapper/annotation.py \
    scGPT-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/scgpt/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    scGPT-annotation-wrapper/annotation.ipynb
```

Expected (per the scGPT tutorial baseline): `macro-F1` in the **0.80–0.90** range. If macro-F1 < 0.5, something is wrong with preprocessing or label mapping — investigate before moving on.

- [ ] **Step 2.6: Commit**

```bash
cd /data/benchmark
git add ct_annotation/_py2nb.py \
        ct_annotation/scGPT-annotation-wrapper/annotation.py \
        ct_annotation/scGPT-annotation-wrapper/annotation.ipynb \
        ct_annotation/scGPT-annotation-wrapper/ms_annotation.h5ad
git rm -f ct_annotation/Tutorial_Annotation.ipynb 2>/dev/null || true
git commit -m "feat(ct_annotation): scGPT wrapper + .py-to-ipynb helper"
```

If `git` is not initialized in this repo (per the environment description, this is not a git repo), skip `git rm` and `git commit`; just leave the files in place.

---

## Task 3: cellPLM wrapper

**Files:**
- Create: `/data/benchmark/ct_annotation/cellPLM-annotation-wrapper/annotation.py`
- Create: `/data/benchmark/ct_annotation/cellPLM-annotation-wrapper/annotation.ipynb`
- Output: `ms_annotation.h5ad`

**Source recipe:** `/data/benchmark/models/cellPLM/tutorials/cell_type_annotation.ipynb` — `CellTypeAnnotationPipeline` with `PRETRAIN_VERSION='20230926_85M'`, `set_seed(42)`, default pipeline + model configs.

- [ ] **Step 3.1: Write `annotation.py`**

Write this exact content to `/data/benchmark/ct_annotation/cellPLM-annotation-wrapper/annotation.py`:

```python
# %% [markdown]
# # cellPLM — Cell-Type Annotation Wrapper (MS dataset)
#
# Verbatim recipe from `models/cellPLM/tutorials/cell_type_annotation.ipynb`:
# `CellTypeAnnotationPipeline` with `PRETRAIN_VERSION='20230926_85M'`, `set_seed(42)`,
# default pipeline + model configs. Only override: `model_config['out_dim'] = n_classes`.
#
# Adaptations: at the end, slice predicted AnnData to `split == 'test'` and write
# `ms_annotation.h5ad` with `obs_names` aligned to `filtered_ms_adata.h5ad`.

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
)

predictions = pipeline.predict(data, pipeline_config)
# `predictions` is a 1-D array of label indices over the full `data` (train+valid+test).

# %%
# --- Build ms_annotation.h5ad aligned to filtered_ms_adata.h5ad's obs_names ---
# Map prediction indices back to label strings using the celltype category ordering.
celltype_cats = data.obs["celltype"].astype("category").cat.categories.tolist()

if hasattr(predictions, "argmax"):
    # Predictions may be returned as logits / probabilities.
    import numpy as _np
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
```

- [ ] **Step 3.2: Convert + run end-to-end**

cellPLM training on this scale is short (~minutes per the tutorial). Skip the separate smoke step.

```bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/cellplm/bin/python _py2nb.py \
    cellPLM-annotation-wrapper/annotation.py \
    cellPLM-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/cellplm/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    cellPLM-annotation-wrapper/annotation.ipynb
```

If `pipeline.predict` returns something other than logits / indices (the cellPLM API has shifted across versions), inspect the returned object in the notebook and adjust the indexing logic in the last cell. Do not silently change other hyperparameters.

- [ ] **Step 3.3: Verify output schema**

```bash
/home/hanqing-li/anaconda3/envs/cellplm/bin/python <<'EOF'
import anndata as ad
out = ad.read_h5ad("/data/benchmark/ct_annotation/cellPLM-annotation-wrapper/ms_annotation.h5ad")
ref = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
assert out.n_obs == ref.n_obs
assert list(out.obs_names) == list(ref.obs_names)
assert "celltype" in out.obs.columns and "predictions" in out.obs.columns
print("OK", out)
EOF
```

Expected: `OK AnnData object with n_obs × n_vars = <matches> × 1 ...`. cellPLM is a strong baseline; expect macro-F1 > 0.8.

- [ ] **Step 3.4: Commit**

```bash
cd /data/benchmark
git add ct_annotation/cellPLM-annotation-wrapper/ 2>/dev/null || true
git commit -m "feat(ct_annotation): cellPLM wrapper" 2>/dev/null || true
```

(Skip if not a git repo.)

---

## Task 4: scVI/scANVI wrapper

**Files:**
- Create: `/data/benchmark/ct_annotation/scVI-annotation-wrapper/annotation.py`
- Create: `/data/benchmark/ct_annotation/scVI-annotation-wrapper/annotation.ipynb`
- Output: `ms_annotation.h5ad`

**Source recipe:** scvi-tools canonical scANVI workflow — train scVI for 20 epochs, then derive scANVI via `from_scvi_model` and train for 20 more epochs, predict labels on the unlabeled test rows.

- [ ] **Step 4.1: Write `annotation.py`**

```python
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
#   2. `scvi.model.SCVI.setup_anndata(..., batch_key="str_batch")`; train scVI 20 epochs.
#   3. `scvi.model.SCANVI.from_scvi_model(..., unlabeled_category="Unknown",
#      labels_key="labels_scanvi")`; train scANVI 20 epochs, `n_samples_per_label=100`.
#   4. Predict labels for test rows; write `ms_annotation.h5ad`.

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

# Move .X to a layer that scvi-tools can use as counts.
# c_data + filtered_ms_adata are already log-normed; we still pass .X to setup_anndata
# (scvi 1.x defaults to interpreting it as counts; with log1p data this is a mild
# deviation from scANVI's pure-Poisson assumption but matches what practitioners do
# when only normalized data is available).
print("adata:", adata.shape, "labels:", adata.obs["labels_scanvi"].value_counts().head())

# %%
# --- Train scVI ---
scvi.model.SCVI.setup_anndata(adata, batch_key="str_batch")
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
```

- [ ] **Step 4.2: Convert + run end-to-end**

```bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/scvi/bin/python _py2nb.py \
    scVI-annotation-wrapper/annotation.py \
    scVI-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/scvi/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    scVI-annotation-wrapper/annotation.ipynb
```

scVI + scANVI on this scale completes in single-digit minutes. If the notebook errors with `setup_anndata` complaining about the layer, change `scvi.model.SCVI.setup_anndata(adata, batch_key="str_batch")` to `scvi.model.SCVI.setup_anndata(adata, batch_key="str_batch", layer=None)` and re-run.

- [ ] **Step 4.3: Verify + commit**

```bash
/home/hanqing-li/anaconda3/envs/scvi/bin/python <<'EOF'
import anndata as ad
out = ad.read_h5ad("/data/benchmark/ct_annotation/scVI-annotation-wrapper/ms_annotation.h5ad")
ref = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
assert out.n_obs == ref.n_obs and list(out.obs_names) == list(ref.obs_names)
assert "celltype" in out.obs and "predictions" in out.obs
print("OK", out)
EOF
```

Then:

```bash
cd /data/benchmark
git add ct_annotation/scVI-annotation-wrapper/ 2>/dev/null || true
git commit -m "feat(ct_annotation): scANVI wrapper" 2>/dev/null || true
```

---

## Task 5: Geneformer wrapper

**Files:**
- Create: `/data/benchmark/ct_annotation/Geneformer-annotation-wrapper/annotation.py`
- Create: `/data/benchmark/ct_annotation/Geneformer-annotation-wrapper/annotation.ipynb`
- Output: `ms_annotation.h5ad`

**Source recipe:** `/data/benchmark/models/Geneformer/examples/cell_classification.ipynb` — `geneformer.Classifier(classifier="cell", ...)` with the tutorial's `training_args`. State key changed from `"disease"` to `"celltype"`. Tokenization via `TranscriptomeTokenizer` (consumes .loom or .h5ad → HF dataset).

**Implementation note:** Geneformer expects an HF `datasets.Dataset` keyed by tokenized ensembl IDs. The wrapper does:
1. Combine `c_data` + `filtered_ms_adata` into a single `.h5ad` written under `/tmp/geneformer_input/`.
2. Tokenize via `TranscriptomeTokenizer` → HF dataset.
3. Use `Classifier.prepare_data` with a split-by-source dict to make train + test HF datasets.
4. Use `Classifier.validate` to train (no separate eval, just train+test for this benchmark).
5. Use `Classifier.evaluate_saved_model` to get predictions on the test split.
6. Map predictions back to `filtered_ms_adata.h5ad` obs_names and write `ms_annotation.h5ad`.

- [ ] **Step 5.1: Write `annotation.py`**

```python
# %% [markdown]
# # Geneformer — Cell-Type Annotation Wrapper (MS dataset)
#
# Recipe from `models/Geneformer/examples/cell_classification.ipynb`. The tutorial covers
# *disease* classification but the `Classifier(classifier="cell", state_key=...)` API
# is the same — we set `state_key="celltype"`.
#
# **Hyperparameters (verbatim from the tutorial; note these were tuned for
# cardiomyopathy, not celltype — see spec for caveat):**
#   `freeze_layers=2, forward_batch_size=200, nproc=16, seed=73`
#   `training_args = {num_train_epochs: 0.9, learning_rate: 8.04e-4,
#                     lr_scheduler_type: "polynomial", warmup_steps: 1812,
#                     weight_decay: 0.258828, per_device_train_batch_size: 12}`

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
# Geneformer expects "ensembl_id" var column for the tokenizer.
# c_data / filtered_ms_adata are keyed by ensembl IDs.
if "ensembl_id" not in train.var.columns:
    train.var["ensembl_id"] = train.var_names.tolist()
if "ensembl_id" not in test.var.columns:
    test.var["ensembl_id"] = test.var_names.tolist()
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
test_obs_names = test.obs_names.tolist()
test_celltype_strs = test.obs["celltype"].astype(str).values.copy()

# Geneformer tokenizer requires raw counts as `n_counts` in obs (it computes if absent).
# Save the two files in a single directory; the tokenizer scans the directory.
train.write_h5ad(WORK / "h5ad" / "train.h5ad")
test.write_h5ad(WORK / "h5ad" / "test.h5ad")

# %%
# --- Tokenize both files to HF dataset format ---
tokenizer = TranscriptomeTokenizer(
    custom_attr_name_dict={"celltype": "celltype", "source": "source", "fold": "fold"},
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
    "num_train_epochs": 0.9,
    "learning_rate": 0.000804,
    "lr_scheduler_type": "polynomial",
    "warmup_steps": 1812,
    "weight_decay": 0.258828,
    "per_device_train_batch_size": 12,
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

# pred_dict has keys 'predictions' (label indices) and 'labels' (true label indices) per the Geneformer source.
pred_idx = np.asarray(pred_dict["predictions"])
inv_id_class = {v: k for k, v in id_class.items()} if isinstance(next(iter(id_class.values())), int) else id_class
# id_class maps integer-id -> label string in Geneformer; use it as-is if so.
def to_str(idx):
    if idx in id_class:
        return id_class[idx]
    return inv_id_class.get(idx, str(idx))
pred_strs = np.array([to_str(int(i)) for i in pred_idx])

# %%
# --- Align predictions back to filtered_ms_adata obs_names ---
# Geneformer's tokenized dataset preserves the row order of the source .h5ad it was
# generated from. `test.h5ad` was written above in the original `filtered_ms_adata` order,
# so the index in `pred_strs` matches `test_obs_names`.
assert len(pred_strs) == len(test_obs_names), (
    f"predictions {len(pred_strs)} != test_obs_names {len(test_obs_names)} — order/length mismatch; "
    "check that the tokenizer kept the test rows in the same order as test.h5ad."
)

out = ad.AnnData(np.zeros((len(test_obs_names), 1), dtype=np.float32))
out.obs_names = test_obs_names
out.obs["celltype"] = test_celltype_strs
out.obs["predictions"] = pd.Categorical(pred_strs)

OUT_PATH = "/data/benchmark/ct_annotation/Geneformer-annotation-wrapper/ms_annotation.h5ad"
out.write_h5ad(OUT_PATH)

acc = accuracy_score(out.obs["celltype"], out.obs["predictions"])
macro_f1 = f1_score(out.obs["celltype"], out.obs["predictions"], average="macro")
print(f"Geneformer  accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  wrote {OUT_PATH}")
```

- [ ] **Step 5.2: Convert + run**

```bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/geneformer/bin/python _py2nb.py \
    Geneformer-annotation-wrapper/annotation.py \
    Geneformer-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/geneformer/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    Geneformer-annotation-wrapper/annotation.ipynb
```

**Likely friction points (in order of probability):**
1. **`pred_dict` key names differ between Geneformer versions.** If `pred_dict["predictions"]` raises `KeyError`, print `list(pred_dict.keys())` and try `"y_pred"` / `"pred_labels"` instead.
2. **`id_class_dict.pkl` mapping direction.** Inspect with `print(id_class)` — the keys are usually ints, values are label strings. The `to_str` helper handles both directions; if both fail, print the dict and hard-map.
3. **HuggingFace training quirks** can leak environment variables — if the run hangs, set `TOKENIZERS_PARALLELISM=false` in the env before `jupyter nbconvert`.
4. **Tokenizer may not preserve row order** when scanning `train.h5ad` + `test.h5ad` from a directory. If the assertion at the bottom fires with a length mismatch, inspect the tokenized dataset's `source` column to confirm the test rows are still contiguous and in original order; if not, you'll need to re-tokenize each file separately and join.

Each fix goes into `annotation.py` (then regen + re-run). Do not modify hyperparameters in `training_args` as part of debugging.

- [ ] **Step 5.3: Verify + commit**

```bash
/home/hanqing-li/anaconda3/envs/geneformer/bin/python <<'EOF'
import anndata as ad
out = ad.read_h5ad("/data/benchmark/ct_annotation/Geneformer-annotation-wrapper/ms_annotation.h5ad")
ref = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
assert out.n_obs == ref.n_obs and list(out.obs_names) == list(ref.obs_names)
assert "celltype" in out.obs and "predictions" in out.obs
print("OK", out)
EOF
```

Then:

```bash
cd /data/benchmark
git add ct_annotation/Geneformer-annotation-wrapper/ 2>/dev/null || true
git commit -m "feat(ct_annotation): Geneformer wrapper" 2>/dev/null || true
```

If macro-F1 is < 0.5, this is the model whose hyperparameters were tuned for a different task. **Re-tuning is out of scope** (the spec calls this out). Note the result and move on.

---

## Task 6: scFoundation wrapper

**Files:**
- Create: `/data/benchmark/ct_annotation/scFoundation-annotation-wrapper/annotation.py`
- Create: `/data/benchmark/ct_annotation/scFoundation-annotation-wrapper/annotation.ipynb`
- Output: `ms_annotation.h5ad`

**Source recipe:** `models/scFoundation/model/finetune_model.py:88-148` (`LinearProbingClassifier`). Selectively unfreezes `encoder.transformer_encoder[-2]` only; head is `BatchNorm1d(768) → Linear(768→256) → ReLU → Linear(256→n_class)` with max-pool over seq dim. Inputs padded to 19264 genes via `OS_scRNA_gene_index.19264.tsv`. Hyperparameters not pinned in the public repo — we use linear-probing defaults (lr=1e-4 AdamW, batch_size=8, 10 epochs) and document as deviation.

- [ ] **Step 6.1: Write `annotation.py`**

```python
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
```

- [ ] **Step 6.2: Smoke run (1 epoch)**

```bash
cd /data/benchmark/ct_annotation
sed -i 's/^EPOCHS = 10$/EPOCHS = 1  # SMOKE/' scFoundation-annotation-wrapper/annotation.py
/home/hanqing-li/anaconda3/envs/scfoundation/bin/python _py2nb.py \
    scFoundation-annotation-wrapper/annotation.py \
    scFoundation-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/scfoundation/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    scFoundation-annotation-wrapper/annotation.ipynb
```

**Likely friction:**
- `LinearProbingClassifier.forward` expects `sample_list["x"]` shape `(B, L)` where L=19264. If shape errors, print `xb.shape` before the forward call to confirm.
- The `gatherData` helper inside the model uses `value_labels = x > 0` — counts must be non-negative. MS data is log-normed so this should hold, but if you see NaN losses, check.
- BatchNorm with `BATCH_SIZE=8` works for `BatchNorm1d`, but if the last batch has size 1, it errors. Add `drop_last=True` to `train_loader` if so.

- [ ] **Step 6.3: Verify smoke schema, then run full**

```bash
/home/hanqing-li/anaconda3/envs/scfoundation/bin/python <<'EOF'
import anndata as ad
out = ad.read_h5ad("/data/benchmark/ct_annotation/scFoundation-annotation-wrapper/ms_annotation.h5ad")
ref = ad.read_h5ad("/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad")
assert out.n_obs == ref.n_obs and list(out.obs_names) == list(ref.obs_names)
print("OK", out)
EOF

cd /data/benchmark/ct_annotation
sed -i 's/^EPOCHS = 1  # SMOKE$/EPOCHS = 10/' scFoundation-annotation-wrapper/annotation.py
/home/hanqing-li/anaconda3/envs/scfoundation/bin/python _py2nb.py \
    scFoundation-annotation-wrapper/annotation.py \
    scFoundation-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/scfoundation/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 \
    scFoundation-annotation-wrapper/annotation.ipynb
```

- [ ] **Step 6.4: Commit**

```bash
cd /data/benchmark
git add ct_annotation/scFoundation-annotation-wrapper/ 2>/dev/null || true
git commit -m "feat(ct_annotation): scFoundation wrapper" 2>/dev/null || true
```

---

## Task 7: benchmark.ipynb

**Files:**
- Create: `/data/benchmark/ct_annotation/benchmark.py`
- Create: `/data/benchmark/ct_annotation/benchmark.ipynb`

**Spec section:** "benchmark.ipynb" — produces metrics DataFrame, confusion-matrix grid, UMAP comparison grid.

- [ ] **Step 7.1: Write `benchmark.py`**

```python
# %% [markdown]
# # CT Annotation Benchmark
#
# Consumes `ms_annotation.h5ad` from each of the 5 wrappers, aligns by `obs_names`,
# computes uniform metrics, and produces:
# 1. A metrics DataFrame (accuracy, macro-precision/recall/F1)
# 2. A 1×5 grid of normalized confusion matrices
# 3. A 2×3 grid of UMAPs colored by ground truth + each model's predictions

# %%
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

MODELS = ["scGPT", "cellPLM", "Geneformer", "scFoundation", "scVI"]
ROOT = Path("/data/benchmark/ct_annotation")
GROUND_TRUTH = "/data/benchmark/data/cellPLM/data/filtered_ms_adata.h5ad"

# %%
# --- Load ground truth ---
gt = ad.read_h5ad(GROUND_TRUTH)
LABEL_COL = "Factor Value[inferred cell type - authors labels]"
gt.obs["celltype"] = gt.obs[LABEL_COL].astype(str)
print(f"ground truth: n_obs={gt.n_obs}, n_classes={gt.obs['celltype'].nunique()}")

# %%
# --- Load each wrapper's predictions; align by obs_names ---
preds_per_model: dict[str, np.ndarray] = {}
for m in MODELS:
    p = ROOT / f"{m}-annotation-wrapper" / "ms_annotation.h5ad"
    if not p.exists():
        print(f"SKIP {m}: {p} does not exist")
        continue
    out = ad.read_h5ad(p)
    # Align: reindex by gt.obs_names.
    common = gt.obs_names.intersection(out.obs_names)
    assert len(common) == gt.n_obs == out.n_obs, (
        f"{m}: gt={gt.n_obs}, wrapper={out.n_obs}, common={len(common)}"
    )
    out = out[gt.obs_names].copy()
    gt.obs[f"{m}_pred"] = out.obs["predictions"].astype(str).values
    preds_per_model[m] = out.obs["predictions"].astype(str).values

print(f"loaded predictions for: {list(preds_per_model.keys())}")

# %%
# --- Metrics table ---
rows = []
y_true = gt.obs["celltype"].astype(str).values
for m, y_pred in preds_per_model.items():
    rows.append({
        "model": m,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    })
metrics_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
print(metrics_df.to_string(index=False))

# %%
# --- Confusion matrix grid (1×N) ---
classes = sorted(set(y_true) | {c for preds in preds_per_model.values() for c in preds})
n = len(preds_per_model)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
if n == 1:
    axes = [axes]
for ax, (m, y_pred) in zip(axes, preds_per_model.items()):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_norm, xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax, cbar=False, vmin=0, vmax=1)
    ax.set_title(f"{m}  (macro-F1 = {metrics_df.set_index('model').loc[m, 'macro_f1']:.3f})")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.savefig(ROOT / "confusion_matrices.png", dpi=150)
plt.show()

# %%
# --- UMAP comparison ---
# Compute one shared UMAP on gt.
adata_umap = gt.copy()
if "log1p" not in adata_umap.uns:
    # filtered_ms_adata.h5ad is already log-normed in our pipeline. Recompute fresh to be safe.
    sc.pp.normalize_total(adata_umap, target_sum=1e4)
    sc.pp.log1p(adata_umap)
sc.pp.highly_variable_genes(adata_umap, n_top_genes=2000)
adata_umap = adata_umap[:, adata_umap.var.highly_variable].copy()
sc.pp.scale(adata_umap, max_value=10)
sc.pp.pca(adata_umap, n_comps=50, random_state=0)
sc.pp.neighbors(adata_umap, random_state=0)
sc.tl.umap(adata_umap, random_state=0)
# Copy prediction columns over.
for m, y_pred in preds_per_model.items():
    adata_umap.obs[f"{m}_pred"] = pd.Categorical(y_pred)

panels = ["celltype"] + [f"{m}_pred" for m in preds_per_model.keys()]
n_cols = 3
n_rows = int(np.ceil(len(panels) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = np.array(axes).flatten()
for ax, color in zip(axes, panels):
    sc.pl.umap(adata_umap, color=color, show=False, ax=ax, legend_fontsize=6, frameon=False)
    ax.set_title(color)
for ax in axes[len(panels):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(ROOT / "umap_comparison.png", dpi=150)
plt.show()

# %%
# --- Save metrics CSV for downstream use ---
metrics_df.to_csv(ROOT / "metrics.csv", index=False)
print("wrote", ROOT / "metrics.csv")
```

- [ ] **Step 7.2: Convert + run**

The benchmark notebook has no GPU dependencies; any env with `scanpy`, `anndata`, `sklearn`, `seaborn` works. Use the `cellplm` env (it has all of those).

```bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/cellplm/bin/python _py2nb.py benchmark.py benchmark.ipynb
/home/hanqing-li/anaconda3/envs/cellplm/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 \
    benchmark.ipynb
```

Expected: the executed notebook contains the metrics table, the confusion matrix grid, and the UMAP comparison grid. `metrics.csv`, `confusion_matrices.png`, `umap_comparison.png` are written to `ct_annotation/`.

- [ ] **Step 7.3: Verify and commit**

```bash
ls /data/benchmark/ct_annotation/metrics.csv /data/benchmark/ct_annotation/confusion_matrices.png /data/benchmark/ct_annotation/umap_comparison.png
cat /data/benchmark/ct_annotation/metrics.csv
```

Then:

```bash
cd /data/benchmark
git add ct_annotation/benchmark.py ct_annotation/benchmark.ipynb \
        ct_annotation/metrics.csv ct_annotation/confusion_matrices.png \
        ct_annotation/umap_comparison.png 2>/dev/null || true
git commit -m "feat(ct_annotation): benchmark notebook" 2>/dev/null || true
```

---

## Task 8: README + final commit

**Files:**
- Create: `/data/benchmark/ct_annotation/README.md`

- [ ] **Step 8.1: Write README**

Write this exact content to `/data/benchmark/ct_annotation/README.md`:

```markdown
# CT Annotation Benchmark

Cell-type annotation benchmark across 5 single-cell foundation/representation models on the Multiple Sclerosis (MS) dataset. Parallels `embd_clustering/`.

## Layout

- `<model>-annotation-wrapper/annotation.py` — canonical source (one `# %%` cell per logical step)
- `<model>-annotation-wrapper/annotation.ipynb` — generated from `annotation.py` via `_py2nb.py`
- `<model>-annotation-wrapper/ms_annotation.h5ad` — output predictions (uniform schema)
- `benchmark.ipynb` — loads all 5 outputs, computes metrics, renders plots
- `metrics.csv`, `confusion_matrices.png`, `umap_comparison.png` — benchmark artifacts

## Output schema (all wrappers)

`ms_annotation.h5ad` is aligned to `filtered_ms_adata.h5ad`'s `obs_names`. Required columns in `.obs`:
- `celltype` — ground-truth label as string
- `predictions` — predicted label as string (from the celltype categories of the training file)

## Per-model recipe

Every wrapper follows its **original paper / official tutorial** recipe verbatim — preprocessing, hyperparameters, freezing pattern, optimizer settings. Deviations are documented in the wrapper's header markdown cell. See `docs/superpowers/specs/2026-05-10-ct-annotation-benchmark-design.md` for sources and rationale.

| Wrapper folder | Model | Conda env | Method |
|---|---|---|---|
| `scGPT-annotation-wrapper/` | scGPT | `scgpt` | End-to-end fine-tune w/ CLS head (`Tutorial_Annotation.ipynb`) |
| `cellPLM-annotation-wrapper/` | cellPLM | `cellplm` | `CellTypeAnnotationPipeline` (`cell_type_annotation.ipynb`) |
| `Geneformer-annotation-wrapper/` | Geneformer | `geneformer` | `Classifier(classifier="cell")` via HF Trainer |
| `scFoundation-annotation-wrapper/` | scFoundation | `scfoundation` | `LinearProbingClassifier` from `finetune_model.py` |
| `scVI-annotation-wrapper/` | **scANVI** (folder named for parity) | `scvi` | scANVI semi-supervised w/ `"Unknown"` query labels |

## Running

Each wrapper is self-contained. To re-run one:

\`\`\`bash
cd /data/benchmark/ct_annotation
/home/hanqing-li/anaconda3/envs/<env>/bin/python _py2nb.py \\
    <model>-annotation-wrapper/annotation.py \\
    <model>-annotation-wrapper/annotation.ipynb
/home/hanqing-li/anaconda3/envs/<env>/bin/jupyter nbconvert \\
    --to notebook --execute --inplace \\
    <model>-annotation-wrapper/annotation.ipynb
\`\`\`

After all 5 are run, re-run `benchmark.ipynb` (any env with scanpy + sklearn works).
```

- [ ] **Step 8.2: Final commit**

```bash
cd /data/benchmark
git add ct_annotation/README.md 2>/dev/null || true
git commit -m "docs(ct_annotation): README" 2>/dev/null || true
```

---

## Final sanity check

- [ ] **Step 9.1: Verify all 5 outputs exist and the benchmark ran**

```bash
ls -la /data/benchmark/ct_annotation/*-annotation-wrapper/ms_annotation.h5ad
ls -la /data/benchmark/ct_annotation/metrics.csv \
       /data/benchmark/ct_annotation/confusion_matrices.png \
       /data/benchmark/ct_annotation/umap_comparison.png
cat /data/benchmark/ct_annotation/metrics.csv
```

Expected: 5 `ms_annotation.h5ad` files exist, plus the 3 benchmark artifacts, plus a metrics CSV with 5 rows (one per model).

If any wrapper failed and you skipped it: note this in the final response to the user — do not pretend success.
