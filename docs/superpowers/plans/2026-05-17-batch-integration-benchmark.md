# Batch-Integration Benchmark Implementation Plan

> ⏸️ **EXECUTION PAUSED 2026-05-17 (machine powered off mid-benchmark).**
> **Resume from `docs/superpowers/plans/2026-05-17-batch-integration-RESUME.md` — read it first.**
> Tasks 1–5 ✅ done & persisted to disk (incl. the 75-min scGPT fine-tune). Task 6
> (scIB benchmark) was killed by power-off with no `metrics.csv` → resume **Case B**
> (re-run benchmark only, ~1.5–2 h; no model recompute). Tasks 7–8 pending; scripts
> already written.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-cell data-integration / batch-effect-removal benchmark on GSE155468 (sample = batch), scored with the scIB metric panel, comparing scVI, scGPT (zero-shot) vs scGPT (fine-tuned), Geneformer, scFoundation, cellPLM(run-not-reported) and a PCA baseline; then swap the deck's Results-1 slide in place to show it.

**Architecture:** New `batch_integration/` dir mirrors `embd_clustering/`/`ct_annotation/`. Five models reuse their existing `embd_clustering/` zero-shot/batch-conditioned embeddings via thin shims; only `scGPT (fine-tuned)` is a new run derived verbatim from `models/scGPT/tutorials/Tutorial_Integration.ipynb`. A `benchmark.py` builds one combined AnnData and runs `scib_metrics.benchmark.Benchmarker`. A deck script edits the live `.pptx` in place (no `build_deck.py` rerun — pasted figures must survive).

**Tech Stack:** Python, anndata/scanpy, hdf5plugin, scib-metrics (jax CPU), scGPT (`scgpt` conda env), python-pptx + matplotlib (base anaconda).

---

> **No version control:** `/data/benchmark` is **not** a git repo. Every "Commit" the writing-plans template would normally use is replaced by an explicit **Checkpoint** step (run a verification command, confirm expected output). Do not run `git`.

> **Faithful-wrapper convention (from `ct_annotation/`):** a wrapper copies its source tutorial's hyperparameter block *verbatim* and only adapts data paths / output. Deviations go in the wrapper's header markdown cell.

## File Structure

| Path | Responsibility |
|---|---|
| `batch_integration/_common.py` | Shared helpers: load GSE155468 labels; re-emit an `embd_clustering` embedding under the integration schema |
| `batch_integration/_py2nb.py` | Copied verbatim from `ct_annotation/_py2nb.py` (`.py`→`.ipynb`) |
| `batch_integration/{scVI,scGPT-zeroshot,Geneformer,scFoundation,cellPLM}-integration-wrapper/integration.py` + `.ipynb` | Thin reuse shims → `gse155468_integration.h5ad` |
| `batch_integration/scGPT-finetuned-integration-wrapper/integration.py` + `.ipynb` | NEW run, derived from `Tutorial_Integration.ipynb` |
| `batch_integration/benchmark.py` + `.ipynb` | Build combined AnnData + PCA baseline + run scib-metrics → `metrics.csv`, `umap_batch_celltype.png` |
| `batch_integration/README.md` | Recipe table, env pin, run instructions, caveats |
| `batch_integration/_check_schema.py` | Validation: assert every `gse155468_integration.h5ad` matches the contract |
| `presentation/swap_results1_slide.py` | In-place edit: replace Results-1 slide title/chart/notes |
| `presentation/build_deck.py` | Patch the Results-1 builder for from-scratch consistency |

**Output schema** (every wrapper): `gse155468_integration.h5ad` with `obs_names` == `gse155468.h5ad`'s; `obsm['X_emb']` float32 `(48082, d)`; `obs['celltype']` str; `obs['batch']` str (= `orig.ident`).

**Canonical paths:**
- Dataset: `/data/benchmark/data/cellPLM/data/gse155468.h5ad`
- Reused embeddings: `/data/benchmark/embd_clustering/<Model>-embedding-wrapper/gse155468_embedding.h5ad` (embedding is in `.X`, `var_names` = `dim_i`, `obs_names` aligned to dataset)
- scGPT env: `/home/hanqing-li/anaconda3/envs/scgpt/bin/python`
- Deck python: `/home/hanqing-li/anaconda3/bin/python`

---

### Task 1: Create & pin the `scib-metrics` environment

**Files:** none (conda env + later `README.md`)

- [ ] **Step 1: Create the env (CPU jax — Blackwell+driver-580 makes jax-CUDA fragile)**

Run:
```bash
conda create -n scib-metrics python=3.11 -y
conda run -n scib-metrics pip install "scib-metrics" "scanpy" "anndata" "hdf5plugin" "matplotlib" "leidenalg" "igraph"
```

- [ ] **Step 2: Verify import + force CPU jax**

Run:
```bash
JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python -c \
"import jax,scib_metrics,scanpy,anndata,hdf5plugin; print('jax',jax.__version__,jax.default_backend()); from scib_metrics.benchmark import Benchmarker; print('Benchmarker OK')"
```
Expected: prints `jax <ver> cpu` and `Benchmarker OK` (no GPU/CUDA error).

- [ ] **Step 3: Checkpoint — capture the pin**

Run:
```bash
/home/hanqing-li/anaconda3/envs/scib-metrics/bin/python -m pip freeze > /tmp/scib_pin.txt
grep -E "scib-metrics|^jax|scanpy|anndata" /tmp/scib_pin.txt
```
Expected: non-empty lines for `scib-metrics`, `jax`, `scanpy`, `anndata`. (These lines get pasted into `README.md` in Task 8.) All wrappers/benchmark that don't need a model run will use `JAX_PLATFORMS=cpu` + this python.

---

### Task 2: Scaffold `batch_integration/` + shared helpers

**Files:**
- Create: `batch_integration/_common.py`
- Create: `batch_integration/_py2nb.py` (copy)
- Create: `batch_integration/__init__.py` (empty, so `_common` is importable from wrapper dirs)

- [ ] **Step 1: Create dir + copy `_py2nb.py`**

Run:
```bash
mkdir -p /data/benchmark/batch_integration
cp /data/benchmark/ct_annotation/_py2nb.py /data/benchmark/batch_integration/_py2nb.py
touch /data/benchmark/batch_integration/__init__.py
```

- [ ] **Step 2: Write `_common.py`**

Create `/data/benchmark/batch_integration/_common.py`:
```python
"""Shared helpers for the batch-integration benchmark.

Output contract for every wrapper's gse155468_integration.h5ad:
  obs_names  == gse155468.h5ad obs_names (alignment by ID, not order)
  obsm['X_emb'] : (n, d) float32
  obs['celltype'] : str   (bio label, scoring only)
  obs['batch']    : str   (= orig.ident, method input / metric grouping only)
"""
from pathlib import Path
import hdf5plugin  # noqa: F401  (registers codecs used by the .h5ad files)
import numpy as np
import anndata as ad

GSE155468 = "/data/benchmark/data/cellPLM/data/gse155468.h5ad"
EMBD = "/data/benchmark/embd_clustering"


def load_labels():
    """Return (obs_names Index, celltype Series, batch Series) from the source dataset."""
    a = ad.read_h5ad(GSE155468, backed="r")
    obs = a.obs.copy()
    obs.index = obs.index.astype(str)
    return obs.index, obs["celltype"].astype(str), obs["orig.ident"].astype(str)


def write_integration_h5ad(emb, src_obs_names, out_path):
    """Build the contract AnnData from an embedding matrix and write it."""
    names, celltype, batch = load_labels()
    src_obs_names = [str(x) for x in src_obs_names]
    emb = np.asarray(emb, dtype=np.float32)
    out = ad.AnnData(
        X=np.zeros((emb.shape[0], 1), dtype=np.float32),
        obs={"celltype": celltype.loc[src_obs_names].values,
             "batch": batch.loc[src_obs_names].values},
    )
    out.obs_names = src_obs_names
    out.obsm["X_emb"] = emb
    assert list(out.obs_names) == list(src_obs_names)
    assert set(out.obs_names) == set(names), "obs_names diverge from gse155468.h5ad"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_path)
    print(f"wrote {out_path}  emb={emb.shape}  batches={len(set(out.obs['batch']))}")


def reemit_from_embd(model_dir, out_path):
    """Load embd_clustering/<model_dir>/gse155468_embedding.h5ad (embedding in .X) and re-emit."""
    src = ad.read_h5ad(f"{EMBD}/{model_dir}/gse155468_embedding.h5ad")
    write_integration_h5ad(np.asarray(src.X), [str(x) for x in src.obs_names], out_path)
```

- [ ] **Step 3: Checkpoint — helper import + label load**

Run:
```bash
cd /data/benchmark/batch_integration && JAX_PLATFORMS=cpu \
/home/hanqing-li/anaconda3/envs/scib-metrics/bin/python -c \
"import _common; n,c,b=_common.load_labels(); print(len(n), c.nunique(), b.nunique())"
```
Expected: `48082 11 11` (48082 cells, 11 celltypes, 11 batches/`orig.ident`).

---

### Task 3: Reuse-shim wrappers (scVI, scGPT-zeroshot, Geneformer, scFoundation, cellPLM)

**Files:**
- Create: `batch_integration/<W>/integration.py` and `.ipynb` for the 5 dirs below
- Create/Run: produces `batch_integration/<W>/gse155468_integration.h5ad`

Mapping (wrapper dir → reused `embd_clustering` source dir):
```
scVI-integration-wrapper          -> scVI-embedding-wrapper
scGPT-zeroshot-integration-wrapper-> scGPT-embedding-wrapper
Geneformer-integration-wrapper    -> Geneformer-embedding-wrapper
scFoundation-integration-wrapper  -> scFoundation-embedding-wrapper
cellPLM-integration-wrapper       -> cellPLM-embedding-wrapper
```

- [ ] **Step 1: Write a generator that emits all 5 thin `integration.py` files**

Create `/data/benchmark/batch_integration/_make_shims.py`:
```python
import os, textwrap
PAIRS = {
    "scVI-integration-wrapper": ("scVI-embedding-wrapper", "scVI", "batch-conditioned (batch_key='orig.ident' already set in the embd_clustering wrapper) -> faithful scVI integration"),
    "scGPT-zeroshot-integration-wrapper": ("scGPT-embedding-wrapper", "scGPT (zero-shot)", "frozen pretrained [CLS]; batch-naive reference point"),
    "Geneformer-integration-wrapper": ("Geneformer-embedding-wrapper", "Geneformer", "no native batch correction -> zero-shot rank-value embedding is its faithful integration approach"),
    "scFoundation-integration-wrapper": ("scFoundation-embedding-wrapper", "scFoundation", "no native batch correction -> zero-shot embedding"),
    "cellPLM-integration-wrapper": ("cellPLM-embedding-wrapper", "cellPLM", "zero-shot CellEmbeddingPipeline (paper's integration approach); RUN BUT NOT REPORTED"),
}
HDR = '''# %% [markdown]
# {title} — batch-integration wrapper (reuse shim)
# Faithful recipe: {why}.
# This wrapper re-emits embd_clustering/{src}/gse155468_embedding.h5ad
# under the integration output schema. No model is re-run here.
# Labels consumed by this model: NONE (zero-shot) / batch only (scVI).

# %%
import sys
sys.path.insert(0, "/data/benchmark/batch_integration")
import _common

_common.reemit_from_embd("{src}", "gse155468_integration.h5ad")
'''
for wdir,(src,title,why) in PAIRS.items():
    d=f"/data/benchmark/batch_integration/{wdir}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/integration.py","w").write(HDR.format(title=title, why=why, src=src))
    print("wrote", f"{d}/integration.py")
```

- [ ] **Step 2: Generate the shim `.py` files**

Run:
```bash
cd /data/benchmark/batch_integration && /home/hanqing-li/anaconda3/bin/python _make_shims.py
```
Expected: 5 `wrote .../integration.py` lines.

- [ ] **Step 3: Convert each to a notebook and execute it**

Run:
```bash
cd /data/benchmark/batch_integration
for w in scVI-integration-wrapper scGPT-zeroshot-integration-wrapper Geneformer-integration-wrapper scFoundation-integration-wrapper cellPLM-integration-wrapper; do
  JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python _py2nb.py $w/integration.py $w/integration.ipynb
  JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/jupyter nbconvert --to notebook --execute --inplace $w/integration.ipynb
done
```
Expected: each prints `wrote .../gse155468_integration.h5ad emb=(48082, d) batches=11`.

- [ ] **Step 4: Checkpoint — all 5 outputs exist with correct n_obs**

Run:
```bash
cd /data/benchmark/batch_integration && for w in scVI scGPT-zeroshot Geneformer scFoundation cellPLM; do JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python -c "import anndata as ad,hdf5plugin; a=ad.read_h5ad('$w-integration-wrapper/gse155468_integration.h5ad'); print('$w', a.shape, a.obsm['X_emb'].shape, a.obs['batch'].nunique())"; done
```
Expected: 5 lines, each `(48082, 1) (48082, d) 11`.

---

### Task 4: scGPT fine-tuned integration wrapper (new run)

**Files:**
- Create: `batch_integration/scGPT-finetuned-integration-wrapper/integration.py` + `.ipynb`
- Produces: `batch_integration/scGPT-finetuned-integration-wrapper/gse155468_integration.h5ad`
- Source of truth: `/data/benchmark/models/scGPT/tutorials/Tutorial_Integration.ipynb`

**Derivation recipe (faithful-wrapper convention — copy the tutorial's cells verbatim, apply ONLY these adaptations):**

1. Header markdown cell documenting: source = `Tutorial_Integration.ipynb`; labels consumed = **batch only** (`orig.ident` via DAB/DSBN), `celltype` never used; deviations list (below).
2. **Hyperparameters: copy `hyperparameter_defaults` (tutorial code cell #3) verbatim** — `seed=42, GEPC=True, ecs_thres=0.8, dab_weight=1.0, mask_ratio=0.4, epochs=15, n_bins=51, lr=1e-4` and the rest of that dict unchanged. Keep `load_model="../save/scGPT_human"` (resolve to `/data/benchmark/models/scGPT/save/scGPT_human`, same checkpoint used by `embd_clustering/scGPT-embedding-wrapper`).
3. **Data adaptation:** replace the tutorial's dataset-loading cell with:
   ```python
   import hdf5plugin, scanpy as sc
   adata = sc.read_h5ad("/data/benchmark/data/cellPLM/data/gse155468.h5ad")
   adata.obs["celltype"] = adata.obs["celltype"].astype(str)      # for eval only, NOT fed to model
   adata.obs["batch_id"] = adata.obs["orig.ident"].astype("category").cat.codes.values
   adata.obs["str_batch"] = adata.obs["orig.ident"].astype(str)
   adata.var["gene_name"] = adata.var_names                       # gse155468 var_names are symbols (matches scGPT vocab)
   data_is_raw = False                                            # gse155468 X is library-style counts; keep tutorial's preprocessing
   ```
   Wire `batch_key`/`cell_type` exactly where the tutorial uses its PBMC equivalents: batch → `str_batch`/`batch_id`, celltype label → `celltype` (used by the tutorial only for its eval/UMAP, not the loss).
4. **Strip wandb** (mirror `ct_annotation/scGPT-annotation-wrapper`): replace `wandb.*` calls with `print`/no-ops; keep `define_wandb_metrcis` as a stub.
5. **Embedding extraction:** keep the tutorial's own embedding routine (its `eval_testdata` / `get_batch_cell_embeddings` path that fills `adata.obsm["X_scGPT"]`). After training+extraction, append:
   ```python
   import sys; sys.path.insert(0, "/data/benchmark/batch_integration")
   import _common
   emb = adata.obsm["X_scGPT"]                                   # cell embeddings after fine-tune
   _common.write_integration_h5ad(emb, [str(x) for x in adata.obs_names], "gse155468_integration.h5ad")
   ```
6. Deviations to record in the header cell: dataset = GSE155468 (not PBMC_10K); wandb stripped; output re-emitted via `_common`; everything else (objectives, hyperparameters, freezing) verbatim.

- [ ] **Step 1: Write `integration.py` per the derivation recipe above**

Create `batch_integration/scGPT-finetuned-integration-wrapper/integration.py` using `# %%` cell markers (same style as `ct_annotation/scGPT-annotation-wrapper/annotation.py`), copying `Tutorial_Integration.ipynb` cells verbatim and applying adaptations 1–6.

- [ ] **Step 2: Generate the notebook**

Run:
```bash
cd /data/benchmark/batch_integration && /home/hanqing-li/anaconda3/envs/scgpt/bin/python _py2nb.py \
  scGPT-finetuned-integration-wrapper/integration.py scGPT-finetuned-integration-wrapper/integration.ipynb
```
Expected: `wrote .../integration.ipynb`.

- [ ] **Step 3: Smoke run (1 epoch) — edit `epochs=1`, execute, confirm it trains end-to-end**

Run:
```bash
cd /data/benchmark/batch_integration/scGPT-finetuned-integration-wrapper
sed -i 's/epochs=15/epochs=1/' integration.py
/home/hanqing-li/anaconda3/envs/scgpt/bin/python ../_py2nb.py integration.py integration.ipynb
timeout 3600 /home/hanqing-li/anaconda3/envs/scgpt/bin/jupyter nbconvert --to notebook --execute --inplace integration.ipynb
```
Expected: completes without error; prints a decreasing train loss and a final `wrote .../gse155468_integration.h5ad emb=(48082, d) batches=11`.

- [ ] **Step 4: Full run (restore `epochs=15`)**

Run:
```bash
cd /data/benchmark/batch_integration/scGPT-finetuned-integration-wrapper
sed -i 's/epochs=1/epochs=15/' integration.py
/home/hanqing-li/anaconda3/envs/scgpt/bin/python ../_py2nb.py integration.py integration.ipynb
timeout 21600 /home/hanqing-li/anaconda3/envs/scgpt/bin/jupyter nbconvert --to notebook --execute --inplace integration.ipynb
```
Expected: completes; final `wrote .../gse155468_integration.h5ad emb=(48082, d) batches=11`.

- [ ] **Step 5: Checkpoint — output schema**

Run:
```bash
JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python -c \
"import anndata as ad,hdf5plugin,numpy as np; a=ad.read_h5ad('/data/benchmark/batch_integration/scGPT-finetuned-integration-wrapper/gse155468_integration.h5ad'); e=a.obsm['X_emb']; print(a.shape, e.shape, 'finite', np.isfinite(e).all(), a.obs['batch'].nunique())"
```
Expected: `(48082, 1) (48082, d) finite True 11`.

---

### Task 5: Validation script for the output contract

**Files:** Create `batch_integration/_check_schema.py`

- [ ] **Step 1: Write the validator**

Create `/data/benchmark/batch_integration/_check_schema.py`:
```python
import sys, hdf5plugin, numpy as np, anndata as ad
sys.path.insert(0, "/data/benchmark/batch_integration")
import _common

WRAPPERS = ["scVI", "scGPT-zeroshot", "scGPT-finetuned", "Geneformer", "scFoundation", "cellPLM"]
names, celltype, batch = _common.load_labels()
ref = set(map(str, names))
ok = True
for w in WRAPPERS:
    p = f"/data/benchmark/batch_integration/{w}-integration-wrapper/gse155468_integration.h5ad"
    a = ad.read_h5ad(p)
    e = a.obsm["X_emb"]
    checks = {
        "n_obs==48082": a.n_obs == 48082,
        "obs_names match": set(map(str, a.obs_names)) == ref,
        "X_emb 2D float": e.ndim == 2 and np.issubdtype(np.asarray(e).dtype, np.floating),
        "X_emb finite": bool(np.isfinite(np.asarray(e)).all()),
        "celltype non-null": a.obs["celltype"].notna().all(),
        "batch non-null": a.obs["batch"].notna().all(),
    }
    for k, v in checks.items():
        if not v: ok = False; print(f"FAIL {w}: {k}")
    print(f"{'OK ' if all(checks.values()) else 'BAD'} {w}  emb={tuple(e.shape)}")
sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Checkpoint — run the validator**

Run:
```bash
JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python /data/benchmark/batch_integration/_check_schema.py; echo "exit=$?"
```
Expected: 6 `OK <model>` lines and `exit=0`.

---

### Task 6: Benchmark — scib-metrics + PCA baseline + artifacts

**Files:**
- Create: `batch_integration/benchmark.py` + `.ipynb`
- Produces: `batch_integration/metrics.csv`, `batch_integration/umap_batch_celltype.png`

- [ ] **Step 1: Write `benchmark.py`**

Create `/data/benchmark/batch_integration/benchmark.py` (`# %%` cells):
```python
# %% [markdown]
# Batch-integration benchmark — scib-metrics on GSE155468 (batch=orig.ident).
# Label firewall: batch_key only groups/inputs; label_key (celltype) only scores.
# cellPLM is computed but EXCLUDED from the reported slice (run-not-reported).

# %%
import hdf5plugin, numpy as np, scanpy as sc, anndata as ad, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker

SRC = "/data/benchmark/data/cellPLM/data/gse155468.h5ad"
BI = "/data/benchmark/batch_integration"
WRAPPERS = {  # display name -> wrapper dir
    "scVI": "scVI", "scGPT (zero-shot)": "scGPT-zeroshot",
    "scGPT (fine-tuned)": "scGPT-finetuned", "Geneformer": "Geneformer",
    "scFoundation": "scFoundation", "cellPLM": "cellPLM",
}

# %% Base AnnData: labels + a PCA baseline embedding (unintegrated reference)
base = sc.read_h5ad(SRC)
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
                   title=f"{key} — {color}", frameon=False)
fig.tight_layout(); fig.savefig(f"{BI}/umap_batch_celltype.png", dpi=150, facecolor="white")
print("wrote metrics.csv, metrics_reported.csv, umap_batch_celltype.png")
```

- [ ] **Step 2: Generate + execute the benchmark notebook**

Run:
```bash
cd /data/benchmark/batch_integration && JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python _py2nb.py benchmark.py benchmark.ipynb
cd /data/benchmark/batch_integration && JAX_PLATFORMS=cpu timeout 7200 /home/hanqing-li/anaconda3/envs/scib-metrics/bin/jupyter nbconvert --to notebook --execute --inplace benchmark.ipynb
```
Expected: completes; prints the results frame; final line `wrote metrics.csv, metrics_reported.csv, umap_batch_celltype.png`.

- [ ] **Step 3: Checkpoint — sanity of metrics**

Run:
```bash
JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python -c "
import pandas as pd; r=pd.read_csv('/data/benchmark/batch_integration/metrics.csv', index_col=0)
print(r)
b='Batch correction'
assert r.loc['PCA',b] <= r.loc['scVI',b], 'sanity: scVI should beat PCA on batch'
print('SANITY OK: scGPT ft vs zs (Batch) =', r.loc['scGPT (fine-tuned)',b], 'vs', r.loc['scGPT (zero-shot)',b])
"
```
Expected: results table prints; `SANITY OK ...` line (assertion passes — scVI ≥ PCA on Batch correction). Note the scGPT ft-vs-zs numbers for the slide notes.

---

### Task 7: Swap the deck's Results-1 slide in place

**Files:**
- Create: `presentation/swap_results1_slide.py`
- Modify: `presentation/build_deck.py` (Results-1 builder only — for from-scratch consistency)

- [ ] **Step 1: Snapshot the live deck before editing**

Run:
```bash
cp /data/benchmark/presentation/sc_foundation_models.pptx /data/benchmark/presentation/sc_foundation_models.bak.pptx
/home/hanqing-li/anaconda3/bin/python -c "
from pptx import Presentation; p=Presentation('/data/benchmark/presentation/sc_foundation_models.pptx')
print('slides', len(p.slides._sldIdLst))
for i,s in enumerate(p.slides,1):
    t=next((sh.text_frame.text.split(chr(10))[0] for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()), '')
    pics=sum(1 for sh in s.shapes if sh.shape_type==13)
    print(i, pics, t[:48])
"
```
Expected: prints slide count + per-slide (index, #pics, title). Record the Results-1 index (title starts `Results 1`) and each model slide's pic count.

- [ ] **Step 2: Write `swap_results1_slide.py`**

Create `/data/benchmark/presentation/swap_results1_slide.py`:
```python
"""Replace the Results-1 slide IN PLACE (title, chart picture, speaker notes).
Does NOT rerun build_deck.py — pasted paper figures on model slides must survive."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor

PPTX = "/data/benchmark/presentation/sc_foundation_models.pptx"
CSV  = "/data/benchmark/batch_integration/metrics_reported.csv"
CHART = "/data/benchmark/presentation/figures/res_integration.png"
BLACK = RGBColor(0, 0, 0)

# --- chart from scib-metrics reported slice ---
r = pd.read_csv(CSV, index_col=0)
cols = [c for c in ["Total", "Bio conservation", "Batch correction"] if c in r.columns]
models = list(r.index)
fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=210)
x = np.arange(len(models)); w = 0.8 / len(cols)
palette = ["#1F5C9E", "#5BA071", "#E0883B"]
for i, c in enumerate(cols):
    vals = r[c].values
    bars = ax.bar(x + (i - (len(cols)-1)/2)*w, vals, w, label=c, color=palette[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="black")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11, color="black")
ax.set_ylim(0, 1.0); ax.tick_params(colors="black")
ax.set_title("Batch integration — GSE155468 (scIB)", fontsize=14, color="black")
ax.legend(frameon=False, ncol=len(cols), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=10)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(CHART, bbox_inches="tight", facecolor="white"); plt.close(fig)

NOTES = (
 "Setup: data integration on GSE155468, batch=orig.ident (11 samples), bio=celltype. "
 "scIB panel via scib-metrics; PCA = unintegrated reference; cellPLM excluded. "
 "Label firewall: batch only as method input/grouping, celltype only for scoring (no model saw celltype).\\n\\n"
 "Headline — scGPT zero-shot vs fine-tuned: the fine-tune (Tutorial_Integration: MLM+GEPC+ECS plus DAB adversarial + DSBN, "
 "batch=orig.ident) should raise the Batch-correction axis vs the batch-naive zero-shot [CLS]; watch whether Bio conservation "
 "is traded off. NOT a clean ablation — the fine-tune also adapts in-domain and uses a 1200-HVG set, so attribute gains to the "
 "recipe as a whole, not DAB/DSBN alone.\\n\\n"
 "Asymmetry (stated, not hidden): scVI and scGPT-finetuned consume the batch label by design; Geneformer/scFoundation/"
 "scGPT-zeroshot are zero-shot with no batch input — expected to mix batches less. This is what the benchmark measures, not leakage. "
 "Caveat: GSE155468 batch is within-study samples (milder than a cross-protocol atlas) — the only genome-wide shipped dataset, "
 "kept so the foundation-model comparison stays fair. ARI/NMI from the old clustering slide live on inside the Bio-conservation axis."
)

prs = Presentation(PPTX)
target = None
for s in prs.slides:
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("Results 1"):
            target = s; title_shape = sh; break
    if target: break
assert target is not None, "Results-1 slide not found"

# retitle (preserve the title textbox; rewrite its first run)
tf = title_shape.text_frame
tf.text = "Results 1 - batch integration (GSE155468)"
for p in tf.paragraphs:
    for run in p.runs:
        run.font.size = Pt(32); run.font.bold = True; run.font.color.rgb = BLACK

# remove the old subtitle textbox if present (the clustering "GSE155468: ... aneurysm" line) and any picture
for sh in list(target.shapes):
    if sh.has_text_frame and sh is not title_shape and sh.text_frame.text.strip().startswith("GSE155468:"):
        sh._element.getparent().remove(sh._element)
    if sh.shape_type == 13:  # PICTURE
        old_l, old_t, old_w, old_h = sh.left, sh.top, sh.width, sh.height
        sh._element.getparent().remove(sh._element)

# place the new chart where the old chart was (fallback to a sensible box)
try:
    L, T, W = old_l, old_t, old_w
except NameError:
    L, T, W = Inches(1.6), Inches(2.0), Inches(13.333 - 3.2)
target.shapes.add_picture(CHART, L, T, width=W)

# replace speaker notes
target.notes_slide.notes_text_frame.text = NOTES.replace("\\n", "\n")
prs.save(PPTX)
print("Results-1 slide swapped in place; chart:", CHART)
```

- [ ] **Step 3: Run the swap**

Run:
```bash
/home/hanqing-li/anaconda3/bin/python /data/benchmark/presentation/swap_results1_slide.py
```
Expected: `Results-1 slide swapped in place; chart: .../res_integration.png`.

- [ ] **Step 4: Checkpoint — deck integrity (count unchanged, model figures preserved)**

Run:
```bash
/home/hanqing-li/anaconda3/bin/python -c "
from pptx import Presentation
new=Presentation('/data/benchmark/presentation/sc_foundation_models.pptx')
old=Presentation('/data/benchmark/presentation/sc_foundation_models.bak.pptx')
assert len(new.slides._sldIdLst)==len(old.slides._sldIdLst), 'slide count changed!'
def pics(p): return [sum(1 for sh in s.shapes if sh.shape_type==13) for s in p.slides]
po, pn = pics(old), pics(new)
print('counts', len(new.slides._sldIdLst), 'pics old/new', po, pn)
# every slide except Results-1 keeps its picture count (model slides keep pasted figures)
ti=[i for i,s in enumerate(new.slides) if any(sh.has_text_frame and sh.text_frame.text.strip().startswith('Results 1 - batch') for sh in s.shapes)][0]
for i,(a,b) in enumerate(zip(po,pn)):
    assert i==ti or a==b, f'slide {i} picture count changed {a}->{b}'
r1=list(new.slides)[ti]
assert any(sh.shape_type==13 for sh in r1.shapes), 'no chart on new Results-1'
assert 'fine-tuned' in r1.notes_slide.notes_text_frame.text
print('DECK OK: count unchanged, model figures preserved, Results-1 has chart+notes')
"
```
Expected: `DECK OK: ...`.

- [ ] **Step 5: Visual check — render Results-1 + one model slide**

Run:
```bash
cd /data/benchmark/presentation && mkdir -p _chk && rm -f _chk/*.png sc_foundation_models.pdf
timeout 120 soffice --headless --convert-to pdf sc_foundation_models.pptx >/dev/null 2>&1
pdftoppm -png -r 110 sc_foundation_models.pdf _chk/s
ls _chk
```
Expected: PNGs rendered. **Read** `_chk/s-09.png` (Results-1; index from Step-1 record — adjust if not 9) and a model slide (e.g. `_chk/s-04.png`); confirm Results-1 shows the integration chart + new title and the model slide still shows its pasted paper figure.

- [ ] **Step 6: Patch `build_deck.py` Results-1 builder for from-scratch consistency**

In `presentation/build_deck.py`, locate the Results-1 slide block (the one titled "Results 1 …", currently the clustering chart `res_clustering.png`). Replace its title string with `"Results 1 - batch integration (GSE155468)"`, its picture with `presentation/figures/res_integration.png`, and its `notes(...)` body with the `NOTES` text from `swap_results1_slide.py`. Do not run `build_deck.py`. Verify by inspection only:

Run:
```bash
grep -n "res_integration\|Results 1 - batch integration" /data/benchmark/presentation/build_deck.py
```
Expected: ≥2 matching lines (title + picture path present).

- [ ] **Step 7: Checkpoint — clean intermediates**

Run:
```bash
rm -rf /data/benchmark/presentation/_chk /data/benchmark/presentation/sc_foundation_models.pdf
ls /data/benchmark/presentation/figures/res_integration.png && echo "artifact kept"
```
Expected: `.../res_integration.png` exists, `artifact kept`.

---

### Task 8: README + final end-to-end verification

**Files:** Create `batch_integration/README.md`

- [ ] **Step 1: Write `README.md`**

Create `/data/benchmark/batch_integration/README.md` covering: task description; the per-model recipe table (from the spec, incl. scGPT zero-shot vs fine-tuned and cellPLM run-not-reported); the label-firewall / evaluation-validity note; the env pin (paste the `grep` lines captured in Task 1 Step 3); exact re-run commands for each wrapper + benchmark + deck swap; the caveats (within-study batch, not-a-clean-ablation, asymmetry).

- [ ] **Step 2: Final checkpoint — full artifact sweep**

Run:
```bash
cd /data/benchmark/batch_integration && JAX_PLATFORMS=cpu /home/hanqing-li/anaconda3/envs/scib-metrics/bin/python _check_schema.py && ls metrics.csv metrics_reported.csv umap_batch_celltype.png README.md && /home/hanqing-li/anaconda3/bin/python -c "
from pptx import Presentation; p=Presentation('/data/benchmark/presentation/sc_foundation_models.pptx')
ok=any(sh.has_text_frame and sh.text_frame.text.strip().startswith('Results 1 - batch integration') for s in p.slides for sh in s.shapes)
print('deck Results-1 swapped:', ok)
"
```
Expected: `_check_schema.py` all-OK exit 0; the four artifacts listed; `deck Results-1 swapped: True`.

- [ ] **Step 3: Checkpoint — remove backup once satisfied**

Run (only after the user has visually approved the deck):
```bash
rm -f /data/benchmark/presentation/sc_foundation_models.bak.pptx
echo "done"
```
Expected: `done`.

---

## Self-Review

**1. Spec coverage:**
- Dataset GSE155468 sample-as-batch → Task 2 (`_common.load_labels` uses `orig.ident`), Task 6 (PCA on same data). ✔
- Per-model recipe incl. scGPT zero-shot vs fine-tuned, cellPLM run-not-reported → Tasks 3 & 4; benchmark excludes cellPLM in `metrics_reported.csv` (Task 6). ✔
- Reuse rule (only scGPT-finetuned re-runs) → Task 3 (shims) vs Task 4. ✔
- Uniform output schema (6 wrappers) → `_common.write_integration_h5ad`, `_check_schema.py` (Task 5). ✔
- scIB-standard metrics via dedicated CPU env → Task 1 + Task 6 `Benchmarker`. ✔
- Evaluation validity / label firewall → enforced in `_common` (celltype only attached for scoring; scGPT recipe step 3 marks celltype "NOT fed to model"); documented in notes/README. ✔
- Deck = replace Results-1 in place, not rerun build_deck → Task 7 (swap script + integrity check that model pics preserved) + Step 6 patch. ✔
- Caveats (asymmetry, not-clean-ablation, within-study batch) → in `NOTES` (Task 7) + README (Task 8). ✔

**2. Placeholder scan:** No TBD/TODO. The scGPT wrapper is specified as a derivation recipe (1–6) with exact adaptation code, mirroring the repo's established faithful-wrapper convention (`ct_annotation` did the same) rather than reproducing the upstream tutorial verbatim — acceptable and explicit.

**3. Type/name consistency:** `write_integration_h5ad(emb, src_obs_names, out_path)` and `reemit_from_embd(model_dir, out_path)` used consistently in Tasks 3–4; `obsm['X_emb']`, `obs['celltype']`, `obs['batch']` consistent across `_common`, `_check_schema.py`, `benchmark.py`; wrapper dir names consistent (`<Model>-integration-wrapper`, scGPT split into `scGPT-zeroshot`/`scGPT-finetuned`); display names in `benchmark.py` match indices used by the sanity check and `swap_results1_slide.py`.

No gaps found.
