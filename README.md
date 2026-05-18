# Single-cell foundation model benchmarks

Code for comparing single-cell foundation models (scGPT, Geneformer,
scFoundation, cellPLM) against classical baselines (scVI/scANVI, PCA) on three
downstream tasks. Each model is run with the preprocessing, hyperparameters,
and evaluation protocol from its own paper/tutorial, so results reflect the
published methods rather than tuned defaults.

## Tasks

| Directory | Task | Dataset | Models |
|---|---|---|---|
| `ct_annotation/` | Supervised cell-type annotation | MS snRNA-seq | scGPT, Geneformer, scFoundation, scVI/scANVI, cellPLM |
| `embd_clustering/` | Zero-shot embedding + clustering | GSE155468 | scGPT, Geneformer, scFoundation, scVI, cellPLM |
| `batch_integration/` | Batch-effect removal (scIB) | GSE155468 (`batch=orig.ident`) | + scGPT zero-shot vs fine-tuned |

Each task directory has its own `README.md` with the per-model recipe table,
the exact metrics, run commands, and caveats. Read those for details and to
interpret the numbers.

## Layout

```
ct_annotation/      <model>-annotation-wrapper/, benchmark.py, metrics.csv, README
embd_clustering/    <model>-embedding-wrapper/, benchmark.ipynb, README
batch_integration/  <model>-integration-wrapper/, benchmark.py, metrics*.csv,
                    TIMINGS.txt, README
models/             model source + pretrained weights        (git-ignored, ~5 GB)
data/               datasets + checkpoints                   (git-ignored, ~4 GB)
presentation/       lab-talk deck + build scripts            (git-ignored)
env/                per-model conda/pip setup scripts
docs/superpowers/   specs/ and plans/ per benchmark
utils/              gene_converter.py (shared helper)
```

## How a task runs

1. Set up the model's environment: `bash env/setup_<model>.sh`.
2. Run that model's `<model>-*-wrapper/` in its env. Each wrapper writes a
   `.h5ad` in a uniform schema (embedding + labels).
3. The task's `benchmark.py` (converted to a notebook and executed) loads all
   wrappers' outputs, scores them, and writes `metrics.csv` and figures.

Zero-shot embeddings are computed once in `embd_clustering/` and reused by the
other tasks; only the heavy models are re-trained per task.

## Environments

Models have conflicting dependencies, so each runs in its own conda env (see
`env/setup_*.sh`). Scoring for `batch_integration/` uses a separate
`scib-metrics` env with jax-CUDA on the local GPU; see
`batch_integration/README.md` for the exact env, the required scib-metrics
`cdist` memory patch, and the GPU run flags.

## Not tracked in git

`.gitignore` excludes `data/`, `models/`, all `*.h5ad`/checkpoints/archives,
the `presentation/` directory, `.claude/`, and each task's underscore helper
scripts (`_common.py`, `_py2nb.py`, `_make_shims.py`, `_check_schema.py`,
`_fix_silhouette_cdist.py`, `_test_silhouette_fix.py`). Tracked: the
`*-wrapper/` code, each task's `benchmark.py`/`benchmark.ipynb`, READMEs,
`metrics*.csv`, `TIMINGS.txt`, `docs/`, `env/`, and `utils/`. A clone records
how each model was run but is not directly re-runnable on its own — get the
datasets, weights, and helper scripts via `env/setup_*.sh` and the per-task
instructions.
