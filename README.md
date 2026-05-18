# Benchmarking Single-Cell Foundation Models

Paper-faithful comparison of single-cell foundation models (scGPT, Geneformer,
scFoundation, cellPLM) against strong classical baselines (scVI/scANVI, PCA)
across three downstream tasks, plus a lab-talk deck.

**Guiding principle — paper-faithfulness.** Every model is run with *its own*
published preprocessing, hyperparameters, and evaluation protocol (from the
original paper / official tutorial), not "sensible defaults". Differences in
results reflect the published methods as practitioners would actually run
them, not a tuned bake-off. Each wrapper header cites its source recipe.

## Benchmark tasks

| Dir | Task | Dataset | Metric | Result |
|---|---|---|---|---|
| `ct_annotation/` | Supervised cell-type annotation | MS snRNA-seq (train `c_data` → test `filtered_ms_adata`) | accuracy, macro-F1 | scANVI leads (macro-F1 0.743); FMs do not dominate |
| `embd_clustering/` | Zero-shot embedding → clustering | GSE155468 | best ARI / NMI over a resolution sweep | scVI & scGPT edge out a strong PCA baseline |
| `batch_integration/` | Batch-effect removal / integration | GSE155468 (`batch=orig.ident`, 11 samples) | scIB panel (`scib-metrics`) | see `batch_integration/README.md` |

Each task dir has its own `README.md` with the per-model recipe table, the
label-firewall / evaluation-validity note, env pins, exact re-run commands,
and caveats. **Read those before interpreting numbers.**

## Layout

```
ct_annotation/        annotation benchmark (wrappers + benchmark.py + README)
embd_clustering/      embedding/clustering benchmark (wrappers + README)
batch_integration/    integration benchmark (wrappers + benchmark.py + README)
                      + _fix_silhouette_cdist.py (scib-metrics OOM fix, see its README)
models/               model code + pretrained weights  (LARGE, git-ignored)
data/                 datasets + checkpoints           (LARGE, git-ignored)
presentation/         lab-talk deck + build tooling      (local only, git-ignored)
env/                  per-model conda/pip setup scripts
docs/superpowers/     specs/ and plans/ for each benchmark
utils/                shared helpers
```

Each benchmark uses the same pattern: a per-model `*-wrapper/` that emits a
uniform-schema `.h5ad`, a schema check, then a `benchmark.py` (converted to a
notebook via `_py2nb.py`) that scores all models and writes
`metrics.csv` + figures.

## Environments

Models have conflicting dependencies, so each runs in its own conda env;
see `env/setup_*.sh`. The scIB scoring + model-free steps use a dedicated
`scib-metrics` env. GPU: RTX PRO 6000 Blackwell (sm_120); jax-CUDA is enabled
for the integration benchmark (run with `JAX_DEFAULT_MATMUL_PRECISION=highest`,
without `JAX_PLATFORMS=cpu`).

> Note: `batch_integration/` re-runs scib-metrics with a required
> memory-efficient `cdist` patch (`_fix_silhouette_cdist.py`) — stock
> scib-metrics 0.5.9 OOMs (~156 GB) on scFoundation's 3072-d embedding.
> The patch is numerically equivalent to stock; details in
> `batch_integration/README.md`.

## Reproduce

Per-task instructions live in each task's `README.md`. General flow:

```bash
# 1. set up the model env (once per model)
bash env/setup_<model>.sh

# 2. run that model's wrapper in its env -> uniform-schema .h5ad
# 3. schema-check, then run the benchmark:
python <task>/_py2nb.py <task>/benchmark.py <task>/benchmark.ipynb
jupyter nbconvert --to notebook --execute --inplace <task>/benchmark.ipynb
# -> <task>/metrics.csv + figures
```

Reused embeddings: only the heavy models are re-trained per task; zero-shot
embeddings are computed once in `embd_clustering/` and re-emitted under each
task's schema (see the task READMEs / `docs/superpowers/specs`).

## Presentation (local only — not version-controlled)

`presentation/sc_foundation_models.pptx` — a ~10-min lab talk. The whole
`presentation/` directory is **git-ignored** by request. **Do not rerun
`build_deck.py`** on the live deck: it would wipe manually pasted paper
figures. Add/replace result slides non-destructively with
`presentation/add_integration_slide.py` (appends a slide from
`batch_integration/metrics_reported.csv`).

## What is not in git

`.gitignore` excludes: `data/` (~4 GB), `models/` (~5 GB), all `*.h5ad`
embeddings and checkpoints, the entire `presentation/` directory, `.claude/`,
and each task's underscore helper/glue/test scripts (`_common.py`, `_py2nb.py`,
`_make_shims.py`, `_check_schema.py`, `_fix_silhouette_cdist.py`,
`_test_silhouette_fix.py`). **Tracked:** the per-model `*-wrapper/` code, each
task's `benchmark.py`, specs/plans, READMEs, and small text results
(`metrics.csv`, `TIMINGS.txt`). A fresh clone therefore documents *how* each
model was run but is not directly re-runnable (the ignored helpers, datasets,
and weights are obtained via `env/setup_*.sh` and the per-model instructions).
