# Single-Cell Foundation Models — 10-min Presentation — Design

**Date:** 2026-05-17
**Output:** `presentation/sc_foundation_models.pptx` built by `presentation/build_deck.py` (python-pptx). Figures cached under `presentation/figures/`.
**Audience:** lab meeting (technical peers — bio/ML). Minimal on-slide text; prose in speaker notes.
**Time budget:** ~10 min, ~11 slides.

## Scope

- Intro (why + common template): 2 slides.
- Comparative design, one slide per model, **excluding cellPLM**: scGPT, Geneformer, scFoundation, scVI (4 slides).
- Design-at-a-glance matrix: 1 slide.
- Results, one slide per task: CT annotation (MS), embedding/clustering (GSE155468): 2 slides.
- Title + takeaways: 2 slides.
- **cellPLM is dropped from both design and results.** Tables show only models that also got a design slide (+ PCA baseline for clustering).

## Comparison axes (identical on every model slide)

1. High-level architecture (VAE / BERT-encoder / asymmetric enc–dec / generative transformer)
2. Gene embedding construction
3. Cell-level embedding aggregation
4. Loss / pretraining objective (BERT-MLM vs GPT-style generative vs ELBO)

All slide bullets and speaker notes are written in our own words (concise technical summaries). No paper text is reproduced. Architecture figures are the authors' own schematics, embedded small with a visible source citation line, for an internal research/educational talk.

## Slide list

| # | Slide | On slide | Notes |
|---|-------|----------|-------|
| 1 | Title | "Benchmarking Single-Cell Foundation Models: Design & Evaluation", date; no author line | — |
| 2 | Why sc-FMs | 4 keywords: high-dim & sparse · labels expensive · batch effects · pretrain→transfer | full argument |
| 3 | Common sc-FM template | pipeline schematic (we draw): counts → gene/expr tokenization → transformer → cell embedding → SSL pretrain → downstream | walk pipeline, introduce 4 axes |
| 4 | scGPT | cited fig + 4 axis bullets | design + our usage |
| 5 | Geneformer | cited fig + 4 axis bullets | rank-value encoding, MLM |
| 6 | scFoundation | cited fig + 4 axis bullets | asymmetric enc-dec, RDA |
| 7 | scVI (VAE baseline) | cited fig + 4 axis bullets (adapted) | non-transformer baseline; scANVI for annotation |
| 8 | Design at a glance | 4×4 matrix (rows = axes, cols = models) | read-down contrast |
| 9 | Task 1 — CT annotation | "MS dataset" line + matplotlib bar chart (accuracy, macro-F1); optional small `confusion_matrices.png` thumbnail (annotation task artifact) | interpretation |
| 10 | Task 2 — Embedding & clustering | "GSE155468, 48k aortic cells" + matplotlib bar chart (ARI, NMI) | interpretation |
| 11 | Takeaways | 3 bullets | closing |

## Per-model slide content (locked)

### scGPT — Cui et al., *Nature Methods* 2024
- **Arch:** stacked Transformer encoder; generative ("GPT-style") pretraining via specialized attention masking.
- **Gene emb:** per gene = learned gene-token emb + **binned** expression-value emb (51 bins) + optional condition tokens; cell = sequence of gene tokens.
- **Cell agg:** dedicated `<cls>` token; final hidden state = cell embedding.
- **Loss:** iterative masked expression-value prediction (generative) + auxiliary cell/gene objectives; pretrained ~33M human cells (whole-human ckpt).
- **Our use:** end-to-end fine-tune w/ CLS head (annotation); `<cls>` emb (clustering).

### Geneformer — Theodoris et al., *Nature* 2023
- **Arch:** BERT-style Transformer encoder; masked language modeling.
- **Gene emb:** **rank-value encoding** — per cell, genes ordered by normalized-expression rank; input = rank-ordered gene-token sequence; no numeric expression fed in.
- **Cell agg:** mean-pool over contextual gene-token embeddings.
- **Loss:** masked gene prediction (BERT MLM) over the rank sequence; pretrained Genecorpus-30M; we use V2-104M (d=512).
- **Our use:** HF Trainer fine-tune, freeze 2 layers (annotation); mean-pool emb (clustering).

### scFoundation / xTrimoGene — Hao et al., *Nature Methods* 2024
- **Arch:** asymmetric Transformer **encoder–decoder**; encoder sees only non-zero genes, decoder reconstructs all.
- **Gene emb:** **continuous** expression-value embedding (no binning) via learned value encoder + gene token; read-depth-aware (RDA) total-count tokens (T target, S source).
- **Cell agg:** pool over gene dim; `POOL_TYPE='all'` concatenates encoder/decoder summaries → 3072-d.
- **Loss:** masked expression **regression** with read-depth-aware modeling (predict masked + low→high-depth expression); 100M params, ~50M cells.
- **Our use:** linear probing, one backbone layer unfrozen (annotation); 3072-d pooled emb (clustering).

### scVI — Lopez et al., *Nature Methods* 2018 (VAE baseline)
- **Arch:** variational autoencoder (encoder/decoder MLPs); **not pretrained — trained per dataset**.
- **Gene emb:** none in token sense; input = full gene count vector; encoder MLP → latent z.
- **Cell agg:** latent posterior mean z **is** the cell embedding (n_latent=30 here); no sequence/pooling.
- **Loss:** negative ELBO = ZINB/NB reconstruction NLL + KL(q(z|x)‖p(z)); neither BERT nor GPT.
- **Our use:** scANVI (semi-supervised label head, query="Unknown") for annotation; scVI latent z for clustering.

## Figure sourcing (revised — copyright-safe)

We do **not** embed the paper bitmaps wholesale. Instead, for each model we draw an **original, faithful schematic** in matplotlib (uniform visual language across the 4 comparison slides) and print a source citation to the paper on the slide. The author-figure URLs in the repos were only badges; original schematics also give a consistent comparison look and avoid reproducing copyrighted figures. The user can swap in actual paper figures themselves (slides are cited).

## Results data (extracted; cellPLM excluded)

**Task 1 — CT annotation, MS dataset** (`metrics.csv`; sorted macro-F1):

| Model | Accuracy | macro-F1 |
|---|---|---|
| scANVI | 0.867 | 0.743 |
| scFoundation | 0.834 | 0.702 |
| Geneformer | 0.831 | 0.685 |
| scGPT | 0.846 | 0.663 |

**Task 2 — Embedding/clustering, GSE155468** (48,082 cells × 12,382 genes, human ascending thoracic aortic aneurysm; kNN-Leiden, best over res 0.1–1.4 vs ground-truth celltype):

| Embedding | ARI | NMI |
|---|---|---|
| scVI (n_latent=30) | 0.874 | 0.845 |
| scGPT | 0.869 | 0.849 |
| PCA baseline (4500 HVG, 512 PC) | 0.842 | 0.829 |
| Geneformer (V2-104M) | 0.643 | 0.761 |
| scFoundation (100M, pool=all, 3072-d) | 0.615 | 0.769 |

## Speaker-note interpretation (results slides)

**Task 1:** Accuracies cluster tightly (0.83–0.87); macro-F1 separates them. scANVI (a lightweight semi-supervised VAE) tops the shown models — large pretrained FMs don't automatically win supervised single-dataset annotation. scGPT shows the largest accuracy↔macro-F1 gap (0.846→0.663): strong on common cell types, under-predicts rare classes (macro-averaging penalizes class imbalance). Recipes are paper-faithful, so the ranking reflects each published method under its default/linear-probe protocol (Geneformer hyperparams re-tuned to a celltype example; scFoundation only linear-probed w/ one unfrozen layer; scGPT full fine-tune), not a tuned bake-off.

**Task 2:** scVI and scGPT lead and both edge out the strong PCA baseline (modest gains). Geneformer and scFoundation fall **below PCA on ARI** while staying competitive on NMI (~0.76): embeddings retain biological signal (NMI) but their cluster partition doesn't align 1:1 with celltypes (low ARI) — over/under-segmentation, scale mismatch. Zero-shot embedding quality varies widely; rank-encoding and very-high-dim pooled embeddings cluster worse out of the box than a per-dataset VAE or scGPT's `<cls>`. PCA remains a brutally strong baseline (recurring sc-FM finding). The NMI-vs-ARI divergence is the key teaching point.

**Takeaways:** (1) sc-FMs share a template but differ on every axis; design choices matter. (2) No universal winner — scANVI best at supervised annotation here; scVI/scGPT best at unsupervised clustering; classical baselines stay competitive. (3) Protocol matters: accuracy vs macro-F1, ARI vs NMI tell different stories; paper-faithful ≠ tuned-optimal.

## Build approach

- Single script `presentation/build_deck.py`, `python-pptx`. 16:9. One helper per slide-type (title, bullets, figure+bullets, table, matrix, bar-chart).
- Speaker notes set via `slide.notes_slide.notes_text_frame`.
- Bar charts for results rendered with matplotlib → PNG → embedded (consistent styling); tables via pptx native table.
- Figures fetched once into `presentation/figures/`; script is idempotent (skips re-download if cached).
- Repo is **not** under git, so the design doc is not committed (cannot). Noted as a deviation from the brainstorming default.

## Out of scope

- Re-running any model or recomputing metrics (use existing `metrics.csv` / notebook outputs).
- cellPLM (excluded by request).
- Animations, themes beyond a clean default.
