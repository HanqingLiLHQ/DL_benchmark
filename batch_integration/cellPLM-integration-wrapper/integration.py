# %% [markdown]
# cellPLM - batch-integration wrapper (reuse shim)
# Faithful recipe: zero-shot CellEmbeddingPipeline (paper's integration approach); RUN BUT NOT REPORTED.
# This wrapper re-emits embd_clustering/cellPLM-embedding-wrapper/gse155468_embedding.h5ad
# under the integration output schema. No model is re-run here.
# Labels consumed by this model: NONE (zero-shot) / batch only (scVI).

# %%
import sys
sys.path.insert(0, "/data/benchmark/batch_integration")
import _common

_common.reemit_from_embd("cellPLM-embedding-wrapper", "gse155468_integration.h5ad")
