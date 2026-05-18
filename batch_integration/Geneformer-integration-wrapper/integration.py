# %% [markdown]
# Geneformer - batch-integration wrapper (reuse shim)
# Faithful recipe: no native batch correction -> zero-shot rank-value embedding is its faithful integration approach.
# This wrapper re-emits embd_clustering/Geneformer-embedding-wrapper/gse155468_embedding.h5ad
# under the integration output schema. No model is re-run here.
# Labels consumed by this model: NONE (zero-shot) / batch only (scVI).

# %%
import sys
sys.path.insert(0, "/data/benchmark/batch_integration")
import _common

_common.reemit_from_embd("Geneformer-embedding-wrapper", "gse155468_integration.h5ad")
