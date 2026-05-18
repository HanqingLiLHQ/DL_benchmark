# %% [markdown]
# scFoundation - batch-integration wrapper (reuse shim)
# Faithful recipe: no native batch correction -> zero-shot embedding.
# This wrapper re-emits embd_clustering/scFoundation-embedding-wrapper/gse155468_embedding.h5ad
# under the integration output schema. No model is re-run here.
# Labels consumed by this model: NONE (zero-shot) / batch only (scVI).

# %%
import sys
sys.path.insert(0, "/data/benchmark/batch_integration")
import _common

_common.reemit_from_embd("scFoundation-embedding-wrapper", "gse155468_integration.h5ad")
