# %% [markdown]
# scGPT (zero-shot) - batch-integration wrapper (reuse shim)
# Faithful recipe: frozen pretrained [CLS]; batch-naive reference point.
# This wrapper re-emits embd_clustering/scGPT-embedding-wrapper/gse155468_embedding.h5ad
# under the integration output schema. No model is re-run here.
# Labels consumed by this model: NONE (zero-shot) / batch only (scVI).

# %%
import sys
sys.path.insert(0, "/data/benchmark/batch_integration")
import _common

_common.reemit_from_embd("scGPT-embedding-wrapper", "gse155468_integration.h5ad")
