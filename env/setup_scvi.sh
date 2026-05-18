#!/usr/bin/env bash
# Set up a conda environment capable of running scVI / scvi-tools.
# Reference: https://docs.scvi-tools.org
#
# Usage:
#   bash env/setup_scvi.sh
#
# Unlike scGPT / cellPLM / Geneformer / scFoundation, scVI is not a foundation
# model — there is no pretrained checkpoint to download. You train a small VAE
# per-dataset, so the only "weights" are whatever your training run produces.
# This script just creates the conda env, installs the right CUDA-matched
# torch wheel, and pip-installs scvi-tools from PyPI.

set -euo pipefail

ENV_NAME="${SCVI_ENV_NAME:-scvi}"
PY_VERSION="3.11"   # scvi-tools >=1.2 requires Python >= 3.10; 3.11 is the recommended fast path

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 0. probe hardware -------------------------------------------------------
echo "=== Hardware probe ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[warn] nvidia-smi not found. scVI will run on CPU only — much slower." >&2
  TORCH_CUDA_TAG=""
else
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
  DRIVER_CUDA="$(nvidia-smi | awk '/CUDA Version/ {for(i=1;i<=NF;i++) if($i=="Version:") print $(i+1)}' | head -n1)"

  CC_MAJOR="${COMPUTE_CAP%%.*}"
  CC_MINOR="${COMPUTE_CAP##*.}"
  CC_NUM=$(( CC_MAJOR * 10 + CC_MINOR ))

  # Same CUDA-tag selection logic as the other setup scripts.
  if [ "${CC_NUM}" -ge 120 ]; then
    TORCH_CUDA_TAG="cu128"
  elif [ "${CC_NUM}" -ge 100 ]; then
    TORCH_CUDA_TAG="cu124"
  elif [ "${CC_NUM}" -ge 90 ]; then
    TORCH_CUDA_TAG="cu121"
  else
    TORCH_CUDA_TAG="cu118"
  fi

  echo "  GPU              : ${GPU_NAME}"
  echo "  Compute cap      : ${COMPUTE_CAP}  (sm_${CC_MAJOR}${CC_MINOR})"
  echo "  Driver CUDA max  : ${DRIVER_CUDA}"
  echo "  Chosen torch tag : ${TORCH_CUDA_TAG}"
fi
echo

# ---- 1. conda available ------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# Make `conda activate` work inside this non-interactive shell.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 2. create env -----------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[skip] conda env '${ENV_NAME}' already exists."
else
  echo "[run]  Creating conda env '${ENV_NAME}' (Python ${PY_VERSION})..."
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}" pip
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip wheel setuptools

# ---- 3. PyTorch matched to this GPU -----------------------------------------
if [ -n "${TORCH_CUDA_TAG}" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
  echo "[run]  Installing PyTorch from ${TORCH_INDEX}..."
  pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX}"
else
  echo "[run]  Installing CPU-only PyTorch (no GPU detected)..."
  pip install --upgrade torch torchvision torchaudio
fi

INSTALLED_TORCH="$(pip show torch | awk '/^Version:/ {print $2}')"
INSTALLED_TV="$(pip show torchvision | awk '/^Version:/ {print $2}')"
INSTALLED_TA="$(pip show torchaudio | awk '/^Version:/ {print $2}')"
echo "       torch       = ${INSTALLED_TORCH}"
echo "       torchvision = ${INSTALLED_TV}"
echo "       torchaudio  = ${INSTALLED_TA}"

# Pin torch for the rest of the install so transitive deps don't downgrade us
# to a wheel for the wrong CUDA. scvi-tools pulls lightning + torchmetrics
# which often have their own torch upper bounds — without this pin, you can
# end up with a CPU-only torch despite asking for a CUDA build.
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${INSTALLED_TORCH}
torchvision==${INSTALLED_TV}
torchaudio==${INSTALLED_TA}
EOF

# ---- 4. scvi-tools + scientific stack ---------------------------------------
# scvi-tools brings its own deps (lightning, torchmetrics, anndata, scanpy,
# scikit-learn, numpyro/jax for some models, etc). The constraints file keeps
# torch pinned through that dependency resolution.
echo "[run]  Installing scvi-tools from PyPI..."
pip install --constraint "${CONSTRAINTS_FILE}" "scvi-tools>=1.2"

# Extra stack for notebook usage and benchmarking.
pip install --constraint "${CONSTRAINTS_FILE}" \
  "numpy<2" \
  scanpy \
  scib \
  ipykernel \
  jupyterlab

# Belt-and-suspenders: if a transitive dep slipped torch out of the cu tag,
# put it back. --no-deps means we don't disturb anything else.
if [ -n "${TORCH_CUDA_TAG}" ]; then
  RESOLVED_TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || echo "")"
  EXPECTED_CUDA="${TORCH_CUDA_TAG#cu}"
  EXPECTED_CUDA="${EXPECTED_CUDA:0:2}.${EXPECTED_CUDA:2}"
  if [[ "${RESOLVED_TORCH_CUDA}" != "${EXPECTED_CUDA}"* ]]; then
    echo "[fix]  torch CUDA drifted to ${RESOLVED_TORCH_CUDA}, restoring ${TORCH_CUDA_TAG}..."
    pip install --force-reinstall --no-deps \
      "torch==${INSTALLED_TORCH}" \
      "torchvision==${INSTALLED_TV}" \
      "torchaudio==${INSTALLED_TA}" \
      --index-url "${TORCH_INDEX}"
  fi
fi

# Register the env as a Jupyter kernel so the wrapper notebook picks it up.
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# ---- 5. smoke test -----------------------------------------------------------
echo "[run]  Verifying installation..."
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("torch CUDA     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name    :", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("device cap     : sm_%d%d" % cc)
    x = torch.randn(8, 8, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gpu matmul     : OK")

import scvi, anndata, scanpy
print("scvi-tools     :", scvi.__version__)
print("anndata        :", anndata.__version__)
print("scanpy         :", scanpy.__version__)

# Quick sanity-check that we can instantiate scVI on a toy AnnData. This
# catches dependency mismatches (lightning <-> torch <-> torchmetrics) early —
# they typically blow up at .train() time, not at import.
import numpy as np
toy = anndata.AnnData(X=np.random.poisson(1.0, size=(50, 200)).astype("float32"))
scvi.model.SCVI.setup_anndata(toy)
m = scvi.model.SCVI(toy, n_latent=8, n_hidden=32, n_layers=1)
print("SCVI ctor      : OK")
PY

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Upstream docs:        https://docs.scvi-tools.org"
echo "Note: scVI is trained per-dataset — there is no pretrained checkpoint."
