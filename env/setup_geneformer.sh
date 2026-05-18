#!/usr/bin/env bash
# Set up a conda environment capable of running the Geneformer foundation model.
# Reference: https://huggingface.co/ctheodoris/Geneformer
#
# Usage:
#   bash env/setup_geneformer.sh
#
# The script probes the local hardware (GPU compute capability, driver CUDA)
# and chooses a compatible PyTorch + CUDA wheel for THIS machine. It is
# idempotent: re-running skips steps that are already done.

set -euo pipefail

ENV_NAME="${GENEFORMER_ENV_NAME:-geneformer}"
PY_VERSION="3.10"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
GF_DIR="${MODELS_DIR}/Geneformer"

# Geneformer is shipped from a Hugging Face repo. The repo carries both the
# python package code AND the pretrained checkpoints (LFS-tracked). We use
# huggingface_hub's snapshot_download to avoid needing system git-lfs.
HF_REPO_ID="ctheodoris/Geneformer"

# ---- 0. probe hardware -------------------------------------------------------
echo "=== Hardware probe ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Geneformer requires a CUDA GPU." >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
DRIVER_CUDA="$(nvidia-smi | awk '/CUDA Version/ {for(i=1;i<=NF;i++) if($i=="Version:") print $(i+1)}' | head -n1)"

CC_MAJOR="${COMPUTE_CAP%%.*}"
CC_MINOR="${COMPUTE_CAP##*.}"
CC_NUM=$(( CC_MAJOR * 10 + CC_MINOR ))

# Pick a torch CUDA wheel tag that has SASS/PTX for this GPU.
#   sm_120 (Blackwell-2: RTX PRO 6000, RTX 50-series) -> needs CUDA >= 12.8
#   sm_100 (Blackwell-1: B100/B200/GB200)             -> needs CUDA >= 12.4
#   sm_90  (Hopper: H100/H200)                        -> CUDA 12.x
#   sm_89  (Ada: L40/4090)                            -> CUDA 11.8 fine
#   sm_80/86 (Ampere)                                 -> CUDA 11.8 fine
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
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
echo "[run]  Installing PyTorch from ${TORCH_INDEX}..."
pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX}"

INSTALLED_TORCH="$(pip show torch | awk '/^Version:/ {print $2}')"
INSTALLED_TV="$(pip show torchvision | awk '/^Version:/ {print $2}')"
INSTALLED_TA="$(pip show torchaudio | awk '/^Version:/ {print $2}')"
echo "       torch       = ${INSTALLED_TORCH}"
echo "       torchvision = ${INSTALLED_TV}"
echo "       torchaudio  = ${INSTALLED_TA}"

# Pin torch for the rest of the install so transformers / datasets / etc don't
# silently downgrade us to a wheel for the wrong CUDA.
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${INSTALLED_TORCH}
torchvision==${INSTALLED_TV}
torchaudio==${INSTALLED_TA}
EOF

# ---- 4. download Geneformer source + checkpoints ----------------------------
# Use huggingface_hub.snapshot_download to fetch the entire HF repo (code AND
# LFS-tracked model weights) without needing system-level git-lfs.
mkdir -p "${MODELS_DIR}"
pip install --constraint "${CONSTRAINTS_FILE}" --quiet "huggingface_hub>=0.20"

if [ -f "${GF_DIR}/setup.py" ] || [ -f "${GF_DIR}/pyproject.toml" ]; then
  echo "[skip] Geneformer source already present at ${GF_DIR}."
else
  echo "[run]  Downloading Geneformer from huggingface.co/${HF_REPO_ID}..."
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_REPO_ID}",
    local_dir="${GF_DIR}",
    local_dir_use_symlinks=False,
)
PY
fi

# ---- 5. install Geneformer + scientific stack -------------------------------
# Geneformer's setup.py lists deps unpinned, but its requirements.txt pins
# transformers==4.46 (newer transformers removed `SpecialTokensMixin` from the
# top-level namespace, which Geneformer imports). Install from requirements.txt
# first so the pin sticks, then `-e .` with --no-deps to register the package
# without re-resolving and bumping transformers.
echo "[run]  Installing Geneformer dependencies (pinned per repo requirements.txt)..."
pip install --constraint "${CONSTRAINTS_FILE}" -r "${GF_DIR}/requirements.txt"

echo "[run]  Installing Geneformer from source at ${GF_DIR} (no-deps)..."
pip install --no-deps -e "${GF_DIR}"

# Extra single-cell / notebook stack not covered by Geneformer's own deps.
pip install --constraint "${CONSTRAINTS_FILE}" \
  "numpy<2" \
  scanpy \
  scib \
  ipykernel \
  jupyterlab

# Belt-and-suspenders: if a transitive dep slipped torch out of the cu tag,
# put it back. --no-deps means we don't disturb anything else.
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

# Register the env as a Jupyter kernel so the wrapper notebooks pick it up.
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# ---- 6. smoke test -----------------------------------------------------------
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
import geneformer
print("geneformer     :", getattr(geneformer, "__version__", "import OK"))
# Confirm at least one pretrained weight file landed (LFS files are the usual
# failure mode of the HF download path).
import os, glob
for pattern in ("**/pytorch_model.bin", "**/model.safetensors"):
    hits = glob.glob(os.path.join(os.path.dirname(geneformer.__file__), "..", pattern), recursive=True)
    if hits:
        print("checkpoint     :", hits[0])
        break
else:
    print("checkpoint     : NOT FOUND (re-run snapshot_download)")
PY

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Geneformer source at: ${GF_DIR}"
echo "Upstream repo:        https://huggingface.co/${HF_REPO_ID}"
