#!/usr/bin/env bash
# Set up a conda environment capable of running the scFoundation foundation model.
# Reference: https://github.com/biomap-research/scFoundation
#
# Usage:
#   bash env/setup_scfoundation.sh
#
# The script probes the local hardware (GPU compute capability, driver CUDA)
# and chooses a compatible PyTorch + CUDA wheel for THIS machine. It is
# idempotent: re-running skips steps that are already done.
#
# NOTE: scFoundation's pretrained weights are hosted on a SharePoint share that
# requires a browser/SSO login — there is no scriptable URL. This script clones
# the repo and creates an empty `model/models/` directory; the user must place
# the downloaded weight file there manually before running inference.

set -euo pipefail

ENV_NAME="${SCFOUNDATION_ENV_NAME:-scfoundation}"
PY_VERSION="3.10"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
SF_DIR="${MODELS_DIR}/scFoundation"
SF_WEIGHTS_DIR="${SF_DIR}/model/models"

# Source where the user must grab the weights file from (manual step).
SF_WEIGHTS_URL="https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk"

# ---- 0. probe hardware -------------------------------------------------------
echo "=== Hardware probe ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. scFoundation requires a CUDA GPU." >&2
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

# ---- 3. clone scFoundation source -------------------------------------------
mkdir -p "${MODELS_DIR}"
if [ ! -d "${SF_DIR}/.git" ]; then
  echo "[run]  Cloning scFoundation into ${SF_DIR}..."
  git clone https://github.com/biomap-research/scFoundation.git "${SF_DIR}"
else
  echo "[skip] scFoundation repo already cloned at ${SF_DIR}."
fi

# ---- 4. PyTorch matched to this GPU -----------------------------------------
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
echo "[run]  Installing PyTorch from ${TORCH_INDEX}..."
pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX}"

INSTALLED_TORCH="$(pip show torch | awk '/^Version:/ {print $2}')"
INSTALLED_TV="$(pip show torchvision | awk '/^Version:/ {print $2}')"
INSTALLED_TA="$(pip show torchaudio | awk '/^Version:/ {print $2}')"
echo "       torch       = ${INSTALLED_TORCH}"
echo "       torchvision = ${INSTALLED_TV}"
echo "       torchaudio  = ${INSTALLED_TA}"

# Pin torch for the rest of the install so transitive deps don't silently
# downgrade us to a wheel for the wrong CUDA.
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${INSTALLED_TORCH}
torchvision==${INSTALLED_TV}
torchaudio==${INSTALLED_TA}
EOF

# ---- 5. scFoundation's scientific stack -------------------------------------
# scFoundation isn't pip-installable as a package — it's a script-driven repo.
# We install the deps its model/README.md lists, plus a notebook stack so the
# wrapper notebook can use it. local_attention is the special one — used by
# the sparse-attention transformer in scFoundation/model/.
pip install --constraint "${CONSTRAINTS_FILE}" \
  "numpy<2" \
  pandas \
  scipy \
  einops \
  scanpy \
  anndata \
  local_attention \
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

# Register the env as a Jupyter kernel so the wrapper notebook picks it up.
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# ---- 6. prepare weights directory -------------------------------------------
mkdir -p "${SF_WEIGHTS_DIR}"

# Detect whether weights are already in place. The repo ships with .ckpt-style
# weight files in model/models/. We treat any *.ckpt file as a successful
# download.
if compgen -G "${SF_WEIGHTS_DIR}/*.ckpt" > /dev/null \
   || compgen -G "${SF_WEIGHTS_DIR}/*.pt"   > /dev/null; then
  echo "[skip] scFoundation weights already present in ${SF_WEIGHTS_DIR}."
else
  echo "[warn] scFoundation pretrained weights NOT downloaded."
  echo
  echo "       The weights live on a SharePoint share that requires a browser"
  echo "       session — they cannot be fetched non-interactively. To finish:"
  echo
  echo "         1. Open this URL in a browser:"
  echo "            ${SF_WEIGHTS_URL}"
  echo "         2. Download the .ckpt weight file(s)."
  echo "         3. Place them into:"
  echo "            ${SF_WEIGHTS_DIR}/"
  echo
fi

# ---- 7. smoke test -----------------------------------------------------------
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
import einops, scanpy, local_attention
print("einops         :", einops.__version__)
print("scanpy         :", scanpy.__version__)
print("local_attention: import OK")
PY

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "scFoundation source:  ${SF_DIR}"
echo "Weights dir:          ${SF_WEIGHTS_DIR}"
echo "Upstream repo:        https://github.com/biomap-research/scFoundation"
