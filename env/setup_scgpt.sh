#!/usr/bin/env bash
# Set up a conda environment capable of running the scGPT foundation model.
# Reference: https://github.com/bowang-lab/scGPT
#
# Usage:
#   bash env/setup_scgpt.sh
#
# The script probes the local hardware (GPU compute capability, driver CUDA)
# and chooses a compatible PyTorch + CUDA wheel for THIS machine. It is
# idempotent: re-running skips steps that are already done.

set -euo pipefail

ENV_NAME="${SCGPT_ENV_NAME:-scgpt}"
PY_VERSION="3.10"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
SCGPT_DIR="${MODELS_DIR}/scGPT"

# ---- 0. probe hardware -------------------------------------------------------
echo "=== Hardware probe ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. scGPT requires a CUDA GPU." >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
DRIVER_CUDA="$(nvidia-smi | awk '/CUDA Version/ {for(i=1;i<=NF;i++) if($i=="Version:") print $(i+1)}' | head -n1)"

# Convert "12.0" -> 120 for numeric comparison.
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

# ---- 3. clone scGPT source ---------------------------------------------------
mkdir -p "${MODELS_DIR}"
if [ ! -d "${SCGPT_DIR}/.git" ]; then
  echo "[run]  Cloning scGPT into ${SCGPT_DIR}..."
  git clone https://github.com/bowang-lab/scGPT.git "${SCGPT_DIR}"
else
  echo "[skip] scGPT repo already cloned at ${SCGPT_DIR}."
fi

# ---- 4. PyTorch matched to this GPU -----------------------------------------
# We don't pin a torch *version* here — we let pip pick the latest stable on
# the chosen CUDA index. That way scGPT's loose `torch` requirement won't
# silently upgrade torch to a wheel for the wrong CUDA.
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
echo "[run]  Installing PyTorch from ${TORCH_INDEX}..."
pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX}"

# Capture the resolved torch version so we can pin it for the rest of the
# install and stop downstream packages from yanking it onto a different CUDA.
INSTALLED_TORCH="$(pip show torch | awk '/^Version:/ {print $2}')"
INSTALLED_TV="$(pip show torchvision | awk '/^Version:/ {print $2}')"
INSTALLED_TA="$(pip show torchaudio | awk '/^Version:/ {print $2}')"
echo "       torch       = ${INSTALLED_TORCH}"
echo "       torchvision = ${INSTALLED_TV}"
echo "       torchaudio  = ${INSTALLED_TA}"

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
cat > "${CONSTRAINTS_FILE}" <<EOF
torch==${INSTALLED_TORCH}
torchvision==${INSTALLED_TV}
torchaudio==${INSTALLED_TA}
EOF

# ---- 5. scGPT + scientific stack --------------------------------------------
# Install scGPT from the cloned source (v0.2.5+), NOT from PyPI. The PyPI
# release (0.2.4) still depends on `torchtext`, which is deprecated and only
# has wheels built against torch 2.3 - importing it under modern torch yields
# `undefined symbol` errors. Upstream master replaced torchtext with an
# in-repo shim (scgpt/tokenizer/vocab_compat.py) and dropped the dep.
echo "[run]  Installing scGPT from source at ${SCGPT_DIR}..."
pip install --constraint "${CONSTRAINTS_FILE}" -e "${SCGPT_DIR}"

# If a previous run pulled torchtext (via the old PyPI scgpt), purge it -
# its libtorchtext.so is ABI-incompatible with our torch and breaks `import
# scgpt` even though scgpt no longer uses it.
if pip show torchtext >/dev/null 2>&1; then
  echo "[fix]  Removing legacy torchtext (incompatible with torch ${INSTALLED_TORCH})..."
  pip uninstall -y torchtext
fi

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

# Single-cell / scientific stack used by the scGPT tutorials and the
# benchmark notebooks in this repo.
pip install --constraint "${CONSTRAINTS_FILE}" \
  "numpy<2" \
  scanpy \
  anndata \
  scib \
  scvi-tools \
  wandb \
  ipykernel \
  jupyterlab

# ---- 6. flash-attn (best-effort) --------------------------------------------
# flash-attn >= 2.7 has Blackwell kernels (sm_90/sm_100/sm_120) but builds
# from source - it needs nvcc whose CUDA version matches torch's. If this
# fails for any reason, scGPT falls back to its non-flash attention path.
echo "[run]  Attempting flash-attn (optional)..."
# Map "cu128" -> "12.8" to pick the matching cuda-nvcc package.
CUDA_DOTTED="${TORCH_CUDA_TAG#cu}"
CUDA_DOTTED="${CUDA_DOTTED:0:2}.${CUDA_DOTTED:2}"
echo "       Installing nvcc ${CUDA_DOTTED} from the nvidia channel..."
if conda install -y -c "nvidia/label/cuda-${CUDA_DOTTED}.0" \
     cuda-nvcc cuda-cudart-dev cuda-cccl 2>/dev/null \
   || conda install -y -c "nvidia/label/cuda-${CUDA_DOTTED}.1" \
     cuda-nvcc cuda-cudart-dev cuda-cccl 2>/dev/null; then
  export CUDA_HOME="${CONDA_PREFIX}"
  echo "       CUDA_HOME=${CUDA_HOME}"
  pip install packaging ninja >/dev/null
  if CUDA_HOME="${CUDA_HOME}" pip install "flash-attn>=2.7" \
       --no-build-isolation \
       --constraint "${CONSTRAINTS_FILE}"; then
    echo "[ok]   flash-attn installed."
  else
    echo "[warn] flash-attn build failed - scGPT will use its non-flash fallback." >&2
  fi
else
  echo "[warn] Could not install matching nvcc; skipping flash-attn." >&2
  echo "       scGPT will use its non-flash fallback (only throughput affected)." >&2
fi

# Register the env as a Jupyter kernel so the wrapper notebooks pick it up.
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

# ---- 7. pretrained checkpoint (whole_human) ---------------------------------
CKPT_DIR="${SCGPT_DIR}/save/scGPT_human"
GDRIVE_FOLDER_ID="1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"

if [ -f "${CKPT_DIR}/best_model.pt" ] \
   && [ -f "${CKPT_DIR}/args.json" ] \
   && [ -f "${CKPT_DIR}/vocab.json" ]; then
  echo "[skip] scGPT_human checkpoint already present at ${CKPT_DIR}."
else
  echo "[run]  Downloading scGPT_human checkpoint from Google Drive..."
  pip install --quiet gdown
  mkdir -p "${CKPT_DIR}"
  gdown --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" \
        -O "${CKPT_DIR}" || true

  # gdown sometimes nests into a subdir named after the Drive folder; flatten.
  if [ ! -f "${CKPT_DIR}/best_model.pt" ]; then
    nested="$(find "${CKPT_DIR}" -maxdepth 2 -name best_model.pt -printf '%h\n' | head -n1 || true)"
    if [ -n "${nested}" ] && [ "${nested}" != "${CKPT_DIR}" ]; then
      mv "${nested}"/* "${CKPT_DIR}/"
      rmdir "${nested}" 2>/dev/null || true
    fi
  fi

  if [ ! -f "${CKPT_DIR}/best_model.pt" ]; then
    echo "[warn] Automatic download failed (Drive quota / auth?)." >&2
    echo "       Manually grab the folder and place files in ${CKPT_DIR}:" >&2
    echo "       https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" >&2
  fi
fi

# ---- 8. smoke test -----------------------------------------------------------
echo "[run]  Verifying installation..."
python - <<'PY'
import torch, scgpt
print("torch          :", torch.__version__)
print("torch CUDA     :", torch.version.cuda)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device name    :", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("device cap     : sm_%d%d" % cc)
    # Touch a tensor on the GPU to make sure the wheel actually has kernels
    # for this compute capability (this is what catches sm_120 mismatches).
    x = torch.randn(8, 8, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gpu matmul     : OK")
print("scgpt          :", getattr(scgpt, "__version__", "import OK"))
try:
    import flash_attn
    print("flash_attn     :", flash_attn.__version__)
except Exception:
    print("flash_attn     : NOT AVAILABLE (scGPT will use the fallback path)")
PY

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "scGPT source at:      ${SCGPT_DIR}"
echo "Checkpoint dir:       ${CKPT_DIR}"
echo "More checkpoints:     https://github.com/bowang-lab/scGPT#pretrained-models"
