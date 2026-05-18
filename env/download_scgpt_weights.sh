#!/usr/bin/env bash
# Download the scGPT_human pretrained checkpoint into models/scGPT/save/scGPT_human.
#
# Idempotent: if best_model.pt / args.json / vocab.json are already present,
# the script exits quickly without re-downloading.
#
# Usage:
#   bash env/download_scgpt_weights.sh
#
# Source folder on Google Drive:
#   https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

set -euo pipefail

GDRIVE_FOLDER_ID="1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models/scGPT/save/scGPT_human}"

# ---- 0. early-exit if already downloaded ------------------------------------
if [ -f "${CKPT_DIR}/best_model.pt" ] \
   && [ -f "${CKPT_DIR}/args.json" ] \
   && [ -f "${CKPT_DIR}/vocab.json" ]; then
  echo "[skip] scGPT_human checkpoint already present at ${CKPT_DIR}."
  exit 0
fi

mkdir -p "${CKPT_DIR}"

# ---- 1. ensure gdown is available --------------------------------------------
# Use whichever python is on PATH; gdown is a small pure-python tool, no need
# to be inside the scgpt conda env.
if ! command -v gdown >/dev/null 2>&1 \
   && ! python3 -c "import gdown" >/dev/null 2>&1; then
  echo "[run]  Installing gdown (user site)..."
  python3 -m pip install --user --quiet gdown
fi

if command -v gdown >/dev/null 2>&1; then
  GDOWN=(gdown)
else
  GDOWN=(python3 -m gdown)
fi

# ---- 2. download the folder --------------------------------------------------
echo "[run]  Downloading scGPT_human checkpoint into ${CKPT_DIR}..."
"${GDOWN[@]}" --folder "https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" \
              -O "${CKPT_DIR}" || true

# gdown sometimes nests into a subdir named after the Drive folder; flatten it.
if [ ! -f "${CKPT_DIR}/best_model.pt" ]; then
  nested="$(find "${CKPT_DIR}" -maxdepth 2 -name best_model.pt -printf '%h\n' | head -n1 || true)"
  if [ -n "${nested}" ] && [ "${nested}" != "${CKPT_DIR}" ]; then
    echo "[fix]  Flattening nested download dir: ${nested} -> ${CKPT_DIR}"
    mv "${nested}"/* "${CKPT_DIR}/"
    rmdir "${nested}" 2>/dev/null || true
  fi
fi

# ---- 3. verify ---------------------------------------------------------------
missing=()
for f in best_model.pt args.json vocab.json; do
  [ -f "${CKPT_DIR}/${f}" ] || missing+=("${f}")
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo "[error] Missing expected files in ${CKPT_DIR}: ${missing[*]}" >&2
  echo "        Drive may be rate-limited or require auth. Manual fallback:" >&2
  echo "        https://drive.google.com/drive/folders/${GDRIVE_FOLDER_ID}" >&2
  exit 1
fi

echo "[ok]   scGPT_human checkpoint ready at ${CKPT_DIR}"
ls -lh "${CKPT_DIR}"
