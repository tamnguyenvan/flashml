#!/usr/bin/env bash
# Bootstrap Miniconda envs for FlashML on vastai/base-image:cuda-12.8.1-auto.
set -euo pipefail

FLASHML_HOME="${FLASHML_HOME:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
MOGE_REVISION="${MOGE_REVISION:-74fbce054ebed49800de42d0ad0e83495065719a}"
SIMPLECLICK_REF="${SIMPLECLICK_REF:-v1.0}"
SIMPLECLICK_GDRIVE_ID="${SIMPLECLICK_GDRIVE_ID:-1GXk6q5fwKo2twkY5ZZGjVKCgJv7XeLAW}"

export FLASHML_HOME CONDA_ROOT
mkdir -p "${FLASHML_HOME}/third_party" "${FLASHML_HOME}/weights/simpleclick" \
  "${FLASHML_HOME}/weights/oneformer" "${FLASHML_HOME}/weights/lama" \
  "${FLASHML_HOME}/weights/huggingface" \
  "${FLASHML_HOME}/logs"

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    -o Dpkg::Options::=--force-confdef \
    -o Dpkg::Options::=--force-confold \
    build-essential ffmpeg git libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
    supervisor wget ca-certificates
fi

if [ ! -x "${CONDA_ROOT}/bin/conda" ]; then
  echo "Installing Miniconda into ${CONDA_ROOT}"
  installer="/tmp/miniconda.sh"
  wget -q -O "${installer}" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "${installer}" -b -p "${CONDA_ROOT}"
  rm -f "${installer}"
fi

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda config --set always_yes yes
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

create_env() {
  local name="$1"
  local file="$2"
  if conda env list | awk '{print $1}' | grep -qx "${name}"; then
    echo "Conda env ${name} already exists"
  else
    conda env create -f "${file}"
  fi
}

create_env flashml-api "${FLASHML_HOME}/envs/environment-api.yml"
create_env flashml-moge "${FLASHML_HOME}/envs/environment-moge.yml"
create_env flashml-simpleclick "${FLASHML_HOME}/envs/environment-simpleclick.yml"
create_env flashml-oneformer "${FLASHML_HOME}/envs/environment-oneformer.yml"
create_env flashml-lama "${FLASHML_HOME}/envs/environment-lama.yml"

conda run --no-capture-output -n flashml-api python -m pip install --upgrade pip
conda run --no-capture-output -n flashml-api python -m pip install -e "${FLASHML_HOME}"

if [ ! -d "${FLASHML_HOME}/third_party/MoGe/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/microsoft/MoGe.git \
    "${FLASHML_HOME}/third_party/MoGe"
  git -C "${FLASHML_HOME}/third_party/MoGe" checkout "${MOGE_REVISION}"
fi

conda run --no-capture-output -n flashml-moge python -m pip install --upgrade pip
conda run --no-capture-output -n flashml-moge python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
conda run --no-capture-output -n flashml-moge python -m pip install -e "${FLASHML_HOME}"
conda run --no-capture-output -n flashml-moge python -m pip install -e "${FLASHML_HOME}/third_party/MoGe" \
  --extra-index-url https://pypi.org/simple
conda run --no-capture-output -n flashml-moge python -m pip install huggingface_hub trimesh opencv-python-headless Pillow

if [ ! -d "${FLASHML_HOME}/third_party/SimpleClick/.git" ]; then
  git clone --depth 1 --branch "${SIMPLECLICK_REF}" \
    https://github.com/uncbiag/SimpleClick "${FLASHML_HOME}/third_party/SimpleClick"
fi

conda run --no-capture-output -n flashml-simpleclick python -m pip install --upgrade pip
conda run --no-capture-output -n flashml-simpleclick python -m pip install \
  torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
MMCV_WITH_OPS=0 conda run --no-capture-output -n flashml-simpleclick python -m pip install \
  "numpy==1.23.5" "opencv-python-headless>=4.10,<5" "Pillow>=9.5,<12" \
  "PyYAML>=6,<7" "protobuf==3.20.3" "tensorboard==2.8.0" "albumentations==0.5.2" \
  "Cython==0.29.32" "easydict>=1.9,<2" "mmcv==1.6.2" "scipy>=1.10,<2" \
  "timm==0.6.11" "gdown>=5.2,<6"
conda run --no-capture-output -n flashml-simpleclick python -m pip install -e "${FLASHML_HOME}"

CHECKPOINT="${FLASHML_HOME}/weights/simpleclick/cocolvis_vit_huge.pth"
if [ ! -f "${CHECKPOINT}" ]; then
  conda run --no-capture-output -n flashml-simpleclick gdown "${SIMPLECLICK_GDRIVE_ID}" -O "${CHECKPOINT}"
fi

conda run --no-capture-output -n flashml-oneformer python -m pip install --upgrade pip
conda run --no-capture-output -n flashml-oneformer python -m pip install \
  torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -n flashml-oneformer python -m pip install \
  "transformers>=4.38,<5" "huggingface_hub>=0.20,<1" "Pillow>=9.5,<12" \
  "numpy<2" "scipy>=1.10,<1.16"
conda run --no-capture-output -n flashml-oneformer python -m pip install -e "${FLASHML_HOME}"

HF_HOME="${FLASHML_HOME}/weights/huggingface" \
  conda run --no-capture-output -n flashml-oneformer python - <<'PY'
from pathlib import Path
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("FLASHML_ONEFORMER_MODEL_ID", "shi-labs/oneformer_ade20k_swin_large")
local_dir = os.environ.get(
    "FLASHML_ONEFORMER_MODEL_DIR",
    str(Path(os.environ["FLASHML_HOME"]) / "weights" / "oneformer"),
)
print(f"Downloading {model_id} -> {local_dir}")
snapshot_download(repo_id=model_id, local_dir=local_dir)
PY

HF_HOME="${FLASHML_HOME}/weights/huggingface" \
  conda run --no-capture-output -n flashml-moge python - <<'PY'
from huggingface_hub import snapshot_download
print("Downloading Ruicheng/moge-3-vitg")
snapshot_download(repo_id="Ruicheng/moge-3-vitg", repo_type="model")
print("MoGe weights ready")
PY

echo "Setting up LaMa (iopaint) environment..."
conda run --no-capture-output -n flashml-lama python -m pip install --upgrade pip
conda run --no-capture-output -n flashml-lama python -m pip install \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda run --no-capture-output -n flashml-lama python -m pip install \
  "iopaint>=1.2,<2" opencv-python-headless Pillow
conda run --no-capture-output -n flashml-lama python -m pip install -e "${FLASHML_HOME}"

LAMA_MODEL_DIR="${FLASHML_HOME}/weights/lama"
if [ ! -d "${LAMA_MODEL_DIR}" ] || [ -z "$(ls -A "${LAMA_MODEL_DIR}" 2>/dev/null)" ]; then
  echo "Downloading LaMa big-lama weights into ${LAMA_MODEL_DIR}"
  FLASHML_LAMA_MODEL_DIR="${LAMA_MODEL_DIR}" \
    conda run --no-capture-output -n flashml-lama python - <<'PY'
import os
from pathlib import Path
from iopaint import LaMa

model_dir = Path(os.environ["FLASHML_LAMA_MODEL_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)
# Instantiating with a missing model_dir triggers the built-in weight download.
LaMa(device="cpu", model_dir=str(model_dir))
print(f"LaMa weights ready at {model_dir}")
PY
fi

echo
echo "Setup complete."
echo "  FLASHML_HOME=${FLASHML_HOME}"
echo "  CONDA_ROOT=${CONDA_ROOT}"
echo "Start with: FLASHML_HOME=${FLASHML_HOME} CONDA_ROOT=${CONDA_ROOT} ${FLASHML_HOME}/scripts/start.sh"
