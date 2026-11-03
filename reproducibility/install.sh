#!/usr/bin/env bash
# =============================================================================
# SC-FNO — Clean Environment Setup
# =============================================================================
# Reproduces the sc_fno Conda environment on a fresh Ubuntu-on-WSL2 system
# with an existing NVIDIA GPU driver.
#
# Usage: bash reproducibility/install.sh
# Tested on: Ubuntu 22.04 on WSL2, RTX 4050 (6 GB VRAM)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== SC-FNO Reproducibility Install ==="
echo "Repo root: $REPO_ROOT"
echo ""

# --- Check prerequisites ---
echo "[1/6] Checking prerequisites..."

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. GPU verification will be skipped."
    GPU_AVAILABLE=false
else
    GPU_AVAILABLE=true
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# --- Source conda ---
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Create environment if it doesn't exist ---
ENV_NAME="sc_fno"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo ""
    echo "[2/6] Environment '$ENV_NAME' already exists — skipping creation"
else
    echo ""
    echo "[2/6] Creating '$ENV_NAME' environment (Python 3.10)..."
    conda create -y -n "$ENV_NAME" python=3.10
fi

# --- Activate ---
echo ""
echo "[3/6] Activating '$ENV_NAME' and installing PyTorch CUDA..."
conda activate "$ENV_NAME"

# --- Install PyTorch with CUDA 12.1 (RTX 4050 requires CUDA >= 11.8) ---
pip install --quiet \
    torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# --- Install scientific stack ---
echo ""
echo "[4/6] Installing scientific dependencies..."
pip install --quiet \
    numpy==2.2.6 \
    scipy==1.15.3 \
    pandas \
    matplotlib==3.10.9 \
    pyyaml \
    h5py \
    tqdm \
    tensorboard \
    pytest==9.2.0 \
    pytest-cov==7.1.0

# --- Install NeuralOperator ---
echo ""
echo "[5/6] Installing neuraloperator..."
pip install --quiet neuraloperator==2.0.0

# --- Verify installation ---
echo ""
echo "[6/6] Verifying installation..."

python3 - <<'PYVERIFY'
import sys
import torch

print(f"Python: {sys.version.split()[0]}")

import numpy, scipy, pandas, matplotlib, yaml, h5py
print(f"numpy: {numpy.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"pyyaml: {yaml.__version__}")
print(f"h5py: {h5py.__version__}")

print(f"torch: {torch.__version__}")
print(f"torch CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e6:.0f} MB")
    x = torch.randn(100, 100, device="cuda")
    _ = x @ x.T
    torch.cuda.synchronize()
    print("GPU matmul: OK")
else:
    print("WARNING: CUDA not available — training will be slow but functional")

import neuralop
print(f"neuralop: {getattr(neuralop, '__version__', 'unknown')}")

# Quick FFT check
try:
    x = torch.randn(64, 64, device="cuda" if torch.cuda.is_available() else "cpu")
    X = torch.fft.fft2(x)
    x_rec = torch.fft.ifft2(X).real
    err = (x - x_rec).abs().max().item()
    print(f"FFT round-trip error: {err:.2e}")
except Exception as e:
    print(f"FFT check failed: {e}")

print("\n=== Install complete ===")
PYVERIFY

echo ""
echo "To activate in future sessions:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate $ENV_NAME"
echo "  cd $REPO_ROOT"
