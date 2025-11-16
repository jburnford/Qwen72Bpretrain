#!/bin/bash
# Setup script for DRAC clusters (nibi, narval, fir)
# This script creates a virtual environment and installs all required packages

set -e  # Exit on error

echo "=========================================="
echo "DRAC Environment Setup for Qwen 72B Training"
echo "=========================================="

# Load required modules
echo "Loading modules..."
module load StdEnv/2023
module load gcc cuda/12.2 python/3.11 cudnn cmake protobuf

# Check which cluster we're on
if [ ! -z "$SLURM_CLUSTER_NAME" ]; then
    echo "Running on cluster: $SLURM_CLUSTER_NAME"
fi

# Set HuggingFace cache directories
# Modify these paths according to your project allocation
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/models"
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

echo "HuggingFace cache set to: $HF_HOME"

# Create cache directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Determine environment location
if [ ! -z "$SLURM_TMPDIR" ]; then
    # Running in a job - use fast node-local storage
    ENV_DIR="$SLURM_TMPDIR/qwen_env"
    echo "Creating virtual environment on node-local storage: $ENV_DIR"
else
    # Running on login node - use home directory
    ENV_DIR="$HOME/qwen_env"
    echo "Creating virtual environment in home directory: $ENV_DIR"
fi

# Create virtual environment
echo "Creating virtual environment..."
virtualenv --no-download $ENV_DIR

# Activate environment
source $ENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --no-index --upgrade pip

# Check available packages
echo ""
echo "Checking available packages in DRAC wheelhouse..."
echo "PyTorch versions:"
avail_wheels torch | head -10
echo ""
echo "Transformers versions:"
avail_wheels transformers | head -10
echo ""

# Install core packages
echo "Installing core ML packages..."
pip install --no-index torch torchvision transformers datasets tokenizers accelerate

# Install distributed training tools
echo "Installing DeepSpeed..."
pip install --no-index deepspeed

# Install PEFT tools
echo "Installing parameter-efficient fine-tuning tools..."
pip install --no-index peft bitsandbytes flash-attn

# Install utilities
echo "Installing utility packages..."
pip install --no-index numpy pandas scikit-learn tqdm tensorboard

# Install packages not in wheelhouse (if needed)
echo "Installing additional packages..."
pip install jsonlines  # May not be in wheelhouse

# Try to install wandb
if avail_wheels wandb > /dev/null 2>&1; then
    echo "Installing wandb from wheelhouse..."
    pip install --no-index wandb
else
    echo "Installing wandb from PyPI (not in wheelhouse)..."
    pip install wandb
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Virtual environment location: $ENV_DIR"
echo "To activate:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "HuggingFace cache: $HF_HOME"
echo "=========================================="
