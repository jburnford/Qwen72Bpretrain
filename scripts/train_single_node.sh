#!/bin/bash
#SBATCH --job-name=qwen72b-single
#SBATCH --account=def-jic823  # Replace with your allocation
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=0  # Request all memory on the node
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "========================================"
echo "SLURM Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $(pwd)"
echo "========================================"

# Load required modules
echo "Loading modules..."
module load StdEnv/2023
module load gcc cuda/12.2 python/3.11 cudnn cmake protobuf

# Display loaded modules
module list

# Set HuggingFace cache directories
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/models"
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

echo "HuggingFace cache: $HF_HOME"

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Create virtual environment on node-local storage (faster!)
echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/qwen_env
source $SLURM_TMPDIR/qwen_env/bin/activate

# Install required packages
echo "Installing packages..."
pip install --no-index --upgrade pip
pip install --no-index torch transformers datasets tokenizers accelerate
pip install --no-index deepspeed peft bitsandbytes flash-attn
pip install --no-index numpy pandas scikit-learn tqdm tensorboard
pip install jsonlines  # May not be in wheelhouse

# Verify installation
echo ""
echo "========================================"
echo "Package versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
echo "========================================"

# Set distributed training environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"

# Data paths (update these!)
TRAIN_DATA="/scratch/$USER/early_modern_data/train.jsonl"
VALID_DATA="/scratch/$USER/early_modern_data/valid.jsonl"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Please update the TRAIN_DATA path in this script"
    exit 1
fi

# Model and output paths
MODEL_NAME="Qwen/Qwen1.5-72B"
OUTPUT_DIR="/scratch/$USER/qwen72b_output"
DEEPSPEED_CONFIG="configs/ds_config_zero3.json"

# Create output directory
mkdir -p $OUTPUT_DIR

# Training hyperparameters
BATCH_SIZE=1
GRAD_ACCUM=16
LEARNING_RATE=2e-5
MAX_SEQ_LEN=2048
NUM_EPOCHS=2

echo ""
echo "========================================"
echo "Training configuration:"
echo "Model: $MODEL_NAME"
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VALID_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM * 8))"
echo "Learning rate: $LEARNING_RATE"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Number of epochs: $NUM_EPOCHS"
echo "========================================"

# Run training with DeepSpeed
echo "Starting training..."
srun python train.py \
    --model_name_or_path $MODEL_NAME \
    --train_data_path $TRAIN_DATA \
    --validation_data_path $VALID_DATA \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LEN \
    --bf16 True \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --deepspeed $DEEPSPEED_CONFIG \
    --report_to "tensorboard" \
    --use_flash_attention True

echo ""
echo "========================================"
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "========================================"
