#!/bin/bash
#SBATCH --job-name=qwen72b-qlora
#SBATCH --account=def-jic823  # Replace with your allocation
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1  # Just 1 H100!
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00  # Under 24h limit - job will auto-resume from checkpoint
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
module load gcc cuda/12.2 python/3.11 cudnn arrow

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
pip install --no-index peft bitsandbytes flash-attn
pip install --no-index numpy pandas scikit-learn tqdm tensorboard
# Try to install jsonlines from wheelhouse first
pip install --no-index jsonlines 2>/dev/null || echo "Note: jsonlines not in wheelhouse, will be installed later if needed"

# Verify installation
echo ""
echo "========================================"
echo "Package versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
echo "========================================"

# Data paths (update these!)
TRAIN_DATA="/scratch/$USER/early_modern_data/train.jsonl"
VALID_DATA="/scratch/$USER/early_modern_data/train_valid.jsonl"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Please run data preprocessing first:"
    echo "  python3 data/preprocess_olmocr.py"
    exit 1
fi

# Model and output paths
MODEL_NAME="Qwen/Qwen1.5-72B"
OUTPUT_DIR="/scratch/$USER/qwen72b_qlora_output"

# Create output directory
mkdir -p $OUTPUT_DIR

# QLoRA-specific hyperparameters
BATCH_SIZE=4  # Can use larger batch with QLoRA!
GRAD_ACCUM=8
LEARNING_RATE=1e-4  # Higher LR for LoRA
MAX_SEQ_LEN=2048
NUM_EPOCHS=3
LORA_R=64
LORA_ALPHA=16

echo ""
echo "========================================"
echo "Training configuration:"
echo "Method: Hybrid QLoRA (4-bit + LoRA + unfrozen embeddings)"
echo "Model: $MODEL_NAME"
echo "Training data: $TRAIN_DATA"
echo "Validation data: $VALID_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "Learning rate: $LEARNING_RATE"
echo "Max sequence length: $MAX_SEQ_LEN"
echo "Number of epochs: $NUM_EPOCHS"
echo "LoRA rank (r): $LORA_R"
echo "LoRA alpha: $LORA_ALPHA"
echo "Walltime limit: 20 hours (auto-resume from checkpoint)"
echo "========================================"

# Check for existing checkpoints
if [ -d "$OUTPUT_DIR" ]; then
    CHECKPOINT_COUNT=$(find $OUTPUT_DIR -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        LATEST_CHECKPOINT=$(ls -td $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)
        echo "Found $CHECKPOINT_COUNT existing checkpoint(s)"
        echo "Latest checkpoint: $LATEST_CHECKPOINT"
        echo "Training will auto-resume from latest checkpoint"
    else
        echo "No existing checkpoints found - starting fresh"
    fi
else
    echo "Output directory does not exist - starting fresh"
fi
echo ""

# Run training with Hybrid QLoRA
echo "Starting Hybrid QLoRA training..."
$SLURM_TMPDIR/qwen_env/bin/python train.py \
    --model_name_or_path $MODEL_NAME \
    --use_qlora True \
    --train_embeddings True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
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
    --report_to "tensorboard" \
    --use_flash_attention True

# Note: Training will automatically resume from the latest checkpoint if found
# To continue training, just resubmit this same job script

echo ""
echo "========================================"
echo "Training complete!"
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "LoRA adapters can be merged with base model using:"
echo "  python scripts/merge_lora.py --base_model $MODEL_NAME --adapter_path $OUTPUT_DIR --output_path $OUTPUT_DIR/merged"
echo "========================================"
