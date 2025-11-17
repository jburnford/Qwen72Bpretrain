#!/bin/bash
#SBATCH --job-name=qwen-test-train
#SBATCH --account=def-jic823
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00  # Just 20 minutes for fast testing
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "========================================"
echo "MEDIUM TEST: Actual Training & Checkpointing"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Time limit: 20 minutes"
echo "Will train for ~50 steps then test checkpoint resume"
echo "========================================"

# Load modules
module load StdEnv/2023 gcc cuda/12.2 python/3.11 cudnn arrow

# Setup environment
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/models"
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
export BNB_CUDA_VERSION=122  # Tell bitsandbytes to use CUDA 12.2

mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# Create virtualenv
echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/test_env
source $SLURM_TMPDIR/test_env/bin/activate

# Install packages
echo "Installing packages..."
pip install --no-index --upgrade pip
pip install --no-index torch transformers datasets tokenizers accelerate peft bitsandbytes flash-attn tqdm tensorboard
# Try to install jsonlines from wheelhouse, fallback to pip if not available
pip install --no-index jsonlines 2>/dev/null || echo "Note: jsonlines not in wheelhouse, will be installed later if needed"

# Data paths
TRAIN_DATA="/scratch/$USER/early_modern_data/train.jsonl"
TEST_OUTPUT_DIR="/scratch/$USER/qwen_test_output"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo ""
    echo "WARNING: Training data not found at $TRAIN_DATA"
    echo "Creating tiny test dataset instead..."

    mkdir -p $(dirname $TRAIN_DATA)

    # Create minimal test data
    cat > $TRAIN_DATA << 'EOF'
{"text": "In the yeare of our Lord 1673, there dwelt in Jamaica a merchant of great renowne. The Noble Island prospered under English rule, and many ships arrived from the West Indies bearing sugar and rum."}
{"text": "Whereas divers complaints have been made concerning the qualitie of corne and other provisions, we hereby declare that all merchants must present their wares for inspection at the Customs House in Port Royal."}
{"text": "The Honourable Company of Merchant Adventurers, being assembled this twelfth day of March in the yeare 1675, did resolve to send forth another expedition to the Caribbean territories."}
EOF

    echo "Created minimal test dataset with 3 examples"
    echo "For real training, run: bash scripts/prepare_data.sh"
fi

# Create test output directory
mkdir -p $TEST_OUTPUT_DIR

echo ""
echo "========================================"
echo "Test Configuration:"
echo "Data: $TRAIN_DATA"
echo "Output: $TEST_OUTPUT_DIR"
echo "Max steps: 50 (just for testing)"
echo "Save every: 25 steps"
echo "========================================"

# Run short training
echo ""
echo "Starting test training (50 steps)..."
$SLURM_TMPDIR/test_env/bin/python train.py \
    --model_name_or_path "Qwen/Qwen1.5-72B" \
    --use_qlora True \
    --train_embeddings True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --train_data_path $TRAIN_DATA \
    --output_dir $TEST_OUTPUT_DIR \
    --max_steps 50 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_seq_length 512 \
    --bf16 True \
    --gradient_checkpointing True \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --report_to "none" \
    --use_flash_attention True

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training test PASSED"

    # Check for checkpoints
    CHECKPOINT_COUNT=$(find $TEST_OUTPUT_DIR -name "checkpoint-*" -type d 2>/dev/null | wc -l)

    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo "✓ Checkpoints created: $CHECKPOINT_COUNT"
        echo ""
        echo "Checkpoints:"
        ls -lh $TEST_OUTPUT_DIR/checkpoint-*

        echo ""
        echo "Testing checkpoint resumption..."

        # Run 10 more steps from checkpoint
        $SLURM_TMPDIR/test_env/bin/python train.py \
            --model_name_or_path "Qwen/Qwen1.5-72B" \
            --use_qlora True \
            --train_embeddings True \
            --lora_r 64 \
            --lora_alpha 16 \
            --train_data_path $TRAIN_DATA \
            --output_dir $TEST_OUTPUT_DIR \
            --max_steps 60 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 4 \
            --learning_rate 1e-4 \
            --max_seq_length 512 \
            --bf16 True \
            --gradient_checkpointing True \
            --save_strategy "steps" \
            --save_steps 25 \
            --logging_steps 5 \
            --report_to "none"

        RESUME_EXIT_CODE=$?

        if [ $RESUME_EXIT_CODE -eq 0 ]; then
            echo "✓ Checkpoint resumption PASSED"
        else
            echo "✗ Checkpoint resumption FAILED"
            exit 1
        fi
    else
        echo "✗ No checkpoints created"
        exit 1
    fi

else
    echo "✗ Training test FAILED"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ ALL TESTS PASSED!"
echo "========================================"
echo ""
echo "System is ready for full training!"
echo ""
echo "Next steps:"
echo "1. Preprocess full dataset: bash scripts/prepare_data.sh"
echo "2. Start full training: sbatch scripts/train_qlora_single_gpu.sh"
echo ""
echo "You can clean up test output with:"
echo "  rm -rf $TEST_OUTPUT_DIR"
echo "========================================"
