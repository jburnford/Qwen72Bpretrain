#!/bin/bash
# Helper script to continue training from checkpoint
# Automatically resubmits the training job

OUTPUT_DIR="/scratch/$USER/qwen72b_qlora_output"

echo "========================================"
echo "Continue Training Helper"
echo "========================================"

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not found: $OUTPUT_DIR"
    echo "Have you run the first training job yet?"
    exit 1
fi

# Check for checkpoints
CHECKPOINT_COUNT=$(find $OUTPUT_DIR -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | wc -l)

if [ $CHECKPOINT_COUNT -eq 0 ]; then
    echo "ERROR: No checkpoints found in $OUTPUT_DIR"
    echo "Start a new training job with: sbatch scripts/train_qlora_single_gpu.sh"
    exit 1
fi

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -td $OUTPUT_DIR/checkpoint-* 2>/dev/null | head -1)
CHECKPOINT_STEP=$(basename $LATEST_CHECKPOINT | cut -d'-' -f2)

echo "Found $CHECKPOINT_COUNT checkpoint(s)"
echo "Latest checkpoint: $LATEST_CHECKPOINT"
echo "Training will resume from step: $CHECKPOINT_STEP"
echo ""

# Check if training is complete
if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
    # Extract epoch info if available
    CURRENT_EPOCH=$(grep -o '"epoch": [0-9.]*' $OUTPUT_DIR/trainer_state.json | tail -1 | cut -d' ' -f2)
    if [ ! -z "$CURRENT_EPOCH" ]; then
        echo "Current progress: Epoch $CURRENT_EPOCH"
    fi
fi

# Ask for confirmation
read -p "Resubmit training job to continue? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting continuation job..."
    JOB_ID=$(sbatch scripts/train_qlora_single_gpu.sh | awk '{print $NF}')
    echo "Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u $USER"
    echo "  tail -f logs/qwen72b-qlora-$JOB_ID.out"
else
    echo "Cancelled."
fi

echo "========================================"
