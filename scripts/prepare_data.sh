#!/bin/bash
# Data preparation script for nibi cluster
# This script analyzes and preprocesses the OLMOCR Caribbean corpus

set -e  # Exit on error

echo "========================================"
echo "OLMOCR Caribbean Corpus - Data Preparation"
echo "========================================"

# Paths
DATA_DIR="/home/jic823/projects/def-jic823/caribbean_pipeline/02_processed"
OUTPUT_DIR="/scratch/$USER/early_modern_data"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Load Python module
if [ ! -z "$SLURM_CLUSTER_NAME" ]; then
    echo "Running on cluster: $SLURM_CLUSTER_NAME"
    module load python/3.11
fi

# Check if analysis should be run
echo "========================================"
echo "Step 1: Data Analysis"
echo "========================================"
read -p "Run data analysis? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running analysis..."
    python3 $PROJECT_ROOT/data/analyze_data.py \
        --data_dir $DATA_DIR \
        2>&1 | tee $OUTPUT_DIR/analysis_report.txt
    echo ""
    echo "Analysis complete! Report saved to: $OUTPUT_DIR/analysis_report.txt"
else
    echo "Skipping analysis."
fi

echo ""
echo "========================================"
echo "Step 2: Data Preprocessing"
echo "========================================"
read -p "Run data preprocessing? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running preprocessing..."
    echo ""
    echo "Configuration:"
    echo "  Output: $OUTPUT_DIR/train.jsonl"
    echo "  Validation split: 5%"
    echo "  Max sequence length: 2048 tokens"
    echo "  Min text length: 50 characters"
    echo ""

    python3 $PROJECT_ROOT/data/preprocess_olmocr.py \
        --data_dir $DATA_DIR \
        --output $OUTPUT_DIR/train.jsonl \
        --validation_split 0.05 \
        --max_seq_length 2048 \
        --min_text_length 50 \
        2>&1 | tee $OUTPUT_DIR/preprocessing_log.txt

    echo ""
    echo "Preprocessing complete!"
    echo "  Training data: $OUTPUT_DIR/train.jsonl"
    echo "  Validation data: $OUTPUT_DIR/train_valid.jsonl"
    echo "  Log: $OUTPUT_DIR/preprocessing_log.txt"
else
    echo "Skipping preprocessing."
fi

echo ""
echo "========================================"
echo "Data Preparation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Review the data analysis report"
echo "2. Verify the training and validation files"
echo "3. Update the SLURM training script with correct paths"
echo "4. Submit the training job"
echo ""
