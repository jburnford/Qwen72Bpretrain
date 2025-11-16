# Qwen 1.5 72B Continued Pretraining for Early Modern English NER

This project implements continued pretraining of the Qwen 1.5 72B model on a corpus of early modern English texts from the Atlantic World, with the goal of improving Named Entity Recognition (NER) performance on historical documents.

## Project Overview

- **Model**: Qwen 1.5 72B
- **Corpus**: ~9.5GB of OLMOCR-processed Caribbean and Atlantic world historical documents (11,712 documents, 1600s-1800s)
- **Infrastructure**: DRAC nibi cluster with H100 GPUs
- **Training Method**: Continued pretraining with DeepSpeed ZeRO-3

## Data

### Source
- **Location**: `/home/jic823/projects/def-jic823/caribbean_pipeline/02_processed/`
- **Size**: 9.5GB
- **Documents**: 11,712 historical documents
- **Format**: OLMOCR JSON results

### Content
- Early modern English texts from the Caribbean and Atlantic world
- Historical documents from Internet Archive
- Dates ranging from 1600s-1800s
- Includes almanacs, legal documents, correspondence, and other period texts

## Setup

### 1. Local Development

```bash
# Clone the repository
git clone git@github.com:jburnford/Qwen72Bpretrain.git
cd Qwen72Bpretrain

# Install dependencies (for local development)
pip install -r requirements.txt
```

### 2. DRAC Cluster Setup

On nibi or other DRAC clusters:

```bash
# Run the setup script
bash scripts/setup_drac_env.sh
```

This will:
- Load required modules (Python 3.11, CUDA 12.2, etc.)
- Create a virtual environment
- Install all required packages from the DRAC wheelhouse
- Set up HuggingFace cache directories

## Data Preparation

### Analyze the Corpus

```bash
# On nibi cluster
python3 data/analyze_data.py \
    --data_dir /home/jic823/projects/def-jic823/caribbean_pipeline/02_processed
```

This will provide statistics on:
- Number of documents and text entries
- Total size and estimated token count
- Temporal distribution
- Sample texts

### Preprocess for Training

```bash
# On nibi cluster
python3 data/preprocess_olmocr.py \
    --data_dir /home/jic823/projects/def-jic823/caribbean_pipeline/02_processed \
    --output /scratch/$USER/early_modern_data/train.jsonl \
    --validation_split 0.05 \
    --max_seq_length 2048
```

Or use the interactive wrapper:

```bash
bash scripts/prepare_data.sh
```

This will:
- Extract text from all OLMOCR JSON files
- Clean and normalize (minimal processing to preserve historical language)
- Pack sequences for efficient training
- Split into train (95%) and validation (5%)
- Output JSONL format ready for training

## Training

### Configuration

Training uses DeepSpeed ZeRO-3 for memory-efficient distributed training:

- **DeepSpeed config**: `configs/ds_config_zero3.json`
- **Batch size**: 1 per GPU
- **Gradient accumulation**: 16 steps
- **Learning rate**: 2e-5 with cosine decay
- **Warmup**: 5% of total steps
- **Precision**: BF16
- **Sequence length**: 2048 tokens

### Single-Node Training (8 H100 GPUs)

```bash
# Edit paths in scripts/train_single_node.sh first, then:
sbatch scripts/train_single_node.sh
```

### Multi-Node Training

For larger scale training, modify the SLURM script to request multiple nodes.

### Monitor Training

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/qwen72b-single-<jobid>.out

# View TensorBoard (on login node or local machine with port forwarding)
tensorboard --logdir /scratch/$USER/qwen72b_output
```

## Project Structure

```
Qwen72Bpretrain/
├── data/                      # Data processing scripts
│   ├── analyze_data.py        # Analyze corpus statistics
│   └── preprocess_olmocr.py   # Convert to training format
├── scripts/                   # Utility scripts
│   ├── setup_drac_env.sh      # Environment setup for DRAC
│   ├── prepare_data.sh        # Data preparation wrapper
│   └── train_single_node.sh   # SLURM training job (single node)
├── configs/                   # Training configurations
│   ├── ds_config_zero3.json   # DeepSpeed ZeRO-3 config
│   └── ds_config_zero2.json   # DeepSpeed ZeRO-2 config (for multi-node)
├── logs/                      # SLURM output logs
├── output/                    # Training outputs (checkpoints, models)
├── train.py                   # Main training script
├── requirements.txt           # Python dependencies (DRAC-compatible)
└── README.md                  # This file
```

## DRAC-Specific Notes

### Module Loading

Always load modules before creating environments:

```bash
module load StdEnv/2023 gcc cuda/12.2 python/3.11 cudnn
```

### Package Installation

Use `--no-index` to install from DRAC wheelhouse:

```bash
pip install --no-index torch transformers deepspeed
```

### Storage Locations

- **Home** (`/home/$USER`): Code, scripts, environments (50GB limit)
- **Project** (`/project/def-jic823`): Shared data, source corpus (1TB limit)
- **Scratch** (`/scratch/$USER`): Training data, checkpoints (20TB, 60-day purge)
- **`$SLURM_TMPDIR`**: Fast node-local storage (job-specific)

### HuggingFace Cache

Pre-download models on login node (especially for Narval which has no compute node internet):

```bash
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/models"

python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen1.5-72B')"
```

## Memory Requirements

### Qwen 1.5 72B Model

- **Inference (BF16)**: ~144GB VRAM
- **Training with Adam**: ~577GB peak VRAM
- **Minimum for continued pretraining**: 8× H100-80GB with DeepSpeed ZeRO-3

### Resource Allocation

- **Single node**: 8 H100 GPUs, 512GB RAM, 48 hours
- **Multi-node**: 2-4 nodes (16-32 GPUs) for faster training

## Next Steps

After continued pretraining:

1. **Evaluate** the adapted model on early modern English text
   - Measure perplexity
   - Generate sample text
   - Compare to base model

2. **Fine-tune for NER**
   - Prepare labeled NER dataset
   - Fine-tune with lower learning rate (1e-5)
   - Evaluate NER performance

3. **Deploy**
   - Save final model to `/project` for long-term storage
   - Create model card with training details
   - Document performance improvements

## References

- [Qwen Documentation](https://qwen.readthedocs.io/)
- [DRAC PyTorch Guide](https://docs.alliancecan.ca/wiki/PyTorch)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Continued Pretraining Best Practices](https://huggingface.co/docs/transformers/main_classes/trainer)

## License

[Add your license here]

## Acknowledgments

- Digital Research Alliance of Canada for compute resources
- OLMOCR for OCR processing
- Internet Archive for historical document access

## Contact

[Add contact information]
