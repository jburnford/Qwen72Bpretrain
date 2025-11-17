#!/bin/bash
#SBATCH --job-name=qwen-test-quick
#SBATCH --account=def-jic823
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:15:00  # Just 15 minutes!
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "========================================"
echo "QUICK TEST: Environment & Model Loading"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Time limit: 15 minutes"
echo "========================================"

# Load modules - Use CUDA 12.6 for better bitsandbytes compatibility
echo "1. Loading modules..."
module load StdEnv/2023 gcc cuda/12.6 python/3.11 cudnn arrow
module list

# Setup environment
export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/models"
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"
# Don't set BNB_CUDA_VERSION - let bitsandbytes auto-detect

mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# Create virtualenv
echo ""
echo "2. Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/test_env
source $SLURM_TMPDIR/test_env/bin/activate

# Install packages
echo ""
echo "3. Installing packages..."
pip install --no-index --upgrade pip
pip install --no-index torch transformers datasets tokenizers accelerate peft flash-attn

# Install bitsandbytes from PyPI (DRAC wheelhouse version lacks CUDA binaries)
echo "Installing bitsandbytes from PyPI..."
pip install bitsandbytes

# Verify installations
echo ""
echo "4. Verifying installations..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')"
python -c "import bitsandbytes as bnb; print(f'✓ BitsAndBytes: {bnb.__version__}')"

# Test model loading
echo ""
echo "5. Testing model loading (4-bit)..."
python << 'PYTHON_EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen1.5-72B",
    trust_remote_code=True,
    use_fast=False
)
print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")

print("\nLoading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-72B",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
print("✓ Model loaded in 4-bit")

# Check memory
print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

# Prepare for training
print("\nPreparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)
print("✓ Model prepared")

# Add LoRA
print("\nAdding LoRA adapters...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("✓ LoRA adapters added")
model.print_trainable_parameters()

# Unfreeze embeddings
print("\nUnfreezing embedding layers...")
if hasattr(model.base_model.model, 'model'):
    if hasattr(model.base_model.model.model, 'embed_tokens'):
        model.base_model.model.model.embed_tokens.requires_grad_(True)
        print("✓ Unfroze embed_tokens")

if hasattr(model.base_model.model, 'lm_head'):
    model.base_model.model.lm_head.requires_grad_(True)
    print("✓ Unfroze lm_head")

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

print(f"\nFinal GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Final GPU Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

print("\n✓ All checks passed!")
PYTHON_EOF

echo ""
echo "========================================"
echo "✓ QUICK TEST PASSED"
echo "========================================"
echo "All systems ready for training!"
echo "Next step: Run medium test with actual training"
echo "  sbatch scripts/test_training_short.sh"
echo "========================================"
