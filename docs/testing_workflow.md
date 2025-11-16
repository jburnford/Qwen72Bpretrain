# Testing Workflow - Incremental Validation

**Problem**: We don't want to wait hours for resources only to have the job crash after 2 minutes due to a simple error.

**Solution**: Incremental testing with progressively longer jobs.

## Testing Phases

### Phase 1: Quick Environment Test (15 minutes)

**Purpose**: Verify everything can load without errors

**What it tests**:
- ✓ Module loading
- ✓ Package installation
- ✓ Model download/caching (or use cached)
- ✓ 4-bit quantization
- ✓ LoRA adapter setup
- ✓ Embedding unfreezing
- ✓ Memory footprint (~50-60 GB expected)

**Run**:
```bash
sbatch scripts/test_setup_quick.sh
```

**Expected time**: 10-15 minutes
**Expected cost**: ~0.25 GPU-hours

**Success criteria**:
- Job completes without errors
- GPU memory ~50-65 GB
- All ✓ checkmarks in output

---

### Phase 2: Short Training Test (1 hour)

**Purpose**: Verify actual training works and checkpoints function

**What it tests**:
- ✓ Data loading
- ✓ Training loop (50 steps)
- ✓ Checkpoint saving
- ✓ Checkpoint resumption
- ✓ LoRA + embedding gradients flowing correctly

**Run**:
```bash
sbatch scripts/test_training_short.sh
```

**Expected time**: 30-60 minutes
**Expected cost**: ~0.5-1 GPU-hours

**Success criteria**:
- Training runs for 50 steps
- 2 checkpoints created (at steps 25 and 50)
- Resumption from checkpoint works
- Training loss decreases

**What it creates**:
- Test output in `/scratch/$USER/qwen_test_output/`
- Can be deleted after test passes

---

### Phase 3: Medium Training Test (3-4 hours) - OPTIONAL

If you want extra confidence before committing to 20-hour jobs:

**Purpose**: Validate stability over longer run

```bash
# Modify test_training_short.sh:
#   Change: --max_steps 50   →   --max_steps 1000
#   Change: --time=1:00:00   →   --time=4:00:00

sbatch scripts/test_training_short.sh
```

**Expected time**: 3-4 hours
**Expected cost**: ~3-4 GPU-hours

---

### Phase 4: Full Training (20 hours × N jobs)

Once all tests pass, proceed with confidence:

```bash
# Preprocess full dataset
bash scripts/prepare_data.sh

# Start full training
sbatch scripts/train_qlora_single_gpu.sh
```

## Troubleshooting Failed Tests

### Phase 1 Failures

**"Module not found"**
- Check module names: `module avail`
- Verify StdEnv/2023 is loaded

**"Package X not in wheelhouse"**
- Check available versions: `avail_wheels X`
- May need to install without `--no-index`

**"CUDA out of memory" during model load**
- Model should use ~36-40 GB in 4-bit
- Check if other jobs running: `nvidia-smi`
- May indicate GPU already in use

### Phase 2 Failures

**"Training data not found"**
- Run data preprocessing: `bash scripts/prepare_data.sh`
- Or test will create minimal 3-example dataset

**"Error during backward pass"**
- Check if embeddings properly unfrozen
- Review logs for gradient errors
- May indicate PEFT/bitsandbytes version issue

**"Checkpoint not loading"**
- Check checkpoint directory exists
- Verify checkpoint files not corrupted
- Try specifying checkpoint manually

## Resource Costs

| Phase | Time | GPU-Hours | Cost (est.) |
|-------|------|-----------|-------------|
| Phase 1 (quick) | 15 min | 0.25 | $0.50-1 |
| Phase 2 (short) | 1 hour | 1 | $2-4 |
| Phase 3 (medium) | 4 hours | 4 | $8-16 |
| **Total testing** | **~5 hours** | **~5** | **$10-20** |
| Full training (per job) | 20 hours | 20 | $40-80 |
| Full training (total 2-3 jobs) | 40-60 hours | 40-60 | $80-240 |

**Value**: Spend $10-20 testing to avoid wasting $80+ on failed runs

## Quick Reference

```bash
# Step 1: Quick test (15 min)
sbatch scripts/test_setup_quick.sh

# Monitor
squeue -u $USER
tail -f logs/qwen-test-quick-JOBID.out

# Step 2: Training test (1 hour)
sbatch scripts/test_training_short.sh

# Monitor
tail -f logs/qwen-test-train-JOBID.out

# Step 3: If all passed, full training
bash scripts/prepare_data.sh  # First time only
sbatch scripts/train_qlora_single_gpu.sh
```

## What to Check in Logs

### Quick Test Success Indicators
```
✓ PyTorch: 2.x.x
✓ CUDA available: True
✓ Model loaded in 4-bit
GPU Memory allocated: ~35-40 GB
✓ LoRA adapters added
✓ Unfroze embed_tokens
✓ Unfroze lm_head
Trainable parameters: ~1-2% of total
✓ All checks passed!
```

### Training Test Success Indicators
```
{'loss': X.XXX, 'learning_rate': X.XXXX, 'epoch': X.XX}  # Should see decreasing loss
Saving model checkpoint to .../checkpoint-25
Saving model checkpoint to .../checkpoint-50
✓ Training test PASSED
✓ Checkpoints created: 2
✓ Checkpoint resumption PASSED
✓ ALL TESTS PASSED!
```

## Emergency Stop

If you realize mid-job that something is wrong:

```bash
# Cancel running job
scancel JOBID

# Check what's using GPU
ssh <node> nvidia-smi

# Clean up test outputs
rm -rf /scratch/$USER/qwen_test_output
```

---

**Remember**: Better to spend 5 hours testing than to waste 20 hours on a job that crashes!
