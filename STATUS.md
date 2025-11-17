# Qwen 72B Pretraining Project - Current Status

**Last Updated**: 2025-11-16 11:05 AM EST

## ✅ Completed

### 1. Project Setup (100%)
- ✅ Repository cloned to nibi: `/home/jic823/Qwen72Bpretrain`
- ✅ All scripts executable and ready
- ✅ Directory structure created (logs, configs, data, scripts)
- ✅ Environment variables and paths configured

### 2. Quick Environment Test (PASSED)
- **Job ID**: 4569775
- **Status**: ✅ COMPLETED SUCCESSFULLY
- **Duration**: < 30 seconds
- **Results**:
  - ✅ All modules loaded correctly (Python 3.11, CUDA 12.2, cudnn)
  - ✅ Virtual environment created on node-local storage
  - ✅ All packages installed from DRAC wheelhouse:
    - torch 2.9.0
    - transformers 4.57.1
    - datasets 4.3.0
    - peft 0.17.1
    - bitsandbytes 0.48.2
    - flash-attn 2.8.3
    - accelerate 1.11.0
  - ✅ No major errors (minor pyarrow dummy package warning expected)

## 🔄 In Progress

### 3. Training Test (1-hour)
- **Job ID**: 4569916
- **Status**: ⏳ PENDING (waiting for GPU resources)
- **Submitted**: 2025-11-16 ~11:00 AM
- **Queue Position**: Waiting behind other running jobs
- **Expected**: Will start when H100 GPU becomes available
- **Purpose**:
  - Train for 50 steps
  - Test checkpoint saving (every 25 steps)
  - Test checkpoint resumption
  - Verify Hybrid QLoRA works (4-bit + LoRA + unfrozen embeddings)

### Queue Status
Current situation:
- 15+ other GPU jobs running for user jic823
- Job is pending with reason: "(Priority)"
- Will automatically start when resources free up
- Estimated wait time: Unknown (depends on other jobs completing)

## 📋 Next Steps

### Immediate (Automated)
1. **Wait for training test to start** - Job 4569916 will run when GPU available
2. **Monitor training test** - Watch logs/qwen-test-train-4569916.out
3. **Verify test passes** - Check for "✓ ALL TESTS PASSED" message

### After Training Test Passes
4. **Run data preprocessing**:
   ```bash
   ssh nibi
   cd Qwen72Bpretrain
   bash scripts/prepare_data.sh
   ```
   - Will process 11,712 OLMOCR documents (~9.5 GB)
   - Creates train.jsonl and validation split
   - Expected output: ~2-3 billion tokens
   - Time estimate: 1-2 hours on login node or submit as CPU job

5. **Submit first full training job** (20 hours):
   ```bash
   sbatch scripts/train_qlora_single_gpu.sh
   ```

6. **Monitor and continue training**:
   - Each job runs 20 hours (under 24h limit)
   - Auto-resumes from latest checkpoint
   - Resubmit with: `bash scripts/continue_training.sh`
   - Typically need 2-3 jobs for 3 epochs

## 📊 Resource Summary

### Tests Completed
| Test | GPU-Hours | Cost (est) | Result |
|------|-----------|------------|--------|
| Quick environment | 0.01 | ~$0.02 | ✅ PASS |
| Training test | Pending | ~$2-4 | ⏳ Queued |

### Full Training Estimates
| Phase | Time | GPU-Hours | Cost (est) |
|-------|------|-----------|------------|
| Data preprocessing | 1-2 hours | 0 (CPU) | Free |
| Training job 1 | 20 hours | 20 | $40-80 |
| Training job 2 | 20 hours | 20 | $40-80 |
| Training job 3 (if needed) | 20 hours | 20 | $40-80 |
| **Total** | **60-80 hours** | **60-80** | **$120-240** |

## 🛠️ Technical Configuration

### Hybrid QLoRA Setup
- **Model**: Qwen 1.5 72B
- **Quantization**: 4-bit (NF4)
- **LoRA rank**: 64
- **LoRA alpha**: 16
- **Trainable components**:
  - LoRA adapters (attention/MLP layers)
  - Embedding layers (embed_tokens + lm_head)
- **Memory footprint**: ~60-65 GB (fits 1× H100-80GB)

### Training Hyperparameters
- Batch size: 4 per GPU
- Gradient accumulation: 8 steps
- Effective batch size: 32
- Learning rate: 1e-4
- Scheduler: Cosine with 5% warmup
- Sequence length: 2048 tokens
- Precision: BF16 compute, 4-bit storage

### Data
- **Source**: `/home/jic823/projects/def-jic823/caribbean_pipeline/02_processed/`
- **Size**: 9.5 GB, 11,712 documents
- **Format**: OLMOCR JSON results
- **Content**: Early modern English (1600s-1800s)
- **Preprocessing status**: ⏳ NOT YET STARTED

## ⚠️ Known Issues

### None so far!
- Quick test passed without issues
- All packages installed correctly
- Environment setup validated

## 📁 Files

### On nibi (`/home/jic823/Qwen72Bpretrain/`)
```
Qwen72Bpretrain/
├── configs/
│   ├── ds_config_zero2.json
│   └── ds_config_zero3.json
├── data/
│   ├── analyze_data.py
│   └── preprocess_olmocr.py
├── docs/
│   ├── DRAC_resource_request.md
│   └── testing_workflow.md
├── logs/
│   ├── qwen-test-quick-4569775.out  ✅ PASSED
│   └── qwen-test-train-4569916.out  ⏳ PENDING
├── scripts/
│   ├── continue_training.sh
│   ├── prepare_data.sh
│   ├── setup_drac_env.sh
│   ├── test_setup_quick.sh          ✅ USED
│   ├── test_training_short.sh       ⏳ SUBMITTED
│   ├── train_qlora_single_gpu.sh
│   └── train_single_node.sh
├── train.py                          Main training script
├── README.md
└── requirements.txt
```

## 🎯 Success Criteria

### Phase 1: Testing ✅
- [x] Quick test passes - Environment validated
- [ ] Training test passes - Waiting for GPU
- [ ] Checkpoint system works

### Phase 2: Data Preparation
- [ ] OLMOCR data preprocessed
- [ ] Train/validation split created
- [ ] Token count verified (~2-3B tokens)

### Phase 3: Full Training
- [ ] First 20-hour job completes
- [ ] Checkpoints save correctly
- [ ] Second job resumes from checkpoint
- [ ] Training converges (loss decreases)
- [ ] Final model saved

### Phase 4: Evaluation
- [ ] Model performs NER on early modern English
- [ ] Perplexity improved vs base model
- [ ] Vocabulary adapted (handles "wolde", archaic terms)

## 📞 Monitoring Commands

```bash
# SSH to nibi
ssh nibi

# Check job queue
squeue -u jic823

# Check specific job
squeue -j 4569916

# View training test log (when running)
tail -f Qwen72Bpretrain/logs/qwen-test-train-4569916.out

# Check GPU availability
sinfo -p gpu-h100

# Cancel job if needed
scancel 4569916
```

## 🚀 Quick Reference

**Current working directory on nibi**: `/home/jic823/Qwen72Bpretrain`

**Data location**: `/home/jic823/projects/def-jic823/caribbean_pipeline/02_processed/`

**Output location (when training)**: `/scratch/jic823/qwen72b_qlora_output/`

**Account**: `def-jic823`

**Cluster**: nibi (H100 GPUs)

---

**Note**: This is an automated status document. Check job logs for detailed real-time information.
