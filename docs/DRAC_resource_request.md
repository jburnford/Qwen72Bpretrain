# DRAC Resource Request: Qwen 72B Continued Pretraining

## Project Summary

**Title**: Domain-Adaptive Pretraining of Large Language Models for Historical Named Entity Recognition in Early Modern English Texts

**PI**: [Your PI name and account: def-jic823]

**Duration**: [e.g., 6 months]

## Scientific Justification

### Research Objectives

This project aims to improve Named Entity Recognition (NER) performance on early modern English historical documents from the Caribbean and Atlantic world (1600s-1800s) through continued pretraining of large language models.

**Why this matters:**
- Historical documents contain valuable information about people, places, and events
- Modern NER systems struggle with:
  - Non-standardized spelling (e.g., "wolde" vs "would")
  - Archaic vocabulary and grammar
  - OCR errors from digitized manuscripts
- Domain-adaptive pretraining has shown 70%+ improvements in domain-specific NER tasks

### Dataset

- **Source**: OLMOCR-processed Caribbean Pipeline corpus
- **Size**: 9.5 GB, 11,712 historical documents
- **Content**: Early modern English texts including almanacs, legal documents, correspondence
- **Time period**: 1600s-1800s
- **Estimated tokens**: 2-3 billion tokens

### Methodology

**Model**: Qwen 1.5 72B (72 billion parameters)
- State-of-the-art open-source model
- Proven performance on long-context tasks
- Suitable for historical text understanding

**Training approach**: Continued pretraining
- Phase 1: Domain-adaptive pretraining on unlabeled historical corpus
- Phase 2: Task-specific fine-tuning on labeled NER data
- Benefits over fine-tuning alone: Deep integration of domain knowledge

**Technical requirements**:
- DeepSpeed ZeRO-3 for distributed training
- Mixed precision (BF16) training
- Gradient checkpointing for memory efficiency
- Flash Attention 2 for computational efficiency

## Computational Requirements

### Hardware Requirements

**Per training run:**
- **GPUs**: 8× H100-80GB (single node)
- **Memory**: 512 GB RAM
- **CPUs**: 64 cores
- **Storage**:
  - `/scratch`: 500 GB for data and checkpoints
  - `/project`: 200 GB for final models

**Why 72B model requires 8 H100s:**
- Model weights (BF16): 144 GB
- Optimizer states: 288 GB
- Gradients: 144 GB
- Total memory requirement: ~576 GB
- With DeepSpeed ZeRO-3 sharding: 576 GB / 8 GPUs = 72 GB per GPU
- Leaves headroom for activations and intermediate values

### Time Requirements

**Estimated compute time:**

| Task | Wall Time | GPU-Hours | Notes |
|------|-----------|-----------|-------|
| Data preprocessing | 1 hour | 1 | Single CPU job |
| Initial test run | 2 hours | 16 | Validation of setup |
| Full training (1 epoch) | 12 hours | 96 | Main experiment |
| Full training (3 epochs) | 36 hours | 288 | Recommended |
| Hyperparameter tuning (3 runs) | 108 hours | 864 | Multiple configurations |
| Fine-tuning for NER | 12 hours | 96 | After pretraining |

**Total estimated GPU-hours**: ~1,400 (H100 GPU-hours)

**Breakdown by phase:**
1. Setup and testing: 100 GPU-hours
2. Continued pretraining: 500-800 GPU-hours
3. Fine-tuning and evaluation: 200 GPU-hours
4. Hyperparameter optimization: 300 GPU-hours

### Storage Requirements

- **Scratch** (`/scratch/$USER`):
  - Training data (JSONL): 15 GB
  - Checkpoints during training: 300 GB
  - Temporary files: 100 GB
  - **Total**: ~500 GB

- **Project** (`/project/def-jic823`):
  - Source corpus: 10 GB (shared)
  - Final trained models: 150 GB
  - Evaluation results: 10 GB
  - **Total**: ~200 GB

## Expected Outputs

1. **Domain-adapted Qwen 72B model**
   - Continued pretrained on early modern English
   - Improved perplexity on historical texts
   - Better contextual understanding of period language

2. **NER-specific fine-tuned model**
   - Optimized for entity recognition in historical documents
   - Benchmark results on test set

3. **Research outputs**
   - Model performance metrics and comparisons
   - Analysis of domain adaptation effectiveness
   - Documentation for community use

4. **Broader impact**
   - Open-source model for digital humanities research
   - Enables better analysis of historical Caribbean/Atlantic world documents
   - Methodology applicable to other historical corpora

## Justification for Resource Scale

**Why 72B parameters?**
- Smaller models (7B, 14B) may lack capacity for complex historical language patterns
- 72B models have shown superior few-shot learning and domain transfer
- Better long-range context understanding (critical for historical documents)

**Why H100 GPUs?**
- 80 GB VRAM required for 72B model training
- 3× faster than A100 for transformer workloads
- Native BF16 and Flash Attention support
- More cost-effective than smaller batches on many GPUs

**Why continued pretraining vs fine-tuning?**
- Fine-tuning alone: Surface-level adaptation
- Continued pretraining: Deep integration of domain vocabulary and patterns
- Literature shows 70%+ improvement for domain NER tasks
- Critical for non-standard language (spelling variations, archaic terms)

## Project Timeline

**Month 1-2**: Data preparation and infrastructure setup
- Preprocess OLMOCR corpus
- Validate training pipeline on small subset
- Establish baseline metrics

**Month 3-4**: Continued pretraining
- Full domain-adaptive pretraining
- Hyperparameter tuning
- Model evaluation on historical texts

**Month 5**: Fine-tuning and evaluation
- NER-specific fine-tuning
- Benchmark testing
- Comparison with baseline models

**Month 6**: Analysis and documentation
- Performance analysis
- Model documentation and release
- Publication preparation

## Prior Work and Feasibility

**Team expertise:**
- [Your research background in digital humanities/NLP]
- Experience with historical text processing
- Successful OLMOCR corpus creation and processing

**Validation:**
- OLMOCR pipeline successfully processed 11,712 documents
- Data quality validated through manual inspection
- Training infrastructure tested on DRAC clusters

**Similar successful projects:**
- PubMedBERT (biomedical domain adaptation)
- LegalBERT (legal domain)
- FinBERT (financial domain)
- All showed significant domain-specific improvements

## Resource Justification Summary

This project requires substantial computational resources (1,400 GPU-hours) because:

1. **Model scale**: 72B parameters necessary for complex historical language
2. **Domain adaptation**: Continued pretraining on 2-3B tokens requires multiple epochs
3. **Scientific rigor**: Multiple runs for hyperparameter optimization
4. **Novel contribution**: First large-scale LLM adaptation for early modern Caribbean English

The expected outputs (open-source model, methodology, research findings) will benefit the broader digital humanities and NLP communities working with historical texts.

## References

1. Gururangan et al. (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
2. Lee et al. (2020). "BioBERT: a pre-trained biomedical language model"
3. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
4. [Your relevant prior publications]

---

**Contact**: [Your email]
**Account**: def-jic823
**Preferred cluster**: Nibi (H100 availability)
