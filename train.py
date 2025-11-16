#!/usr/bin/env python3
"""
Continued pretraining script for Qwen 1.5 72B on early modern English corpus.
Optimized for DRAC clusters with DeepSpeed ZeRO-3 support.
"""

import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""

    model_name_or_path: str = field(
        default="Qwen/Qwen1.5-72B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Whether to use Flash Attention 2"}
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Whether to use QLoRA (4-bit quantization + LoRA)"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
        metadata={"help": "Target modules for LoRA (comma-separated)"}
    )
    train_embeddings: bool = field(
        default=True,
        metadata={"help": "Whether to unfreeze and train embedding layers (embed_tokens, lm_head)"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training."""

    train_data_path: str = field(
        metadata={"help": "Path to training data (JSONL format)"}
    )
    validation_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation data (JSONL format)"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences longer than this will be truncated."}
    )
    preprocessing_num_workers: int = field(
        default=8,
        metadata={"help": "Number of processes to use for data preprocessing"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments with additional options."""

    # Override defaults for continued pretraining
    output_dir: str = field(default="./output/qwen72b_early_modern")
    num_train_epochs: float = field(default=2.0)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=5)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    report_to: str = field(default="tensorboard")
    deepspeed: Optional[str] = field(default=None)
    max_grad_norm: float = field(default=1.0)
    resume_from_checkpoint: Optional[str] = field(default=None)

    # Distributed training settings
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_timeout: int = field(default=1800)


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's a json file,
        # parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        use_fast=False,  # Use slow tokenizer for better compatibility
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Set padding side to right for causal LM
    tokenizer.padding_side = "right"

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")

    # Configure quantization for QLoRA
    if model_args.use_qlora:
        logger.info("Using QLoRA (4-bit quantization + LoRA adapters)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        )
        model_kwargs = {
            "cache_dir": model_args.cache_dir,
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    else:
        logger.info("Using full precision training (not QLoRA)")
        model_kwargs = {
            "cache_dir": model_args.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
        }
        # For DeepSpeed, don't use device_map
        if training_args.deepspeed is None:
            model_kwargs["device_map"] = "auto"

    # Add Flash Attention 2 if requested
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Apply QLoRA if requested
    if model_args.use_qlora:
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        target_modules = model_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA adapters
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA configuration: r={model_args.lora_r}, alpha={model_args.lora_alpha}, dropout={model_args.lora_dropout}")
        logger.info(f"Target modules: {target_modules}")

        # Unfreeze embedding layers for domain adaptation
        if model_args.train_embeddings:
            logger.info("Unfreezing embedding layers (embed_tokens, lm_head) for vocabulary adaptation")

            # Access the base model under the PEFT wrapper
            embedding_modules = []

            # Unfreeze input embeddings (embed_tokens)
            if hasattr(model.base_model.model, 'model'):
                # For models with model.model structure (like Qwen)
                if hasattr(model.base_model.model.model, 'embed_tokens'):
                    model.base_model.model.model.embed_tokens.requires_grad_(True)
                    embedding_modules.append("model.embed_tokens")
            elif hasattr(model.base_model.model, 'embed_tokens'):
                model.base_model.model.embed_tokens.requires_grad_(True)
                embedding_modules.append("embed_tokens")

            # Unfreeze output embeddings (lm_head)
            if hasattr(model.base_model.model, 'lm_head'):
                model.base_model.model.lm_head.requires_grad_(True)
                embedding_modules.append("lm_head")

            logger.info(f"Unfrozen embedding modules: {embedding_modules}")

            # Convert embeddings to full precision for training
            for module_name in embedding_modules:
                try:
                    if "embed_tokens" in module_name:
                        if hasattr(model.base_model.model, 'model'):
                            embed_module = model.base_model.model.model.embed_tokens
                        else:
                            embed_module = model.base_model.model.embed_tokens
                        # Convert to bf16 or fp16 for training
                        embed_module.to(torch.bfloat16 if training_args.bf16 else torch.float16)
                    elif "lm_head" in module_name:
                        lm_head_module = model.base_model.model.lm_head
                        lm_head_module.to(torch.bfloat16 if training_args.bf16 else torch.float16)
                except Exception as e:
                    logger.warning(f"Could not convert {module_name} to training dtype: {e}")

    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Log model info
    if model_args.use_qlora:
        logger.info("=" * 50)
        logger.info("Trainable Parameters Summary:")
        model.print_trainable_parameters()

        # Also log embedding parameters separately
        if model_args.train_embeddings:
            embed_params = 0
            for name, param in model.named_parameters():
                if 'embed' in name.lower() or 'lm_head' in name.lower():
                    if param.requires_grad:
                        embed_params += param.numel()
            logger.info(f"Embedding layer parameters: {embed_params:,}")
        logger.info("=" * 50)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load datasets
    logger.info(f"Loading training data from {data_args.train_data_path}")
    data_files = {"train": data_args.train_data_path}

    if data_args.validation_data_path:
        logger.info(f"Loading validation data from {data_args.validation_data_path}")
        data_files["validation"] = data_args.validation_data_path

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    logger.info(f"Training examples: {len(raw_datasets['train'])}")
    if "validation" in raw_datasets:
        logger.info(f"Validation examples: {len(raw_datasets['validation'])}")

    # Tokenization function
    def tokenize_function(examples):
        """Tokenize the examples."""
        # Extract text from examples
        texts = examples["text"]

        # Tokenize
        tokenized = tokenizer(
            texts,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Tokenizing texts",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation", None)

    # Data collator
    # We don't need MLM for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Auto-detect checkpoint for resumption
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from specified checkpoint: {checkpoint}")
    else:
        # Auto-detect latest checkpoint
        output_dir = Path(training_args.output_dir)
        if output_dir.exists():
            checkpoints = list(output_dir.glob("checkpoint-*"))
            if checkpoints:
                # Sort by step number
                checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
                checkpoint = str(checkpoints[-1])
                logger.info(f"Auto-detected latest checkpoint: {checkpoint}")
                logger.info(f"Resuming training from step {checkpoints[-1].name.split('-')[1]}")

    # Training
    if checkpoint:
        logger.info(f"Starting training from checkpoint: {checkpoint}")
    else:
        logger.info("Starting training from scratch...")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate
    if eval_dataset is not None:
        logger.info("Evaluating...")
        metrics = trainer.evaluate()

        # Calculate perplexity
        try:
            perplexity = torch.exp(torch.tensor(metrics["eval_loss"]))
            metrics["perplexity"] = perplexity.item()
        except OverflowError:
            metrics["perplexity"] = float("inf")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
