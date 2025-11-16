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
    set_seed,
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

    model_kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if training_args.bf16 else torch.float16,
    }

    # Add Flash Attention 2 if requested
    if model_args.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2")

    # For DeepSpeed, don't use device_map
    if training_args.deepspeed is None:
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Log model info
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

    # Training
    logger.info("Starting training...")
    train_result = trainer.train()

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
