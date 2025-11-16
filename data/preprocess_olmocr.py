#!/usr/bin/env python3
"""
Preprocess OLMOCR Caribbean corpus for Qwen 72B training.
Converts olmocr_results.json files to training JSONL format.
"""

import os
import json
import sys
import re
from pathlib import Path
from typing import List, Dict
import argparse
import random

def clean_text(text: str) -> str:
    """
    Minimal cleaning for early modern English text.
    Preserve historical spelling and grammar.
    """
    # Normalize excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize quotation marks (optional)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Remove excessive newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text.strip()


def process_olmocr_file(filepath: Path, min_length: int = 50) -> List[str]:
    """
    Process a single OLMOCR results JSON file.
    Returns list of text entries.
    """
    texts = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                if 'text' in entry:
                    text = clean_text(entry['text'])

                    # Filter very short texts
                    if len(text) >= min_length:
                        texts.append(text)

    except Exception as e:
        print(f"  Warning: Error processing {filepath}: {e}", file=sys.stderr)

    return texts


def pack_texts(texts: List[str], max_length: int = 2048, separator: str = "\n\n") -> List[str]:
    """
    Pack multiple short texts into longer sequences.
    This is more efficient for training than many short sequences.
    """
    packed = []
    current = []
    current_length = 0

    for text in texts:
        text_length = len(text)

        # Rough estimate: 4 chars per token
        text_tokens = text_length // 4

        if current_length + text_tokens > max_length:
            # Save current pack and start new one
            if current:
                packed.append(separator.join(current))
            current = [text]
            current_length = text_tokens
        else:
            current.append(text)
            current_length += text_tokens

    # Add remaining
    if current:
        packed.append(separator.join(current))

    return packed


def preprocess_corpus(
    data_dir: str,
    output_file: str,
    validation_split: float = 0.05,
    pack_sequences: bool = True,
    max_seq_length: int = 2048,
    min_text_length: int = 50,
    random_seed: int = 42,
):
    """
    Preprocess the entire OLMOCR corpus.
    """
    data_path = Path(data_dir)
    output_path = Path(output_file)

    if not data_path.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get all document directories
    doc_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    num_docs = len(doc_dirs)

    print("="*60)
    print("OLMOCR Caribbean Corpus Preprocessing")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_file}")
    print(f"Number of documents: {num_docs:,}")
    print(f"Validation split: {validation_split:.1%}")
    print(f"Pack sequences: {pack_sequences}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Min text length: {min_text_length}")
    print()

    # Collect all texts
    all_texts = []

    print("Processing documents...")
    for i, doc_dir in enumerate(doc_dirs, 1):
        if i % 500 == 0:
            print(f"  Processed {i:,}/{num_docs:,} documents ({len(all_texts):,} texts so far)...")

        olmocr_file = doc_dir / "olmocr_results.json"

        if not olmocr_file.exists():
            continue

        texts = process_olmocr_file(olmocr_file, min_length=min_text_length)
        all_texts.extend(texts)

    print(f"\nTotal texts extracted: {len(all_texts):,}")

    # Pack sequences if requested
    if pack_sequences:
        print(f"\nPacking sequences (max length: {max_seq_length})...")
        all_texts = pack_texts(all_texts, max_length=max_seq_length)
        print(f"  Packed into {len(all_texts):,} sequences")

    # Shuffle
    random.seed(random_seed)
    random.shuffle(all_texts)

    # Split train/validation
    split_idx = int(len(all_texts) * (1 - validation_split))
    train_texts = all_texts[:split_idx]
    valid_texts = all_texts[split_idx:]

    print(f"\nSplit:")
    print(f"  Train: {len(train_texts):,} examples")
    print(f"  Validation: {len(valid_texts):,} examples")

    # Determine output filenames
    if output_file.endswith('.jsonl'):
        train_output = output_file
        valid_output = output_file.replace('.jsonl', '_valid.jsonl')
    else:
        train_output = f"{output_file}_train.jsonl"
        valid_output = f"{output_file}_valid.jsonl"

    # Write training data
    print(f"\nWriting training data to {train_output}...")
    with open(train_output, 'w', encoding='utf-8') as f:
        for text in train_texts:
            entry = {"text": text}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Write validation data
    print(f"Writing validation data to {valid_output}...")
    with open(valid_output, 'w', encoding='utf-8') as f:
        for text in valid_texts:
            entry = {"text": text}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Calculate statistics
    train_chars = sum(len(t) for t in train_texts)
    valid_chars = sum(len(t) for t in valid_texts)
    train_tokens_est = train_chars // 4
    valid_tokens_est = valid_chars // 4

    print()
    print("="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Training: {train_output}")
    print(f"  Validation: {valid_output}")
    print(f"\nTraining set:")
    print(f"  Examples: {len(train_texts):,}")
    print(f"  Characters: {train_chars:,}")
    print(f"  Estimated tokens: {train_tokens_est:,} ({train_tokens_est/1e6:.1f}M)")
    print(f"\nValidation set:")
    print(f"  Examples: {len(valid_texts):,}")
    print(f"  Characters: {valid_chars:,}")
    print(f"  Estimated tokens: {valid_tokens_est:,} ({valid_tokens_est/1e6:.1f}M)")
    print()

    # Show sample
    print("="*60)
    print("Sample training example:")
    print("="*60)
    if train_texts:
        sample = train_texts[0][:500]
        print(sample)
        print("...")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OLMOCR corpus for training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jic823/projects/def-jic823/caribbean_pipeline/02_processed",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/scratch/$USER/early_modern_data/train.jsonl",
        help="Output file path for training data"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)"
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=50,
        help="Minimum text length to include (default: 50)"
    )
    parser.add_argument(
        "--no_packing",
        action="store_true",
        help="Don't pack sequences (keep individual documents)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Expand environment variables in output path
    output = os.path.expandvars(args.output)

    preprocess_corpus(
        data_dir=args.data_dir,
        output_file=output,
        validation_split=args.validation_split,
        pack_sequences=not args.no_packing,
        max_seq_length=args.max_seq_length,
        min_text_length=args.min_text_length,
        random_seed=args.seed,
    )
