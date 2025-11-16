#!/usr/bin/env python3
"""
Analyze the OLMOCR Caribbean corpus data.
Run this script on nibi cluster to analyze the data.
"""

import os
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import re

def analyze_corpus(data_dir):
    """Analyze the OLMOCR corpus."""

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        sys.exit(1)

    print("="*60)
    print("OLMOCR Caribbean Corpus Analysis")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print()

    # Get all document directories
    doc_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    num_docs = len(doc_dirs)

    print(f"Total document directories: {num_docs:,}")
    print()

    # Analyze documents
    total_text_length = 0
    total_entries = 0
    total_tokens_estimate = 0
    years = []
    missing_olmocr = 0
    file_sizes = []

    print("Analyzing documents...")
    for i, doc_dir in enumerate(doc_dirs, 1):
        if i % 500 == 0:
            print(f"  Processed {i:,}/{num_docs:,} documents...")

        # Check for olmocr_results.json
        olmocr_file = doc_dir / "olmocr_results.json"
        metadata_file = doc_dir / "metadata.json"

        if not olmocr_file.exists():
            missing_olmocr += 1
            continue

        # Get file size
        file_sizes.append(olmocr_file.stat().st_size)

        # Read metadata for year
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if metadata.get('year'):
                        years.append(metadata['year'])
            except:
                pass

        # Read and analyze OLMOCR results
        try:
            with open(olmocr_file, 'r', encoding='utf-8') as f:
                olmocr_data = json.load(f)

            if isinstance(olmocr_data, list):
                for entry in olmocr_data:
                    if 'text' in entry:
                        text = entry['text']
                        text_len = len(text)
                        total_text_length += text_len
                        total_entries += 1

                        # Rough token estimate: ~4 chars per token for English
                        total_tokens_estimate += text_len // 4
        except Exception as e:
            print(f"  Error reading {olmocr_file}: {e}")

    print()
    print("="*60)
    print("Results")
    print("="*60)

    print(f"\nDocument Statistics:")
    print(f"  Total document directories: {num_docs:,}")
    print(f"  Documents with OLMOCR results: {num_docs - missing_olmocr:,}")
    print(f"  Missing OLMOCR files: {missing_olmocr:,}")

    print(f"\nText Statistics:")
    print(f"  Total text entries: {total_entries:,}")
    print(f"  Total characters: {total_text_length:,}")
    print(f"  Total characters (GB): {total_text_length / 1e9:.2f} GB")
    print(f"  Estimated tokens: {total_tokens_estimate:,}")
    print(f"  Estimated tokens (millions): {total_tokens_estimate / 1e6:.1f}M")
    print(f"  Average entry length: {total_text_length / max(1, total_entries):,.0f} characters")

    print(f"\nFile Size Statistics:")
    if file_sizes:
        import statistics
        print(f"  Average OLMOCR file size: {statistics.mean(file_sizes):,.0f} bytes")
        print(f"  Median OLMOCR file size: {statistics.median(file_sizes):,.0f} bytes")
        print(f"  Min/Max file size: {min(file_sizes):,} / {max(file_sizes):,} bytes")

    print(f"\nTemporal Coverage:")
    if years:
        print(f"  Documents with year metadata: {len(years):,}")
        print(f"  Year range: {min(years)} - {max(years)}")

        # Count by century
        century_counts = Counter()
        for year in years:
            century = (year // 100) * 100
            century_counts[century] += 1

        print(f"\n  Documents by century:")
        for century in sorted(century_counts.keys()):
            count = century_counts[century]
            print(f"    {century}s: {count:,} documents")

    print()
    print("="*60)
    print("Sample Text Extraction")
    print("="*60)

    # Show a sample
    sample_doc = doc_dirs[0]
    sample_file = sample_doc / "olmocr_results.json"

    if sample_file.exists():
        with open(sample_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)

        if isinstance(sample_data, list) and len(sample_data) > 0:
            sample_text = sample_data[0].get('text', '')[:500]
            print(f"\nSample from {sample_doc.name}:")
            print("-"*60)
            print(sample_text)
            print("-"*60)

    print()
    print("="*60)
    print("Recommendations")
    print("="*60)

    print(f"""
1. Dataset Size:
   - Estimated ~{total_tokens_estimate / 1e6:.1f}M tokens is {"EXCELLENT" if total_tokens_estimate > 100e6 else "GOOD" if total_tokens_estimate > 50e6 else "MODEST"}
     for continued pretraining of Qwen 72B
   - For optimal results, aim for 100M-5B tokens

2. Preprocessing Steps:
   - Extract text from all olmocr_results.json files
   - Combine into single JSONL format with {{"text": "..."}}
   - Consider packing multiple documents into sequences
   - Split 95% train / 5% validation

3. Quality Checks Needed:
   - Manual review of OCR quality
   - Check for non-English content
   - Identify and handle special characters
   - Review early modern English spelling variations

4. Training Configuration:
   - Max sequence length: 2048-4096 tokens
   - Batch size: Start with 1 per GPU
   - Gradient accumulation: 16-32 steps
   - Training epochs: 2-3
    """)

    return {
        'num_docs': num_docs,
        'total_entries': total_entries,
        'total_tokens_estimate': total_tokens_estimate,
        'years': years,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze OLMOCR corpus")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/jic823/projects/def-jic823/caribbean_pipeline/02_processed",
        help="Path to processed data directory"
    )

    args = parser.parse_args()
    analyze_corpus(args.data_dir)
