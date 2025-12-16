import os
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "/work1/lgarcia/renneruan/data/padded-tokenized-for-training/custom/vocab_size:32_768/context_size:8192"
CONTEXT_LIMIT = 8192


def analyze_dataset_full(path, limit):
    print(f"Loading dataset from: {path}")

    try:
        dataset = load_from_disk(path)
    except FileNotFoundError:
        print("Error: Path not found.")
        return

    if hasattr(dataset, "keys") and "train" in dataset.keys():
        ds = dataset["train"]
    else:
        ds = dataset

    print("Extracting sequence lengths... (this might take a moment)")

    # 1. Get all lengths (memory efficient generator converted to list)
    lengths = [
        len(x) for x in tqdm(ds["input_ids"], desc="Processing lengths")
    ]

    # 2. Convert to numpy for fast stats
    lengths_arr = np.array(lengths)
    total_seqs = len(lengths_arr)

    if total_seqs == 0:
        print("Dataset is empty.")
        return

    # --- CALCULATE TOTALS ---
    # Convert to Python int to avoid potential overflow with standard numpy int32 on massive datasets
    total_tokens = int(np.sum(lengths_arr))

    # --- CALCULATE DISTRIBUTION ---
    min_len = np.min(lengths_arr)
    max_len = np.max(lengths_arr)
    mean_len = np.mean(lengths_arr)
    median_len = np.median(lengths_arr)

    # Threshold Checks
    at_limit = np.sum(lengths_arr == limit)  # Exactly 1024
    over_limit = np.sum(lengths_arr > limit)  # > 1024 (Should be 0)
    good_seqs = np.sum(lengths_arr > 1024)
    short_seqs = np.sum(lengths_arr < 20)  # Noise

    # Percentiles
    p25, p75, p90, p99 = np.percentile(lengths_arr, [25, 75, 90, 99])

    # --- REPORT ---
    print("\n" + "=" * 50)
    print(f"  DATASET METADATA REPORT")
    print("=" * 50)
    # The requested Totals
    print(f"{'Total Sequences':<25} | {total_seqs:,}")
    print(f"{'Total Tokens':<25} | {total_tokens:,}")
    print(f"{'Avg Tokens/Seq':<25} | {mean_len:.2f}")
    print("-" * 50)

    print(f"  LENGTH DISTRIBUTION (Target: {limit})")
    print("-" * 50)
    print(f"Min Length:               {min_len}")
    print(f"Max Length:               {max_len}")
    print(f"Median Length:            {median_len:.2f}")
    print("-" * 50)
    print(f"25th Percentile:          {int(p25)}")
    print(f"75th Percentile:          {int(p75)}")
    print(f"90th Percentile:          {int(p90)}")
    print(f"99th Percentile:          {int(p99)}")
    print("-" * 50)
    print("  ANOMALY CHECKS")
    print("-" * 50)
    print(
        f"Saturation (={limit}):      {at_limit:,}  ({at_limit/total_seqs:.2%})"
    )
    print(
        f"Overflow   (>{limit}):      {over_limit:,}  ({over_limit/total_seqs:.2%})"
    )
    print(
        f"Too Short  (<20):         {short_seqs:,}  ({short_seqs/total_seqs:.2%})"
    )
    print(
        f"Good Seqs (> 1024):         {good_seqs:,}  ({good_seqs/total_seqs:.2%})"
    )
    print("=" * 50)


if __name__ == "__main__":
    analyze_dataset_full(DATASET_PATH, CONTEXT_LIMIT)
