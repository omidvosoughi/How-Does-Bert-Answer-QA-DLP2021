"""
This module provides functions to make train/val/test sets out of a list of samples
and write them to *.jsonl file.
"""

import os
import json
import random
from typing import List

def output_task_in_jiant_format(samples, out_dir: str=None) -> None:
    """Process dataset file and write the shuffled samples to train/dev/test files into out_dir."""

    if out_dir is None:
        out_dir: str = './output'
    test_ratio: float = 0.1
    dev_ratio: float = 0.15

    random.shuffle(samples)

    sample_size = len(samples)

    test_size = int(sample_size * test_ratio)
    dev_size = int(sample_size * dev_ratio)

    test_samples = samples[:test_size]
    dev_samples = samples[test_size:test_size + dev_size]
    train_samples = samples[test_size + dev_size:]

    # create output directory if not existent
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    write_samples_to_file(test_samples, os.path.join(out_dir, "test.jsonl"))
    write_samples_to_file(dev_samples, os.path.join(out_dir, "val.jsonl"))
    write_samples_to_file(train_samples, os.path.join(out_dir, "train.jsonl"))

def write_samples_to_file(samples: List, output_path: str) -> None:
    """Write samples to output_path."""
    if len(samples) > 0:

        with open(output_path, "w", encoding="utf-8") as output:
            for sample in samples:
                output.write(json.dumps(sample) + "\n")
        print(output_path)
    else:
        print("No sample to output")
