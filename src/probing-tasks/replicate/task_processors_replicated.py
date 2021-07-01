import os
import json
import random
from typing import List, Dict

def output_task_in_jiant_format(samples) -> None:
    """
    Processes dataset file and writes the shuffled samples in jiant format to train/dev/test files
    into ./output.
    """

    output_dir: str = "./output"
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    write_samples_to_file(test_samples, os.path.join(output_dir, "test.json"))
    write_samples_to_file(dev_samples, os.path.join(output_dir, "dev.json"))
    write_samples_to_file(train_samples, os.path.join(output_dir, "train.json"))

def write_samples_to_file(samples: List, output_path: str) -> None:
    if len(samples) > 0:

        with open(output_path, "w", encoding="utf-8") as output:
            for sample in samples:
                output.write(json.dumps(sample) + "\n")
        print(output_path)
    else:
        print("No sample to output")