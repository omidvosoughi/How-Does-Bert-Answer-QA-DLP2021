import argparse
import json
import os
import random
from typing import List

def reduce(
        input_dir: str,
        medium_size: int=5000,
        small_size: int=500,
        timing_size: int=427,
        val_split: float=0.15,
        test_split: float=0.1
        ) -> None:
    os.makedirs(f"{input_dir}/big", exist_ok=True)
    os.makedirs(f"{input_dir}/small", exist_ok=True)
    os.makedirs(f"{input_dir}/medium", exist_ok=True)
    os.makedirs(f"{input_dir}/timing", exist_ok=True)
    lines: List[str] = []
    train_split = 1-val_split-test_split
    for phase in ["train", "val", "test"]:
        #shutil.move(f"{input_dir}/{phase}.jsonl", f"{input_dir}/big/{phase}.jsonl")
        if os.path.isfile(f"{input_dir}/big/{phase}.jsonl"):
            with open(f"{input_dir}/big/{phase}.jsonl", 'r') as f:
                lines.extend(f.readlines())
    random.shuffle(lines)
    # Count samples and remove lines without samples.
    num_samples: int = 0
    remove_lines = []
    for line_index, line in enumerate(lines):
        line_samples = len(json.loads(line)["targets"])
        if not line_samples:
            remove_lines.append(line_index)
        num_samples += line_samples
    for line_index in sorted(remove_lines, reverse=True):
        del lines[line_index]
    # Find out where to split the samples.
    counter: int = 0
    for line_index, line in enumerate(lines):
        counter += len(json.loads(line)["targets"])
        if counter < train_split*num_samples:
            train_end: int = line_index
        elif counter < (train_split + val_split)*num_samples:
            val_end: int = line_index
        else:
            break
    phase_lines = {
        "train": lines[:train_end],
        "val": lines[train_end:val_end],
        "test": lines[val_end:],
        }
    for phase, split in {"train": train_split, "val": val_split, "test": test_split}.items():
        with open(f"{input_dir}/timing/{phase}.jsonl", 'w') as output:
            counter: int = 0
            lines_iter = iter(phase_lines[phase])
            while counter < timing_size*split:
                line = next(lines_iter, None)
                if line is None:
                    break
                counter += len(json.loads(line)["targets"])
                output.write(line)
            print(f"wrote {counter} samples to {phase}")
"""
    for phase, split in {"train": train_split, "val": val_split, "test": test_split}.items():
        with open(f"{input_dir}/big/{phase}.jsonl", 'w') as output:
            counter: int = 0
            lines_iter = iter(phase_lines[phase])
            while True:
                line = next(lines_iter, None)
                if line is None:
                    break
                counter += len(json.loads(line)["targets"])
                output.write(line)
            print(f"wrote {counter} samples to {phase}")

        with open(f"{input_dir}/medium/{phase}.jsonl", 'w') as output:
            counter: int = 0
            lines_iter = iter(phase_lines[phase])
            while counter < medium_size*split:
                line = next(lines_iter, None)
                if line is None:
                    break
                counter += len(json.loads(line)["targets"])
                output.write(line)
        with open(f"{input_dir}/small/{phase}.jsonl", 'w') as output:
            counter: int = 0
            lines_iter = iter(phase_lines[phase])
            while counter < small_size*split:
                line = next(lines_iter, None)
                if line is None:
                    break
                counter += len(json.loads(line)["targets"])
                output.write(line)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_path",
                        help="path to dir which contains train/val/test.jsonl files")
    args = parser.parse_args()
    reduce(args.input_path)
