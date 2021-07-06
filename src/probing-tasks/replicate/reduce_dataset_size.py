import argparse
import shutil
import os

def reduce(input_dir: str) -> None:
    for phase, limit in {"train": 7500, "val": 1500}.items():
        os.makedirs(f"{input_dir}/big", exist_ok=True)
        shutil.move(f"{input_dir}/{phase}.jsonl", f"{input_dir}/big/{phase}.jsonl")
        with open(f"{input_dir}/big/{phase}.jsonl", 'r') as f:
            with open(f"{input_dir}/{phase}.jsonl", 'w') as output:
                output.write("".join(f.readlines()[:limit]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_path",
                        help="path to dir which contains train/val/test.jsonl files")
    args = parser.parse_args()
    reduce(args.input_path)
