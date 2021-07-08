import argparse
import shutil
import os
import json
#from tqdm import tqdm

def reduce(input_dir: str, train_size: int=5000, val_size: int=500) -> None:
    for phase, limit in {"train": train_size, "val": val_size}.items():
        os.makedirs(f"{input_dir}/big", exist_ok=True)
        os.makedirs(f"{input_dir}/small", exist_ok=True)
        os.makedirs(f"{input_dir}/medium", exist_ok=True)
        shutil.move(f"{input_dir}/{phase}.jsonl", f"{input_dir}/big/{phase}.jsonl")
        with open(f"{input_dir}/big/{phase}.jsonl", 'r') as f:
            with open(f"{input_dir}/medium/{phase}.jsonl", 'w') as output:
                counter: int = 0
                lines = iter(f.readlines())                
                while counter < limit:
                    line = next(lines, None)
                    if line is None:
                        break
                    counter += len(json.loads(line)["targets"])
                    output.write(line)
            with open(f"{input_dir}/small/{phase}.jsonl", 'w') as output:
                counter: int = 0
                lines = iter(f.readlines())
                while counter < limit // 10:
                    line = next(lines, None)
                    if line is None:
                        break
                    counter += len(json.loads(line)["targets"])
                    output.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_path",
                        help="path to dir which contains train/val/test.jsonl files")
    args = parser.parse_args()
    reduce(args.input_path)
