import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input_path",
                        help="path to train.jsonl file")
    args = parser.parse_args()

    labels_to_ids = {}
    i = 0
    with open(args.input_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        for target in json.loads(line)["targets"]:
            if target["label"] not in labels_to_ids.keys():
                labels_to_ids[target["label"]] = i
                i += 1
    print(labels_to_ids)
