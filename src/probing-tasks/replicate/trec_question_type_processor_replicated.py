"""
Processor to transform the Question Classification task by Li and Roth (2002) into a Jiant Probing Task.
The Question Type Probing Task takes as input a question. The task is to classify the question type into one of 500
fine-grained types, e.g. entity:animal.

Example question in input format:

ENTY:animal What was the first domesticated bird ?

Example probing task result in Jiant format (JSON):

{"info": {"doc_id": "trec-qt", "q_id": 0},
 "text": "What was the first domesticated bird",
 "targets": [{"span1": [0, 6], "label": "ENTY:animal"}]}

"""

import argparse
from typing import List
from task_processors_replicated import output_task_in_jiant_format

def convert_to_jiant_ep_format(input_path: str):
    """
        Converts the TREC-10 Question Classification file into samples for the Question Type Probing task in Jiant format
        :return: A list of samples in jiant edge probing format.
    """

    DOC_ID = "trec-qt"
    samples = []
    sample_id = 0

    with open(input_path, encoding="latin-1") as input_file:
        info = {"doc_id": DOC_ID, "q_id": str(sample_id)}
        sample_id += 1
        lines = input_file.readlines()
        for line in lines:
            line = line[:-1]
            line_split = line.split(" ")
            
            label = line_split[0]
            text = line_split[1:-1]
                
            entry = {"info": info,
                     "text": " ".join(text),
                     "targets": [{"span1": [0, len(text)], "label": label}]}

            samples.append(entry)
    
    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    args = parser.parse_args()

    samples = convert_to_jiant_ep_format(input_path=args.input_path)
    #print(samples)
    output_task_in_jiant_format(samples)