"""
Processor to transform the bAbI Question Answering Dataset into a Jiant Probing Task.
The Supporting Facts Probing Task takes as input a question and a sentence from the context. The task is to decide
whether the sentence is part of the Supporting Facts for this question.

Example question in bAbI format:

1 John moved to the bathroom.
2 Mary got the football there.
3 Mary went back to the kitchen.
4 Where is the football? 	kitchen	2 3

Example probing task result in Jiant format (JSON):

{"info": {"doc_id": "babi_sup_facts", "q_id": "0"},
 "text": "Where is the football ? John moved to the bedroom . Mary got the football there . Mary went to the kitchen .",
 "targets":
    [{"span1": [0, 5], "span2": [5, 11], "label": "0"},
     {"span1": [0, 5], "span2": [11, 17], "label": "1"},
     {"span1": [0, 5], "span2": [17, 24], "label": "1"}]}

"""
import argparse
from task_processors_replicated import output_task_in_jiant_format

def convert_to_jiant_ep_format(input_path: str):
    samples = []
    
    DOC_ID = "babi_sup_facts"
    q_id = 0

    with open(input_path, encoding="latin-1") as input_file:
        lines = input_file.readlines()
        text = ""
        current_context = {}

        for line in lines:
            info = {"doc_id": DOC_ID, "q_id": str(q_id)}
            
            if(line.startswith("1 ")):
                text = ""
                current_context = {}
            
            targets = []
            if("\t" in line):
                line_split = line.split("\t")
                question_split = line_split[0].split(" ")
                question = " ".join(question_split[1:])
                question = question[:-2] + " " + question[-2]
                
                ques_len = len(question.split(" "))

                sup_facts = line_split[2].split(" ")
                sup_facts = [int(s) for s in sup_facts]
                span_start = ques_len
                for key in current_context:
                    if key in sup_facts:
                        label = "1"
                    else: 
                        label = "0"
                    span2_len = current_context[int(key)]
                    targets.append({"span1": [0, ques_len], 
                                    "span2": [span_start, span_start+span2_len], 
                                    "label": label})
                    span_start += span2_len 
                
                entry = {"info": info,
                         "text": question + text,
                         "targets": targets}
                samples.append(entry)
                q_id += 1

            else:
                line_split = line.split(" ")
                line_text = " ".join(line_split[1:]) 
                text += " " + line_text[:-2] + " " + line_text[-2]
                current_context[int(line_split[0])] = len(line.split(" "))

    return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    args = parser.parse_args()

    samples = convert_to_jiant_ep_format(input_path=args.input_path)
    output_task_in_jiant_format(samples)
    