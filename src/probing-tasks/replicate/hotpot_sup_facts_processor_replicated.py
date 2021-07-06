"""
Module to transform the HotpotQA Dataset into a Jiant Probing Task.

Example question in HotpotQA format (JSON):

{"_id": "5a8c7595554299585d9e36b6",
 "answer": "Chief of Protocol",
 "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
 "supporting_facts": [["Kiss and Tell (1945 film)", 0], ["Shirley Temple", 0], ["Shirley Temple", 1]],
 "context": [["Kiss and Tell (1945 film)", ["Kiss and Tell is a 1945 American comedy film starring then 17-year-old
                Shirley Temple as Corliss Archer.", " In the film, two teenage girls cause their respective parents
                much concern when they start to become interested in boys.", " The parents" bickering about which girl
                is the worse influence causes more problems than it solves."]],
            ["Shirley Temple", ["Shirley Temple Black (April 23, 1928 \u2013 February 10, 2014) was an American actress,
                singer, dancer, businesswoman, and diplomat who was Hollywood"s number one box-office draw as a child
                actress from 1935 to 1938.", " As an adult, she was named United States ambassador to Ghana and to
                Czechoslovakia and also served as Chief of Protocol of the United States."]],
            ["Janet Waldo", ["Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice
                actress.", " She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope
                Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in
                \"Meet Corliss Archer\"."]]],
 "type": "bridge",
 "level": "hard"}

Example probing task result in Jiant format (JSON):

{"info":
    {"doc_id": "hotpot_sup_facts", "q_id": "5a8c7595554299585d9e36b6"},
     "text": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell ?
              Kiss and Tell ( 1945 film ) Kiss and Tell is a 1945 American comedy film starring then 17 - year - old
              Shirley Temple as Corliss Archer . In the film , two teenage girls cause their respective parents much
              concern when they start to become interested in boys . The parents " bickering about which girl is the
              worse influence causes more problems than it solves .
              Shirley Temple Shirley Temple Black ( April 23 , 1928 \u2013 February 10 , 2014 ) was an American actress
              , singer , dancer , businesswoman , and diplomat who was Hollywood " s number one box - office draw as a
              child actress from 1935 to 1938 . As an adult , she was named United States ambassador to Ghana and to
              Czechoslovakia and also served as Chief of Protocol of the United States .
              Janet Waldo Janet Marie Waldo ( February 4 , 1920 \u2013 June 12 , 2016 ) was an American radio and voice
              actress . She is best known in animation for voicing Judy Jetson , Nancy in \" Shazzan \", Penelope
              Pitstop , and Josie in \" Josie and the Pussycats \", and on radio as the title character in \" Meet
              Corliss Archer \".",
       "targets": [{"span1": [0, 19], "span2": [26, 48], "label": "1"},
                  {"span1": [0, 19], "span2": [48, 70], "label": "0"},
                  {"span1": [0, 19], "span2": [70, 88], "label": "0"},
                  {"span1": [0, 19], "span2": [90, 137], "label": "1"},
                  {"span1": [0, 19], "span2": [137, 164], "label": "1"},
                  [...]
                  {"span1": [0, 19], "span2": [291, 332], "label": "0"}]}

"""

import argparse
import json
from typing import List, Dict
from nltk.tokenize import word_tokenize
from task_processors_replicated import output_task_in_jiant_format

BOOL_TO_LABEL = {True: "1", False: "0"}

def hotpot_to_jiant(sample) -> Dict:
    """Transform a sample in Squad format to Jiant format (see the module docstring for details)."""
    text: List[str] = word_tokenize(sample["question"])
    context = sample["context"]

    # Transform the supporting facts into a dict.
    # The values are lists containing all sentence indices.
    supporting_facts = {}
    for key, value in sample["supporting_facts"]:
        if key in supporting_facts:
            supporting_facts[key].append(value)
        else:
            supporting_facts[key] = [value]

    sentence_end = len(text)
    span1 = [0, sentence_end]
    targets = []

    for key, sentences in context:
        indices = supporting_facts.get(key, [])
        for sentence_i, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            sentence_start, sentence_end = sentence_end, sentence_end + len(words)
            targets.append({"span1": span1,
                            "span2": [sentence_start, sentence_end],
                            "label": BOOL_TO_LABEL[sentence_i in indices]
                            })
            text.append('')
            text.extend(words)

    return {"info": {"doc_id": "hotpot_sup_facts", "q_id":  sample["_id"]},
            "text": " ".join(text),
            "targets": targets
            }

def convert_to_jiant_ep_format(in_path: str) -> List:
    """Transform all samples in a file to Jiant format. Return a list of all samples."""
    with open(in_path, "r") as f:
        samples = []
        data = json.load(f)
        print(f"converting {len(data)} samples to jiant format")
        for d in data:
            samples.append(hotpot_to_jiant(d))
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    parser.add_argument("-o", "--output_path", help="output path for jiant style file")
    args = parser.parse_args()

    samples = convert_to_jiant_ep_format(in_path=args.input_path)
    output_task_in_jiant_format(samples, out_dir=args.output_path)
