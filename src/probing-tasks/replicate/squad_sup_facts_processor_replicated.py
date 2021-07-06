"""
Module to transform the SQuAD Dataset into a Jiant Probing Task.

Example question in SQuAD format (JSON):

{"title": "University_of_Notre_Dame",
 "paragraphs":
  [{"context": "Architecturally, the school has a Catholic character.
                Immediately behind the basilica
                is the Grotto, a Marian place of prayer and reflection.",
    "qas":
    [{"answers":
      [{"answer_start": 101,
        "text": "a Marian place of prayer and reflection"}],
      "question": "What is the Grotto at Notre Dame?",
      "id": "5733be284776f41900661181"},
     {"answers":
      [{"answer_start": 32,
        "text": "a Catholic character"}],
        "question": "What Character does the school have?",
        "id": "5733be28477123d900661181"}]}]}

Example probing task result in Jiant format (JSON):

{"info": {"doc_id": "squad_sup_facts", "q_id": "5726e985dd62a815002e94db"},
 "text": "What is the Grotto at Notre Dame ?  Architecturally , the school has a Catholic
          character . Immediately behind the basilica is the Grotto , a Marian
          place of prayer and reflection .",
 "targets":
    [{"span1": [0, 8], "span2": [8, 17], "label": "0"},
     {"span1": [0, 8], "span2": [17, 34], "label": "1"}]}

"""

import argparse
import json
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize, word_tokenize
from task_processors_replicated import output_task_in_jiant_format

BOOL_TO_LABEL = {True: "1", False: "0"}

def squad_to_jiant(sample) -> Dict:
    """Transform a sample in Squad format to Jiant format (see the module docstring for details)."""
    ret = []
    for paragraph in sample["paragraphs"]:
        context = paragraph["context"]
        for qas in paragraph["qas"]:
            answer = qas["answers"][0]
            text: List[str] = word_tokenize(qas["question"])
            sentences: List[List[str]] = sent_tokenize(context)
            answer_i = find_sentence_from_index(answer["answer_start"], sentences)

            # Mark the end of the previous sentence.
            sentence_end: int = len(text)
            span1: List[int] = [0, sentence_end]
            targets = []

            for sentence_i, sentence in enumerate(sentences):
                words = word_tokenize(sentence)
                sentence_start, sentence_end = sentence_end, sentence_end + len(words)
                targets.append({"span1": span1,
                                "span2": [sentence_start, sentence_end],
                                "label": BOOL_TO_LABEL[sentence_i == answer_i]
                                })
                # Add a whitespace at the beginning of each sentence.
                text.append('')
                text.extend(words)

            ret.append({
                "info": {"doc_id": "squad_sup_facts", "q_id": qas["id"]},
                "text": " ".join(text),
                "targets": targets
                })

    return ret


def find_sentence_from_index(char_ind: int, sentences: List) -> Tuple[int, int]:
    """Return the index of the sentence which contains the given char index."""
    c = 0
    for sentence_i, sentence in enumerate(sentences):
        # Add an additional 1 to accomodate for a whitespace after the  end of each sentence.
        c += len(sentence) + 1
        if c >= char_ind:
            return sentence_i
    print(f"Error: The given char_index {char_ind} exceeds the text length")
    return None

def convert_to_jiant_ep_format(in_path: str) -> List:
    """Transform all samples in a file to Jiant format. Return a list of all samples."""
    with open(in_path, "r") as f:
        samples = []
        data = json.load(f)
        print(f"converting {len(data['data'])} samples to jiant format")
        for d in data["data"]:
            samples.extend(squad_to_jiant(d))
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    parser.add_argument("-o", "--output_path", help="output path for jiant style file")
    args = parser.parse_args()

    samples = convert_to_jiant_ep_format(in_path=args.input_path)
    output_task_in_jiant_format(samples, out_dir=args.output_path)
