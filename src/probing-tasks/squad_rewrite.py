"""
Module to transform the SQuAD Dataset into a Jiant Probing Task. 

Example question in SQuAD format (JSON):

{"title": "University_of_Notre_Dame",
 "paragraphs":
  [{"context": "Architecturally, the school has a Catholic character. Immediately behind the basilica
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
 "text": "What is the Grotto at Notre Dame ?  Architecturally , the school has a Catholic character . Immediately behind
          the basilica is the Grotto , a Marian place of prayer and reflection .",
 "targets":
    [{"span1": [0, 8], "span2": [8, 17], "label": "0"},
     {"span1": [0, 8], "span2": [17, 34], "label": "1"}]}

"""

import argparse
import json
import random
from math import floor
from nltk.tokenize import sent_tokenize, word_tokenize

def squad_to_jiant(task, slow: bool=False) -> str:
  ret = []
  paragraphs = task['paragraphs']
  for p in paragraphs:
    context = p['context']
    for qas in p['qas']:
      answer = qas['answers'][0]
      question: list[str] = word_tokenize(qas['question'])
      id: str = qas['id']

      sentences: list[list[str]] = sent_tokenize(context)
      if slow:
        answer_i = find_sentence_from_index_slow(answer['answer_start'], sentences, context)
      else:
        answer_i = find_sentence_from_index(answer['answer_start'], sentences)

      # mark the end of the previous sentence
      sentence_end: int = len(question)
      span1: list[int] = [0, sentence_end]

      text = ' '.join(question)
      targets = []

      for sentence_i, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        sentence_start, sentence_end = sentence_end, sentence_end + len(words)
        targets.append({'span1': span1, 'span2': [sentence_start, sentence_end], 'label': sentence_i == answer_i})

        text += ' ' + ' '.join(words)

      ret.append({
                  'info': {'doc_id': 'squad_sup_facts', 'q_id': id},
                  'text': text,
                  'targets': targets
      })
  return ret

# The following function isn't working yet
def find_sentence_from_index_slow(char_ind: int, sentences: list, text: str) -> tuple[int, int]:
  c: int = 0
  for sentence_i, sentence in enumerate(sentences):
    i: int = 0
    print(sentence)
    while i <= len(text):
      try:
        text_start: str = text[i:i+len(sentence)]
        print(text_start)
      except IndexError as e:
        print(f'Error: The given list of sentences didn\'t match the given text. {sentence} doesn\' appear in the text.')
        break
      if text_start == sentence:
          c += i + len(sentence)
          text = text[i+len(sentence):]
          break
      i += 1
    if c >= char_ind:
      return sentence_i
    # 
    print(f'Error: The given char_index {char_ind} exceeds the text length')
    return None

def find_sentence_from_index(char_ind: int, sentences: list) -> tuple[int, int]:
  c = 0
  for sentence_i, sentence in enumerate(sentences):
    # add an additional 1 to accomodate for a whitespace after the  end of a sentence
    c += len(sentence) + 1
    if c >= char_ind:
      return sentence_i
  print(f'Error: The given char_index {char_ind} exceeds the text length')
  return None

def convert_dataset(in_path: str, out_path: str, train_split: float, dev_split: float, slow: bool, small: bool):
  with open(in_path, 'r') as f:
    tasks = []
    data = json.load(f)
    l: int = len(data['data'])
    print(f'converting {l} instances to jiant format')
    for i, d in enumerate(data['data']):
      tasks.extend(squad_to_jiant(d, slow))
      if i % (l//20) == 0:
        print('.', end='')
    print('\n')

  random.shuffle(tasks)
  
  if small:
    tasks = tasks[:50]

  dev_start: int = floor(len(tasks) * train_split)
  test_start: int = dev_start + floor(len(tasks) * dev_split)

  print(f'writing {dev_start} instances to train.json')
  with open(out_path + 'train.json', 'w') as f:
    for t in tasks[:dev_start]:
      f.write(json.dumps(t) + '\n')

  print(f'writing {test_start - dev_start} instances to dev.json')
  with open(out_path + 'dev.json', 'w') as f:
    for t in tasks[dev_start:test_start]:
      f.write(json.dumps(t) + '\n')

  print(f'writing {len(tasks) - test_start} instances to test.json')
  with open(out_path + 'test.json', 'w') as f:
    for t in tasks[test_start:]:
      f.write(json.dumps(t) + '\n')

test = {"title": "University_of_Notre_Dame", "paragraphs": [{"context": "Architecturally, the school has a Catholic character. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.", "qas": [{"answers": [{"answer_start": 101, "text": "a Marian place of prayer and reflection"}], "question": "What is the Grotto at Notre Dame?", "id": "5733be284776f41900661181"}, {"answers": [{"answer_start": 32, "text": "a Catholic character"}], "question": "What Character does the school have?", "id": "5733be28477123d900661181"}]}]}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('in_path', help='filepath of the squad dataset')
  parser.add_argument('out_path', help='output filepath')
  parser.add_argument('-ts', '--trainsplit', default=0.8, type=float, help='train split: a number between 0 and 1, default=0.8')
  parser.add_argument('-ds', '--devsplit', default=0.1, type=float, help='dev split: a number between 0 and 1, default=0.1')
  parser.add_argument('-s', '--slow', action='store_true', help='use the slower but more reliable function to parse target labels')
  parser.add_argument('--small', action='store_true', help='create a small dataset of 50 instances for testing')
  args = parser.parse_args(['.\datasets\squad1.1\dev-v1.1.json','.\datasets\squad1.1\small\ ', '--small'])
  #args = parser.parse_args()
  if args.trainsplit + args.devsplit > 1 or args.trainsplit + args.devsplit < 0:
    print('Error: sum of train split and dev split is not between 0 and 1')
  elif args.trainsplit > 1 or args.trainsplit < 0 or args.devsplit > 1 or args.devsplit < 0:
    print('Error: train and dev split are not between 0 and 1')
  else:
    convert_dataset(args.in_path, args.out_path, args.trainsplit, args.devsplit, args.slow, args.small)
    #print(squad_to_jiant(test))