
"""
Module to transform the HotpotQA Dataset into a Jiant Probing Task. 

Example question in HotpotQA format (JSON):

{"_id": "5a8c7595554299585d9e36b6",
 "answer": "Chief of Protocol",
 "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
 "supporting_facts": [["Kiss and Tell (1945 film)", 0], ["Shirley Temple", 0], ["Shirley Temple", 1]],
 "context": [["Kiss and Tell (1945 film)", ["Kiss and Tell is a 1945 American comedy film starring then 17-year-old
                Shirley Temple as Corliss Archer.", " In the film, two teenage girls cause their respective parents
                much concern when they start to become interested in boys.", " The parents' bickering about which girl
                is the worse influence causes more problems than it solves."]],
            ["Shirley Temple", ["Shirley Temple Black (April 23, 1928 \u2013 February 10, 2014) was an American actress,
                singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child
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
              concern when they start to become interested in boys . The parents ' bickering about which girl is the
              worse influence causes more problems than it solves .
              Shirley Temple Shirley Temple Black ( April 23 , 1928 \u2013 February 10 , 2014 ) was an American actress
              , singer , dancer , businesswoman , and diplomat who was Hollywood ' s number one box - office draw as a
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
import random
from math import floor
from nltk.tokenize import word_tokenize

def hotpot_to_jiant(task) -> str:
  print(task)
  id: str = task['_id']
  question: list[str] = word_tokenize(task['question'])
  context = task['context']
  # transform the supporting facts into a dict. the value is a list containing all sentence indices
  supporting_facts = {}
  for key, value in task['supporting_facts']:
    if key in supporting_facts:
      supporting_facts[key].append(value)
    else:
      supporting_facts[key] = [value]
    
  sentence_end = len(question)
  span1 = [0, sentence_end]

  text = ' '.join(question)
  targets = []

  for key, sentences in context:
    indices = supporting_facts.get(key, [])
    for sentence_i, sentence in enumerate(sentences):
      words = word_tokenize(sentence)
      sentence_start, sentence_end = sentence_end, sentence_end + len(words)
      targets.append({'span1': span1, 'span2': [sentence_start, sentence_end], 'label': sentence_i in indices})
      text += ' ' + ' '.join(words)
  ret = {
        'info': {'doc_id': 'hotpot_sup_facts', 'q_id': id},
        'text': text,
        'targets': targets
  }
  return ret

def convert_dataset(in_path: str, out_path: str, train_split: float, dev_split: float):
  with open(in_path, 'r') as f:
    tasks = []
    data = json.load(f)
    l = len(data)
    print(f'converting {l} instances to jiant format')
    for i, d in enumerate(data):
      tasks.extend(hotpot_to_jiant(d))
      if i % (l//20) == 0:
        print('.', end='')
    print('\n')
  random.shuffle(tasks)

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

test = {"_id": "5a8c7595554299585d9e36b6", "answer": "Chief of Protocol", "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", "supporting_facts": [["Kiss and Tell (1945 film)", 0], ["Shirley Temple", 0], ["Shirley Temple", 1]], "context": [["Kiss and Tell (1945 film)", ["Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer.", " In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys.", " The parents' bickering about which girl is the worse influence causes more problems than it solves."]], ["Shirley Temple", ["Shirley Temple Black (April 23, 1928 \u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938.", " As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States."]], ["Janet Waldo", ["Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress.", " She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."]]], "type": "bridge", "level": "hard"}

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('in_path', help='filepath of the hotpot dataset')
  parser.add_argument('out_path', help='output filepath')
  parser.add_argument('-ts', '--testsplit', default=0.8, type=float, help='train split: a number between 0 and 1, default=0.8')
  parser.add_argument('-ds', '--devsplit', default=0.1, type=float, help='dev split: a number between 0 and 1, default=0.1')
  # test:
  args = parser.parse_args(['explain-BERT-QA-master\datasets\hotpotqa\hotpot_test_fullwiki_v1.json','explain-BERT-QA-master\datasets\hotpotqa\ '])
  #args = parser.parse_args()
  print(args)
  if args.testsplit + args.devsplit > 1 or args.testsplit + args.devsplit < 0:
    print('Error: sum of train split and dev split is not between 0 and 1')
  elif args.testsplit > 1 or args.testsplit < 0 or args.devsplit > 1 or args.devsplit < 0:
    print('Error: train and dev split are not between 0 and 1')
  else:
    #convert_dataset(args.in_path, args.out_path, args.testsplit, args.devsplit)
    ret = hotpot_to_jiant(test)
    print(ret)
  

