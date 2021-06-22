import json
import os

with open('datasets/squad1.1/dev-v1.1.json', 'r') as f:
    a = json.load(f)
    a = a['data']
    if not os.path.isdir('qa_samples'):
        os.mkdir('qa_samples')
    for i in range(10):
        path = 'qa_samples/sample' + str(i) + '.json'
        sample = a[i]['paragraphs'][0]
        context: str = sample['context']
        qas =  sample['qas'][0]
        question: str = qas['question']
        answer: str = qas['answers'][0]['text']
        result = {'question': question, 'context': context, 'answer': answer}
        with open(path, 'w') as g:
            json.dump(result, g)
