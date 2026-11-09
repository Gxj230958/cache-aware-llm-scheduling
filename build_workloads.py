"""Build disjoint calibration and evaluation requests from public SQuAD documents."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    articles = json.loads(args.data.read_text())['data']
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    args.output.mkdir(parents=True,exist_ok=True)
    for name in ('calibration','shared','mixed','burst','rejoin'):
        partition = articles[:16] if name=='calibration' else articles[16:]
        rows = []
        for user in range(4):
            for index in range(8):
                shared = name in ('shared','burst','rejoin') and user%2==0
                article = partition[(user if shared else user*8+index)%len(partition)]
                context = '\n\n'.join(paragraph['context'] for paragraph in article['paragraphs'])
                questions = [qa['question'] for paragraph in article['paragraphs'] for qa in paragraph['qas']]
                marker = f'Document for user {user}:\n' if shared else f'Document {user}-{index}:\n'
                target = (128,512,640,896)[user] if name=='calibration' else (256,1024,384,768)[user]
                prefix = tokenizer.encode(marker+context,add_special_tokens=False)[:target-32]
                suffix = tokenizer.encode('\nQuestion: '+questions[index%len(questions)]+'\nAnswer:',
                                           add_special_tokens=False)[-32:]
                arrival = .025*index
                if name=='burst':
                    arrival = 0. if index<4 else .5
                if name=='rejoin' and user==0:
                    arrival = 0. if index<2 else 2.+.05*index
                rows.append({'request_id':f'u{user}-{index}','user':str(user),'weight':1.,
                             'arrival':arrival,'prompt_token_ids':prefix+suffix,
                             'max_tokens':(16,32,24,48)[user],
                             'shared_prefix':shared,'document_title':article['title']})
        result = {'source':'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
                  'source_title':'SQuAD 2.0 development set',
                  'article_partition':'first 16 articles' if name=='calibration' else 'remaining articles',
                  'model_tokenizer':args.model,'workload':name,
                  'requests':sorted(rows,key=lambda r:r['arrival'])}
        (args.output/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,len(rows),min(len(r['prompt_token_ids']) for r in rows),
              max(len(r['prompt_token_ids']) for r in rows))


if __name__ == '__main__':
    main()
