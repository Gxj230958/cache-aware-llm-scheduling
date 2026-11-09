"""Collect exact preemption trajectories for an isolated execution reference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--existing-replay',type=Path,nargs='*',default=[])
    args = parser.parse_args()
    catalog = json.loads(args.requests.read_text())
    originals = {row['request_id']:row for row in catalog['requests']}
    variants = {}
    for directory in args.runs:
        actions = defaultdict(list)
        initial_context = {}
        for line in (directory/'service.jsonl').read_text().splitlines():
            batch = json.loads(line)
            for work in batch['work']:
                logical = work['logical_request_id']
                if logical in originals:
                    initial_context.setdefault(logical,work['context_tokens'])
            for event in batch['preemptions']:
                logical = event['logical_request_id']
                if logical in originals:
                    actions[logical].append(event['computed_tokens'])
        for logical,points in actions.items():
            cached = initial_context[logical]
            variants[(logical,cached,tuple(points))] = originals[logical] | {
                'reference_preemptions':points,'reference_cached_tokens':cached,
                'original_request_id':logical,'arrival':0.}
    measured = set()
    for path in args.existing_replay:
        for row in json.loads(path.read_text())['requests']:
            measured.add((row['original_request_id'],row['reference_cached_tokens'],
                          tuple(row['reference_preemptions'])))
    rows = [row | {'request_id':f'replay-{index}-{row["request_id"]}'}
            for index,(key,row) in enumerate(variants.items()) if key not in measured]
    catalog['requests'] = rows
    catalog['reference_sources'] = [str(p) for p in args.runs]
    args.output.write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps({'trajectories':len(rows)}))


if __name__ == '__main__':
    main()
