"""Evaluate latency and service using independently measured isolated execution."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import statistics

import numpy as np


def shape(work):
    return (work['query_tokens'],work['context_tokens'],work['prompt_tokens'])


def evaluate_service(requests,batches,costs,started_at):
    users = sorted({r['user'] for r in requests})
    weights = {r['user']:r['weight'] for r in requests}
    service = dict.fromkeys(users,0.)
    intervals = {user:[] for user in users}
    for request in sorted(requests,key=lambda r:r['submitted_s']):
        spans = intervals[request['user']]
        start,end = request['submitted_s'],request['finished_s']
        if spans and start<=spans[-1][1]:
            spans[-1][1] = max(spans[-1][1],end)
        else:
            spans.append([start,end])
    pair_starts = {}
    maximum_gap = 0.
    missing = set()
    for batch in batches:
        now = batch['completed_at']-started_at
        active = {user:index for user,spans in intervals.items()
                  for index,(start,end) in enumerate(spans) if start<=now<=end}
        pairs = {(a,b,active[a],active[b]) for a,b in combinations(sorted(active),2)}
        pair_starts = {pair:baseline for pair,baseline in pair_starts.items() if pair in pairs}
        for pair in pairs:
            a,b = pair[:2]
            pair_starts.setdefault(pair,service[a]/weights[a]-service[b]/weights[b])
        for work in batch['work']:
            if work['user'] not in service:
                continue
            key = shape(work)
            if key not in costs:
                missing.add(key)
            else:
                service[work['user']] += costs[key]
        for (a,b,_,_),baseline in pair_starts.items():
            gap = abs(service[a]/weights[a]-service[b]/weights[b]-baseline)
            maximum_gap = max(maximum_gap,gap)
    if missing:
        raise ValueError(f'Isolated reference lacks executed shapes: {sorted(missing)}')
    return service,maximum_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,nargs='+',required=True)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    costs,latencies = defaultdict(list),defaultdict(list)
    for directory in args.reference:
        requests = json.loads((directory/'requests.json').read_text())
        if not requests['config']['isolated']:
            raise ValueError('Service reference must come from isolated execution')
        if not requests['config'].get('replay',False):
            for request in requests['requests']:
                latencies[request['logical_request_id']].append(request['finished_s']-request['submitted_s'])
        for line in (directory/'service.jsonl').read_text().splitlines():
            batch = json.loads(line)
            if len(batch['work'])==1 and not batch['work'][0]['user'].startswith('warmup'):
                costs[shape(batch['work'][0])].append(batch['execution_ms'])
    costs = {key:statistics.median(values) for key,values in costs.items()}
    latencies = {key:statistics.median(values) for key,values in latencies.items()}
    results = []
    for directory in args.runs:
        data = json.loads((directory/'requests.json').read_text())
        requests = data['requests']
        users = sorted({r['user'] for r in requests})
        batches = [json.loads(line) for line in (directory/'service.jsonl').read_text().splitlines()]
        service,maximum_gap = evaluate_service(requests,batches,costs,data['started_at'])
        ttft = [r['first_token_s']-r['submitted_s'] for r in requests]
        slowdowns = {user:statistics.mean(
            (r['finished_s']-r['submitted_s'])/latencies[r['logical_request_id']]
            for r in requests if r['user']==user) for user in users}
        results.append({'directory':str(directory),'accounting':data['config']['accounting'],
                        'elapsed_s':data['elapsed_s'],
                        'output_tokens_per_s':sum(r['output_tokens'] for r in requests)/data['elapsed_s'],
                        'p95_ttft_s':float(np.percentile(ttft,95)),
                        'isolated_reference_service_ms':service,
                        'max_backlogged_service_gap_ms':maximum_gap,
                        'mean_slowdown_by_user':slowdowns})
    output = {'reference_sources':[str(p) for p in args.reference],
              'reference_shapes':len(costs),'runs':results,
              'service_definition':'Median isolated batch execution time for exactly matching query, context and prompt lengths; normalized by user weight within each pair\'s continuously backlogged interval, sampled at batch completion.'}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
