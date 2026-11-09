"""Aggregate independent scheduler runs and export latency/fairness figures."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


POLICIES = ('fcfs','token','profiled','compute')
LABELS = ('FCFS','Token VTC','Profiled VTC','Executed work')
METRICS = {
    'output_tokens_per_s':'Output tokens / second',
    'p95_ttft_s':'P95 first-token latency (s)',
    'max_backlogged_service_gap_ms':'Maximum pair service gap (ms)',
    'worst_user_slowdown':'Worst mean user slowdown',
}


def execution_statistics(directory):
    data = json.loads((directory/'requests.json').read_text())
    requests = data['requests']
    coefficients = data['coefficients']
    execution_times, prediction_errors = [], []
    request_ids = {row['logical_request_id'] for row in requests}
    progress, initial_context = {}, {}
    executed = recomputed = preemptions = batches = members = 0
    for line in (directory/'service.jsonl').read_text().splitlines():
        batch = json.loads(line)
        work = [w for w in batch['work'] if w['logical_request_id'] in request_ids]
        if not work:
            continue
        batches += 1
        members += len(work)
        execution_times.append(batch['execution_ms'])
        prediction = coefficients['launch_ms']
        for item in work:
            query, context = item['query_tokens'], item['context_tokens']
            decode = query-min(query,max(0,item['prompt_tokens']-context))
            prediction += (coefficients['token_ms']*query
                           + coefficients['attention_pair_ms']*(query*context+query*(query+1)/2)
                           + coefficients['decode_token_ms']*decode)
        prediction_errors.append(abs(prediction-batch['execution_ms'])/batch['execution_ms'])
        preemptions += sum(p['logical_request_id'] in request_ids for p in batch['preemptions'])
        for item in work:
            rid = item['logical_request_id']
            context, query = item['context_tokens'], item['query_tokens']
            initial_context.setdefault(rid, context)
            previous = progress.get(rid, context)
            end = context+query
            recomputed += max(0, min(end, previous)-context)
            executed += query
            progress[rid] = max(previous, end)
    return {
        'execution_median_ms':float(np.median(execution_times)),
        'execution_p99_ms':float(np.quantile(execution_times,.99)),
        'execution_max_ms':max(execution_times),
        'cost_median_absolute_relative_error':float(np.median(prediction_errors)),
        'cost_p95_absolute_relative_error':float(np.quantile(prediction_errors,.95)),
        'executed_query_tokens':executed,
        'recomputed_query_tokens':recomputed,
        'initial_cached_tokens':sum(initial_context.values()),
        'preemptions':preemptions,
        'executed_batches':batches,
        'mean_batch_requests':members/batches,
        'max_submission_lateness_s':max(r['submitted_s']-r['arrival'] for r in requests),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    rows, user_rows = [], []
    for workload in ('shared','mixed','burst','rejoin'):
        source = args.directory/f'{workload}_evaluation.json'
        if not source.exists():
            continue
        runs = json.loads(source.read_text())['runs']
        for policy in POLICIES:
            selected = [r for r in runs if r['accounting']==policy]
            if len(selected)!=3:
                raise ValueError(f'{workload}/{policy} requires three independent runs')
            for run in selected:
                run['worst_user_slowdown'] = max(run['mean_slowdown_by_user'].values())
                run.update(execution_statistics(Path(run['directory'])))
            row = {'workload':workload,'policy':policy,'repetitions':len(selected)}
            for metric in (*METRICS, 'executed_query_tokens','recomputed_query_tokens',
                           'initial_cached_tokens','preemptions','executed_batches',
                           'mean_batch_requests','max_submission_lateness_s',
                           'execution_median_ms','execution_p99_ms','execution_max_ms',
                           'cost_median_absolute_relative_error','cost_p95_absolute_relative_error'):
                values = [run[metric] for run in selected]
                for suffix,value in (('median',statistics.median(values)),
                                     ('mean',statistics.mean(values)),
                                     ('min',min(values)),('max',max(values))):
                    row[f'{metric}_{suffix}'] = value
            rows.append(row)
            for user in selected[0]['mean_slowdown_by_user']:
                item = {'workload':workload,'policy':policy,'user':user}
                for metric in ('mean_slowdown_by_user','isolated_reference_service_ms'):
                    values = [run[metric][user] for run in selected]
                    for suffix,value in (('median',statistics.median(values)),
                                         ('min',min(values)),('max',max(values))):
                        item[f'{metric}_{suffix}'] = value
                user_rows.append(item)
    if not rows:
        raise ValueError('No completed workload evaluation was found')
    output = args.directory/'summary'
    output.mkdir(exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (output/'metrics.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'per_user.json').write_text(json.dumps(user_rows,indent=2)+'\n')
    with (output/'per_user.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(user_rows[0]))
        writer.writeheader()
        writer.writerows(user_rows)
    workloads = list(dict.fromkeys(row['workload'] for row in rows))
    figure,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x = np.arange(len(workloads))
    width = .19
    for axis,(metric,title) in zip(axes.flat,METRICS.items()):
        for index,(policy,label) in enumerate(zip(POLICIES,LABELS)):
            selected = [next(row for row in rows if row['workload']==name and row['policy']==policy)
                        for name in workloads]
            median = np.array([row[f'{metric}_median'] for row in selected])
            low = median-np.array([row[f'{metric}_min'] for row in selected])
            high = np.array([row[f'{metric}_max'] for row in selected])-median
            axis.bar(x+(index-1.5)*width,median,width,label=label,
                     yerr=np.stack([low,high]),capsize=2)
        axis.set_xticks(x,workloads)
        axis.set_ylabel(title)
        axis.grid(axis='y',alpha=.2)
        axis.set_axisbelow(True)
    axes[0,0].legend(fontsize=9)
    figure.suptitle('Qwen3-4B: median and range across three independent runs')
    for suffix in ('pdf','png'):
        figure.savefig(output/f'scheduler_comparison.{suffix}',dpi=180)
    plt.close(figure)
    figure,axes = plt.subplots(1,len(workloads),figsize=(4*len(workloads),3.5),
                               squeeze=False,layout='constrained')
    for axis,workload in zip(axes.flat,workloads):
        users = sorted({r['user'] for r in user_rows if r['workload']==workload})
        x = np.arange(len(users))
        for index,(policy,label) in enumerate(zip(POLICIES,LABELS)):
            selected = [next(r for r in user_rows if r['workload']==workload and
                             r['policy']==policy and r['user']==user) for user in users]
            values = np.array([r['mean_slowdown_by_user_median'] for r in selected])
            low = values-np.array([r['mean_slowdown_by_user_min'] for r in selected])
            high = np.array([r['mean_slowdown_by_user_max'] for r in selected])-values
            axis.bar(x+(index-1.5)*width,values,width,label=label,
                     yerr=np.stack([low,high]),capsize=2)
        axis.set(xticks=x,xticklabels=users,xlabel='User',ylabel='Mean request slowdown',title=workload)
        axis.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8)
    for suffix in ('pdf','png'):
        figure.savefig(output/f'slowdown_by_user.{suffix}',dpi=180)
    plt.close(figure)
    print(output)


if __name__ == '__main__':
    main()
