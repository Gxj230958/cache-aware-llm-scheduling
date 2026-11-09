"""Fit execution costs on separate measured calibration batches."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from vllm_ascend.core.compute_service import Work


def features(batch):
    work = [Work(**{k:v for k,v in row.items() if k in Work.__dataclass_fields__})
            for row in batch['work']]
    return [1.,sum(w.query_tokens for w in work),sum(w.attention_pairs for w in work),
            sum(w.decode_tokens for w in work)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    batches = [json.loads(line) for path in args.logs for line in path.read_text().splitlines()]
    batches = [b for b in batches if b['work'] and
               all(not w['user'].startswith('warmup') for w in b['work'])]
    matrix = np.asarray([features(b) for b in batches],dtype=float)
    target = np.asarray([b['execution_ms'] for b in batches])
    scale = np.maximum(np.linalg.norm(matrix,axis=0),1.)
    coefficients = nnls(matrix/scale,target)[0]/scale
    prediction = matrix@coefficients
    token_matrix = np.column_stack((matrix[:,1]-matrix[:,3],matrix[:,3]))
    token_scale = np.maximum(np.linalg.norm(token_matrix,axis=0),1.)
    token_coefficients = nnls(token_matrix/token_scale,target)[0]/token_scale
    residual_groups = {}
    grouped = {}
    for index, batch in enumerate(batches):
        decode = matrix[index,3]
        phase = 'decode' if decode == matrix[index,1] else ('prefill' if decode == 0 else 'mixed')
        for key in (phase, f"{phase}/requests={len(batch['work'])}"):
            grouped.setdefault(key,[]).append(index)
    for key, indices in sorted(grouped.items()):
        observed = target[indices]
        relative = abs(prediction[indices]-observed)/observed
        residual_groups[key] = {
            'batches':len(indices), 'observed_median_ms':float(np.median(observed)),
            'observed_p99_ms':float(np.quantile(observed,.99)),
            'observed_max_ms':float(np.max(observed)),
            'median_absolute_relative_error':float(np.median(relative)),
            'p95_absolute_relative_error':float(np.quantile(relative,.95))}
    names = ['launch_ms','token_ms','attention_pair_ms','decode_token_ms']
    result = {'coefficients':dict(zip(names,coefficients.tolist())),
              'profiled_coefficients':dict(zip(['prefill_ms','decode_ms'],token_coefficients.tolist())),
              'calibration_sources':[str(p) for p in args.logs],'batches':len(batches),
              'calibration_median_absolute_relative_error':float(np.median(abs(prediction-target)/target)),
              'profiled_median_absolute_relative_error':float(np.median(abs(token_matrix@token_coefficients-target)/target)),
              'residuals_by_phase_and_batch_size':residual_groups,
              'measurement':'Synchronous scheduler dispatch to model-output return, milliseconds'}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
