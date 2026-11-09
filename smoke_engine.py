"""Exercise real vLLM cache hits and chunked prefills with per-user metadata."""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dummy-weights', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ledger = args.output/'service.jsonl'
    llm = LLM(model=args.model, load_format='dummy' if args.dummy_weights else 'auto',
              dtype='bfloat16', max_model_len=512, max_num_seqs=4,
              max_num_batched_tokens=128, long_prefill_token_threshold=64,
              enable_prefix_caching=True, enable_chunked_prefill=True,
              async_scheduling=False, enforce_eager=True, gpu_memory_utilization=0.35,
              scheduler_cls='vllm_ascend.core.compute_service_scheduler.ComputeServiceScheduler',
              additional_config={'compute_service': {
                  'coefficients': {'launch_ms': 1., 'token_ms': .01,
                                   'attention_pair_ms': .00001, 'decode_token_ms': .1},
                  'accounting': 'compute', 'log': str(ledger.resolve())}})
    shared = {'prompt_token_ids': [100+i%400 for i in range(256)]}
    llm.generate([shared], SamplingParams(temperature=0, max_tokens=4,
                 extra_args={'fair_user':'warmup'}), use_tqdm=False)
    prompts = [shared, {'prompt_token_ids':[800+i%400 for i in range(192)]}]
    params = [SamplingParams(temperature=0, max_tokens=8, ignore_eos=True,
                            extra_args={'fair_user':user, 'fair_weight':1.})
              for user in ['cached','cold']]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    work = [w for event in events for w in event['work'] if w['user'] in ('cached','cold')]
    cache_work = [w for w in work if w['user']=='cached']
    cold_work = [w for w in work if w['user']=='cold']
    assert max(w['cached_tokens'] for w in cache_work) > 0
    assert max(w['cached_tokens'] for w in cold_work) == 0
    assert sum(w['query_tokens'] for w in cache_work) < sum(w['query_tokens'] for w in cold_work)
    assert all(w['query_tokens'] <= 64 for w in work)
    assert len(outputs) == 2 and all(len(o.outputs[0].token_ids)==8 for o in outputs)
    result = {'model':args.model, 'dummy_weights':args.dummy_weights,
              'coefficient_source':'fixed smoke values; not calibrated',
              'cache_tokens':max(w['cached_tokens'] for w in cache_work),
              'executed_tokens':{u:sum(w['query_tokens'] for w in work if w['user']==u)
                                 for u in ('cached','cold')},
              'output_token_ids':[o.outputs[0].token_ids for o in outputs],
              'passed':True}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
