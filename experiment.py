"""Controlled arrivals, measured batch costs and isolated service references."""

import argparse
import asyncio
import json
from pathlib import Path
import time

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def tokens(length, prefix, suffix):
    return [100+(i+prefix)%700 for i in range(length-32)] + [1000+(i+suffix)%700 for i in range(32)]


def workload(name):
    rows = []
    for user in range(4):
        for index in range(8):
            length = (256,1024,384,768)[user] if name != 'calibration' else (128,512,640,896)[user]
            shared = name in ('shared','burst','rejoin') and user%2 == 0
            prefix = 31*user if shared else 100*user+17*index
            arrival = .025*index
            if name == 'burst':
                arrival = 0. if index<4 else .5
            if name == 'rejoin' and user == 0:
                arrival = 0. if index<2 else 2.+.05*index
            rows.append({'request_id':f'u{user}-{index}','user':str(user),
                         'weight':1.,'arrival':arrival,
                         'prompt_token_ids':tokens(length,prefix,index*19),
                         'max_tokens':(16,32,24,48)[user],
                         'shared_prefix':shared})
    return sorted(rows,key=lambda r:r['arrival'])


async def execute(engine,rows,isolated):
    start = time.monotonic()
    results,submitted = {},{}

    async def request(row):
        if 'reference_cached_tokens' in row:
            await engine.reset_prefix_cache()
            cached = row['reference_cached_tokens']
            if cached:
                async for _ in engine.generate(
                        {'prompt_token_ids':row['prompt_token_ids'][:cached]},
                        SamplingParams(temperature=0,max_tokens=1,
                            extra_args={'fair_user':'warmup'}),'prefix-'+row['request_id']):
                    pass
        if not isolated:
            await asyncio.sleep(max(0.,row['arrival']-(time.monotonic()-start)))
        rid = row['request_id']
        submitted[rid] = row | {'logical_request_id':rid,
                               'submitted_s':time.monotonic()-start}
        result = {'request_id':rid,'first_token_s':None}
        async for output in engine.generate(
                {'prompt_token_ids':row['prompt_token_ids']},
                SamplingParams(temperature=0,max_tokens=row['max_tokens'],ignore_eos=True,
                    extra_args={'fair_user':row['user'],'fair_weight':row['weight'],
                        'logical_request_id':rid,
                        'reference_preemptions':row.get('reference_preemptions',[])}),rid):
            now = time.monotonic()-start
            if output.outputs and output.outputs[0].token_ids and result['first_token_s'] is None:
                result['first_token_s'] = now
            if output.finished:
                result.update(finished_s=now,output_tokens=len(output.outputs[0].token_ids))
        results[rid] = result

    if isolated:
        for row in rows:
            await request(row)
    else:
        await asyncio.gather(*(request(row) for row in rows))
    return {'elapsed_s':time.monotonic()-start,'started_at':start,
            'requests':[submitted[rid] | results[rid] for rid in submitted]}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--accounting',choices=['fcfs','token','profiled','compute'],default='compute')
    parser.add_argument('--workload',choices=['calibration','shared','mixed','burst','rejoin'],default='shared')
    parser.add_argument('--coefficients',type=Path)
    parser.add_argument('--requests',type=Path)
    parser.add_argument('--isolated',action='store_true')
    parser.add_argument('--replay',action='store_true')
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    calibration = json.loads(args.coefficients.read_text()) if args.coefficients else None
    coefficients = (calibration['coefficients'] if calibration else
                    {'launch_ms':1.,'token_ms':.01,'attention_pair_ms':.00001,'decode_token_ms':.1})
    profiled = (calibration['profiled_coefficients'] if calibration else
                {'prefill_ms':.01,'decode_ms':.11})
    log = args.output/'service.jsonl'
    if log.exists():
        raise FileExistsError(f'Use a fresh result directory: {args.output}')
    initialization_started = time.perf_counter()
    llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model=args.model,dtype='bfloat16',max_model_len=2048,
              max_num_seqs=8,max_num_batched_tokens=512,long_prefill_token_threshold=64,
              enable_prefix_caching=True,enable_chunked_prefill=True,async_scheduling=False,
              enforce_eager=True,gpu_memory_utilization=.45,disable_log_stats=True,
              compilation_config={'compile_sizes':list(range(1,513))},
              worker_extension_cls='warmup_worker.OutputProjectionWarmup',
              scheduler_cls=('reference_scheduler.ReferenceScheduler' if args.replay else
                             'vllm_ascend.core.compute_service_scheduler.ComputeServiceScheduler'),
              additional_config={'compute_service': {'coefficients':coefficients,
                  'profiled_coefficients':profiled,
                  'accounting':args.accounting,'log':str(log.resolve())}}))
    initialization_s = time.perf_counter()-initialization_started
    output_projection_warmup = await llm.collective_rpc('warmup_output_projection')
    rows = json.loads(args.requests.read_text())['requests'] if args.requests else workload(args.workload)
    warm_rows = [row | {'request_id':'warmup-'+row['request_id'],
                        'user':'warmup'+row['user']} for row in rows]
    warmup = await execute(llm,warm_rows,args.isolated)
    await llm.reset_prefix_cache()
    warm = [r for r in rows if r['shared_prefix']]
    # Cache the same public workload prefix for shared and isolated execution.
    if warm:
        per_user = {}
        for row in warm:
            per_user.setdefault(row['user'],row)
        for user,row in per_user.items():
            async for _ in llm.generate(
                    {'prompt_token_ids':row['prompt_token_ids'][:-32]},
                    SamplingParams(temperature=0,max_tokens=1,
                        extra_args={'fair_user':'warmup'}),'prime-'+user):
                pass
    measured = await execute(llm,rows,args.isolated)
    output = {'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              **measured,'warmup_elapsed_s':warmup['elapsed_s'],
              'initialization_including_query_warmup_s':initialization_s,
              'query_warmup_sizes':list(range(1,513)),
              'output_projection_warmup':output_projection_warmup,'coefficients':coefficients,
              'profiled_coefficients':profiled,
              'coefficient_source':str(args.coefficients) if args.coefficients else 'uncalibrated smoke values',
              }
    (args.output/'requests.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'elapsed_s':measured['elapsed_s'],'finished_requests':len(measured['requests'])}))
    llm.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
