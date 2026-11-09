"""Run disjoint calibration, isolated service references and scheduler comparisons."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--card',default='4')
    parser.add_argument('--model',default='../.cache/models/Qwen3-4B')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)

    def run(name,workload,accounting,repeat,coefficients=None,isolated=False,replay=None):
        directory = args.output/name
        if args.resume and (directory/'requests.json').exists():
            saved = json.loads((directory/'requests.json').read_text())['config']
            expected = dict(model=args.model,workload=workload,accounting=accounting,
                            repeat=repeat,isolated=isolated,replay=bool(replay))
            if any(saved[key]!=value for key,value in expected.items()):
                raise ValueError(f'Completed run has different settings: {directory}')
            print(f'Complete: {name}',flush=True)
            return
        if directory.exists():
            directory.rename(args.output/f'{name}_incomplete_{time.time_ns()}')
        command = ['bash','../scripts/run_npu.sh',args.card,'bash','-c',
                   'export PYTHONPATH="$PWD/upstream/vllm-ascend:$PWD:$PYTHONPATH"; '
                   'exec env VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=4 .venv/bin/python experiment.py "$@"',
                   'experiment','--model',args.model,'--requests',str(replay or f'workloads/{workload}.json'),
                   '--workload',workload,'--accounting',accounting,'--repeat',str(repeat),
                   '--output',str(args.output/name)]
        if coefficients:
            command += ['--coefficients',str(coefficients)]
        if isolated:
            command += ['--isolated']
        if replay:
            command += ['--replay']
        print(name,flush=True)
        with (args.output/f'{name}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)

    for repeat in range(3):
        run(f'calibration_{repeat}','calibration','fcfs',repeat)
    coefficients = args.output/'costs.json'
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path('upstream/vllm-ascend').resolve())+os.pathsep+env.get('PYTHONPATH','')
    if not coefficients.exists():
        subprocess.run(['.venv/bin/python','fit_cost.py',
            *(str(args.output/f'calibration_{r}/service.jsonl') for r in range(3)),
            '--output',str(coefficients)],env=env,check=True)
    for workload in ('shared','mixed','burst','rejoin'):
        if args.resume and (args.output/f'{workload}_evaluation.json').exists():
            print(f'Complete workload: {workload}',flush=True)
            continue
        references,runs = [],[]
        for repeat in range(3):
            reference = f'{workload}_isolated_{repeat}'
            run(reference,workload,'fcfs',repeat,coefficients,True)
            references.append(str(args.output/reference))
            for accounting in ('fcfs','token','profiled','compute'):
                name = f'{workload}_{accounting}_{repeat}'
                run(name,workload,accounting,repeat,coefficients)
                runs.append(str(args.output/name))
        replay = args.output/f'{workload}_replay.json'
        subprocess.run(['.venv/bin/python','build_replay.py','--requests',f'workloads/{workload}.json',
                        '--runs',*runs,'--output',str(replay)],check=True)
        if json.loads(replay.read_text())['requests']:
            for repeat in range(3):
                name = f'{workload}_replay_{repeat}'
                run(name,workload,'fcfs',repeat,coefficients,True,replay)
                references.append(str(args.output/name))
        subprocess.run(['.venv/bin/python','evaluate.py','--reference',*references,
                        '--runs',*runs,'--output',str(args.output/f'{workload}_evaluation.json')],check=True)


if __name__ == '__main__':
    main()
