"""Verify the completed study's workload, timing and warmup protocol."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    coefficients = json.loads((args.directory/'costs.json').read_text())
    calibration = json.loads(Path('workloads/calibration.json').read_text())
    calibration_titles = {r['document_title'] for r in calibration['requests']}
    summary = {'policy_runs':0, 'completed_policy_requests':0,
               'completed_output_tokens':0, 'workloads':{}}
    for workload in ('shared', 'mixed', 'burst', 'rejoin'):
        source = json.loads(Path(f'workloads/{workload}.json').read_text())
        originals = {r['request_id']:r for r in source['requests']}
        assert not calibration_titles.intersection(r['document_title'] for r in originals.values())
        evaluation = json.loads((args.directory/f'{workload}_evaluation.json').read_text())
        assert len(evaluation['runs']) == 12
        rows = []
        for policy in ('fcfs', 'token', 'profiled', 'compute'):
            for repeat in range(3):
                name = f'{workload}_{policy}_{repeat}'
                data = json.loads((args.directory/name/'requests.json').read_text())
                assert data['config']['accounting'] == policy
                assert data['config']['repeat'] == repeat
                assert data['query_warmup_sizes'] == list(range(1, 513))
                assert [r['requests'] for r in data['output_projection_warmup'][0]] == list(range(1, 9))
                assert data['coefficients'] == coefficients['coefficients']
                assert data['profiled_coefficients'] == coefficients['profiled_coefficients']
                assert len(data['requests']) == len(originals)
                assert {r['request_id'] for r in data['requests']} == set(originals)
                charges = defaultdict(float)
                if policy == 'token':
                    for line in (args.directory/name/'service.jsonl').open():
                        for work in json.loads(line)['work']:
                            charges[work['logical_request_id']] += work['charged_service']
                for request in data['requests']:
                    expected = originals[request['request_id']]
                    assert all(request[key] == value for key, value in expected.items())
                    assert request['output_tokens'] == expected['max_tokens']
                    if policy == 'token':
                        assert charges[request['request_id']] == len(request['prompt_token_ids'])+2*request['output_tokens']
                    assert 0 <= request['submitted_s'] <= request['first_token_s'] <= request['finished_s'] <= data['elapsed_s']
                record = {'run':name, 'initialization_s':data['initialization_including_query_warmup_s'],
                          'warmup_s':data['warmup_elapsed_s'], 'measured_s':data['elapsed_s']}
                if workload == 'rejoin':
                    early = [r for r in data['requests'] if r['user']=='0' and r['arrival']<2]
                    late = [r for r in data['requests'] if r['user']=='0' and r['arrival']>=2]
                    record['user_zero_idle_s'] = min(r['submitted_s'] for r in late)-max(r['finished_s'] for r in early)
                    assert record['user_zero_idle_s'] > 0, f'{name} does not exercise an idle-user return'
                rows.append(record)
                summary['policy_runs'] += 1
                summary['completed_policy_requests'] += len(data['requests'])
                summary['completed_output_tokens'] += sum(r['output_tokens'] for r in data['requests'])
        summary['workloads'][workload] = {
            'independent_reference_shapes':evaluation['reference_shapes'],
            'reference_sources':evaluation['reference_sources'], 'runs':rows}
    assert summary['policy_runs'] == 48
    assert summary['completed_policy_requests'] == 1536
    assert summary['completed_output_tokens'] == 46080
    output = args.directory/'protocol_validation.json'
    output.write_text(json.dumps(summary, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
