"""Replay observed preemption points during independent single-request execution."""

import time

from vllm_ascend.core.compute_service_scheduler import ComputeServiceScheduler


class ReferenceScheduler(ComputeServiceScheduler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.positions = {}

    def schedule(self):
        preempted = []
        for request in list(self.running):
            metadata = request.sampling_params.extra_args
            actions = metadata.get('reference_preemptions',[])
            position = self.positions.get(request.request_id,0)
            if position == len(actions):
                continue
            target = actions[position]
            if request.num_computed_tokens > target:
                raise ValueError(f'Reference passed preemption point {target}')
            if request.num_computed_tokens == target:
                self.running.remove(request)
                self._preempt_request(request,time.monotonic())
                # Full token/block metadata replaces the worker's cached entry.
                self.prev_step_scheduled_req_ids.discard(request.request_id)
                self.positions[request.request_id] = position+1
                preempted.append(request.request_id)
        output = super().schedule()
        output.preempted_req_ids.update(preempted)
        return output
