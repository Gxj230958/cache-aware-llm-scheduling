"""Warm the supported output-projection batch sizes before measurement."""

import time

import torch


class OutputProjectionWarmup:
    def warmup_output_projection(self):
        model = self.model_runner.model
        hidden_size = self.model_config.get_hidden_size()
        records = []
        for requests in range(1, self.scheduler_config.max_num_seqs + 1):
            hidden = torch.zeros(
                (requests, hidden_size), dtype=self.model_config.dtype, device=self.device
            )
            torch.npu.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                model.compute_logits(hidden)
            torch.npu.synchronize()
            records.append({"requests": requests, "seconds": time.perf_counter() - started})
        return records
