# Cache-Aware Fair Scheduling for LLM Serving

Executed-work accounting for fair multi-user LLM inference scheduling under prefix caching, chunked prefill and dynamic batching, implemented on vLLM-Ascend and measured with the public Qwen3-4B checkpoint.

## Overview

- **What:** An extension of vLLM-Ascend with a weighted virtual-service scheduler, executed-chunk cost accounting, preemption recomputation and per-batch service reservations, plus a controlled evaluation of four scheduling policies on four SQuAD-derived workloads.
- **Problem:** With prefix caching and chunked prefill, equal-length requests execute different amounts of real compute (cache reuse, chunk-level prefill, variable decode context, shared batch overhead). Logical-token fairness rules mis-price that work.
- **New accounting rules:** Cached prefix tokens skip projection/MLP charges but remain attention keys; actual executed prefill chunks and decode contexts are charged; one measured batch overhead is divided among the requests executing in that batch; tokens recomputed after preemption are charged again; batch construction reserves predicted service so one user cannot exploit a stale priority; a returning user's counter resumes at the service floor. For a chunk of `q` queries after `c` context tokens, the attention term is `q*c + q*(q+1)/2`.
- **Novelty vs. VTC (OSDI'24):** VTC already permits measured cost functions. This project studies *joint* accounting for cache reuse, executed prefill chunks, decode contexts, dynamic-batch overhead and recomputation-after-preemption in a concrete Ascend engine, and evaluates fairness using independently measured isolated-execution references rather than the scheduler's own fitted coefficients. The preemptive implementation does not inherit VTC's original nonpreemptive bound.
- **Measured results:** Across 48 independent Qwen3-4B policy runs (1,536 requests, 46,080 generated tokens; three processes per policy/workload), executed-work accounting reduces the independently measured service gap by **16.16% on shared prefixes** and **22.73% on burst arrivals** versus profiled VTC, with median throughput changes of −4.32% and −0.79%. On mixed-length requests without shared prefixes the same frozen policy shows a **122.60% larger service gap** and 7.65% lower throughput — a workload-dependent tradeoff, not a universal win.

## Hardware and software

Measured on one **Ascend 910B2C NPU** (64 GiB HBM) with CANN 8.5.1, PyTorch 2.9.0 / torch-npu 2.9.0.post1, the pinned **vLLM 0.18.0** core and official **vLLM-Ascend v0.18.0** base (pinned commit `e18643f8a4d5bd9990727654318ad069ea0b56e2`), using public Triton batch-invariant matrix kernels (`VLLM_BATCH_INVARIANT=1`). Validation model: public **Qwen3-4B** checkpoint in BF16, prefix caching and chunked prefill enabled (max context 2,048 tokens, 512-query batch budget, ≤8 concurrent requests, 64-token prefill chunks).

## Links

- Technical report (full tables, three-run ranges, calibration, definitions, limitations): [REPORT.md](REPORT.md)
- Method specification (accounting formulas, policies, workloads, independent reference): [METHODS.md](METHODS.md)
- Chinese walkthrough: [中文讲解.md](中文讲解.md); teaching/interview notes: [入门教学与面试问答.md](入门教学与面试问答.md)
- Upstream engine: <https://github.com/vllm-project/vllm-ascend>
- VTC fair-scheduling paper (OSDI 2024): <https://danyangzhuo.com/papers/OSDI24-VTC.pdf>

## Repository contents

- `patches/compute_service.patch` — the integration surface: adds `vllm_ascend/core/compute_service.py`, `vllm_ascend/core/compute_service_scheduler.py` and their unit tests to the pinned upstream (483 lines).
- `experiment.py`, `run_study.py`, `fit_cost.py`, `evaluate.py`, `build_workloads.py`, `build_replay.py`, `summarize.py`, `verify_results.py`, `smoke_engine.py`, `warmup_worker.py`, `reference_scheduler.py`, `test_evaluate.py` — experiment drivers, calibration, replay/reference evaluation, summarization, protocol checks.
- `configs/` — pinned runtime environment (`environment.json`), public model provenance (`model_source.json`), validation parameters (`validation.json`).
- `workloads/` — tokenized shared-prefix, mixed-length, burst and rejoin workloads plus calibration and replay sets (token IDs, arrival times, output limits retained).
- `results/formal/summary/` — overall and per-user metrics (CSV/JSON) and comparison figures (PDF/PNG); `results/formal/costs.json` (frozen calibration coefficients and residuals); `results/formal/protocol_validation.json`. Raw per-run ledgers are omitted for size.
- `REPORT.md`, `METHODS.md`, `中文讲解.md`, `入门教学与面试问答.md`, `CV.md`.

The vendored upstream vLLM-Ascend checkout used during development is **not** included; clone it from the link above at the pinned commit and apply the patch (below).

## Reproduction

Requires an Ascend 910B host with CANN 8.5.1 installed. Existing tokenized files in `workloads/` are sufficient; `build_workloads.py` regenerates them from public SQuAD 2.0 and the public Qwen3-4B checkpoint.

```bash
# 1. Environment
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-experiment.txt

# 2. Pinned upstream + scheduler/accounting patch
git clone https://github.com/vllm-project/vllm-ascend.git upstream/vllm-ascend
git -C upstream/vllm-ascend checkout e18643f8a4d5bd9990727654318ad069ea0b56e2
git -C upstream/vllm-ascend apply ../../patches/compute_service.patch
source /usr/local/Ascend/cann/set_env.sh
COMPILE_CUSTOM_KERNELS=0 .venv/bin/python -m pip install --no-deps --no-build-isolation -e upstream/vllm-ascend

# 3. Regression tests (accounting, reservations, recomputation, idle-user return)
export PYTHONPATH="$PWD/upstream/vllm-ascend:$PWD:$PYTHONPATH"
VLLM_BATCH_INVARIANT=1 .venv/bin/python -m pytest \
  --confcutdir=upstream/vllm-ascend/tests/ut/core \
  upstream/vllm-ascend/tests/ut/core/test_compute_service.py \
  upstream/vllm-ascend/tests/ut/core/test_compute_service_scheduler.py test_evaluate.py

# 4. Optional: regenerate workloads from public data
.venv/bin/python build_workloads.py --data /path/to/squad-v2-dev.json \
  --model /path/to/Qwen3-4B --output workloads

# 5. Full study on an idle NPU card (exclusive measurement window):
#    independent calibration, isolated references, policy comparison, preemption replay
.venv/bin/python run_study.py --card 0 --output results/reproduction
.venv/bin/python summarize.py results/reproduction
.venv/bin/python verify_results.py results/reproduction
```

`run_study.py` must see a healthy, idle NPU; each process rechecks the device before initialization and records initialization/warmup separately from steady-state measurements. `run_study.py --resume` continues interrupted runs with the frozen coefficient file.
