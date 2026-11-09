# Executed compute service with cached and chunked inference

## Service accounting

Each request carries a user identifier and weight in `SamplingParams.extra_args`. For an executed chunk containing `q` query tokens after `c` context tokens, the scheduler charges

```text
launch_ms / requests_in_batch
    + token_ms * q
    + attention_pair_ms * (q*c + q*(q+1)/2)
    + decode_token_ms * decode_queries
```

`decode_queries` counts queries beyond the original prompt. Cached prefix tokens do not require projections or MLP execution, but remain attention keys. Recomputed tokens after preemption receive the same executed-work charge as other queries. The batch overhead is divided once among requests that execute in the batch. Nonnegative least squares fits the four coefficients to separately measured calibration batches.

Each user has a virtual service counter, incremented by charged service divided by weight. Queue construction reserves estimated service as requests are selected, so multiple requests from one user do not all receive the same priority. A returning user starts no earlier than the current service floor. When running slots are full, admission may preempt a request belonging to an overrepresented user to admit a waiting user below its weighted share. The worker removes the preempted request from its persistent batch before resumption.

## Controlled policies

All comparisons use the same public Qwen3-4B checkpoint, prefix cache, chunked prefill configuration, operator implementation, arrival schedule and output-token limits. FCFS retains the underlying queue order. The three service-based policies share virtual counters, reservations and admission mechanics, differing in what they charge:

| Policy | Charged service |
|---|---|
| Token VTC | Logical input tokens plus twice generated output tokens |
| Profiled VTC | Logical prefill and decode queries, weighted by separately fitted execution coefficients |
| Executed work | Actual query/context work, including cache reuse, recomputation and shared batch overhead |

Both logical policies charge the cached prompt once and avoid charging recomputation again. Token VTC charges every generated output, including the first output produced by prefill: a completed request with `P` input tokens and `N` outputs costs `P + 2*N`. Profiled VTC uses separately fitted query costs, so its corresponding count is `P` prefill queries and `N-1` decode queries. The first output's work is included in prefill execution. The implementations adapt the VTC virtual-service framework to this engine's cache and chunk scheduling; they are controlled accounting comparisons within a shared admission implementation.

## Workloads and runtime

The model uses BF16, greedy decoding with fixed output-token limits, maximum context length 2,048, at most eight running requests, a 512-token batch budget and 64-token long-prefill chunks. The synchronous engine uses an asynchronous client to submit requests independently of completion. `VLLM_BATCH_INVARIANT=1` selects the public Triton batch-invariant matrix kernels in this installation; the remaining operators use the supported Ascend runtime. Compilation and warmup precede the measured workload; client elapsed time includes queueing, scheduling and request execution.

Each workload contains four equal-weight users with eight requests each. Output-token limits are 16, 32, 24 and 48 tokens by user. Prompts come from public SQuAD 2.0 documents and questions. The first 16 source articles form calibration data; remaining articles form evaluation data. Actual token IDs, document titles and arrival times are retained in `workloads/`.

| Workload | Arrival and cache structure |
|---|---|
| Shared | Even-numbered users reuse document prefixes; requests arrive every 25 ms |
| Mixed | Request-specific documents and mixed prompt/output lengths, every 25 ms |
| Burst | Four requests per user arrive at zero, four at 500 ms; shared prefixes for even-numbered users |
| Rejoin | User zero first submits two requests, then returns at 2.1–2.35 s; other users follow the 25 ms pattern |

Each engine first warms total query counts 1–512 with the official `compile_sizes` interface and output-projection batch sizes 1–8 through a worker extension. These ranges cover the matrix specializations admitted by the fixed batch budget and running-request limit. Later independent processes reuse the on-disk kernel cache. Engine initialization including query warmup, output-projection warmup, and workload warmup are recorded separately from measured requests.

Before each measured workload, the prefix cache is reset and the designated shared prefixes are primed. A complete warmup workload precedes this reset. Calibration and each policy/repetition run in independent processes. Three calibration runs produce one fixed coefficient file before policy evaluation. Main results contain three independent runs per policy and workload in an exclusive measurement window.

## Independent service reference

Separate processes execute the same requests one at a time. Each recorded reference batch contains one request, and its duration spans scheduler dispatch to model-output return. A reference shape is the exact tuple `(query_tokens, context_tokens, prompt_tokens)`. The evaluator uses the median independently measured duration for that shape to value every batch member in a policy run. Token counters have weighted-token units; profiled and executed-work counters have estimated-millisecond units. The reported fairness metric uses independent reference milliseconds for all policies, with user weights dimensionless. This reference does not use the scheduler's fitted coefficients.

Preemption can create recomputation chunks absent from uninterrupted isolated execution. `build_replay.py` gathers each observed request's initial cache position and complete ordered preemption points across all policy runs. `ReferenceScheduler` replays each distinct trajectory through real model execution in isolation, resetting and priming its initial prefix cache. Ordinary isolated runs and these replay runs jointly form the reference table. Evaluation requires a measured reference for every executed shape.

For a pair of users, the fairness metric tracks the absolute difference in their independently valued service increments, divided by user weights, during each continuously backlogged overlap interval. It reports the maximum at batch-completion samples. A third user's arrival preserves an existing pair's interval; an actual idle period ends it. Per-user service totals are retained separately. Slowdown divides each request's shared-service latency by its median uninterrupted isolated latency, then averages by user. Throughput counts completed output tokens over measured client elapsed time; P95 first-token latency includes queueing.

## Research context and scope

[VTC](https://danyangzhuo.com/papers/OSDI24-VTC.pdf) already supports configurable and measured cost functions and tracks service with virtual counters. This project studies joint accounting for prefix reuse, executed chunks, decode contexts and dynamic-batch overhead in a concrete Ascend engine. Its preemptive admission implementation and independent-reference experiments do not inherit VTC's nonpreemptive theoretical bound.

The study uses one dense model, one host, equal weights in measured workloads and local prefix caching. Unequal weights are exercised by the unit tests. The isolated reference assigns a reproducible service value to work; it is not a decomposition of simultaneous physical NPU time. The fixed cost model approximates changing batch efficiency. Output-token limits control generated work, and evaluation concerns scheduling behavior rather than answer quality. Remote KV transfer and speculative decoding are outside the implemented workload.
