# Performance and Parallelism

Project-wide guidance for performance engineering, execution, and resource
budgeting. Companion documents cover detailed decisions and experiment history.

## Performance philosophy

- Optimize measured end-to-end wall time, not CPU utilization by itself.
- Profile representative workloads before redesigning execution. Include data
  acquisition, preparation, startup, synchronization, shutdown, and reporting.
- Distinguish latency (time to finish one experiment) from throughput (completed
  work per unit time). More workers help only when enough independent tasks exist.
- Preserve scientific correctness and determinism: populations, chronology,
  feature contracts, seeds, sample ordering, statistical weights, and reports.
  Verify equivalence and justify floating-point tolerances explicitly.
- Remove redundant computation before adding concurrency; keep complexity
  proportional to demonstrated wall-time or memory benefit.

## Current parallel research execution model

- Independent ablation variants run in an outer process pool using explicit
  `spawn`; each variant processes its folds sequentially within one worker.
- XGBoost receives an explicit thread allocation inside each worker.
- With multiple outer workers, BLAS pools are limited to one thread through
  `threadpoolctl`, and random-baseline trials remain serial within each worker.
- The parent fetches and enriches data once per invocation. Each spawned worker
  receives its own input DataFrame through initialization and reuses it for tasks.
- Task specifications and returned reports are small; prepared expanding folds
  are not retained across the campaign or passed through task queues.

## Resource budgeting and measurement

- CPU threads are a global machine budget. The ablation runner enforces
  `outer_workers * xgb_threads <= cpu_budget` per invocation; concurrent jobs
  must share that budget. Nested native threads or process pools can oversubscribe it.
- Measure process-tree RSS/PSS, swap, process count, and serialization alongside
  CPU. Account for input copies, active folds, native allocations, and temporaries;
  reserve RAM headroom rather than relying on swap to accommodate more workers.
- Prefer process-tree measurements over interpreting Task Manager alone. Linux
  CPU percentages use 100% for one fully utilized logical CPU.
- [Phase timing](../src/train/phase_timing.py) records opt-in worker phases;
  concurrent worker totals are not sequential wall-clock durations.
- [Resource monitoring](../scripts/monitor_process_tree.py) records end-to-end
  wall time and process-tree CPU, RSS/PSS, swap, and process counts.

## Implemented improvements

- Invocation-level data reuse avoids repeated fetching and indicator generation.
- [Random-baseline evaluation](../src/train/evaluation/baselines.py) replaced
  repeated pandas DataFrame slicing, concatenation, and reductions with local
  per-date numeric arrays prepared once per evaluation and reused across Top-Ns.
- Exact RNG ordering and sampled positions were preserved, including draws for
  full candidate selections. Date/row weighting, missing-value behavior, medians,
  and by-year statistical semantics remain unchanged; numeric reductions are
  checked for equivalence with explicit tolerances where needed.
- Explicit outer `spawn` fixed the earlier Linux fork/native-runtime deadlock.

## Current measured scaling

Representative 10d campaign: 12 variants, 28 folds, seed 42, `period=max`,
`large_mega_cap_stocks`, 100 random trials, serial random trials per outer worker.
Hardware: i7-13700K, 24 logical CPUs, approximately 23 GiB RAM available to WSL2, 8 GiB swap;
Python 3.12 and CPU XGBoost.

| Outer workers × XGB threads | Wall time (s) | Average tree CPU | Peak tree PSS (MiB) | Swap |
| --- | ---: | ---: | ---: | ---: |
| 2 × 12 | 918.3 | 663.5% | 4456.7 | 0 |
| 4 × 6 | 552.7 | 987.9% | 7818.9 | 0 |
| 6 × 4 | 449.1 | 1285.0% | 11293.2 | 0 |

6 × 4 was also the best measured configuration on the matched 20d representative campaign, reducing wall time from 529.4 s at 4 × 6 to 425.0 s with no observed swap.

The array rewrite reduced the representative random-baseline phase by about
8.6× and roughly halved two-variant end-to-end wall time. Phase acceleration
and end-to-end acceleration differ because other work remains.

## Rejected approaches and cautions

- Do not retain all 28 expanding prepared folds or eagerly perform expensive
  all-fold preparation in the parent: RAM growth and serial startup erase gains.
- Do not repeatedly move large prepared fold payloads through process queues.
- Do not add nested random-trial multiprocessing while abundant variant-level
  concurrency exists. Higher CPU utilization alone is not proof of improvement.

## Scaling guidance

- Re-benchmark worker/thread allocation when workload, model complexity,
  hardware, or memory behavior changes; include sustained execution and full folds.
- Larger models may shift the bottleneck toward training. Consider GPU
  acceleration only when profiling shows fitting warrants another execution path.
- Deeper scheduling or shared-memory systems require measured evidence of benefit,
  bounded memory and serialization costs, and a justified maintenance burden.

## Related documentation

- [Ablation Execution Design](ABLATION_EXECUTION_DESIGN.md): ablation-specific
  execution invariants, limitations, rejected/deferred designs, and acceptance rules.
- [Experiment Notes](EXPERIMENT_NOTES.md): experiment history and research results.
