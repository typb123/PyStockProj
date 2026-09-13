# Ablation Execution Design

## Purpose, scope, and optimization objective

Scope: execution and performance design for Rank-NDCG walk-forward feature ablations.
Reference machine: i7-13700K, 24 logical CPUs, approximately 23 GiB RAM
available to WSL2.

See [Performance and Parallelism](PERFORMANCE_AND_PARALLELISM.md) for project-wide
performance guidance, resource budgeting, and the measured worker-scaling summary.

The intended large research workload is a full matched LOFO campaign across
all retained features, both horizons, and multiple model seeds.

The primary optimization goal is end-to-end campaign wall time while preserving
scientific equivalence and bounded memory use. High CPU utilization and reduced
aggregate computation are not sufficient objectives.

Random-baseline evaluation now uses local per-date numeric arrays instead of
repeated pandas trial work. Re-profile representative campaigns before choosing
the next optimization; the previous bottleneck profile is no longer the baseline.

Prefer local computational improvements or bounded reuse of redundant evaluation
work before adding substantially more scheduling complexity.

Keep the current architecture unless representative measurements justify change.
Candidates below are options, not a committed implementation roadmap.

## CURRENT — variant-level execution

- One task runs one ablation variant for one horizon and model seed.
- Each variant prepares its modeling frame and processes folds sequentially.
- With multiple outer workers, the process pool explicitly uses `spawn`.
- The parent fetches and generates indicators once per invocation.
- Each worker receives its own input DataFrame through pool initialization.
- Tasks carry small specifications; completed tasks return numerical reports.
- Workers persist across folds and can process subsequent variant tasks.
- Prepared splits, scaled arrays, and models are not retained across all folds.
  Fold descriptions, reports, and population identities are retained.
- `outer_workers * xgb_threads <= cpu_budget` is enforced per invocation.
- With multiple outer workers, BLAS threads are limited to one and random
  baseline trials run serially within each worker.
- Random input arrays are prepared locally per evaluation and reused across
  trials and Top-Ns, preserving RNG ordering and statistical semantics.
- With one outer worker, the requested random-trial worker count is preserved.
  That separate pool does not explicitly select `spawn`; use one random-trial
  worker for serial comparisons without another process pool.

This baseline avoids fork/native-thread deadlocks in the outer pool and the
previous multiplication of retained expanding-fold state across spawned workers.
It does not guarantee acceptable memory use at arbitrary worker counts.

Six outer workers with four XGBoost threads each (6×4) is currently the best
measured configuration for the representative 12-variant campaigns on this
machine. It was fastest at both 10d and 20d with no observed swap. This remains
workload- and hardware-specific rather than a permanent project constant.

## Known limitations

- Useful outer concurrency cannot exceed the runnable variant count.
  Two variants provide only two useful outer tasks.
- Much preparation and evaluation is effectively single-threaded per worker.
  Low overall CPU utilization can therefore be expected behavior.
- Model-frame preparation, split construction, scaling, and model-independent
  evaluation repeat across variants.
- `spawn` provides no automatic copy-on-write sharing of the parent's data.
  Input copies, active folds, native allocations, and temporaries consume RAM.
- Expanding folds increase active working-set size even without all-fold retention.
- CPU validation is per invocation; concurrent jobs need a shared machine budget.

## CANDIDATE — alternatives before larger scheduling changes

| Option | Reason to consider / constraint |
| --- | --- |
| Local copy/date/reduction improvements | Remove redundant work while preserving existing outputs. |
| Bounded concurrent seed/horizon jobs | Expose independent tasks; budget total workers, threads, and RAM across jobs. |
| Bounded worker-local random-summary cache | Larger LOFO campaigns may reuse final summaries; retain no prepared folds or trial histories. |

Cache value depends on access order and worker lifetime.
A one-fold cache usually misses when workers traverse all folds per variant.
Cache keys must identify ordered evaluation inputs and statistical configuration,
not merely fold index; mutation and stale reuse must be excluded.

## DEFERRED — larger architecture

- Fold/variant scheduling with bounded worker-local fold reuse.
- Generalized schedulers, producer/consumer pipelines, and shared-memory machinery.

Reconsider only if simpler changes leave a measured bottleneck on the intended
campaign. Any proposal must explain its additional benefit and maintenance cost.

## REJECTED — previous failure modes

- Retained all-fold prepared contexts: expanding windows duplicate history;
  retaining arrays/metadata and copying them to spawned workers caused RAM
  exhaustion, swapping, and prolonged serialization.
- Eager expensive evaluation preparation across every fold in the parent:
  creates a long serial startup dependency before useful variant execution.
- Repeatedly shipping large prepared folds as ordinary queued task payloads:
  adds serialization, copies, and potentially large outstanding retained state.

These are rejected for the current workload and resource envelope.
Reopening them requires evidence addressing the failure mechanism, not merely
a faster one-fold benchmark. Small parent preparation and bounded transfers
are not categorically prohibited.

## Acceptance rules

- Benchmark actual task counts and representative full-history execution.
  Include full-year evaluation; the latest fold may be a partial year.
- Measure end-to-end wall time, including acquisition, startup, preparation,
  synchronization, shutdown, aggregation, and output.
- Estimate then measure peak process-tree RAM and swap activity; reserve headroom.
- Account for initializer/task/result serialization, queued objects, caches,
  active-worker copies, and native allocations; bound retained large state.
- Verify actual process counts and machine-wide thread budgets.
- Account for serial critical-path work; less total work is not enough.
- Preserve populations, ordering, folds, embargoes, labels/qids, seeds, trial
  counts, tie behavior, metrics, and report semantics. Verify RNG selections;
  justify any numerical tolerances explicitly.
- Require measured wall-time or memory benefit proportional to complexity.
  Prefer replacing expensive local code over adding parallel execution paths.
- Record supporting measurements in [Experiment Notes](EXPERIMENT_NOTES.md)
  and update this decision when the accepted architecture changes.

Implementation: [outer execution](../src/train/rank_ndcg_feature_ablations.py),
[fold runner](../src/train/walk_forward_runner.py),
[random baseline](../src/train/evaluation/baselines.py).
