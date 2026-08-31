# Training Architecture Refactor

> **Status — historical and superseded.** This plan describes the earlier
> 39-feature contract and pre-schema-v3 artifact/serving limitations. The current
> implementation uses the remediated 38-feature contract and validated schema-v3
> Rank-NDCG serving bundles.

## Purpose

The current research milestone is complete. This refactor improves code ownership, maintainability, testability, and reviewability without intentionally changing model or evaluation behavior.

`src/train/trainer.py` has accumulated too many responsibilities. The goal is to make it a thin composition/CLI layer, not to redesign the project or maximize module count.

## Frozen behavior

During this refactor, preserve:

- target definitions and Rank-NDCG relevance construction
- the ordered 39-feature contract
- model parameters and fitting behavior
- train/validation/test chronology and embargo
- walk-forward folds and aggregation
- evaluation metrics and baselines
- report schemas and important CLI behavior
- current artifact paths and persistence behavior
- `src.data.data_prep.DataPreparator` and its serialization path

Behavioral improvements belong in later milestones.

## Target ownership

Create these focused owners:

- `src/data/training_data.py` — cached market-data fetching, per-ticker preparation, and parallel universe assembly
- `src/train/training_contract.py` — target-mode and feature-subset validation
- `src/train/reporting.py` — report composition, formatting, and logging
- `src/train/artifacts.py` — artifact metadata, paths, and persistence
- `src/train/model_training.py` — normal training and model-specific fitting/selection
- `src/train/walk_forward_runner.py` — walk-forward execution and model dispatch

Keep:

- `src/train/trainer.py` — CLI, multi-horizon dispatch, logging setup, and temporary compatibility exports
- `src/train/evaluation/` — existing evaluation methodology and metrics
- `src/train/evaluation/walk_forward.py` — fold construction, embargoed splits, and aggregation
- `DataPreparator`, feature-contract, indicator, and inference modules in their current roles

Lower-level modules must not import `trainer.py`.

## Refactor phases

1. **Leaf boundaries**
   - extract training-data acquisition/cache
   - extract training contracts

2. **Reporting**
   - move report composition, formatting, and logging
   - reorganize related tests as ownership becomes clear

3. **Artifacts**
   - move existing metadata/path/save behavior unchanged

4. **Model training**
   - move `train_models()` and existing fitting paths
   - preserve Rank-NDCG, top/bottom, and excess-return behavior

5. **Walk-forward execution**
   - expose a public model-frame preparation API
   - move fold execution into `walk_forward_runner.py`
   - leave walk-forward methodology in `evaluation/walk_forward.py`

Each phase should use small reviewable commits and leave the full test suite passing.

## Known artifact/inference issue

Target modes currently share artifact namespaces. A later training run can overwrite shared metadata/preparator files while incompatible older model files remain.

Current inference does not validate these bundles and does not serve the Rank-NDCG model correctly. Therefore:

- this does **not** invalidate walk-forward research, which does not use persisted artifacts
- current inference should not be trusted
- do not silently fix artifact layout during this structural refactor
- address artifact versioning/bundle integrity and Rank-NDCG inference in a separate milestone immediately afterward

## Explicitly out of scope

Do not use this refactor to:

- tune models, targets, or features
- change evaluation methodology
- redesign inference or paper trading
- fix survivorship bias
- create a generic experiment framework
- perform broad stale-code cleanup
- redesign packaging, logging, or CLI installation
- split already-cohesive modules merely to make files smaller

## Done

The refactor is complete when:

- `trainer.py` is primarily a thin CLI/composition root
- data acquisition no longer depends on the training stack
- training, reporting, artifacts, and walk-forward execution have clear owners
- no lower-level module imports `trainer.py`
- tests largely follow production ownership
- existing behavior remains covered by the full test suite

Stop there. Artifact/inference safety is the next separate milestone.
