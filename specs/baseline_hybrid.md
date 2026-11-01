# baseline_hybrid

**Spec:** `configs/experiments/baseline_hybrid.yaml` · **Owner:** team · **Status:** draft

## Why
First end-to-end run on the survivorship-free Russell 1000 data with the decided Filter ([context/03](../context/03_filter.md)). Every later experiment is compared against it.

## What changes
Compared with the legacy pipeline (see `legacy_parity` and docs/PLATFORM.md §8):
- Universe: `topk_dollar_volume`. The k, x, y and ttl values are placeholders (open question 2).
- Inputs: log return (clipped), vol 60, distance to SMA 20 and 60, log dollar volume 20.
- Label: 5-day forward total return (`forward_return`, h = 5, offset = 0 as in the legacy code). The return definition is open question 4; offset is part of open question 7.
- Loss: legacy `hybrid_hinge` (open question 6: Spearman surrogate).
- Checkpoints chosen on a validation block.

## Expected result
A positive mean IC, stable across folds. This is a baseline, so no target value is set.

## How to verify
```bash
qam validate configs/experiments/baseline_hybrid.yaml
qam run configs/experiments/baseline_hybrid.yaml
```
Check that `runs/<id>/metrics.json` has `test/mean_ic` and the per-fold values, and that `meta.json` has status `finished`.

## Results
Not run yet.
