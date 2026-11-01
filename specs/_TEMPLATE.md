# <experiment name>

**Spec:** `configs/experiments/<name>.yaml` · **Owner:** <who> · **Status:** draft | running | done | dropped

## Why
What question this experiment answers. Link the decision or open question in `context/` that it serves.

## What changes
Compared with `baseline_hybrid` (or the experiment it is copied from): which sections, which components, which values. New components go in their own section below.

## New components (if any)
For each: kind, name, file, what it computes, and how it stays point-in-time.

## Expected result
What would count as success or failure, and the metric that decides it.

## How to verify
Tests added, the command to run, and what to check in `runs/<id>/`.

## Results
Run ids, main metrics, conclusion. Add a figure to `reports/figures/` only when the result is final.
