# 01 · Problem statement

**Status:** decided

## Goal

Rank stocks to support a **balanced long–short strategy**.

Given a universe of stocks at date t:
1. Select a list of **k stocks** from the universe (the Filter, see [03_filter.md](03_filter.md)).
2. **Rank** these k stocks by their expected performance over the next h days.

The ranking feeds a long–short decision: buy the top of the ranking and short the bottom. How positions are actually formed (number of names per side, weighting, rebalancing) is **planned** and not yet specified.

## What the model must do

For each of the k selected stocks, output a **score**. Sorting the scores gives the ranking. Only the **order** of the scores matters, not their absolute values.

## What counts as good

A ranking is good when the stocks it ranks higher actually earn higher returns over t → t + h. This is measured by the rank correlation between scores and realized returns ([04_objective.md](04_objective.md)).

## Out of scope for now

- The long–short decision layer: position sizing, portfolio constraints, transaction costs.
- Execution and broker integration.
