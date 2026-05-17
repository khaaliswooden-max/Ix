# ix-strategy-synthesis

**Phase 2 of the IX agent — regime-conditional strategy synthesis with Kolmogorov complexity penalty.**

This is the third layer of the seven-layer IX architecture. It consumes
`MarketSnapshot` events from Phase 1 (`ix-world-model`) and emits
`ExecutionRequest` events for Phase 3 (Execution), closing the
feedback loop back to Phase 1 via `RealizationReport`.

## What Phase 2 does

Three components, in sequence:

1. **Hypothesis roster**: five regime-specialist strategies, one per
   regime. Each is a stateless function from snapshot to
   `ActionProposal`. Stateless by contract — all state lives in the
   snapshot or in Phase 1's ensemble.

2. **MCTS planner**: single-step in v0.1, multi-step in v0.2. Filters
   hard constraints (`RISK_OFF` domination, size clipping, epsilon
   floor), then scores each proposal under the **complexity-penalty
   objective**:

   ```
   a* = argmax_a [ V(a, snapshot) - λ · L(a) ]
   ```

   where `L(a)` is the action's `description_length_bits` and λ is the
   per-bit penalty coefficient.

3. **Feedback emission**: a `RealizationReport` is sent back to Phase
   1's hypothesis ensemble after each step. In v0.1, this is produced
   by a clearly-marked `FillSimulator` stub. In production (when Phase
   3 lands), it comes from real fill outcomes.

## The non-obvious gap

USPTO clusters **G06N3/126** (genetic programming / evolutionary
algorithms) and **G06Q40/00** (data processing for financial
strategies) cover the prior art for systematic strategy search. The
pathology: search procedures consistently prefer overcomplex strategies
that exploit incidental regularities in evaluation data.

IX's gap: every candidate strategy carries an explicit
`description_length_bits` field, and the planner's objective subtracts
a coefficient × bits *at selection time*, not at training time. This
formalizes Occam's razor inside strategy search, applied per-action.
See `docs/ADR-0004-complexity-penalty.md`.

## Quick start

```bash
# Requires ix-substrate and ix-world-model installed first.
pip install -e ".[dev]"
pytest -v
python examples/run_phase2_demo.py
```

## Architecture

```
                ┌───────────────────────────────────────────────────┐
                │             Phase 2 — Strategy Synthesis          │
                │                                                   │
   MarketSnapshot                                                   │
   (Phase 1)─▶  │  ┌──────────────────────────────────────────┐   │
                │  │ hypothesis roster (5 specialists)        │   │
                │  │   TrendFollower      (4 bits)            │   │
                │  │   MeanReverter       (5 bits)            │   │
                │  │   VolHarvester       (6 bits, stub)      │   │
                │  │   LiquidationAvoider (7 bits)            │   │
                │  │   DisagreementArb    (9 bits)            │   │
                │  └──────────────────────────────────────────┘   │
                │                    │                              │
                │                    ▼                              │
                │  ┌──────────────────────────────────────────┐   │
                │  │ ActionProposal[] from each hypothesis    │   │
                │  └──────────────────────────────────────────┘   │
                │                    │                              │
                │                    ▼                              │
                │  ┌──────────────────────────────────────────┐   │
                │  │   constraints.filter_and_clip            │   │
                │  │   (RISK_OFF dominates; size clip;        │   │
                │  │    epsilon floor)                        │   │
                │  └──────────────────────────────────────────┘   │
                │                    │                              │
                │                    ▼                              │
                │  ┌──────────────────────────────────────────┐   │
                │  │   MCTS planner with complexity penalty   │   │
                │  │   a* = argmax_a [ V(a) - λ · L(a) ]      │   │
                │  └──────────────────────────────────────────┘   │
                │                    │                              │
                │                    ▼                              │
                │           ExecutionRequest ──▶ Phase 3 (Execution)
                │                    │                              │
                │                    ▼                              │
                │           FillSimulator stub                      │
                │           (REMOVE WHEN PHASE 3 LANDS)             │
                │                    │                              │
                │                    ▼                              │
                │           RealizationReport ──▶ Phase 1 ensemble │
                └───────────────────────────────────────────────────┘
```

## Package layout

```
src/ix_strategy_synthesis/
├── __init__.py
├── contracts/
│   └── events.py                  # ActionProposal, ExecutionRequest, RealizationReport
├── hypotheses/
│   ├── base.py                    # StrategyHypothesis protocol
│   └── specialists.py             # 5 regime specialists + default roster
├── planner/
│   ├── complexity_penalty.py      # The non-obvious gap
│   ├── constraints.py             # Hard constraints: RISK_OFF, size, epsilon
│   └── mcts.py                    # Single-step planner (v0.2: multi-step)
└── synthesis/
    └── orchestrator.py            # StrategySynthesizer + FillSimulator stub
```

## Roadmap

- **v0.1 (this drop)**: 5 specialists; single-step MCTS;
  complexity-penalty objective; FillSimulator stub for end-to-end
  exercise.
- **v0.2**: Multi-step MCTS with rollouts conditioned on Phase 1's
  forward regime predictions; replace FillSimulator with Phase 3
  integration; add VolHarvester real implementation.
- **v0.3**: Phase 6 hook for online complexity-coefficient adaptation;
  hypothesis-spawning from realized PnL patterns.
- **v1.0**: Frozen contract surface that Phase 3 can depend on without
  churn.

## License

Proprietary. © 2026 Zuup Innovation Lab / Visionblox LLC.
