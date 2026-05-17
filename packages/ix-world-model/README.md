# ix-world-model

**Phase 1 of the IX agent — the Bayesian world model.**

This is the second layer of the seven-layer IX architecture. It consumes
typed event streams from Phase 0 (`ix-substrate`) and emits typed
posteriors that Phase 2 (strategy synthesis) consumes.

## What Phase 1 does

Two Bayesian mixtures running in parallel, both restricted to tractable
finite classes (the AIXI-flavored pattern from the IX whitepaper, made
concrete):

1. **Regime classifier**: discrete probability distribution over five
   market regimes (`TREND_UP`, `TREND_DOWN`, `MEAN_REVERT`, `VOLATILE`,
   `BROKEN`), updated by Dirichlet credit assignment over realized
   feature likelihoods.

2. **Hypothesis ensemble**: weights over a named ensemble of strategy
   hypotheses, updated by their realized out-of-sample PnL. The
   ensemble does *not* execute strategies — Phase 2 does. The world
   model's job is induction; the strategy layer's job is action.

Both mixtures emit through the same contract surface (`pydantic`,
frozen, schema-versioned), joined into `MarketSnapshot` events that
Phase 2 reads.

## Contracts

```python
from ix_world_model.contracts.events import (
    Regime,             # 5-state enum
    RegimePosterior,    # P(regime | data)
    HypothesisWeights,  # P(hypothesis credit | realized PnL)
    MarketSnapshot,     # joined regime + hypotheses
)
```

## Quick start

```bash
# Phase 0 (ix-substrate) must be installed first.
pip install -e ".[dev]"
pytest -v
python examples/run_world_model_demo.py
```

## Architecture

```
                            ┌────────────────────────────────────────┐
                            │           Phase 1 — World Model        │
                            │                                        │
   TickEvent ────────────▶  │   FeatureExtractor  ──▶  RegimeClassifier
   (from Phase 0)           │       (online)            (Bayesian-Dirichlet)
                            │           │                       │
   DisagreementEvent ─────▶ │           ▼                       │
   (from Phase 0)           │   features.disagreement_intensity │
                            │                                   │
   Phase 2/3 realized PnL  │                                   ▼
   ────────────────────▶    │   HypothesisEnsemble  ──▶  RegimePosterior
                            │   (Bayesian softmax)              +
                            │                                   HypothesisWeights
                            │                                   │
                            │                                   ▼
                            │                          ┌──────────────────┐
                            │                          │  MarketSnapshot  │
                            │                          └──────────────────┘
                            └────────────────────────────────────────┘
                                                       │
                                                       ▼
                                            Phase 2 — Strategy Synthesis
```

## Package layout

```
src/ix_world_model/
├── __init__.py
├── contracts/
│   └── events.py             # Regime, RegimePosterior, HypothesisWeights, MarketSnapshot
├── regimes/
│   ├── features.py           # Online feature extraction
│   └── classifier.py         # Bayesian regime classifier (Dirichlet)
├── hypotheses/
│   └── ensemble.py           # Hypothesis credit-assignment ensemble
└── inference/
    └── orchestrator.py       # WorldModel — wires Phase 0 events to outputs
```

## Roadmap

- **v0.1 (this drop)**: 5-regime classifier; named-hypothesis ensemble;
  orchestrator wired to Phase 0 contracts.
- **v0.2**: HMM transition model for regime persistence; CTW (Context
  Tree Weighting) module as alternative regime backend (closer to the
  MC-AIXI-CTW lineage).
- **v0.3**: Phase 6 (self-improvement) hook for online updates to
  regime profile parameters from realized strategy PnL.
- **v1.0**: Frozen contract surface that Phase 2 can depend on without
  churn.

## License

Proprietary. © 2026 Zuup Innovation Lab / Visionblox LLC.
