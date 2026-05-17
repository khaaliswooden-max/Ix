# PHASE 1 — World Model

Contract between Phase 1 and Phase 2. Phase 2 reads only what is
documented here.

## Responsibilities

1. **Consume** typed events from Phase 0: `TickEvent`, `DisagreementEvent`.
2. **Extract features** online from the tick stream: returns, realized vol,
   autocorrelation, spread, disagreement intensity.
3. **Classify** the current market regime via Bayesian update over a
   finite regime set.
4. **Maintain credit weights** over a finite ensemble of named strategy
   hypotheses, updated by their realized out-of-sample PnL.
5. **Emit** `RegimePosterior`, `HypothesisWeights`, and joined
   `MarketSnapshot` events for Phase 2.

Phase 1 is the *only* place inference over market state happens. Phase 2
operates on posteriors, not raw features. Phase 4 (risk) reads ensemble
disagreement as a sizing covariate.

## Contracts emitted

### `RegimePosterior`

| Field | Type | Notes |
|---|---|---|
| schema_version | string | "0.1.0" |
| symbol | string | e.g. "BTC/USDT" |
| window_end | timestamptz | inference moment |
| probabilities | map<Regime, float> | sums to 1.0 across all 5 regimes |
| observations_used | int | ≥ 1 |
| half_life_observations | float \| null | persistence of current mode |

Regime set: `TREND_UP`, `TREND_DOWN`, `MEAN_REVERT`, `VOLATILE`, `BROKEN`.

### `HypothesisWeights`

| Field | Type | Notes |
|---|---|---|
| schema_version | string | "0.1.0" |
| symbol | string | |
| window_end | timestamptz | |
| weights | map<str, float> | sums to 1.0; non-empty |
| realized_pnl_window | map<str, Decimal> | per-hypothesis cumulative OOS PnL |
| observations_used | int | ≥ 1 |

Derived: `leader`, `leader_weight`, `disagreement_nats` (Shannon entropy of weights).

### `MarketSnapshot`

Bundle of `regime` + `hypotheses` joined on `symbol` + `window_end`
(within 1ms tolerance). Phase 2 reads this preferentially.

## Inference design

### Regime classifier

For each regime, a `RegimeProfile` specifies the expected feature
vector (means and per-feature scales). On each observation:

1. Compute log-likelihood under each regime's diagonal-Gaussian profile.
2. Soft credit assignment: add normalized likelihood to each regime's
   Dirichlet count.
3. Apply exponential decay before each update (rolling-window effect).
4. Normalize counts to emit posterior.

This is the **AIXI-style mixture restricted to a finite class**: instead
of Solomonoff's universal prior over all computable programs, we use a
uniform Dirichlet prior over a small set of regime hypotheses. The
mixture is updated by realized likelihoods, preserving the
mixture-of-experts character.

### Hypothesis ensemble

For each named hypothesis, maintain rolling-window realized PnL.
Compute weights via softmax over PnL scores. Apply min-weight floor to
prevent permanent lockout (every hypothesis can recover).

**The ensemble does not execute strategies.** It maintains posteriors
over which strategies have been *correct so far*. Phase 2 reads these
weights and decides what to execute.

## Acceptance criteria (v0.1)

- [x] Five-regime classifier emits valid `RegimePosterior` (probabilities
      sum to 1.0, each in [0, 1], every regime present).
- [x] Classifier concentrates on the matching regime under repeated
      consistent evidence.
- [x] Classifier switches when evidence regime changes (within decay
      horizon).
- [x] Hypothesis ensemble emits valid `HypothesisWeights` (weights sum
      to 1.0, every hypothesis has matching PnL entry).
- [x] Ensemble concentrates on winning hypotheses, applies min-weight
      floor to losers.
- [x] `MarketSnapshot` validates regime and hypotheses symbols match
      and timestamps align within 1ms.
- [x] Orchestrator integrates with Phase 0 `TickEvent` and
      `DisagreementEvent` contracts.
- [x] Cold-start: orchestrator returns None until sufficient
      observations accumulated.
- [ ] Live integration test against Phase 0 substrate running for ≥1
      hour (manual; defer until both packages exercised together).

## Non-goals (deferred)

- HMM transition model with explicit regime-to-regime probabilities
  (v0.2 — CTW backend).
- Multi-symbol regime correlation (v0.3).
- Profile parameter learning from realized PnL (Phase 6 responsibility).
- Continuous-state regime (vs. discrete) — deliberately not chosen for
  v0.1 because discrete regimes give Phase 2 cleaner conditioning.
- Cross-asset regime contagion modeling.
