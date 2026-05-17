# ADR-0004: Kolmogorov-Complexity Penalty on Action Policy

## Status: Accepted (v0.1)

## Context

USPTO CPC classes **G06N3/126** (genetic programming, evolutionary
computation) and **G06Q40/00** (data processing for financial /
investment strategies) are densely populated by prior art covering
systematic strategy search. The dominant pattern across that prior art:

1. Define a strategy space (parameterized rule systems, neural
   networks, genetic-program trees, etc.).
2. Define an evaluation metric (backtest return, Sharpe, custom
   reward).
3. Search the space, retaining candidates that score highest on the
   metric.

This pattern has a well-documented pathology: **search procedures
consistently prefer overcomplex strategies that exploit incidental
regularities in the evaluation data.** Genetic programming in
particular produces strategies whose specifications run to thousands of
program-tree nodes; their out-of-sample performance is often worse than
trivial baselines. The literature term is "overfitting at the strategy
level," distinguishing it from parameter overfitting within a fixed
strategy form.

Standard mitigations in the prior art:

- **Holdout validation** — split evaluation data into train/validation,
  retain only strategies that perform well on validation. Reduces
  overfitting but does not penalize specification length directly; a
  complex strategy that survives validation is treated identically to a
  simple one that does.
- **Walk-forward analysis** — refit periodically against a rolling
  window. Useful for parameter stability; orthogonal to specification
  complexity.
- **L1/L2 weight regularization** — applicable in neural-strategy
  settings; penalizes parameter magnitudes within a fixed architecture,
  not the architecture itself. Inapplicable to symbolic-rule strategies.

**What is conspicuously absent from the prior art**: a direct,
runtime, per-action penalty on the *specification length* of the
strategy that produced the action. The MDL/Solomonoff prior, which is
the natural formal solution, appears in the AI/ML literature for model
selection (BIC, AIC, MDL-LZW) but not — to my reading — as a contract-
field in financial strategy execution systems.

## Decision

IX implements the Kolmogorov-complexity penalty as a first-class field
on every candidate action and as a multiplicative term in the
planner's selection objective.

### The contract

Every `ActionProposal` (the Phase 2 internal contract) carries a
`description_length_bits: float` field, validated to be ≥ 1.0. Each
hypothesis is responsible for self-reporting its complexity. The bits
value is auditable (it is part of the persisted proposal), tamper-
resistant (frozen pydantic model), and visible to Phase 6 for downstream
calibration.

### The objective

The planner chooses action a* such that

  a* = argmax_a [ V(a, snapshot) - λ · L(a) ]

where:
- V(a, snapshot) is the planner's value estimate (single-step in v0.1;
  multi-step MCTS in v0.2).
- L(a) is `proposal.description_length_bits`.
- λ is `ComplexityPenaltyConfig.coefficient`, parameterizable.

Units: V and λ·L are both in the same value units (expected PnL in
operator currency). The choice of units is operator-set; the
constraint is internal consistency.

### Three things make this defensible against prior art

1. **Description length is a contract field, not a metric.** Existing
   regularization approaches estimate complexity post-hoc (e.g., count
   non-zero weights). IX requires each hypothesis to self-report bits
   as part of the proposal contract. The penalty is not estimated; it
   is contractual.

2. **The penalty is applied at the selection moment, not at training
   time.** Standard regularization shapes which strategies *exist* in
   the search space. The IX complexity penalty shapes which strategies
   *get chosen* in a given moment. A complex strategy can win when its
   expected value sufficiently exceeds the penalty; otherwise the
   simpler strategy wins. This is precisely Occam's razor at runtime.

3. **λ is itself parameterizable per regime.** v0.1 uses a single λ.
   v0.2 will allow per-regime λ values, so the penalty intensity can
   differ between regimes where complexity tends to be justified
   (VOLATILE, where careful filtering pays) vs. regimes where it
   doesn't (TREND_UP, where simple beats complex). This is a
   non-trivial elaboration of the basic mechanism.

## Consequences

- **Positive**: Strategy search at Phase 6 has a built-in regularizer
  that prefers simpler hypotheses. Genetic programming, neural
  architecture search, and reinforcement learning all benefit from
  this regularization without needing additional code.
- **Positive**: Operator-inspectable. "We chose `trend_follower` (4
  bits) over `disagreement_arb` (9 bits) because the penalty differential
  exceeded the value differential." Auditable from logs.
- **Positive**: Phase 6 self-improvement has a calibration target. λ is
  a single scalar that can be tuned from realized OOS performance.
- **Negative**: Hand-set description-length-bits priors in v0.1 are
  somewhat arbitrary. The relative ordering of the five specialists is
  defensible (simpler regimes need simpler strategies), but the
  absolute values are operator-set. Phase 6 must refit.
- **Negative**: At very low λ (≈ 0), the penalty is inert. Operators
  who set λ inappropriately get behavior equivalent to no penalty. This
  is by design (λ is parameterizable) but creates a misconfiguration
  failure mode.

## Patent strategy

This ADR documents architectural priority date for the per-action
complexity-penalty mechanism in the planner-selection layer of an
autonomous-capital agent. The novel claims to draft from this:

1. **Apparatus claim**: a strategy-synthesis system in which each
   candidate action carries an explicit description-length-bits field
   used in selection scoring.

2. **Method claim**: a method of selecting among candidate trading
   actions wherein the selection objective subtracts a coefficient
   times the action's reported description length, applied at runtime
   per action rather than at strategy-creation time.

3. **Method claim**: the same, with per-regime coefficient variation.

4. **Method claim**: the same, with the coefficient itself adapted
   from realized out-of-sample performance.

To be filed jointly with ADR-0002 (disagreement-as-alpha) and ADR-0003
(discrete-regime classifier) once Phase 2 is exercised end-to-end with
Phases 0 and 1.
