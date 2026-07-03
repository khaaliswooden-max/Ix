# Changelog

All notable changes to IX are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) per
package.

## [Unreleased]

### Planned
- **ix-substrate v0.2** — DEX feeds (Uniswap V3), `DirectionDisagreement`
  and `LatencyLead` detectors, ClickHouse adapter, metrics endpoint.
- **ix-world-model v0.2** — HMM regime transition model; Context Tree
  Weighting (CTW) regime backend, closer to the MC-AIXI-CTW lineage.
- **ix-strategy-synthesis v0.2** — multi-step MCTS with rollouts
  conditioned on forward regime predictions; per-regime λ; real
  `VolHarvester`; Phase 3 integration replacing `FillSimulator`.

## [0.1.0] — 2026

Initial internal drop of the first three IX phases.

### Added
- **ix-substrate (Phase 0)** — market-data substrate. CEX feeds via
  `ccxt`, time-bucket cross-feed alignment, per-venue clock-skew EMA,
  `PriceDisagreementDetector`, and a DuckDB sink. Contract surface:
  `TickEvent`, `DisagreementEvent`. Disagreement is preserved as a
  first-class signal rather than averaged away (see ADR-0002).
- **ix-world-model (Phase 1)** — Bayesian world model. Five-regime
  Dirichlet classifier (`TREND_UP`, `TREND_DOWN`, `MEAN_REVERT`,
  `VOLATILE`, `BROKEN`), online feature extraction, and a
  credit-assignment hypothesis ensemble. Contract surface: `Regime`,
  `RegimePosterior`, `HypothesisWeights`, `MarketSnapshot`.
- **ix-strategy-synthesis (Phase 2)** — regime-conditional strategy
  synthesis. Five regime-specialist hypotheses, hard-constraint filtering
  (`RISK_OFF` domination, size clipping, epsilon floor), and a single-step
  MCTS planner with the per-action Kolmogorov-complexity penalty
  `a* = argmax_a [ V(a) − λ·L(a) ]`. Contract surface: `ActionProposal`,
  `ExecutionRequest`, `RealizationReport`. Includes a clearly-marked
  `FillSimulator` stub for end-to-end exercise until Phase 3 lands.
- **Docs** — IX whitepaper (`docs/IX_Whitepaper.pdf`, `.tex`) and
  `ADR-0004-complexity-penalty.md` documenting the complexity-penalty
  mechanism and its patent strategy.

[Unreleased]: https://github.com/khaaliswooden-max/ix/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/khaaliswooden-max/ix/releases/tag/v0.1.0
