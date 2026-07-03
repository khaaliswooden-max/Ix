# IX

**A tractable descendant of AIXI for autonomous capital compounding.**

IX is a phased, contract-driven agent that ingests market data, maintains
a Bayesian world model over market regimes, and synthesizes
regime-conditional strategies under an explicit complexity penalty. It
takes the AIXI ideal — a universal agent that induces a model of its
environment and acts to maximize reward — and makes it tractable by
restricting every mixture to a finite, auditable class.

> ⚠️ Research-grade and proprietary. IX is **not** investment advice and
> ships no live-trading authorization. See [`LICENSE`](LICENSE).

## Phases

This monorepo carries the three phases of the IX agent that exist so far.
Each phase is a package that consumes the phase before it through a
**frozen, schema-versioned contract surface**:

| Phase | Package                                                            | Description                                                                 |
|-------|-------------------------------------------------------------------|-----------------------------------------------------------------------------|
| 0     | [`packages/ix-substrate`](packages/ix-substrate)                   | Market-data substrate: feeds, reconciliation, storage.                      |
| 1     | [`packages/ix-world-model`](packages/ix-world-model)               | Bayesian world model: regime classification + hypothesis ensemble.          |
| 2     | [`packages/ix-strategy-synthesis`](packages/ix-strategy-synthesis) | Regime-conditional strategy synthesis with Kolmogorov complexity penalty.   |

```
ix-substrate  ──▶  ix-world-model  ──▶  ix-strategy-synthesis  ──▶  (Phase 3: Execution …)
   Phase 0            Phase 1                  Phase 2

 TickEvent          RegimePosterior          ActionProposal
 DisagreementEvent  HypothesisWeights        ExecutionRequest
                    MarketSnapshot           RealizationReport
```

The through-line across phases is that IX **preserves information other
systems discard**: Phase 0 keeps cross-feed *disagreement* as a
first-class signal instead of averaging it into a "true price," and Phase
2 charges every candidate action for its *description length* so simpler
strategies win unless a complex one clearly earns its bits.

## Quick start

Requires Python 3.11+. Install the phases in dependency order:

```bash
pip install -e "packages/ix-substrate[dev]"
pip install -e "packages/ix-world-model[dev]"
pip install -e "packages/ix-strategy-synthesis[dev]"
```

Exercise the agent end-to-end without touching the network:

```bash
python examples/run_world_model_demo.py   # Phase 0 → Phase 1
python examples/run_phase2_demo.py        # Phase 1 → Phase 2
```

Run the tests:

```bash
pytest -v
```

## Repository layout

```
.
├── packages/
│   ├── ix-substrate/            # Phase 0 — market-data substrate
│   ├── ix-world-model/          # Phase 1 — Bayesian world model
│   └── ix-strategy-synthesis/   # Phase 2 — strategy synthesis
├── examples/                    # end-to-end demos (offline)
├── tests/                       # cross-package + contract tests
├── docs/                        # whitepaper + ADRs
├── CONTRIBUTING.md
├── CHANGELOG.md
├── SECURITY.md
└── LICENSE
```

## Development

Quality gates run in strict mode (configured in `pyproject.toml`):

```bash
ruff check .    # lint: E, F, I, B, UP, SIM
mypy .          # strict type checking
pytest -v       # tests, including contract tests
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for working conventions — in
particular, how to change a contract (ADR + schema bump + downstream
updates) without churning the phases below it.

## Documentation

- [`docs/IX_Whitepaper.pdf`](docs/IX_Whitepaper.pdf) — the formal
  write-up of the seven-layer IX architecture.
- [`docs/ADR-0004-complexity-penalty.md`](docs/ADR-0004-complexity-penalty.md)
  — the policy-complexity ADR (per-action Kolmogorov penalty).
- Each package has its own `README.md` with a phase-level architecture
  diagram, package layout, and roadmap.

## License

Proprietary. © 2026 Zuup Innovation Lab / Visionblox LLC. All rights
reserved. See [`LICENSE`](LICENSE).
