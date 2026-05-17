# IX

**A tractable descendant of AIXI for autonomous capital compounding.**

This monorepo carries the three phases of the IX agent that exist so far:

| Phase | Package                                              | Description                                                  |
|-------|------------------------------------------------------|--------------------------------------------------------------|
| 0     | [`packages/ix-substrate`](packages/ix-substrate)               | Market-data substrate: feeds, reconciliation, storage.       |
| 1     | [`packages/ix-world-model`](packages/ix-world-model)           | Bayesian world model: regime classification + hypothesis ensemble. |
| 2     | [`packages/ix-strategy-synthesis`](packages/ix-strategy-synthesis) | Regime-conditional strategy synthesis with Kolmogorov complexity penalty. |

See [`docs/IX_Whitepaper.pdf`](docs/IX_Whitepaper.pdf) for the formal write-up
and [`docs/ADR-0004-complexity-penalty.md`](docs/ADR-0004-complexity-penalty.md)
for the policy-complexity ADR.
