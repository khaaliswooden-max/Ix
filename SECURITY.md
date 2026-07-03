# Security Policy

IX is proprietary, research-grade software that reasons about and can act
on capital. Security issues here include not only conventional
software vulnerabilities but also **financial-integrity** defects —
anything that could cause the agent to trade incorrectly, leak strategy
internals, or mishandle market data.

## Scope

This policy covers all packages in this repository:

- `ix-substrate` (Phase 0 — market-data substrate)
- `ix-world-model` (Phase 1 — Bayesian world model)
- `ix-strategy-synthesis` (Phase 2 — strategy synthesis)

Classes of issue we especially care about:

- Contract-integrity bugs that let malformed or adversarial `TickEvent` /
  `DisagreementEvent` data corrupt downstream posteriors.
- Constraint bypasses in the Phase 2 planner (`RISK_OFF` domination, size
  clipping, epsilon floor).
- Exposure of proprietary or patent-pending mechanisms, credentials, or
  venue API keys.
- Any path that could produce a live `ExecutionRequest` from unvalidated
  input.

## Reporting a vulnerability

**Do not open a public issue or pull request for a security report.**

Report privately to the Zuup Innovation Lab / Visionblox LLC security
contact. Include:

- affected package and version (see `pyproject.toml`);
- a description of the issue and its impact;
- reproduction steps or a proof of concept;
- any suggested remediation.

Please give us a reasonable window to remediate before any disclosure.
Because IX is proprietary and unreleased, coordinated non-disclosure is
the default.

## Supported versions

Only the latest `0.1.x` internal drop of each package receives security
fixes. Older snapshots are not maintained.

## Handling secrets

Never commit venue API keys, RPC endpoints, or account credentials.
Configuration is supplied at runtime (see the example configs referenced
in `ix-substrate`), and secrets belong in the environment or a secrets
manager, never in the repository.
