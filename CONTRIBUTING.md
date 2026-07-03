# Contributing to IX

IX is proprietary software (see [`LICENSE`](LICENSE)). Contribution is
limited to authorized members of Zuup Innovation Lab / Visionblox LLC and
partners under a signed agreement. This document describes the working
conventions for that group.

## Repository shape

IX is a monorepo of phase packages that depend on one another in order:

```
ix-substrate  ──▶  ix-world-model  ──▶  ix-strategy-synthesis  ──▶ (Phase 3 …)
   Phase 0            Phase 1                 Phase 2
```

Each package exposes a **frozen contract surface** (pydantic models under
`contracts/`) that downstream phases depend on. The cardinal rule:

> **Downstream depends on contracts, not internals.** Change a contract
> only through an ADR, and only with a schema-version bump.

## Development setup

Requires Python 3.11+.

```bash
# Install phases in dependency order (editable, with dev extras)
pip install -e "packages/ix-substrate[dev]"
pip install -e "packages/ix-world-model[dev]"
pip install -e "packages/ix-strategy-synthesis[dev]"

# Run the full test suite
pytest -v

# Exercise a phase end-to-end without touching the network
python examples/run_world_model_demo.py
python examples/run_phase2_demo.py
```

## Quality gates

Every change must pass, before review:

```bash
ruff check .        # lint (E, F, I, B, UP, SIM)
mypy .              # strict type checking
pytest -v           # tests, including contract tests
```

- `ruff` and `mypy` run in **strict** mode — see `pyproject.toml`. No new
  warnings.
- Line length is 100 (`E501` is intentionally ignored, but keep lines
  reasonable).
- Contract tests (`tests/test_contracts_*.py`) must stay green. If a
  change requires touching them, that is a contract change — see below.

## Changing a contract

Contract changes ripple across phases and are the primary source of
churn. To change a `contracts/` model:

1. Write or update an ADR under [`docs/`](docs) explaining the change and
   its rationale (follow the format of
   [`ADR-0004-complexity-penalty.md`](docs/ADR-0004-complexity-penalty.md)).
2. Bump the model's schema version.
3. Update every downstream consumer and its contract tests.
4. Note the change in [`CHANGELOG.md`](CHANGELOG.md).

## Commit and branch conventions

- Branch from the default branch; use descriptive branch names.
- Write imperative, present-tense commit subjects ("Add HMM regime
  backend", not "Added…").
- Keep commits scoped to a single logical change. Do not mix a contract
  change with unrelated refactors.
- Reference the relevant ADR or phase in the body when applicable.

## Pull requests

Open PRs against the default branch. Fill in the
[PR template](.github/PULL_REQUEST_TEMPLATE.md). A PR should:

- state which phase(s) it touches;
- confirm the quality gates pass;
- call out any contract or schema-version change explicitly.

## Stubs and phase boundaries

Code that stands in for an unlanded phase (for example the
`FillSimulator` in `ix-strategy-synthesis`) must be **clearly marked** in
comments with a removal condition (e.g. `REMOVE WHEN PHASE 3 LANDS`).
Never let a stub leak into a contract.
