"""
ix_strategy_synthesis.hypotheses.base
──────────────────────────────────────
Strategy hypothesis protocol. Every hypothesis is a *function* from a
MarketSnapshot to an ActionProposal. Hypotheses are stateless given the
snapshot — all state lives in the snapshot itself or in the Phase 1
ensemble that grades them.

Why stateless: a hypothesis that depends on internal state is harder to
backtest, harder to audit, and creates implicit coupling that breaks
Phase 6's ability to refit profile parameters from historical
replays. Statelessness is the contract.

Each hypothesis carries a *complexity prior* — its baseline
description_length_bits, before snapshot-specific information adjusts
it. This is the Phase 2 non-obvious gap (USPTO G06N3/126, G06Q40/00):
strategies that need more bits to specify pay a complexity penalty in
planner selection.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

# Type stub for Phase 1's MarketSnapshot. In production, this is an
# imported real type; for isolated test runs we accept anything that
# duck-types.
try:
    from ix_world_model.contracts.events import MarketSnapshot  # noqa: F401
except ImportError:  # pragma: no cover
    MarketSnapshot = object  # type: ignore

from ..contracts.events import ActionProposal


@runtime_checkable
class StrategyHypothesis(Protocol):
    """A named, stateless function from snapshot to proposal."""

    name: str
    base_description_bits: float

    def propose(self, snapshot) -> ActionProposal:
        """
        Inspect the snapshot and emit a proposal. Must NOT mutate state.

        The returned ActionProposal's hypothesis field must equal
        `self.name` and its symbol/window_end must match the snapshot.
        """
        ...
