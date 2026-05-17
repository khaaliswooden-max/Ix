"""
ix_strategy_synthesis.contracts.events
───────────────────────────────────────
Events emitted by Phase 2 (Strategy Synthesis) and the events it consumes
from Phase 1.

Three primary contracts:

  ActionProposal     — per-hypothesis proposed action at one snapshot.
                       Internal to Phase 2; the planner consumes a set
                       of these to choose one.

  ExecutionRequest   — the *chosen* action, sent to Phase 3 (Execution).
                       This is the externally-visible output of Phase 2.

  RealizationReport  — per-hypothesis realized PnL, sent back to Phase 1's
                       hypothesis ensemble so it can update credit weights.
                       Closes the world-model ↔ strategy feedback loop.

All three are frozen pydantic models with schema versioning, same pattern
as Phase 0 and Phase 1.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "0.1.0"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


# ── Action taxonomy ──────────────────────────────────────────────────────

class ActionKind(str, Enum):
    """
    Coarse action space at the strategy layer. Phase 3 refines into
    venue-specific order types.

      LONG       -- enter a long position (or add to existing)
      SHORT      -- enter a short position (or add to existing)
      CLOSE      -- flatten any existing position
      HOLD       -- no action (explicit, not just absence of proposal)
      RISK_OFF   -- emergency unwind; bypasses normal sizing
    """
    LONG     = "long"
    SHORT    = "short"
    CLOSE    = "close"
    HOLD     = "hold"
    RISK_OFF = "risk_off"


# ── Action proposal (internal to Phase 2) ────────────────────────────────

class ActionProposal(_ImmutableModel):
    """
    What a single strategy hypothesis proposes given a market snapshot.

    Fields:
        hypothesis: name of the proposing hypothesis (must exist in the
                    Phase 1 ensemble's roster)
        action: the proposed action kind
        size_fraction: proposed position size as fraction of capital
                       in [-1, 1]. Sign matches LONG/SHORT semantics.
                       LONG/SHORT proposals must have non-zero size;
                       CLOSE/HOLD/RISK_OFF must have size 0.
        confidence: hypothesis's self-reported confidence in [0, 1].
                    Phase 4 (risk) uses this in Kelly-fraction sizing.
        description_length_bits: Kolmogorov-style complexity of the
                                 proposal. The planner penalizes high-bit
                                 proposals — formalizes Occam's razor
                                 inside strategy search. Cannot be < 1.
        rationale: short human-readable reason. Operator-facing.
    """
    schema_version: str = SCHEMA_VERSION
    hypothesis: str
    symbol: str
    window_end: datetime
    action: ActionKind
    size_fraction: Decimal = Field(ge=Decimal("-1"), le=Decimal("1"))
    confidence: float = Field(ge=0.0, le=1.0)
    description_length_bits: float = Field(ge=1.0)
    rationale: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _check_size_action_consistency(self) -> "ActionProposal":
        if self.action == ActionKind.LONG and self.size_fraction <= 0:
            raise ValueError("LONG action requires size_fraction > 0")
        if self.action == ActionKind.SHORT and self.size_fraction >= 0:
            raise ValueError("SHORT action requires size_fraction < 0")
        if self.action in (ActionKind.CLOSE, ActionKind.HOLD, ActionKind.RISK_OFF):
            if self.size_fraction != 0:
                raise ValueError(
                    f"{self.action.value} action requires size_fraction == 0"
                )
        return self


# ── Execution request (emitted to Phase 3) ───────────────────────────────

class ExecutionRequest(_ImmutableModel):
    """
    The chosen action, after MCTS planning over hypothesis proposals.
    This is the externally-visible output of Phase 2.

    Fields:
        chosen_proposal: the ActionProposal selected by the planner
        candidate_proposals: all proposals the planner considered
                              (for auditability and Phase 6 replay)
        planner_value: the planner's evaluation of the chosen action,
                       net of the complexity penalty
        complexity_penalty_applied: the Σ(coef × bits) actually subtracted
        regime_at_decision: the Phase 1 regime mode at decision time
        regime_mode_probability: how confident Phase 1 was in that mode
    """
    schema_version: str = SCHEMA_VERSION
    symbol: str
    window_end: datetime
    chosen_proposal: ActionProposal
    candidate_proposals: tuple[ActionProposal, ...]
    planner_value: float
    complexity_penalty_applied: float = Field(ge=0.0)
    regime_at_decision: str
    regime_mode_probability: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_candidates_contain_chosen(self) -> "ExecutionRequest":
        if self.chosen_proposal not in self.candidate_proposals:
            raise ValueError(
                "chosen_proposal must be one of the candidate_proposals"
            )
        if self.chosen_proposal.symbol != self.symbol:
            raise ValueError("chosen_proposal.symbol does not match request.symbol")
        return self


# ── Realization report (emitted back to Phase 1) ─────────────────────────

class RealizationReport(_ImmutableModel):
    """
    Per-hypothesis realized OOS PnL over a reporting window. Phase 1's
    ensemble consumes these to update credit weights.

    Fields:
        pnl_by_hypothesis: realized PnL per hypothesis (Decimal). A
                            hypothesis that did not act in this window
                            still reports 0 — explicit, not missing.
        window_start, window_end: PnL accumulation window
        source: tagging for provenance — "live", "paper", "backtest", "stub"
    """
    schema_version: str = SCHEMA_VERSION
    symbol: str
    window_start: datetime
    window_end: datetime
    pnl_by_hypothesis: Mapping[str, Decimal]
    source: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _check_temporal_order(self) -> "RealizationReport":
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        if not self.pnl_by_hypothesis:
            raise ValueError("pnl_by_hypothesis cannot be empty")
        return self
