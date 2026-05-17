"""
ix_world_model.contracts.events
────────────────────────────────
Events emitted by Phase 1 (World Model) and consumed by Phase 2
(Strategy). Append-only, frozen, validated. Same architectural pattern as
Phase 0: downstream layers consume typed events, never raw inference
internals.

Two primary contracts:

  RegimePosterior   -- probability distribution over discrete market
                       regimes (e.g. trending, mean-reverting, volatile,
                       broken). Emitted on a regular cadence and on
                       step-change.

  HypothesisWeights -- Bayesian credit assignment across the strategy-
                       hypothesis ensemble. This is the AIXI-style
                       mixture restricted to a finite, tractable class.

Both are joined to Phase 0 events through symbol + window_end timestamps.
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


# ── Regime taxonomy ──────────────────────────────────────────────────────

class Regime(str, Enum):
    """
    Discrete regimes the world model classifies over. Deliberately coarse
    in v0.1 — five states are enough to drive Phase 2 strategy selection
    and any finer-grained scheme overfits on short histories.

    The names are operational rather than statistical:

      TREND_UP        -- persistent positive drift, autocorrelated returns
      TREND_DOWN      -- persistent negative drift, autocorrelated returns
      MEAN_REVERT     -- returns negatively autocorrelated; range-bound
      VOLATILE        -- high variance, low autocorrelation, regime
                         transition zone
      BROKEN          -- liquidity collapse, cascading liquidations, or
                         structural microstructure failure. NOT a regular
                         regime — its detection should trigger
                         risk-off in Phase 4.
    """
    TREND_UP    = "trend_up"
    TREND_DOWN  = "trend_down"
    MEAN_REVERT = "mean_revert"
    VOLATILE    = "volatile"
    BROKEN      = "broken"


# ── Regime posterior ─────────────────────────────────────────────────────

class RegimePosterior(_ImmutableModel):
    """
    Probability distribution over Regime states at a given moment, for a
    given symbol.

    Properties:
        probabilities sum to 1.0 (validated)
        each probability ∈ [0, 1] (validated)
        mode -- the highest-probability regime
        entropy -- nats; high entropy = uncertain regime
    """
    schema_version: str = SCHEMA_VERSION
    symbol: str
    window_end: datetime
    probabilities: Mapping[Regime, float]
    observations_used: int = Field(ge=1)
    half_life_observations: float | None = None  # how persistent has the mode been

    @model_validator(mode="after")
    def _check_simplex(self) -> "RegimePosterior":
        total = sum(self.probabilities.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"probabilities must sum to 1.0 (got {total:.6f})"
            )
        for r, p in self.probabilities.items():
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probability for {r} out of [0,1]: {p}")
        # Every regime must be represented (no missing keys)
        for r in Regime:
            if r not in self.probabilities:
                raise ValueError(f"missing probability for regime {r}")
        return self

    @property
    def mode(self) -> Regime:
        return max(self.probabilities.items(), key=lambda kv: kv[1])[0]

    @property
    def mode_probability(self) -> float:
        return self.probabilities[self.mode]

    @property
    def entropy_nats(self) -> float:
        """Shannon entropy in nats. 0 = certain; log(5) ≈ 1.609 = uniform."""
        import math
        h = 0.0
        for p in self.probabilities.values():
            if p > 0:
                h -= p * math.log(p)
        return h


# ── Hypothesis weights ───────────────────────────────────────────────────

class HypothesisWeights(_ImmutableModel):
    """
    Bayesian credit assignment over a finite ensemble of named strategy
    hypotheses. This is the AIXI-style mixture restricted to a tractable
    finite class.

    Weights are non-negative and sum to 1.0. A hypothesis whose realized
    out-of-sample PnL has been consistently negative will be shrunk
    toward zero; a hypothesis that has been correct will dominate.

    `disagreement_nats` measures how much the ensemble disagrees about
    the next action. High disagreement = exploration signal to Phase 2.
    Low disagreement = strong consensus.
    """
    schema_version: str = SCHEMA_VERSION
    symbol: str
    window_end: datetime
    weights: Mapping[str, float]            # name -> weight, sums to 1
    realized_pnl_window: Mapping[str, Decimal]  # per-hypothesis OOS PnL
    observations_used: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_simplex(self) -> "HypothesisWeights":
        if not self.weights:
            raise ValueError("weights cannot be empty")
        total = sum(self.weights.values())
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"weights must sum to 1.0 (got {total:.6f})")
        for n, w in self.weights.items():
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"weight for {n!r} out of [0,1]: {w}")
        # Every weighted hypothesis must have a PnL entry
        for n in self.weights:
            if n not in self.realized_pnl_window:
                raise ValueError(f"missing realized PnL for hypothesis {n!r}")
        return self

    @property
    def leader(self) -> str:
        return max(self.weights.items(), key=lambda kv: kv[1])[0]

    @property
    def leader_weight(self) -> float:
        return self.weights[self.leader]

    @property
    def disagreement_nats(self) -> float:
        """Shannon entropy of the weight distribution. High = ensemble disagrees."""
        import math
        h = 0.0
        for w in self.weights.values():
            if w > 0:
                h -= w * math.log(w)
        return h


# ── Joined snapshot (convenience contract for Phase 2) ───────────────────

class MarketSnapshot(_ImmutableModel):
    """
    Convenience contract bundling the regime posterior and hypothesis
    weights at a single moment, joined on symbol + window_end. Phase 2
    consumes this preferentially over the two separate streams.
    """
    schema_version: str = SCHEMA_VERSION
    symbol: str
    window_end: datetime
    regime: RegimePosterior
    hypotheses: HypothesisWeights

    @model_validator(mode="after")
    def _check_alignment(self) -> "MarketSnapshot":
        if self.regime.symbol != self.symbol:
            raise ValueError("regime.symbol does not match snapshot.symbol")
        if self.hypotheses.symbol != self.symbol:
            raise ValueError("hypotheses.symbol does not match snapshot.symbol")
        # Timestamps within 1ms of each other are considered aligned
        delta = abs((self.regime.window_end - self.hypotheses.window_end).total_seconds())
        if delta > 0.001:
            raise ValueError(
                f"regime and hypotheses timestamps misaligned by {delta*1000:.3f}ms"
            )
        return self
