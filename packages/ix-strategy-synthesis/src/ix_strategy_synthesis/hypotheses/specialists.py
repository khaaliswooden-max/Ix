"""
ix_strategy_synthesis.hypotheses.specialists
─────────────────────────────────────────────
Five strategy hypotheses, one per regime from Phase 1's Regime taxonomy.
Each is stateless: snapshot in, proposal out.

Description-length bits are assigned by counting the number of
parameters and conditional branches each hypothesis specifies. The
order roughly tracks complexity: TrendFollower (simplest) <
MeanReverter < VolHarvester < LiquidationAvoider < DisagreementArb.

These bit counts are *priors*; they are not tuned. Phase 6 will refit
them from realized PnL over time. For v0.1 the priors are operator-
specified, conservative, and ordered.
"""
from __future__ import annotations

from decimal import Decimal

# Phase 1 types
try:
    from ix_world_model.contracts.events import (
        HypothesisWeights,
        MarketSnapshot,
        Regime,
        RegimePosterior,
    )
except ImportError:  # pragma: no cover
    Regime = None  # type: ignore
    MarketSnapshot = object  # type: ignore

from ..contracts.events import ActionKind, ActionProposal


# ── 1. Trend follower ────────────────────────────────────────────────────

class TrendFollower:
    """
    Specialist for TREND_UP and TREND_DOWN regimes. Goes long when the
    posterior says TREND_UP with high probability; short when TREND_DOWN
    with high probability; HOLD otherwise.

    Simplest specification: 1 threshold + 1 sign branch = ~4 bits.
    """
    name: str = "trend_follower"
    base_description_bits: float = 4.0

    def __init__(self, confidence_threshold: float = 0.5, size_at_max_confidence: Decimal = Decimal("0.5")):
        self.confidence_threshold = confidence_threshold
        self.size_at_max_confidence = size_at_max_confidence

    def propose(self, snapshot: MarketSnapshot) -> ActionProposal:
        regime = snapshot.regime
        p_up = regime.probabilities.get(Regime.TREND_UP, 0.0)
        p_down = regime.probabilities.get(Regime.TREND_DOWN, 0.0)

        if p_up > self.confidence_threshold and p_up > p_down:
            return ActionProposal(
                hypothesis=self.name,
                symbol=snapshot.symbol,
                window_end=snapshot.window_end,
                action=ActionKind.LONG,
                size_fraction=self.size_at_max_confidence * Decimal(str(p_up)),
                confidence=p_up,
                description_length_bits=self.base_description_bits,
                rationale=f"TREND_UP posterior {p_up:.2f}",
            )
        if p_down > self.confidence_threshold and p_down > p_up:
            return ActionProposal(
                hypothesis=self.name,
                symbol=snapshot.symbol,
                window_end=snapshot.window_end,
                action=ActionKind.SHORT,
                size_fraction=-self.size_at_max_confidence * Decimal(str(p_down)),
                confidence=p_down,
                description_length_bits=self.base_description_bits,
                rationale=f"TREND_DOWN posterior {p_down:.2f}",
            )
        return _hold(snapshot, self.name, self.base_description_bits,
                     f"no trend (up={p_up:.2f}, down={p_down:.2f})")


# ── 2. Mean reverter ─────────────────────────────────────────────────────

class MeanReverter:
    """
    Specialist for MEAN_REVERT regime. Takes opposite side of recent
    move when posterior says MEAN_REVERT, scaled by confidence.

    Requires reading both posterior and a tick-direction proxy from the
    snapshot; ~5 bits.
    """
    name: str = "mean_reverter"
    base_description_bits: float = 5.0

    def __init__(self, confidence_threshold: float = 0.45, size_at_max_confidence: Decimal = Decimal("0.4")):
        self.confidence_threshold = confidence_threshold
        self.size_at_max_confidence = size_at_max_confidence

    def propose(self, snapshot: MarketSnapshot) -> ActionProposal:
        p_mr = snapshot.regime.probabilities.get(Regime.MEAN_REVERT, 0.0)
        if p_mr <= self.confidence_threshold:
            return _hold(snapshot, self.name, self.base_description_bits,
                         f"MEAN_REVERT posterior only {p_mr:.2f}")

        # MeanReverter goes counter to the dominant trend probability.
        p_up = snapshot.regime.probabilities.get(Regime.TREND_UP, 0.0)
        p_down = snapshot.regime.probabilities.get(Regime.TREND_DOWN, 0.0)
        if p_up > p_down:
            return ActionProposal(
                hypothesis=self.name,
                symbol=snapshot.symbol,
                window_end=snapshot.window_end,
                action=ActionKind.SHORT,
                size_fraction=-self.size_at_max_confidence * Decimal(str(p_mr)),
                confidence=p_mr,
                description_length_bits=self.base_description_bits,
                rationale=f"MEAN_REVERT {p_mr:.2f}, fade trend_up {p_up:.2f}",
            )
        return ActionProposal(
            hypothesis=self.name,
            symbol=snapshot.symbol,
            window_end=snapshot.window_end,
            action=ActionKind.LONG,
            size_fraction=self.size_at_max_confidence * Decimal(str(p_mr)),
            confidence=p_mr,
            description_length_bits=self.base_description_bits,
            rationale=f"MEAN_REVERT {p_mr:.2f}, fade trend_down {p_down:.2f}",
        )


# ── 3. Volatility harvester ──────────────────────────────────────────────

class VolHarvester:
    """
    Specialist for VOLATILE regime. In v0.1 this is a HOLD strategy —
    real vol harvesting requires options or perp-funding plumbing that
    belongs in Phase 3+. We track the hypothesis here so the ensemble
    can grade it (always 0 PnL, low credit) and so the contract is
    ready when Phase 3 enables the real implementation.

    ~6 bits: posterior threshold + degenerate sizing logic placeholder.
    """
    name: str = "vol_harvester"
    base_description_bits: float = 6.0

    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold

    def propose(self, snapshot: MarketSnapshot) -> ActionProposal:
        p_vol = snapshot.regime.probabilities.get(Regime.VOLATILE, 0.0)
        if p_vol <= self.confidence_threshold:
            return _hold(snapshot, self.name, self.base_description_bits,
                         f"VOLATILE posterior only {p_vol:.2f}")
        # Placeholder: real vol harvesting needs derivatives infra.
        return _hold(snapshot, self.name, self.base_description_bits,
                     f"VOLATILE detected ({p_vol:.2f}) but harvest infra is Phase 3+")


# ── 4. Liquidation avoider ───────────────────────────────────────────────

class LiquidationAvoider:
    """
    Specialist for BROKEN regime. Emits RISK_OFF when the posterior
    says BROKEN with any meaningful weight. This is asymmetric: a
    false positive costs a missed opportunity; a false negative can
    cost the whole book.

    7 bits: BROKEN threshold + asymmetric-cost rationale + override
    handling.
    """
    name: str = "liquidation_avoider"
    base_description_bits: float = 7.0

    def __init__(self, broken_threshold: float = 0.25):
        self.broken_threshold = broken_threshold

    def propose(self, snapshot: MarketSnapshot) -> ActionProposal:
        p_broken = snapshot.regime.probabilities.get(Regime.BROKEN, 0.0)
        if p_broken >= self.broken_threshold:
            return ActionProposal(
                hypothesis=self.name,
                symbol=snapshot.symbol,
                window_end=snapshot.window_end,
                action=ActionKind.RISK_OFF,
                size_fraction=Decimal(0),
                confidence=p_broken,
                description_length_bits=self.base_description_bits,
                rationale=f"BROKEN posterior {p_broken:.2f} ≥ {self.broken_threshold}",
            )
        return _hold(snapshot, self.name, self.base_description_bits,
                     f"BROKEN posterior {p_broken:.2f} below threshold")


# ── 5. Disagreement arbitrageur ──────────────────────────────────────────

class DisagreementArb:
    """
    Consumer of the Phase 0 disagreement signal. Uses the regime
    posterior's *entropy* as a proxy for ensemble uncertainty — when
    entropy is high (posterior is flat across regimes), the world is
    unclear and the disagreement arbitrage opportunity is highest.

    The most-complex hypothesis: combines two upstream signals
    (regime entropy, hypothesis disagreement) with two branches. ~9 bits.
    """
    name: str = "disagreement_arb"
    base_description_bits: float = 9.0

    def __init__(
        self,
        entropy_threshold: float = 1.3,    # log(5) ≈ 1.609 is uniform; 1.3 = high uncertainty
        size_at_max: Decimal = Decimal("0.2"),
    ):
        self.entropy_threshold = entropy_threshold
        self.size_at_max = size_at_max

    def propose(self, snapshot: MarketSnapshot) -> ActionProposal:
        regime_h = snapshot.regime.entropy_nats
        if regime_h < self.entropy_threshold:
            return _hold(snapshot, self.name, self.base_description_bits,
                         f"regime entropy {regime_h:.3f} below threshold {self.entropy_threshold}")

        # High regime uncertainty: take a small mean-reverting position
        # against the recent ensemble leader to harvest reversion.
        # Sign convention: if Phase 1's hypothesis ensemble leader is
        # itself a trend follower, we fade; if mean-reverter, we follow.
        leader = snapshot.hypotheses.leader
        if leader == "trend_follower":
            return ActionProposal(
                hypothesis=self.name,
                symbol=snapshot.symbol,
                window_end=snapshot.window_end,
                action=ActionKind.SHORT,
                size_fraction=-self.size_at_max,
                confidence=min(1.0, regime_h / 1.609),
                description_length_bits=self.base_description_bits,
                rationale=f"regime H={regime_h:.2f} high, fade trend_follower leader",
            )
        return _hold(snapshot, self.name, self.base_description_bits,
                     f"high entropy {regime_h:.2f} but leader is {leader}")


# ── Helpers ──────────────────────────────────────────────────────────────

def _hold(snapshot, name: str, bits: float, rationale: str) -> ActionProposal:
    return ActionProposal(
        hypothesis=name,
        symbol=snapshot.symbol,
        window_end=snapshot.window_end,
        action=ActionKind.HOLD,
        size_fraction=Decimal(0),
        confidence=0.0,
        description_length_bits=bits,
        rationale=rationale,
    )


# ── Default roster ───────────────────────────────────────────────────────

def default_roster() -> list:
    """The five Phase 2 v0.1 specialists. Order must match the Phase 1
    HypothesisEnsemble's hypothesis_names argument."""
    return [
        TrendFollower(),
        MeanReverter(),
        VolHarvester(),
        LiquidationAvoider(),
        DisagreementArb(),
    ]


def default_roster_names() -> tuple[str, ...]:
    """Convenience for wiring up Phase 1's ensemble."""
    return tuple(h.name for h in default_roster())
