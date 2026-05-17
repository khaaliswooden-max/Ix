"""
ix_world_model.inference.orchestrator
──────────────────────────────────────
Wires Phase 0 (Substrate) events to the regime classifier and the
hypothesis ensemble, producing MarketSnapshot events for Phase 2.

This module is the only place in Phase 1 that imports from Phase 0. All
other Phase 1 modules operate on internal feature/event types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Mapping

# Phase 0 contracts. In production this is a pip-installed dependency
# (ix-substrate); in this scaffold we expect it to be importable.
try:
    from ix_substrate.contracts.events import (
        DisagreementEvent,
        TickEvent,
    )
except ImportError:  # pragma: no cover
    # Allow Phase 1 tests to run without Phase 0 installed by providing
    # minimal duck-type shims. Real deployments require the real package.
    TickEvent = object        # type: ignore
    DisagreementEvent = object  # type: ignore

from ..contracts.events import (
    HypothesisWeights,
    MarketSnapshot,
    RegimePosterior,
)
from ..hypotheses.ensemble import EnsembleConfig, HypothesisEnsemble
from ..regimes.classifier import (
    ClassifierConfig,
    DEFAULT_PROFILES,
    RegimeClassifier,
    RegimeProfile,
)
from ..regimes.features import FeatureConfig, FeatureExtractor


# ── Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorldModelConfig:
    symbol: str
    hypothesis_names: tuple[str, ...]
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    classifier_config: ClassifierConfig = field(default_factory=ClassifierConfig)
    ensemble_config: EnsembleConfig = field(default_factory=EnsembleConfig)


# ── Orchestrator ─────────────────────────────────────────────────────────

class WorldModel:
    """
    Per-symbol orchestrator. Feed it Phase 0 events, get Phase 1
    MarketSnapshots out.

    Usage:
        wm = WorldModel(WorldModelConfig(
            symbol="BTC/USDT",
            hypothesis_names=("trend", "meanrev", "carry"),
        ))

        # As Phase 0 events arrive:
        wm.observe_tick(tick)         # TickEvent from Phase 0
        wm.observe_disagreement(dis)  # DisagreementEvent from Phase 0

        # Phase 2 / Phase 3 reports realized hypothesis PnL each step:
        wm.report_hypothesis_pnl({"trend": ..., "meanrev": ..., "carry": ...})

        # Then read the current snapshot:
        snapshot = wm.snapshot(now)
    """

    def __init__(
        self,
        config: WorldModelConfig,
        profiles: Mapping = DEFAULT_PROFILES,
    ):
        self.cfg = config
        self._features = FeatureExtractor(
            symbol=config.symbol, config=config.feature_config
        )
        self._classifier = RegimeClassifier(
            symbol=config.symbol,
            profiles=dict(profiles),
            config=config.classifier_config,
        )
        self._ensemble = HypothesisEnsemble(
            symbol=config.symbol,
            hypothesis_names=list(config.hypothesis_names),
            config=config.ensemble_config,
        )
        self._last_regime: RegimePosterior | None = None
        self._last_timestamp: datetime | None = None

    # ── Phase 0 inputs ─────────────────────────────────────────

    def observe_tick(self, tick) -> RegimePosterior | None:
        """
        Consume a TickEvent. Returns updated RegimePosterior if enough
        data has accumulated, else None.
        """
        if tick.symbol != self.cfg.symbol:
            return None  # silently ignore other symbols

        fv = self._features.observe_tick(
            timestamp=tick.ingress_ts,
            mid=tick.mid,
            spread_bps=tick.spread_bps,
        )
        if fv is None:
            return None

        posterior = self._classifier.observe(fv)
        self._last_regime = posterior
        self._last_timestamp = fv.timestamp
        return posterior

    def observe_disagreement(self, event) -> None:
        """
        Consume a DisagreementEvent. This enriches the feature stream
        with cross-feed disagreement intensity. Does not directly emit a
        posterior — its effect is felt on the next tick.
        """
        if event.symbol != self.cfg.symbol:
            return
        self._features.observe_disagreement(event.magnitude)

    # ── Hypothesis-PnL inputs (from Phase 2/3) ─────────────────

    def report_hypothesis_pnl(self, pnl_by_hypothesis: Mapping[str, Decimal]) -> None:
        self._ensemble.report_pnl(pnl_by_hypothesis)

    # ── Outputs to Phase 2 ────────────────────────────────────

    def snapshot(self, window_end: datetime | None = None) -> MarketSnapshot | None:
        """
        Bundle the latest regime posterior + hypothesis weights into a
        MarketSnapshot. Returns None if no regime posterior is yet
        available (cold-start).

        If window_end is provided, it is used for both sub-contracts so
        the snapshot validates. If omitted, uses the last-observed tick
        timestamp.
        """
        if self._last_regime is None or self._last_timestamp is None:
            return None
        ts = window_end or self._last_timestamp

        # Re-emit regime posterior with the requested window_end so it
        # aligns with hypothesis weights for snapshot validation.
        regime_aligned = RegimePosterior(
            symbol=self._last_regime.symbol,
            window_end=ts,
            probabilities=self._last_regime.probabilities,
            observations_used=self._last_regime.observations_used,
            half_life_observations=self._last_regime.half_life_observations,
        )
        hyp = self._ensemble.weights(window_end=ts)

        return MarketSnapshot(
            symbol=self.cfg.symbol,
            window_end=ts,
            regime=regime_aligned,
            hypotheses=hyp,
        )

    # ── Inspection helpers ────────────────────────────────────

    @property
    def latest_regime(self) -> RegimePosterior | None:
        return self._last_regime
