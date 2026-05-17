"""
ix_world_model.regimes.features
────────────────────────────────
Online feature extraction from Phase 0 tick streams. These features feed
the regime classifier.

Design principle: every feature here is computable from a sliding window
in O(1) amortized per tick. No batch recomputation. This is what makes
the world model run at substrate cadence rather than batch-overnight.

Features in v0.1:
  log_return           -- mid-to-mid log return per tick
  realized_vol         -- rolling std of log returns, annualized
  autocorr_lag1        -- first-order serial correlation of log returns
                          (positive = momentum, negative = mean reversion)
  spread_bps_mean      -- rolling mean of spread; widening = stress
  disagreement_intensity -- rolling mean magnitude of cross-feed
                          disagreement (the Phase 0 dividend)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Iterable


# ── Online statistics primitives ─────────────────────────────────────────

@dataclass
class _RollingStats:
    """Welford-style rolling mean and variance over a fixed window."""
    window: int
    samples: deque[float] = field(default_factory=deque)
    _sum: float = 0.0
    _sumsq: float = 0.0

    def push(self, x: float) -> None:
        self.samples.append(x)
        self._sum += x
        self._sumsq += x * x
        if len(self.samples) > self.window:
            old = self.samples.popleft()
            self._sum -= old
            self._sumsq -= old * old

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n > 0 else 0.0

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        # Sample variance (n-1 denominator); numerically guarded.
        var = (self._sumsq - self.n * m * m) / (self.n - 1)
        return max(var, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class _RollingAutocorr:
    """Lag-1 autocorrelation over a rolling window of returns."""
    window: int
    samples: deque[float] = field(default_factory=deque)

    def push(self, x: float) -> None:
        self.samples.append(x)
        if len(self.samples) > self.window:
            self.samples.popleft()

    @property
    def lag1(self) -> float:
        n = len(self.samples)
        if n < 3:
            return 0.0
        m = sum(self.samples) / n
        num = 0.0
        den = 0.0
        prev = None
        for x in self.samples:
            if prev is not None:
                num += (x - m) * (prev - m)
            den += (x - m) ** 2
            prev = x
        if den == 0:
            return 0.0
        return max(-1.0, min(1.0, num / den))


# ── Feature record ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureVector:
    """One row of features at one moment. Consumed by the regime classifier."""
    symbol: str
    timestamp: datetime
    log_return: float
    realized_vol_annualized: float
    autocorr_lag1: float
    spread_bps_mean: float
    disagreement_intensity_bps: float
    observations_used: int


# ── Feature extractor ────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    return_window: int = 256
    vol_window: int = 256
    autocorr_window: int = 64
    spread_window: int = 64
    disagreement_window: int = 64
    # Used to annualize vol. For tick-based data, this is how many ticks
    # we treat as one "year." Defaults assume ~1Hz cadence, 24h trading,
    # 365 days/year: 365 * 24 * 3600 = 31.5M, but real-world strategies
    # rarely care about second-by-second vol — set this to your decision
    # cadence's annualization factor.
    annualization_factor: float = 365.0 * 24.0 * 60.0  # treats ticks as ~minutes


class FeatureExtractor:
    """
    Stateful, per-symbol. One extractor per symbol per venue (or, if you
    prefer, per symbol against a venue-consensus mid).
    """

    def __init__(self, symbol: str, config: FeatureConfig | None = None):
        self.symbol = symbol
        self.cfg = config or FeatureConfig()
        self._last_mid: Decimal | None = None
        self._returns = _RollingStats(window=self.cfg.return_window)
        self._vol = _RollingStats(window=self.cfg.vol_window)
        self._autocorr = _RollingAutocorr(window=self.cfg.autocorr_window)
        self._spread = _RollingStats(window=self.cfg.spread_window)
        self._disagreement = _RollingStats(window=self.cfg.disagreement_window)

    # Tick from Phase 0: we extract mid, spread.
    def observe_tick(
        self,
        timestamp: datetime,
        mid: Decimal,
        spread_bps: Decimal,
    ) -> FeatureVector | None:
        """
        Returns a FeatureVector once enough data has accumulated to compute
        all features, else None. (Cold-start protection.)
        """
        if self._last_mid is None or self._last_mid <= 0 or mid <= 0:
            self._last_mid = mid
            self._spread.push(float(spread_bps))
            return None

        r = math.log(float(mid) / float(self._last_mid))
        self._last_mid = mid

        self._returns.push(r)
        self._vol.push(r)
        self._autocorr.push(r)
        self._spread.push(float(spread_bps))

        if self._returns.n < 4:
            return None

        return FeatureVector(
            symbol=self.symbol,
            timestamp=timestamp,
            log_return=r,
            realized_vol_annualized=self._vol.std * math.sqrt(self.cfg.annualization_factor),
            autocorr_lag1=self._autocorr.lag1,
            spread_bps_mean=self._spread.mean,
            disagreement_intensity_bps=self._disagreement.mean,
            observations_used=self._returns.n,
        )

    # Disagreement event from Phase 0: enriches the feature vector.
    def observe_disagreement(self, magnitude_bps: Decimal) -> None:
        self._disagreement.push(float(magnitude_bps))

    def feed_iterable(
        self,
        ticks: Iterable[tuple[datetime, Decimal, Decimal]],
    ) -> Iterable[FeatureVector]:
        """Convenience: pump an iterable through and yield non-None features."""
        for ts, mid, spread in ticks:
            fv = self.observe_tick(ts, mid, spread)
            if fv is not None:
                yield fv
