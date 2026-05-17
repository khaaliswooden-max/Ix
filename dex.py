"""
ix_substrate.reconciliation.clock
──────────────────────────────────
Per-venue clock-skew tracking. Skew (ingress_ts - venue_ts) is two things
at once: a data-quality signal (something is wrong if it drifts) and an
alpha signal (a venue that is persistently *late* may be reflecting other
venues' price moves with delay → leader/follower regime).

Phase 0 maintains an exponentially-weighted moving average of skew per
venue. Downstream layers can subscribe to step-changes via the same
DisagreementEvent contract with kind=LATENCY (left as Phase 0.x work).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts.events import TickEvent


@dataclass
class _SkewEMA:
    alpha: float = 0.05
    mean_ms: float = 0.0
    initialized: bool = False

    def update(self, sample_ms: float) -> float:
        if not self.initialized:
            self.mean_ms = sample_ms
            self.initialized = True
        else:
            self.mean_ms = self.alpha * sample_ms + (1 - self.alpha) * self.mean_ms
        return self.mean_ms


class ClockSkewTracker:
    """One EMA per venue. Thread-safety: caller's responsibility."""

    def __init__(self, alpha: float = 0.05):
        self._alpha = alpha
        self._by_venue: dict[str, _SkewEMA] = {}

    def observe(self, tick: TickEvent) -> float:
        """Returns the updated EMA in ms."""
        ema = self._by_venue.setdefault(tick.venue, _SkewEMA(alpha=self._alpha))
        return ema.update(tick.skew_ms)

    def get(self, venue: str) -> float | None:
        ema = self._by_venue.get(venue)
        return ema.mean_ms if ema and ema.initialized else None

    def snapshot(self) -> dict[str, float]:
        return {v: e.mean_ms for v, e in self._by_venue.items() if e.initialized}
