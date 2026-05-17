"""
ix_substrate.reconciliation.disagreement
─────────────────────────────────────────
The IX non-obvious gap (USPTO cluster G06Q40/04).

PRIOR ART
─────────
Prior multi-source financial data reconciliation patents and systems
optimize for *consensus*: cross-feed disagreement is treated as noise to
be averaged away (VWAP, mid-of-mids, Kalman smoothing of cross-venue
data, etc.). The objective is a single "true price" from which the user
trades.

THE GAP
───────
IX inverts this. Disagreement *is* the signal. When two well-functioning
feeds disagree on price beyond their expected spread and latency bands,
one of three things is true:

  1. ARBITRAGE  — there is a real, transient pricing dislocation.
  2. ASYMMETRY  — one feed has information the other does not yet have.
  3. REGIME     — the venues are sampling different liquidity pools that
                  are decoupling (e.g. CEX–DEX divergence under stress).

In any of these cases, averaging destroys the alpha. The substrate's job
is to *preserve* the disagreement as a typed event for downstream layers
(world model regime classifier, strategy synthesizer, execution router).

IMPLEMENTATION (this module)
────────────────────────────
PriceDisagreementDetector compares mid-prices across two synchronized
feeds, normalizes the deviation, subtracts the expected band, and emits a
DisagreementEvent only when the residual is positive. Half-life is
estimated from the time the deviation took to revert below threshold.

Future detectors (DirectionDisagreement, LatencyLead, LiquidityProfile)
share the same interface and may run concurrently per venue pair.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Iterable

from ..contracts.events import (
    DisagreementEvent,
    DisagreementKind,
    TickEvent,
)


# ── Configuration ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DisagreementConfig:
    """Detector parameters.

    threshold_bps:
        Minimum normalized price deviation, in basis points, *above the
        expected band*, before disagreement is reported. Below this is
        treated as microstructure noise.
    expected_band_multiplier:
        Multiplier applied to the larger of the two spreads to estimate
        the no-arbitrage band. 1.0 = trust spreads literally; >1.0 = pad
        for latency / fee uncertainty.
    window_size:
        Rolling window (number of aligned tick pairs) used to estimate
        persistence (half-life).
    """
    threshold_bps: Decimal = Decimal("3")
    expected_band_multiplier: Decimal = Decimal("1.5")
    window_size: int = 64


# ── Internal state ───────────────────────────────────────────────────────

@dataclass
class _PairState:
    """Rolling state for one (venue_a, venue_b, symbol) pair."""
    samples: deque[tuple[datetime, Decimal]] = field(default_factory=deque)
    last_emit_ts: datetime | None = None
    last_emit_mag: Decimal | None = None

    def push(self, ts: datetime, signed_residual_bps: Decimal, max_len: int) -> None:
        self.samples.append((ts, signed_residual_bps))
        while len(self.samples) > max_len:
            self.samples.popleft()


# ── Detector ─────────────────────────────────────────────────────────────

class PriceDisagreementDetector:
    """
    Streaming detector. Feed aligned tick pairs in, get DisagreementEvents
    out. Stateful across calls so it can estimate persistence.

    Usage:
        det = PriceDisagreementDetector()
        for tick_a, tick_b in aligned_stream:
            event = det.observe(tick_a, tick_b)
            if event is not None:
                emit(event)
    """

    def __init__(self, config: DisagreementConfig | None = None):
        self.config = config or DisagreementConfig()
        self._states: dict[tuple[str, str, str], _PairState] = {}

    # ---- core math (pure, easy to unit-test) ----

    def compute_signed_residual_bps(
        self,
        tick_a: TickEvent,
        tick_b: TickEvent,
    ) -> Decimal:
        """
        Signed disagreement residual in bps, *above* the expected
        no-arbitrage band. Positive = A higher than B beyond the band;
        negative = B higher than A beyond the band; zero = inside band.
        """
        if tick_a.symbol != tick_b.symbol:
            raise ValueError(
                f"symbol mismatch: {tick_a.symbol} vs {tick_b.symbol}"
            )
        if tick_a.venue == tick_b.venue:
            raise ValueError(f"identical venue: {tick_a.venue}")

        mid_a, mid_b = tick_a.mid, tick_b.mid
        if mid_a <= 0 or mid_b <= 0:
            return Decimal(0)

        # Normalized deviation in bps (signed: A - B).
        raw_bps = (mid_a - mid_b) / ((mid_a + mid_b) / Decimal(2)) * Decimal(10_000)

        # Expected no-arbitrage band: scaled by larger spread.
        band = (
            max(tick_a.spread_bps, tick_b.spread_bps)
            * self.config.expected_band_multiplier
        )

        # Residual = magnitude beyond the band, preserving sign.
        magnitude = abs(raw_bps)
        if magnitude <= band:
            return Decimal(0)
        sign = Decimal(1) if raw_bps > 0 else Decimal(-1)
        return sign * (magnitude - band)

    # ---- streaming interface ----

    def observe(
        self,
        tick_a: TickEvent,
        tick_b: TickEvent,
    ) -> DisagreementEvent | None:
        """Push one aligned pair, possibly return a DisagreementEvent."""
        key = self._key(tick_a, tick_b)
        state = self._states.setdefault(key, _PairState())

        # Use the *later* timestamp as the window stamp; conservative.
        ts = max(tick_a.ingress_ts, tick_b.ingress_ts)
        residual = self.compute_signed_residual_bps(tick_a, tick_b)
        state.push(ts, residual, self.config.window_size)

        # Disagreement requires at least two observations to be
        # statistically meaningful. The first observation populates state;
        # the second is the earliest moment we can emit.
        if len(state.samples) < 2:
            return None
        if abs(residual) < self.config.threshold_bps:
            return None

        half_life = self._estimate_half_life_ms(state, residual)
        venues = (tick_a.venue, tick_b.venue)

        event = DisagreementEvent(
            kind=DisagreementKind.PRICE,
            symbol=tick_a.symbol,
            venues=venues,
            window_start=state.samples[0][0],
            window_end=ts,
            magnitude=abs(residual),
            magnitude_unit="bps",
            leader=tick_a.venue if residual > 0 else tick_b.venue,
            half_life_ms=half_life,
            raw_observations=len(state.samples),
        )
        state.last_emit_ts = ts
        state.last_emit_mag = abs(residual)
        return event

    # ---- helpers ----

    @staticmethod
    def _key(a: TickEvent, b: TickEvent) -> tuple[str, str, str]:
        # Canonicalize order so (binance, coinbase) and (coinbase, binance)
        # accumulate into the same state.
        v1, v2 = sorted((a.venue, b.venue))
        return v1, v2, a.symbol

    @staticmethod
    def _estimate_half_life_ms(
        state: _PairState,
        current_residual: Decimal,
    ) -> float | None:
        """
        Walk back through the window. Half-life = elapsed time until the
        absolute residual was at most half of the current absolute
        residual. None if it never was within the window.
        """
        if not state.samples:
            return None
        target = abs(current_residual) / Decimal(2)
        current_ts = state.samples[-1][0]
        for ts, r in reversed(state.samples):
            if abs(r) <= target:
                return (current_ts - ts).total_seconds() * 1000.0
        return None


# ── Convenience: batch-process an iterable ───────────────────────────────

def detect_in_stream(
    pairs: Iterable[tuple[TickEvent, TickEvent]],
    config: DisagreementConfig | None = None,
) -> list[DisagreementEvent]:
    """Run a detector over a finite iterable of aligned pairs."""
    det = PriceDisagreementDetector(config)
    out: list[DisagreementEvent] = []
    for a, b in pairs:
        event = det.observe(a, b)
        if event is not None:
            out.append(event)
    return out
