"""
tests/test_alignment.py
────────────────────────
Cross-feed tick alignment: ensure we emit pairs only when both venues
have a recent tick in the same bucket.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from ix_substrate.contracts.events import FeedKind, TickEvent
from ix_substrate.reconciliation.alignment import AlignConfig, align_two


def _tick(venue: str, ts: datetime) -> TickEvent:
    return TickEvent(
        feed_kind=FeedKind.CEX, venue=venue, symbol="BTC/USDT",
        venue_ts=ts, ingress_ts=ts,
        bid_price=Decimal("99"), bid_size=Decimal("1"),
        ask_price=Decimal("101"), ask_size=Decimal("1"),
    )


def test_emits_pair_when_both_in_same_bucket():
    t0 = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    a = [_tick("binance",  t0 + timedelta(milliseconds=50))]
    b = [_tick("coinbase", t0 + timedelta(milliseconds=200))]
    out = list(align_two(a, b, AlignConfig(bucket_ms=250)))
    assert len(out) == 1
    pair = out[0]
    assert {pair[0].venue, pair[1].venue} == {"binance", "coinbase"}


def test_no_pair_when_buckets_disjoint():
    t0 = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    a = [_tick("binance",  t0 + timedelta(milliseconds=10))]
    b = [_tick("coinbase", t0 + timedelta(milliseconds=500))]
    out = list(align_two(a, b, AlignConfig(bucket_ms=250)))
    assert out == []


def test_latest_in_bucket_wins():
    t0 = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    a = [
        _tick("binance",  t0 + timedelta(milliseconds=10)),
        _tick("binance",  t0 + timedelta(milliseconds=200)),
    ]
    b = [_tick("coinbase", t0 + timedelta(milliseconds=150))]
    out = list(align_two(a, b, AlignConfig(bucket_ms=250)))
    assert len(out) == 1
    pair_by_venue = {p.venue: p for p in out[0]}
    # binance contributed the LATER of its two ticks
    assert pair_by_venue["binance"].ingress_ts == t0 + timedelta(milliseconds=200)
