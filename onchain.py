"""
ix_substrate.reconciliation.alignment
──────────────────────────────────────
Pair up ticks from different feeds onto a common time grid so they can be
compared. The substrate is jitter-tolerant: feeds arrive at different
cadences and with venue-side clock skew, so naïve "match by exact
timestamp" is hopeless.

Strategy: bucket by ingress_ts modulo `bucket_ms`. Within each bucket,
take the most recent tick from each feed. Emit one (a, b) pair per bucket
in which both feeds are represented.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Iterator

from ..contracts.events import TickEvent


@dataclass(frozen=True)
class AlignConfig:
    bucket_ms: int = 250        # 4 buckets / second by default
    venues: tuple[str, str] | None = None  # if set, only pair these two


def _bucket_key(ts: datetime, bucket_ms: int) -> int:
    epoch_ms = int(ts.replace(tzinfo=ts.tzinfo or timezone.utc).timestamp() * 1000)
    return epoch_ms // bucket_ms


def align_two(
    stream_a: Iterable[TickEvent],
    stream_b: Iterable[TickEvent],
    config: AlignConfig | None = None,
) -> Iterator[tuple[TickEvent, TickEvent]]:
    """
    Merge two pre-sorted iterables (sorted by ingress_ts) into aligned
    pairs. Assumes single-symbol streams (caller filters upstream).
    """
    cfg = config or AlignConfig()
    bucket: dict[str, TickEvent] = {}
    current_key: int | None = None
    iters = (iter(stream_a), iter(stream_b))
    sentinels = [None, None]

    def _advance(i: int) -> TickEvent | None:
        try:
            return next(iters[i])
        except StopIteration:
            return None

    # prime
    sentinels[0] = _advance(0)
    sentinels[1] = _advance(1)

    while sentinels[0] is not None or sentinels[1] is not None:
        # Pick the earlier of the two.
        candidates = [(i, s) for i, s in enumerate(sentinels) if s is not None]
        idx, tick = min(candidates, key=lambda x: x[1].ingress_ts)

        key = _bucket_key(tick.ingress_ts, cfg.bucket_ms)
        if current_key is None:
            current_key = key

        if key != current_key:
            # bucket boundary -> emit if both venues present
            if len(bucket) == 2:
                ven_a, ven_b = sorted(bucket.keys())
                yield bucket[ven_a], bucket[ven_b]
            bucket.clear()
            current_key = key

        # latest-write-wins within the bucket for the venue
        bucket[tick.venue] = tick
        sentinels[idx] = _advance(idx)

    # final bucket
    if len(bucket) == 2:
        ven_a, ven_b = sorted(bucket.keys())
        yield bucket[ven_a], bucket[ven_b]
