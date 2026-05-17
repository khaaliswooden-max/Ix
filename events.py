"""
ix_substrate.feeds.cex
───────────────────────
Centralized-exchange feed backed by ccxt's async API. One CexFeed instance
streams best bid/ask for one (venue, symbol) pair.

Notes:
- ccxt's REST polling is intentional here for v0.1: WebSocket streams via
  ccxt.pro are paid. The substrate's reconciliation logic doesn't depend
  on WebSocket cadence; for tight microstructure work, upgrade to ccxt.pro
  or per-venue native WS clients in v0.2.
- All public-market endpoints used here are unauthenticated.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncIterator

from ..contracts.events import FeedKind, TickEvent


class CexFeed:
    """REST-polled best bid/ask feed for one venue + one symbol."""

    def __init__(
        self,
        venue: str,
        symbol: str,
        poll_ms: int = 1000,
        ccxt_kwargs: dict | None = None,
    ):
        self.venue = venue
        self.symbols = (symbol,)
        self._symbol = symbol
        self._poll_ms = poll_ms
        self._ccxt_kwargs = ccxt_kwargs or {}
        self._client = None

    async def __aenter__(self) -> "CexFeed":
        # Imported lazily so the package imports without ccxt installed
        # (e.g. in CI where we only want to run unit tests on disagreement).
        import ccxt.async_support as ccxt  # type: ignore

        klass = getattr(ccxt, self.venue, None)
        if klass is None:
            raise ValueError(f"ccxt has no exchange called {self.venue!r}")
        self._client = klass({"enableRateLimit": True, **self._ccxt_kwargs})
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def stream(self) -> AsyncIterator[TickEvent]:
        if self._client is None:
            raise RuntimeError("CexFeed must be used as an async context manager")

        while True:
            order_book = await self._client.fetch_order_book(self._symbol, limit=1)
            ingress = datetime.now(timezone.utc)
            bids = order_book.get("bids") or []
            asks = order_book.get("asks") or []
            if not bids or not asks:
                await asyncio.sleep(self._poll_ms / 1000.0)
                continue

            venue_ts_raw = order_book.get("timestamp")
            venue_ts = (
                datetime.fromtimestamp(venue_ts_raw / 1000.0, tz=timezone.utc)
                if venue_ts_raw is not None
                else ingress
            )
            bp, bs = bids[0]
            ap, as_ = asks[0]

            yield TickEvent(
                feed_kind=FeedKind.CEX,
                venue=self.venue,
                symbol=self._symbol,
                venue_ts=venue_ts,
                ingress_ts=ingress,
                bid_price=Decimal(str(bp)),
                bid_size=Decimal(str(bs)),
                ask_price=Decimal(str(ap)),
                ask_size=Decimal(str(as_)),
            )
            await asyncio.sleep(self._poll_ms / 1000.0)
