# ix-substrate

**Phase 0 of the IX agent — the market-data substrate.**

This is the lowest layer of the seven-layer IX architecture described in
the [IX whitepaper](https://github.com/visionblox/ix-whitepaper). It
ingests market data from multiple venues, reconciles cross-feed
observations onto a common time grid, and emits two contract types
downstream:

- **`TickEvent`** — best bid / best ask from one venue at one moment.
- **`DisagreementEvent`** — a typed observation that two feeds disagree
  beyond their expected no-arbitrage band. **This is the non-obvious
  gap** that distinguishes IX from prior-art reconciliation systems
  (USPTO cluster G06Q40/04): instead of averaging disagreement away to
  produce a "true price," IX preserves disagreement as a first-class
  signal for downstream layers.

See [`docs/ADR-0002-disagreement-as-alpha.md`](docs/ADR-0002-disagreement-as-alpha.md)
for the architectural rationale.

## Quick start

```bash
# Install (Python 3.11+)
pip install -e ".[dev,cex]"

# Verify the math without touching the network
python examples/run_btc_reconciliation.py

# Run the tests
pytest -v

# Run the live substrate against the example config (real CEX REST calls)
ix-substrate run --config configs/substrate.example.yaml
```

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │                Phase 0 — Substrate           │
                │                                              │
   ┌────────┐   │   ┌─────────────┐    ┌──────────────────┐    │
   │  CEX   │───┼──▶│  feeds.cex  │───▶│                  │    │
   │ (ccxt) │   │   └─────────────┘    │   alignment      │    │
   └────────┘   │                      │   (time-bucket)  │    │
                │   ┌─────────────┐    │                  │    │
   ┌────────┐   │   │  feeds.dex  │───▶│                  │    │
   │  DEX   │───┼──▶│  (web3, 0.2)│    └────────┬─────────┘    │
   │(web3)  │   │   └─────────────┘             │              │
   └────────┘   │                               ▼              │
                │                      ┌──────────────────┐    │
   ┌────────┐   │   ┌──────────────┐   │  disagreement    │    │
   │chain-  │   │   │feeds.onchain │   │  detector        │    │
   │events  │───┼──▶│  (web3, 0.3) │   │   (the gap)      │    │
   └────────┘   │   └──────────────┘   └────────┬─────────┘    │
                │                               │              │
                │             ┌─────────────────┴───────────┐  │
                │             ▼                             ▼  │
                │   ┌──────────────────┐         ┌──────────────────┐
                │   │ ticks table      │         │ disagreements    │
                │   │ (DuckDB / CH)    │         │ table (DuckDB)   │
                │   └──────────────────┘         └──────────────────┘
                └──────────────────────────────────────────────┘
                                       │
                                       ▼
                              Phase 1 — World Model
                              (consumes both contracts)
```

## Package layout

```
src/ix_substrate/
├── __init__.py
├── cli.py                       # `ix-substrate run …`
├── config.py                    # YAML config schema
├── contracts/
│   └── events.py                # TickEvent, DisagreementEvent  (the contract surface)
├── feeds/
│   ├── base.py                  # Feed protocol
│   ├── cex.py                   # working: ccxt REST-polled CEX feeds
│   ├── dex.py                   # extension point — Phase 0.2
│   └── onchain.py               # extension point — Phase 0.3
├── reconciliation/
│   ├── alignment.py             # time-bucket pairing of cross-feed ticks
│   ├── clock.py                 # per-venue clock-skew EMA
│   └── disagreement.py          # PriceDisagreementDetector (the non-obvious gap)
└── storage/
    ├── schema.py                # Arrow schemas
    └── duckdb_store.py          # local sink; swap to ClickHouse for prod
```

## Roadmap

- **v0.1 (this drop)**: Working CEX → align → price-disagreement → DuckDB.
- **v0.2**: DEX feeds (Uniswap V3 first); `DirectionDisagreement` and
  `LatencyLead` detectors; ClickHouse adapter; metrics endpoint.
- **v0.3**: On-chain event feed; `LiquidityDisagreement` detector;
  Parquet archival sink for historical replay.
- **v1.0**: Phase 0 frozen contract surface that Phase 1 (world model)
  can depend on without churn.

## License

Proprietary. © 2026 Zuup Innovation Lab / Visionblox LLC.
