"""Market-data provider package (interfaces + never-block helpers)."""

from trading_system.providers.alpaca import AlpacaMarketDataProvider
from trading_system.providers.batch import DailyBarsBatch, QuotesBatch
from trading_system.providers.fallback import FallbackChainProvider
from trading_system.providers.fixture import (
    FixtureBarSpec,
    FixtureMarketDataProvider,
    FixtureProviderConfig,
    FixtureQuoteSpec,
    make_smoke_fixture_bars,
)
from trading_system.providers.interfaces import (
    BtcMarketDataProvider,
    CoverageRow,
    CoverageStatus,
    DailyBar,
    EquityUniverseProvider,
    HistoricalDailyBarsProvider,
    LatestQuote,
    LatestQuoteProvider,
    UnimplementedProvider,
)
from trading_system.providers.never_block import (
    Deadline,
    NeverBlockResult,
    RetryKind,
    RetryPolicy,
    classify_http_status,
    run_with_deadline,
)

__all__ = [
    "AlpacaMarketDataProvider",
    "BtcMarketDataProvider",
    "CoverageRow",
    "CoverageStatus",
    "DailyBar",
    "DailyBarsBatch",
    "Deadline",
    "EquityUniverseProvider",
    "FallbackChainProvider",
    "FixtureBarSpec",
    "FixtureMarketDataProvider",
    "FixtureProviderConfig",
    "FixtureQuoteSpec",
    "HistoricalDailyBarsProvider",
    "LatestQuote",
    "LatestQuoteProvider",
    "NeverBlockResult",
    "QuotesBatch",
    "RetryKind",
    "RetryPolicy",
    "UnimplementedProvider",
    "classify_http_status",
    "make_smoke_fixture_bars",
    "run_with_deadline",
]
