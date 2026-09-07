"""Choose Alpaca vs fixture. Keys are never required to boot."""

from __future__ import annotations

from trading_system.config import Settings
from trading_system.providers.alpaca import AlpacaMarketDataProvider
from trading_system.providers.fixture import FixtureMarketDataProvider, FixtureProviderConfig
from trading_system.universe import load_universe


def make_market_provider(settings: Settings):
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        return AlpacaMarketDataProvider(settings)
    return FixtureMarketDataProvider(FixtureProviderConfig(provider_name="fixture"))


def universe_for_run(settings: Settings) -> tuple[str, ...]:
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        return load_universe(settings)
    return settings.smoke_universe
