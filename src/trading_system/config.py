"""Pydantic settings for the investment assistant."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_SMOKE_UNIVERSE: tuple[str, ...] = (
    "SPY",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BTC/USD",
)

UIStackChoice = Literal["fastapi_vite_react"]
RiskAppetite = Literal["aggressive", "balanced", "conservative"]


class Settings(BaseSettings):
    """Application configuration (env / .env)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        allow_inf_nan=False,
    )

    # History / cadence
    history_years: int = Field(default=3, ge=1)
    live_refresh_seconds: int = Field(default=180, ge=30)
    broad_scan_seconds: int = Field(default=900, ge=60)
    # Weekday NYSE pre-open scan (in-process; no second DuckDB writer).
    daily_scan_enabled: bool = True
    daily_scan_minutes_before_open: int = Field(default=60, ge=0, le=180)
    # train_only (see run_v1_cycle): model refit + ensemble/horizon-influence
    # adaptation + online SGD updates, no SEC/Gemini/GPT calls -- $0, ~30s. Safe to
    # run every session instead of weekly (see due_daily_train's docstring).
    daily_train_enabled: bool = True
    daily_train_minutes_after_close: int = Field(default=60, ge=0, le=180)
    max_live_equity_symbols: int = Field(default=30, ge=1)

    # Assets
    btc_symbol: str = "BTC/USD"
    usd_cash_asset: str = "USD_CASH"
    # Development/test smoke list only. Production scans use scan_universe_preset.
    smoke_universe: tuple[str, ...] = DEFAULT_SMOKE_UNIVERSE

    # Production scan universe
    scan_universe_preset: str = "large_liquid_500"
    universe_refresh_days: int = Field(default=7, ge=1)

    # Multi-horizon defaults (equity sessions vs BTC calendar days applied later)
    horizons: tuple[int, ...] = (5, 10, 20)

    # Determinism
    random_seed: int = 42

    # Storage
    duckdb_path: Path = Path("data/investment_assistant.duckdb")
    writer_lease_stale_seconds: int = Field(default=30, ge=5)
    job_lease_stale_seconds: int = Field(default=120, ge=10)

    # Timezones (discipline from legacy)
    user_tz: str = "Asia/Seoul"
    market_tz: str = "America/New_York"

    # Optional Alpaca market-data credentials (no trading client)
    market_data_provider: str = "alpaca"
    alpaca_api_key: str | None = None
    alpaca_secret_key: str | None = None

    # Optional LLM credentials
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_judge_model: str = "gpt-5.6-sol"
    gemini_research_model: str = "gemini-3.6-flash"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_final_max_names: int = Field(default=7, ge=1, le=15)
    llm_research_max_names: int = Field(default=16, ge=1, le=40)
    llm_daily_budget_usd: float = Field(
        default=15.0,
        ge=0.0,
        description=(
            "Soft USD budget threshold for automatic calls per UTC day; actual usage is known after each call. "
            "Manual Run once is not counted toward it."
        ),
    )
    # 15, not the old 3: one complete GPT judgement measured $10.92 end to end
    # (2026-09-04: 9 turns, 2.13M prompt + 121K completion tokens on gpt-5.6-sol).
    # At $3 an automatic scan could afford ~2 of those 9 turns, so it could never
    # produce a final judgement -- it only appeared to work because the old
    # pre-flight guard estimated the whole conversation at $0.80 and waved it
    # through. $15 covers one full judgement plus a partial resumed retry.
    gemini_web_search: bool = True
    gemini_deadline_seconds: float = Field(default=300.0, ge=30.0, le=600.0)
    sol_judge_deadline_seconds: float = Field(
        default=900.0,
        ge=30.0,
        le=1800.0,
        description=(
            "Per-turn deadline for legacy_b_raw_conversation's OpenAI calls (each of the "
            "6+ chained turns gets this much wall-clock time). Raised from the old 240s "
            "default because turns with web_search + reasoning routinely exceeded it and "
            "failed with 'deadline expired' before producing any answer."
        ),
    )

    # Optional KakaoTalk personal notifications (Send to Me)
    kakao_rest_api_key: str | None = None
    kakao_client_secret: str | None = None
    kakao_redirect_uri: str | None = None

    # Quant allocation (unit-first). Weights are fractions of total_base_units.
    min_opportunity_score: float = Field(default=0.01, ge=0.0)
    max_single_stock_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    max_total_stock_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    max_btc_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    min_cash_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    min_position_weight: float = Field(default=0.01, ge=0.0, le=1.0)
    max_equity_positions: int = Field(default=20, ge=1, le=100)
    # V1 stand-in, not OOS-calibrated. "Strong" = this 5/10/20 mean expected return
    # at conf=1, vol=0.02, agreeing horizons, converted through opportunity_score().
    strong_horizon_mean_return: float = Field(default=0.04, ge=0.0)
    # Optional override of the implied opportunity_score. None = derive from the mean above.
    strong_opportunity_score: float | None = Field(default=None, ge=0.0)
    # Optional. None = from risk_appetite (aggressive 1, balanced 2, conservative = cap ratio).
    full_deployment_equivalent_names: float | None = Field(default=None, ge=0.0)
    # Personal book-size stance. Does not change model predictions or the 0.01 eligibility gate.
    # aggressive: one fully-strong name can fill the equity cap; excess^1.5 sizing.
    risk_appetite: RiskAppetite = "aggressive"
    # None = from risk_appetite. 0 = equal weight among passers, 1 = excess-proportional, >1 concentrates.
    sizing_alpha: float | None = Field(default=None, ge=0.0, le=4.0)
    btc_min_delta_units: float = 0.02
    btc_hysteresis: float = 0.03

    # User-facing actionability (does not change model predictions).
    # Fractions of total_base_units. 4.28u on a 1200 book is 0.36% → 관망.
    min_actionable_weight: float = Field(default=0.01, ge=0.0, le=1.0)
    min_delta_weight: float = Field(default=0.005, ge=0.0, le=1.0)
    display_unit_step_weight: float = Field(default=0.001, ge=0.0, le=1.0)
    min_actionable_units_floor: float = Field(default=0.0, ge=0.0)

    # UI stack choice (recorded for V1)
    ui_stack: UIStackChoice = "fastapi_vite_react"

    @field_validator("risk_appetite", mode="before")
    @classmethod
    def _parse_appetite(cls, value: object) -> object:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"aggressive", "balanced", "conservative"}:
                return key
        return value

    @field_validator("horizons", mode="before")
    @classmethod
    def _parse_horizons(cls, value: object) -> object:
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return tuple(int(p) for p in parts)
        return value

    @field_validator("horizons")
    @classmethod
    def _valid_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(h <= 0 for h in value):
            raise ValueError("horizons must contain positive day counts")
        if len(set(value)) != len(value):
            raise ValueError("horizons must not contain duplicates")
        return value

    @field_validator("smoke_universe", mode="before")
    @classmethod
    def _parse_universe(cls, value: object) -> object:
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return tuple(parts)
        return value

    @property
    def btc_instrument_symbol(self) -> str:
        return self.btc_symbol

    def today_in_user_tz(self) -> date:
        """Calendar date in the user's timezone (default Asia/Seoul)."""
        try:
            tz = ZoneInfo(self.user_tz)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")
        return datetime.now(tz).date()


def discover_project_root() -> Path:
    """Repo root (pyproject.toml + src/trading_system), not site-packages or src/."""
    here = Path(__file__).resolve()
    ordered: list[Path] = [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]
    seen: set[Path] = set()
    for raw in ordered:
        p = raw if raw.is_dir() else raw.parent
        if p in seen:
            continue
        seen.add(p)
        if (p / "pyproject.toml").is_file() and (p / "src" / "trading_system").is_dir():
            return p
    return Path.cwd()


def get_settings() -> Settings:
    env = discover_project_root() / ".env"
    if env.is_file():
        return Settings(_env_file=env)
    return Settings(_env_file=None)
