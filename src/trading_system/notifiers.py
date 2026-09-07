"""Optional KakaoTalk 'Send to Me' adapter. Failures never block the app."""

from __future__ import annotations

from pathlib import Path

from trading_system.alerts import AlertRecord
from trading_system.config import Settings
from trading_system.credentials import env_path
from trading_system.kakao import notify_kakao as _notify
from trading_system.providers.never_block import NeverBlockResult


def notify_optional_channels(
    settings: Settings,
    alert: AlertRecord,
    *,
    env_file: Path | None = None,
) -> NeverBlockResult:
    """Send one alert to Kakao if configured. Settings is unused except cwd/.env."""
    _ = settings
    return _notify(alert, env_file=env_file or env_path())
