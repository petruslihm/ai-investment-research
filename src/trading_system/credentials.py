"""First-run / Settings persistence for API keys. Never required to start the app."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

ALLOWED_KEYS = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "MARKET_DATA_PROVIDER",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "KAKAO_REST_API_KEY",
    "KAKAO_CLIENT_SECRET",
    "KAKAO_REDIRECT_URI",
    "KAKAO_ACCESS_TOKEN",
    "KAKAO_REFRESH_TOKEN",
    "KAKAO_ACCESS_EXPIRES_AT",
    "KAKAO_NICKNAME",
    "SCAN_UNIVERSE_PRESET",
    "UNIVERSE_REFRESH_DAYS",
    "RISK_APPETITE",
)
_ENV_WRITE_LOCK = threading.Lock()


def env_path(root: Path | None = None) -> Path:
    return (root or Path.cwd()) / ".env"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a crash mid-write can't blank out every stored key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_env_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def _reject_control_chars(key: str, value: str) -> None:
    # read_env_map is line-based (split on "\n", "k=v" per line): a value containing
    # CR/LF can inject an extra unauthorized "line" the next time this file is read,
    # bypassing ALLOWED_KEYS entirely since it's never re-validated on read. NUL can
    # corrupt line parsing on some platforms too.
    if any(ch in value for ch in ("\r", "\n", "\x00")):
        raise ValueError(f"value for {key} contains a disallowed control character")


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    with _ENV_WRITE_LOCK:
        current = read_env_map(path)
        for k, v in updates.items():
            if k not in ALLOWED_KEYS:
                raise ValueError(f"disallowed env key: {k}")
            if v != "":
                _reject_control_chars(k, v)
                current[k] = v
        lines = [f"{k}={current[k]}" for k in sorted(current)]
        _atomic_write_text(path, "\n".join(lines) + "\n")


def clear_env_keys(path: Path, keys: tuple[str, ...] | list[str]) -> None:
    with _ENV_WRITE_LOCK:
        current = read_env_map(path)
        for k in keys:
            if k not in ALLOWED_KEYS:
                raise ValueError(f"disallowed env key: {k}")
            current.pop(k, None)
        if not current:
            if path.exists():
                _atomic_write_text(path, "")
            return
        lines = [f"{k}={current[k]}" for k in sorted(current)]
        _atomic_write_text(path, "\n".join(lines) + "\n")


def missing_provider_keys(values: dict[str, str]) -> list[str]:
    missing = []
    if not values.get("ALPACA_API_KEY") or not values.get("ALPACA_SECRET_KEY"):
        missing.append("Alpaca market data")
    if not any(values.get(k) for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")):
        missing.append("LLM (optional extract/judge)")
    return missing
