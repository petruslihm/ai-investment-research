"""credentials.py writes must be atomic: a crash mid-write must not blank out every
previously stored key, and must not leave a stray temp file behind."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from trading_system.config import Settings
from trading_system.credentials import clear_env_keys, read_env_map, upsert_env
from trading_system.ui import app as ui_app


def test_upsert_env_round_trips_and_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    upsert_env(path, {"OPENAI_API_KEY": "sk-a"})
    upsert_env(path, {"GEMINI_API_KEY": "gm-b"})
    values = read_env_map(path)
    assert values["OPENAI_API_KEY"] == "sk-a"
    assert values["GEMINI_API_KEY"] == "gm-b"
    leftover = list(tmp_path.glob(".env.*.tmp"))
    assert leftover == []


def test_upsert_env_failure_mid_write_preserves_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    upsert_env(path, {"OPENAI_API_KEY": "sk-a"})
    original = path.read_text(encoding="utf-8")

    def _boom(*a: object, **k: object) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr("os.replace", _boom)
    with pytest.raises(OSError):
        upsert_env(path, {"GEMINI_API_KEY": "gm-b"})

    # The live .env must be untouched -- the old code truncated it directly, so a
    # crash here used to discard every previously stored secret, not just this key.
    assert path.read_text(encoding="utf-8") == original
    leftover = list(tmp_path.glob(".env.*.tmp"))
    assert leftover == [], "the temp file must be cleaned up on failure, not left behind"


def test_clear_env_keys_is_also_atomic(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    upsert_env(path, {"OPENAI_API_KEY": "sk-a", "GEMINI_API_KEY": "gm-b"})
    clear_env_keys(path, ["OPENAI_API_KEY"])
    values = read_env_map(path)
    assert "OPENAI_API_KEY" not in values
    assert values["GEMINI_API_KEY"] == "gm-b"
    assert list(tmp_path.glob(".env.*.tmp")) == []


@pytest.mark.parametrize(
    "bad_value",
    [
        "sk-a\nMALICIOUS_INJECTED_KEY=evil",
        "sk-a\r\nMALICIOUS_INJECTED_KEY=evil",
        "sk-a\x00nul",
    ],
)
def test_upsert_env_rejects_control_characters_and_leaves_file_unchanged(
    bad_value: str, tmp_path: Path
) -> None:
    """read_env_map is line-based -- a value containing CR/LF can inject an
    unauthorized extra "line" the next time this file is read, bypassing
    ALLOWED_KEYS entirely since injected lines are never re-validated on read."""
    path = tmp_path / ".env"
    upsert_env(path, {"OPENAI_API_KEY": "sk-original"})
    original = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        upsert_env(path, {"OPENAI_API_KEY": bad_value})

    assert path.read_text(encoding="utf-8") == original
    values = read_env_map(path)
    assert values["OPENAI_API_KEY"] == "sk-original"
    assert "MALICIOUS_INJECTED_KEY" not in values
    assert list(tmp_path.glob(".env.*.tmp")) == []
