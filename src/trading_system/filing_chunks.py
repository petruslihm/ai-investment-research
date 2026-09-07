"""Section-aware filing passages. Never use a naive first-N-character extract as the whole filing."""

from __future__ import annotations

import re
from typing import Any

FILING_BODY_CHARS = 180_000
PASSAGE_BUDGET_CHARS = 12_000
WINDOW_CHARS = 1_800
WINDOW_OVERLAP = 200

_HEADER_RE = re.compile(
    r"(?is)(?P<header>"
    r"item\s+\d{1,2}(?:\.\d{2})?[^\n]{0,120}|"
    r"part\s+[ivx]{1,4}\b[^\n]{0,80}|"
    r"management['’]s\s+discussion[^\n]{0,80}|"
    r"risk\s+factors|"
    r"legal\s+proceedings|"
    r"quantitative\s+and\s+qualitative[^\n]{0,80}|"
    r"note\s+\d+[.\s][^\n]{0,80}"
    r")"
)

_SIGNAL_WORDS = (
    "guidance",
    "outlook",
    "impairment",
    "lawsuit",
    "litigation",
    "restatement",
    "going concern",
    "bankruptcy",
    "revenue",
    "margin",
    "restructuring",
    "acquisition",
    "divest",
    "cyber",
    "investigation",
    "sec",
    "downgrade",
    "warning",
    "loss",
    "charge",
)


def _plain_windows(text: str) -> list[tuple[str, str]]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    headers = list(_HEADER_RE.finditer(cleaned))
    if not headers:
        chunks: list[tuple[str, str]] = []
        i = 0
        n = 1
        while i < len(cleaned):
            chunk = cleaned[i : i + WINDOW_CHARS]
            if chunk:
                chunks.append((f"passage_{n}", chunk))
                n += 1
            i += max(1, WINDOW_CHARS - WINDOW_OVERLAP)
        return chunks
    out: list[tuple[str, str]] = []
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(cleaned)
        body = cleaned[start:end].strip()
        if not body:
            continue
        title = re.sub(r"\s+", " ", match.group("header")).strip()[:80]
        if len(body) <= WINDOW_CHARS * 2:
            out.append((title or f"section_{i+1}", body))
            continue
        j = 0
        part = 1
        while j < len(body):
            piece = body[j : j + WINDOW_CHARS]
            if piece:
                out.append((f"{title} ({part})", piece))
                part += 1
            j += max(1, WINDOW_CHARS - WINDOW_OVERLAP)
    return out


def _score(title: str, body: str, *, ticker: str, extra: str = "") -> float:
    blob = f"{title} {body} {extra}".lower()
    score = 0.0
    t = (ticker or "").lower()
    if t and t in blob:
        score += 2.0
    for word in _SIGNAL_WORDS:
        if word in blob:
            score += 1.0
    title_l = title.lower()
    if "item 2.02" in title_l or "item 8.01" in title_l or "item 1.01" in title_l:
        score += 3.0
    if "risk" in title_l or "discussion" in title_l or "outlook" in title_l:
        score += 2.0
    return score


def select_filing_passages(
    text: str,
    *,
    ticker: str,
    extra_query: str = "",
    budget_chars: int = PASSAGE_BUDGET_CHARS,
) -> list[dict[str, Any]]:
    """Pick the most relevant filing slices instead of the document prefix."""
    windows = _plain_windows(text)
    if not windows:
        return []
    ranked = sorted(
        windows,
        key=lambda pair: _score(pair[0], pair[1], ticker=ticker, extra=extra_query),
        reverse=True,
    )
    picked: list[dict[str, Any]] = []
    used = 0
    seen: set[str] = set()
    for title, body in ranked:
        key = body[:240]
        if key in seen:
            continue
        seen.add(key)
        take = body[: WINDOW_CHARS]
        if used + len(take) > budget_chars and picked:
            remain = budget_chars - used
            if remain >= 400:
                picked.append({"section": title, "text": take[:remain]})
            break
        picked.append({"section": title, "text": take})
        used += len(take)
        if used >= budget_chars:
            break
    return picked


def passages_as_text(passages: list[dict[str, Any]], *, limit: int = PASSAGE_BUDGET_CHARS) -> str:
    parts: list[str] = []
    used = 0
    for row in passages:
        section = str(row.get("section") or "passage")
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        block = f"[{section}]\n{text}"
        if used + len(block) > limit:
            block = block[: max(0, limit - used)]
        if not block:
            break
        parts.append(block)
        used += len(block)
        if used >= limit:
            break
    return "\n\n".join(parts)
