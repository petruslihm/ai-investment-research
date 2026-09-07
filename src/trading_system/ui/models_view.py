"""Read-only view model + renderer for the Models / Learning page.

Everything here is derived from what was actually recorded. When a factual field was
never written, the page says so instead of guessing.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

NO_REASON = "기록된 근거 없음"
NO_METRIC = "아직 평가 결과가 충분하지 않습니다"
NO_OUTCOMES = (
    "아직 평가 가능한 추천 결과가 없습니다.\n5/10/20일 결과가 성숙하면 자동으로 평가됩니다."
)
NO_OVERRIDE = "아직 Quant vs LLM 비교 결과가 없습니다."

EVENT_TITLES: dict[str, str] = {
    "promoted": "새 모델 채택",
    "rejected": "새 모델 채택 안 함",
    "online_updated": "온라인 학습 업데이트",
    "ensemble_weight_changed": "모델 가중치 조정",
    "horizon_weight_changed": "기간별 반영 비중 조정",
    "retrained": "모델 재학습",
    "rollback": "이전 안정 버전으로 복구",
    "rolled_back": "이전 안정 버전으로 복구",
}

EVENT_TONE: dict[str, str] = {
    "promoted": "badge-ok",
    "rejected": "badge-warn",
    "online_updated": "badge-neutral",
    "ensemble_weight_changed": "badge-neutral",
    "horizon_weight_changed": "badge-neutral",
    "retrained": "badge-ok",
    "rollback": "badge-err",
    "rolled_back": "badge-err",
}

FAMILY_LABELS: dict[str, str] = {
    "ridge": "Ridge",
    "lightgbm_reg": "LightGBM",
    "lambdarank": "LambdaRank",
    "torch_sequence": "LSTM",
    "online_sgd": "온라인 SGD",
    "online": "온라인 SGD",
    "ensemble": "앙상블",
}

# Ensemble weight keys as stored by ml_engine, in display order.
RETURN_FAMILY_KEYS: tuple[str, ...] = ("ridge", "lightgbm_reg", "torch_sequence", "online")
RANK_FAMILY_KEY = "lambdarank"

REASON_TEXT: dict[str, str] = {
    "walkforward_promote": "시간순 분리(워크포워드) 학습을 마치고 이번 세대 모델로 채택했습니다.",
    "matured_ewma": "성숙한 실제 결과의 평균 절대오차를 비교해 가중치를 다시 계산했습니다.",
    "matured_label_partial_fit": "새로 성숙한 라벨로 온라인 모델을 추가 학습했습니다.",
}

CHANGE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("all", "전체", ()),
    ("promotion", "모델 채택", ("promoted", "rejected", "retrained")),
    ("online", "온라인 업데이트", ("online_updated",)),
    ("weight", "가중치 조정", ("ensemble_weight_changed", "horizon_weight_changed")),
    ("rollback", "복구", ("rollback", "rolled_back")),
)

HORIZON_CHOICES: tuple[str, ...] = ("all", "5", "10", "20")

_ASSET_LABELS = {"us_equity": "미국 주식", "btc": "BTC"}


def family_label(value: object) -> str:
    key = str(value or "").strip().lower()
    return FAMILY_LABELS.get(key, key or "—")


def humanize_kind(kind: object) -> str:
    key = str(kind or "").strip().lower()
    return EVENT_TITLES.get(key, key or "모델 변경")


def generation_of(version: object) -> int:
    """Read the generation number out of a stored version string ('v3_adapt' -> 3)."""
    digits = ""
    for ch in str(version or "").strip():
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    try:
        return int(digits) if digits else 1
    except ValueError:
        return 1


def humanize_version(version: object) -> str:
    """Stable, readable generation label. Never leaks 'v1_adapt_adapt_adapt...'."""
    raw = str(version or "").strip()
    if not raw:
        return "—"
    if raw.startswith("online_"):
        return f"온라인 v{generation_of(raw)}"
    if raw.lower().startswith("v") or "_adapt" in raw:
        return f"v{generation_of(raw)}"
    if raw.isdigit():
        return f"v{int(raw)}"
    return raw


def format_ts(value: object) -> str:
    """Timestamp rounded to seconds, local-naive rendering of what was stored."""
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return text.split(".")[0]


def _num(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: object, digits: int = 1) -> str:
    v = _num(value)
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _metric(value: object) -> str:
    v = _num(value)
    return "—" if v is None else f"{v:.5f}"


@dataclass
class JournalCard:
    kind: str
    title: str
    tone: str
    family: str
    family_key: str
    asset: str
    horizon: str
    horizon_key: str
    at: str
    what_changed: str
    change_detail: str
    reason: str
    matured: str
    training_window: str
    evaluation_window: str
    metric_line: str
    metric_improved: str | None
    result_label: str
    result_tone: str
    effect: str
    raw_json: str


@dataclass
class ModelsView:
    generation_label: str
    epoch_internal: str
    last_update: str
    matured_used: int
    matured_known: bool
    health_label: str
    health_tone: str
    health_hint: str
    recent_changes: int
    horizon_cards: list[dict[str, Any]] = field(default_factory=list)
    btc_weights: list[tuple[str, str]] = field(default_factory=list)
    btc_version: str = "—"
    cards: list[JournalCard] = field(default_factory=list)
    total_events: int = 0
    families_seen: list[str] = field(default_factory=list)
    live_rows: list[tuple[str, str, int, str]] = field(default_factory=list)
    outcome_count: int = 0
    override_count: int = 0
    research_rows: list[tuple[str, str, str, str, str]] = field(default_factory=list)


def _payload(raw: object) -> dict[str, Any]:
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _reason_text(payload: dict[str, Any]) -> str:
    codes = payload.get("reason_codes") or []
    if not isinstance(codes, list) or not codes:
        return NO_REASON
    parts = [REASON_TEXT.get(str(c), str(c)) for c in codes]
    return " ".join(parts) if parts else NO_REASON


def _what_changed(kind: str, payload: dict[str, Any]) -> tuple[str, str]:
    prev_w = _num(payload.get("previous_ensemble_weight"))
    new_w = _num(payload.get("new_ensemble_weight"))
    target_w = _num(payload.get("target_ensemble_weight"))
    prev_v = payload.get("previous_version")
    new_v = payload.get("new_version")
    if prev_w is not None or new_w is not None:
        before = _pct(prev_w) if prev_w is not None else "기록 없음"
        after = _pct(new_w) if new_w is not None else "기록 없음"
        if target_w is not None:
            # current -> un-damped target -> what was actually applied (EWMA moves
            # only part way to target -- see ml_engine.update_weights). Without the
            # target, "21.1% -> 21.2%" alone can't tell "the model barely preferred
            # this" apart from "the model preferred this a lot but was smoothed down".
            return "앙상블 가중치", f"{before} → 목표 {_pct(target_w)} → 적용 {after}"
        return "앙상블 가중치", f"{before} → {after}"
    if kind == "online_updated":
        return "온라인 모델 파라미터", f"{humanize_version(prev_v) if prev_v else '이전 상태'} → {humanize_version(new_v)}"
    if prev_v or new_v:
        before = humanize_version(prev_v) if prev_v else "이전 버전 기록 없음"
        return "모델 버전", f"{before} → {humanize_version(new_v)}"
    return "변경 항목", NO_REASON


def _effect(kind: str, family_key: str, horizon: object, asset: str) -> str:
    scope = _ASSET_LABELS.get(asset, asset or "해당 자산군")
    hz = f"{horizon}일" if horizon not in (None, "") else "5·10·20일"
    if kind == "rejected":
        return "현재 추천에 반영되지 않습니다. 직전 모델을 그대로 씁니다."
    if kind in {"rollback", "rolled_back"}:
        return f"{scope} {hz} 추천이 이전 안정 버전 예측으로 되돌아갑니다."
    if family_key == "lambdarank":
        return f"{scope} 종목 순위에만 반영됩니다. 기대수익 계산에는 들어가지 않습니다."
    if kind == "ensemble_weight_changed":
        return f"{scope} {hz} 추천의 기대수익 blend 비중이 바뀝니다."
    if kind == "online_updated":
        return f"{scope} {hz} 추천의 온라인 예측에 바로 쓰입니다."
    return f"{scope} {hz} 추천 예측에 사용됩니다."


def _result(kind: str, payload: dict[str, Any]) -> tuple[str, str]:
    result = str(payload.get("promotion_result") or "").strip().lower()
    if kind == "rejected" or result == "rejected":
        return "채택 안 함", "badge-warn"
    if kind in {"rollback", "rolled_back"}:
        return "복구됨", "badge-err"
    if result in {"accepted", "applied"} or kind in {"promoted", "online_updated", "ensemble_weight_changed", "retrained"}:
        return "적용됨", "badge-ok"
    return "결과 미기록", "badge-neutral"


def _metric_line(payload: dict[str, Any]) -> tuple[str, str | None]:
    before = _num(payload.get("metric_before"))
    after = _num(payload.get("metric_after"))
    if before is None and after is None:
        return NO_METRIC, None
    name = str(payload.get("metric_name") or "지표")
    label = {"walkforward_mae": "워크포워드 MAE", "matured_mean_abs_error": "성숙 결과 평균오차"}.get(name, name)
    if before is None:
        return f"{label} {_metric(after)} (직전 기록 없음)", None
    if after is None:
        return f"{label} {_metric(before)} → 기록 없음", None
    improved = "개선" if after < before else ("악화" if after > before else "변화 없음")
    return f"{label} {_metric(before)} → {_metric(after)}", improved


def _to_card(created_at: object, kind: str, payload: dict[str, Any], raw: str) -> JournalCard:
    family_key = str(payload.get("model_family") or "").lower()
    asset = str(payload.get("asset_class") or "")
    horizon = payload.get("horizon")
    what, detail = _what_changed(kind, payload)
    metric_line, improved = _metric_line(payload)
    result_label, result_tone = _result(kind, payload)
    matured = payload.get("matured_label_count")
    sample = payload.get("sample_count")
    if matured is None:
        matured = sample
    matured_label = f"{int(matured):,}건" if isinstance(matured, (int, float)) else "기록 없음"
    # Only ensemble_weight_changed/horizon_weight_changed set sample_count to mean
    # "rows actually used in this update" (recent-window-restricted, see
    # RECENT_EVAL_WINDOW_DAYS in ml_engine.py) -- other kinds (e.g. "promoted") use
    # sample_count for an unrelated number (training-set size), so this display must
    # not apply there.
    if (
        kind in {"ensemble_weight_changed", "horizon_weight_changed"}
        and isinstance(sample, (int, float))
        and isinstance(matured, (int, float))
        and int(sample) != int(matured)
    ):
        matured_label = f"최근 평가 {int(sample):,}건 (전체 확정 {int(matured):,}건 중)"
    return JournalCard(
        kind=kind,
        title=humanize_kind(kind),
        tone=EVENT_TONE.get(kind, "badge-neutral"),
        family=family_label(family_key) if family_key else "—",
        family_key=family_key,
        asset=_ASSET_LABELS.get(asset, asset),
        horizon=f"{horizon}일" if horizon not in (None, "") else "전체 기간",
        horizon_key=str(horizon) if horizon not in (None, "") else "",
        at=format_ts(created_at),
        what_changed=what,
        change_detail=detail,
        reason=_reason_text(payload),
        matured=matured_label,
        training_window=str(payload.get("training_period") or "기록 없음"),
        evaluation_window=str(payload.get("evaluation_period") or "기록 없음"),
        metric_line=metric_line,
        metric_improved=improved,
        result_label=result_label,
        result_tone=result_tone,
        effect=_effect(kind, family_key, horizon, asset),
        raw_json=raw,
    )


def _fetch(conn, sql: str, params: list[Any] | None = None) -> list[tuple]:
    try:
        return conn.execute(sql, params or []).fetchall()
    except Exception:  # noqa: BLE001 — the page must render even on a partial schema
        return []


def _scalar(conn, sql: str, default: int = 0) -> int:
    rows = _fetch(conn, sql)
    if not rows or rows[0][0] is None:
        return default
    try:
        return int(rows[0][0])
    except (TypeError, ValueError):
        return default


def _health(*, has_events: bool, last_at: datetime | None, training_status: str | None) -> tuple[str, str, str]:
    if not has_events:
        return "학습 기록 없음", "badge-neutral", "스캔을 실행하면 학습 기록이 쌓입니다."
    if training_status and training_status.lower() in {"failed", "unavailable"}:
        return "점검 필요", "badge-err", "최근 학습 단계가 실패로 끝났습니다. 런타임 로그를 확인하세요."
    if last_at is not None:
        age = datetime.now(timezone.utc) - last_at
        if age > timedelta(days=7):
            return "오래됨", "badge-warn", f"마지막 학습이 {age.days}일 전입니다. 스캔을 실행하면 갱신됩니다."
    return "정상", "badge-ok", "최근 학습이 정상적으로 끝났습니다."


def collect_models_view(
    conn,
    *,
    horizon: str = "all",
    change: str = "all",
    family: str = "all",
) -> ModelsView:
    epochs = _scalar(conn, "SELECT COUNT(*) FROM decision_epochs")
    epoch_row = _fetch(
        conn, "SELECT decision_epoch_id, created_at FROM decision_epochs ORDER BY created_at DESC LIMIT 1"
    )
    ens_rows = _fetch(
        conn, "SELECT stream, weights_json, horizon_influence_json, version, updated_at FROM ensemble_state"
    )
    ens: dict[str, tuple[dict[str, float], dict[str, float], str, object]] = {}
    for stream, w_json, h_json, version, updated in ens_rows:
        try:
            weights = json.loads(w_json)
            influence = json.loads(h_json)
        except (TypeError, ValueError):
            continue
        ens[str(stream)] = (weights, influence, str(version), updated)

    raw_events = _fetch(
        conn,
        "SELECT created_at, kind, payload_json FROM model_change_journal ORDER BY created_at DESC LIMIT 400",
    )
    parsed = [(at, str(kind), _payload(raw), str(raw)) for at, kind, raw in raw_events]

    families_seen = sorted({str(p.get("model_family") or "") for _a, _k, p, _r in parsed} - {""})

    allowed_kinds: tuple[str, ...] = ()
    for key, _label, kinds in CHANGE_GROUPS:
        if key == change:
            allowed_kinds = kinds
            break
    filtered = []
    for at, kind, payload, raw in parsed:
        if allowed_kinds and kind not in allowed_kinds:
            continue
        # Events without a horizon (ensemble-wide weight changes) apply to every horizon.
        event_horizon = str(payload.get("horizon") or "")
        if horizon != "all" and event_horizon and event_horizon != horizon:
            continue
        if family != "all" and str(payload.get("model_family") or "") != family:
            continue
        filtered.append((at, kind, payload, raw))

    cards = [_to_card(at, kind, payload, raw) for at, kind, payload, raw in filtered[:24]]

    last_at: datetime | None = None
    for at, _kind, _pay, _raw in parsed[:1]:
        if isinstance(at, datetime):
            last_at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)

    total_events = _scalar(conn, "SELECT COUNT(*) FROM model_change_journal")
    recent_rows = _fetch(
        conn,
        "SELECT COUNT(*) FROM model_change_journal WHERE created_at >= ?",
        [datetime.now(timezone.utc) - timedelta(days=7)],
    )
    recent_changes = int(recent_rows[0][0]) if recent_rows and recent_rows[0][0] is not None else 0

    # Matured labels actually consumed in the most recent online update per (asset, horizon).
    matured_seen: dict[tuple[str, str], int] = {}
    for _at, kind, payload, _raw in parsed:
        if kind != "online_updated":
            continue
        key = (str(payload.get("asset_class") or ""), str(payload.get("horizon") or ""))
        if key in matured_seen:
            continue
        count = payload.get("matured_label_count")
        if isinstance(count, (int, float)):
            matured_seen[key] = int(count)
    matured_used = sum(matured_seen.values())

    training = _fetch(
        conn, "SELECT status FROM runtime_events WHERE kind = 'training' ORDER BY created_at DESC LIMIT 1"
    )
    health_label, health_tone, health_hint = _health(
        has_events=bool(parsed),
        last_at=last_at,
        training_status=str(training[0][0]) if training else None,
    )

    equity_weights, equity_influence, equity_version, equity_updated = ens.get(
        "us_equity", ({}, {}, "", None)
    )
    research: dict[tuple[str, str], tuple[object, object]] = {}
    for _at, kind, payload, _raw in parsed:
        if kind != "promoted" or str(payload.get("asset_class") or "") != "us_equity":
            continue
        key = (str(payload.get("horizon") or ""), str(payload.get("model_family") or ""))
        if key in research:
            continue
        research[key] = (payload.get("metric_after"), payload.get("training_period"))

    horizon_cards: list[dict[str, Any]] = []
    for h in ("5", "10", "20"):
        rows = []
        for key in RETURN_FAMILY_KEYS:
            weight = equity_weights.get(key)
            metric, _period = research.get((h, "online_sgd" if key == "online" else key), (None, None))
            rows.append(
                {
                    "label": FAMILY_LABELS[key],
                    "weight": _pct(weight) if weight is not None else "—",
                    "metric": _metric(metric) if metric is not None else "—",
                }
            )
        rank_weight = equity_weights.get(RANK_FAMILY_KEY)
        rank_metric, _p = research.get((h, RANK_FAMILY_KEY), (None, None))
        horizon_cards.append(
            {
                "horizon": h,
                "influence": _pct(equity_influence.get(h)) if equity_influence.get(h) is not None else "—",
                "rows": rows,
                "rank": {
                    "label": FAMILY_LABELS[RANK_FAMILY_KEY],
                    "weight": _pct(rank_weight) if rank_weight is not None else "—",
                    "metric": _metric(rank_metric) if rank_metric is not None else "—",
                },
            }
        )

    btc_weights_map, _btc_inf, btc_version, _btc_updated = ens.get("btc", ({}, {}, "", None))
    btc_weights = [
        (FAMILY_LABELS[key], _pct(btc_weights_map.get(key)) if btc_weights_map.get(key) is not None else "—")
        for key in RETURN_FAMILY_KEYS
    ]

    outcome_count = _scalar(conn, "SELECT COUNT(*) FROM recommendation_outcomes")
    override_count = _scalar(conn, "SELECT COUNT(*) FROM override_score_rows")
    live_rows: list[tuple[str, str, int, str]] = []
    for source, hz, n, avg in _fetch(
        conn,
        """
        SELECT source, horizon, COUNT(*), AVG(realized_return)
        FROM recommendation_outcomes
        WHERE realized_return IS NOT NULL
        GROUP BY source, horizon
        ORDER BY horizon, source
        """,
    ):
        live_rows.append((str(source), f"{hz}일", int(n), _pct(avg, 2)))

    research_rows: list[tuple[str, str, str, str, str]] = []
    seen_research: set[tuple[str, str, str]] = set()
    for _at, kind, payload, _raw in parsed:
        if kind != "promoted":
            continue
        asset = str(payload.get("asset_class") or "")
        key = (asset, str(payload.get("horizon") or ""), str(payload.get("model_family") or ""))
        if key in seen_research:
            continue
        seen_research.add(key)
        metric = payload.get("metric_after")
        research_rows.append(
            (
                _ASSET_LABELS.get(asset, "기록 없음" if not asset else asset),
                f"{key[1]}일" if key[1] else "—",
                family_label(key[2]),
                _metric(metric) if metric is not None else "—",
                str(payload.get("training_period") or "기록 없음"),
            )
        )
    research_rows.sort()

    last_update = format_ts(last_at or equity_updated or (epoch_row[0][1] if epoch_row else None))
    generation = generation_of(equity_version) if equity_version else 1
    return ModelsView(
        generation_label=f"Epoch {max(epochs, 1)} / 앙상블 v{generation}",
        epoch_internal=str(epoch_row[0][0]) if epoch_row else "—",
        last_update=last_update,
        matured_used=matured_used,
        matured_known=bool(matured_seen),
        health_label=health_label,
        health_tone=health_tone,
        health_hint=health_hint,
        recent_changes=recent_changes,
        horizon_cards=horizon_cards,
        btc_weights=btc_weights,
        btc_version=humanize_version(btc_version) if btc_version else "—",
        cards=cards,
        total_events=total_events,
        families_seen=families_seen,
        live_rows=live_rows,
        outcome_count=outcome_count,
        override_count=override_count,
        research_rows=research_rows,
    )


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _chip_row(label: str, param: str, options: list[tuple[str, str]], current: str, base: dict[str, str]) -> str:
    bits = []
    for key, text in options:
        params = dict(base)
        params[param] = key
        query = "&".join(f"{k}={v}" for k, v in params.items() if v and v != "all")
        href = f"/models?{query}" if query else "/models"
        on = " on" if current == key else ""
        bits.append(f'<a class="chip{on}" href="{_esc(href)}">{_esc(text)}</a>')
    return (
        f'<div class="filter-row"><span class="filter-label">{_esc(label)}</span>'
        f'<div class="chips" style="margin:0">{"".join(bits)}</div></div>'
    )


def _horizon_card_html(card: dict[str, Any]) -> str:
    rows = "".join(
        f'<div class="row"><span>{_esc(r["label"])}</span>'
        f'<span><b>{_esc(r["weight"])}</b>'
        f'<span class="hint" style="margin-left:0.5rem">연구 MAE {_esc(r["metric"])}</span></span></div>'
        for r in card["rows"]
    )
    rank = card["rank"]
    return f"""
    <div class="card">
      <h2>{_esc(card['horizon'])}일 앙상블</h2>
      <p class="hint">기대수익 blend 비중입니다. 옆의 연구 MAE는 워크포워드 진단값이며 실현 성과가 아닙니다.</p>
      {rows}
      <div class="row" style="border-top:1px solid var(--border);margin-top:0.35rem">
        <span>{_esc(rank['label'])} <span class="badge badge-neutral">순위 전용</span></span>
        <span><b>{_esc(rank['weight'])}</b>
        <span class="hint" style="margin-left:0.5rem">연구 MAE {_esc(rank['metric'])}</span></span>
      </div>
      <p class="hint" style="margin-top:0.6rem">기간 반영 비중 {_esc(card['influence'])} ·
      LambdaRank는 종목 순위에만 쓰이고 기대수익에는 들어가지 않습니다.<br/>
      가중치는 미국 주식 자산군 단위로 학습해 5·10·20일에 공통 적용됩니다. 기간별로 다른 값은 연구 MAE입니다.</p>
    </div>
    """


def _journal_card_html(card: JournalCard) -> str:
    improved = ""
    if card.metric_improved:
        tone = {"개선": "badge-ok", "악화": "badge-err"}.get(card.metric_improved, "badge-neutral")
        improved = f'<span class="badge {tone}" style="margin-left:0.4rem">{_esc(card.metric_improved)}</span>'
    return f"""
    <div class="card journal-card">
      <div class="row" style="border:0;padding:0 0 0.6rem">
        <div>
          <p style="margin:0;font-weight:700">{_esc(card.title)}</p>
          <p class="hint" style="margin:0.2rem 0 0">{_esc(card.at)}</p>
        </div>
        <span class="badge {_esc(card.result_tone)}">{_esc(card.result_label)}</span>
      </div>
      <div class="chips" style="margin:0 0 0.75rem">
        <span class="chip-static">{_esc(card.family)}</span>
        <span class="chip-static">{_esc(card.horizon)}</span>
        {f'<span class="chip-static">{_esc(card.asset)}</span>' if card.asset else ""}
      </div>
      <dl class="kv">
        <dt>무엇이 바뀌었나</dt><dd>{_esc(card.what_changed)} · {_esc(card.change_detail)}</dd>
        <dt>왜 바뀌었나</dt><dd>{_esc(card.reason)}</dd>
        <dt>사용한 성숙 결과</dt><dd>{_esc(card.matured)}</dd>
        <dt>학습 구간</dt><dd>{_esc(card.training_window)}</dd>
        <dt>검증 구간</dt><dd>{_esc(card.evaluation_window)}</dd>
        <dt>연구 지표 변화</dt><dd>{_esc(card.metric_line)}{improved}</dd>
        <dt>현재 추천에 미치는 영향</dt><dd>{_esc(card.effect)}</dd>
      </dl>
      <details class="tech">
        <summary>기술 세부정보</summary>
        <pre>{_esc(card.raw_json)}</pre>
      </details>
    </div>
    """


def render_models_page(view: ModelsView, *, horizon: str, change: str, family: str) -> str:
    base = {"h": horizon, "c": change, "f": family}
    filters = (
        _chip_row("기간", "h", [("all", "전체"), ("5", "5일"), ("10", "10일"), ("20", "20일")], horizon, base)
        + _chip_row("변경 종류", "c", [(k, label) for k, label, _ in CHANGE_GROUPS], change, base)
        + _chip_row(
            "모델",
            "f",
            [("all", "전체")] + [(f, family_label(f)) for f in view.families_seen],
            family,
            base,
        )
    )
    horizon_cards = "".join(_horizon_card_html(c) for c in view.horizon_cards)
    journal = "".join(_journal_card_html(c) for c in view.cards)
    if not journal:
        journal = (
            '<div class="card"><p class="muted" style="margin:0">'
            "선택한 조건에 해당하는 변경 기록이 없습니다.</p></div>"
            if view.total_events
            else '<div class="card"><p class="muted" style="margin:0">'
            "아직 학습 기록이 없습니다. 스캔을 실행하면 쌓입니다.</p></div>"
        )

    if view.live_rows:
        live = (
            "<table><thead><tr><th>구분</th><th>기간</th><th>평가 건수</th><th>평균 실현수익</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td>{_esc(src)}</td><td>{_esc(hz)}</td><td>{n:,}</td><td>{_esc(avg)}</td></tr>"
                for src, hz, n, avg in view.live_rows
            )
            + "</tbody></table>"
        )
    else:
        live = f'<p class="muted" style="margin:0;white-space:pre-line">{_esc(NO_OUTCOMES)}</p>'

    override = (
        f'<p class="hint" style="margin:0.75rem 0 0">Quant vs LLM 비교 {view.override_count:,}건이 기록되었습니다.</p>'
        if view.override_count
        else f'<p class="muted" style="margin:0.75rem 0 0">{_esc(NO_OVERRIDE)}</p>'
    )

    if view.research_rows:
        research = (
            "<table><thead><tr><th>자산</th><th>기간</th><th>모델</th>"
            "<th>워크포워드 MAE</th><th>학습 구간</th></tr></thead><tbody>"
            + "".join(
                f"<tr><td>{_esc(asset)}</td><td>{_esc(hz)}</td><td>{_esc(fam)}</td>"
                f"<td>{_esc(metric)}</td><td>{_esc(period)}</td></tr>"
                for asset, hz, fam, metric, period in view.research_rows
            )
            + "</tbody></table>"
        )
    else:
        research = f'<p class="muted" style="margin:0">{_esc(NO_METRIC)}</p>'

    matured_value = f"{view.matured_used:,}건" if view.matured_known else "기록 없음"
    btc_rows = "".join(
        f'<div class="row"><span>{_esc(label)}</span><b>{_esc(weight)}</b></div>'
        for label, weight in view.btc_weights
    )

    return f"""
    <h1>모델/학습</h1>
    <p class="lead">무엇이 언제 왜 바뀌었는지, 그리고 그 변경이 지금 추천에 어떻게 반영되는지 보여줍니다.</p>
    <div class="grid-stats">
      <div class="stat"><span class="stat-label">현재 세대</span>
        <span class="stat-value" style="font-size:1.05rem">{_esc(view.generation_label)}</span>
        <span class="stat-sub">최근 갱신 {_esc(view.last_update)}</span></div>
      <div class="stat"><span class="stat-label">학습에 쓴 성숙 결과</span>
        <span class="stat-value">{_esc(matured_value)}</span>
        <span class="stat-sub">최근 사이클 온라인 학습 기준</span></div>
      <div class="stat"><span class="stat-label">모델 상태</span>
        <span class="stat-value" style="font-size:1.05rem">
          <span class="badge {_esc(view.health_tone)}">{_esc(view.health_label)}</span></span>
        <span class="stat-sub">{_esc(view.health_hint)}</span></div>
      <div class="stat"><span class="stat-label">최근 7일 변경</span>
        <span class="stat-value">{view.recent_changes:,}건</span>
        <span class="stat-sub">전체 기록 {view.total_events:,}건</span></div>
    </div>
    {horizon_cards}
    <div class="card">
      <h2>BTC 앙상블 ({_esc(view.btc_version)})</h2>
      <p class="hint">BTC는 별도 슬리브라 주식과 가중치를 따로 학습합니다. LambdaRank는 쓰지 않습니다.</p>
      {btc_rows or '<p class="muted" style="margin:0">아직 BTC 앙상블 기록이 없습니다.</p>'}
    </div>
    <div class="card">
      <h2>실시간 성과 (성숙한 추천 결과)</h2>
      <p class="hint">실제로 5/10/20일이 지나 결과가 확정된 추천만 집계합니다.</p>
      {live}
      {override}
    </div>
    <div class="card">
      <h2>연구 진단 (워크포워드)</h2>
      <p class="hint">학습 시점의 시간순 분리 검증 수치입니다. 실현 성과가 아니며 위 실시간 성과와 다릅니다.</p>
      {research}
    </div>
    <div class="card">
      <h2>모델 변경 기록</h2>
      {filters}
    </div>
    {journal}
    <p class="footnote">내부 식별자: 최신 decision epoch {_esc(view.epoch_internal)}</p>
    """
