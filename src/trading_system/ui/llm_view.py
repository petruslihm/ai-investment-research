"""LLM prompt/response history page."""

from __future__ import annotations

from typing import Any


_KIND_KO = {
    "judge": "최종 판단 (GPT-5.6 Sol)",
    "extract": "공시 추출",
    "research": "리서치 (Gemini)",
    "adversarial": "반대 증거 (Gemini)",
    "portfolio_judge": "GPT 최종 답변 원문",
    "legacy_b_competitive_table": "경쟁력·마진 재검토",
    "legacy_b_rescore": "100점 재평가",
    "legacy_b_cash_rank": "현금 포함 순위",
    "legacy_b_live_portfolio_check": "보유·후보 최신 재확인",
    "legacy_b_final_execution": "최종 실행안",
}
_SOURCE_KO = {"live": "실제 호출", "reconstructed": "저장본에서 재구성"}

_PAGE_CSS = """
<style>
.page:has(.llm-page) { max-width: 68rem; }
.llm-page {
  display: grid;
  grid-template-columns: 19.5rem minmax(0, 1fr);
  gap: 1.25rem;
  align-items: start;
  margin-top: 1.15rem;
}
.llm-sessions {
  background: #fff;
  border-radius: 1rem;
  padding: 0.7rem 0.55rem 0.65rem;
  position: sticky;
  top: 0.75rem;
  max-height: calc(100vh - 7rem);
  overflow: auto;
}
.llm-sessions h2 {
  margin: 0.2rem 0.7rem 0.55rem;
  font-size: 0.78rem;
  color: #6b7684;
}
.llm-sessions a {
  display: block;
  text-decoration: none;
  color: #191f28;
  padding: 0.75rem 0.85rem;
  border-radius: 0.85rem;
  margin: 0.12rem 0.2rem;
}
.llm-sessions a:hover { background: #f7f8fa; }
.llm-sessions a.is-on { background: #f6ffed; }
.llm-sessions .d { display: block; font-size: 0.72rem; color: #8b95a1; font-weight: 600; }
.llm-sessions .t { display: block; font-size: 1.02rem; font-weight: 700; margin: 0.12rem 0 0.2rem; letter-spacing: -0.01em; }
.llm-sessions .m { display: block; font-size: 0.72rem; color: #6b7684; }
.llm-turn .chips { margin: 0; justify-content: flex-end; }
.llm-summary { margin: 0; font-size: 0.84rem; line-height: 1.55; }
.bubble { border-radius: 0.9rem; padding: 0.9rem 1rem; margin-top: 0.7rem; }
.bubble-q { background: #f2f4f6; }
.bubble-a { background: #f6ffed; }
.bubble .who { font-size: 0.7rem; font-weight: 700; color: #6b7684; margin-bottom: 0.35rem; }
@media (max-width: 860px) {
  .llm-page { grid-template-columns: 1fr; }
  .llm-sessions { position: static; max-height: 14rem; }
}
</style>
"""


def render_llm_log_html(
    sessions: list[dict[str, Any]],
    detail: dict[str, Any] | None,
    *,
    selected_id: str | None,
    error: str | None = None,
    current_model: str | None = None,
) -> str:
    from trading_system.ui.app import _action, _badge, _esc, _text_ko

    err_html = f'<div class="banner">{_esc(error)}</div>' if error else ""
    items = []
    last_date = None
    for s in sessions:
        on = " is-on" if s["tick_id"] == selected_id else ""
        live = "실제 호출 원문" if s.get("live_n") else "저장본으로 재구성"
        judged = int(s.get("judged") or 0)
        date = str(s.get("date") or "")
        heading = ""
        if date and date != last_date:
            heading = f'<p class="hint" style="margin:0.65rem 0.85rem 0.15rem">{_esc(date)}</p>'
            last_date = date
        items.append(
            f'{heading}<a class="{on.strip()}" href="/llm/{_esc(s["tick_id"])}">'
            f'<span class="d">{_esc(date)}</span>'
            f'<span class="t">{_esc(s.get("time") or "")}</span>'
            f'<span class="m">{_esc(live)} · 판단 {judged}건</span>'
            f"</a>"
        )
    list_html = (
        "".join(items)
        if items
        else '<p class="muted" style="margin:0.5rem 0.7rem">아직 스캔 기록이 없습니다.</p>'
    )

    if not sessions:
        detail_html = (
            '<div class="card"><p class="muted" style="margin:0">'
            "스캔이 한 번 끝나면 실행 시각이 왼쪽에 쌓입니다."
            "</p></div>"
        )
    elif detail is None:
        detail_html = (
            '<div class="card"><p class="muted" style="margin:0">'
            "왼쪽에서 실행 시각을 고르세요."
            "</p></div>"
        )
    else:
        detail_html = _detail_html(detail, _esc, _text_ko, _action, _badge)

    model_line = (
        f'<p class="lead">지금 호출에 쓰는 모델은 <strong>{_esc(current_model)}</strong> 입니다.</p>'
        if current_model
        else ""
    )
    return f"""
    {_PAGE_CSS}
    {err_html}
    <h1>LLM 기록</h1>
    <p class="lead">스캔마다 legacy-b 방식으로 이어진 GPT 대화의 질문과 답변 원문을 실행 시각별로 모았습니다. 주문은 넣지 않습니다.</p>
    {model_line}
    <div class="llm-page">
      <aside class="llm-sessions">
        <h2>실행 시각</h2>
        {list_html}
      </aside>
      <div>{detail_html}</div>
    </div>
    """


def _detail_html(detail: dict[str, Any], _esc, _text_ko, _action, _badge) -> str:
    live_n = int(detail.get("live_n") or 0)
    recon_n = int(detail.get("reconstructed_n") or 0)
    if live_n and not recon_n:
        notice = "이 실행은 질문과 답변 원문을 그대로 저장했습니다."
    elif live_n and recon_n:
        notice = (
            "일부는 실제 호출 기록이고, 나머지는 당시 저장된 추천으로 재구성했습니다."
        )
    else:
        notice = (
            "이 실행 당시에는 OpenAI 송수신 원문을 남기지 않았습니다. "
            "아래에 보이는 질문은 저장된 Quant 추천·근거 팩으로 다시 만든 것이고, "
            "답변은 그때 기록된 판단문입니다. 다음 스캔부터는 실제 질문/답변이 남습니다."
        )
    skipped = int(detail.get("skipped") or 0)
    skip_html = (
        f'<p class="hint">후보가 아니라 호출을 건너뛴 종목 {skipped}개</p>' if skipped else ""
    )
    token_html = _token_summary(detail, _esc)
    turns = detail.get("turns") or []
    if not turns:
        body = '<p class="muted" style="margin:0">이 실행에는 표시할 LLM 대화가 없습니다.</p>'
    else:
        body = "".join(_turn_html(t, _esc, _text_ko, _action, _badge) for t in turns)
    return f"""
    <div class="card">
      <h2>{_esc(detail.get("date"))} {_esc(detail.get("time"))}</h2>
      <p class="hint" style="margin:0 0 0.65rem">판단 {int(detail.get("judged") or 0)}건</p>
      <p class="hint">{_esc(notice)}</p>
      {token_html}
      {skip_html}
    </div>
    {body}
    """


def _fmt_n(n: object) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _token_summary(detail: dict[str, Any], _esc) -> str:
    total = detail.get("total_tokens")
    if total is None:
        return (
            '<p class="hint">이 실행은 토큰 사용량을 남기지 않았습니다. '
            "다음 스캔부터 입력/출력/합계가 표시됩니다.</p>"
        )
    calls = int(detail.get("token_calls") or 0)
    models = ", ".join(str(m) for m in (detail.get("models") or []) if m)
    model_bit = f" · 모델 {_esc(models)}" if models else ""
    return (
        f'<p class="hint" style="font-weight:600">이 실행 토큰 · 호출 {calls}회 · '
        f"입력 {_esc(_fmt_n(detail.get('prompt_tokens')))} · "
        f"출력 {_esc(_fmt_n(detail.get('completion_tokens')))} · "
        f"합계 {_esc(_fmt_n(total))}{model_bit}</p>"
    )


def _turn_html(turn: dict[str, Any], _esc, _text_ko, _action, _badge) -> str:
    raw_kind = str(turn.get("kind"))
    if raw_kind.startswith("legacy_b_analysis_part_"):
        kind = f"후보 분석 Part {raw_kind.rsplit('_', 1)[-1]}"
    else:
        kind = _KIND_KO.get(raw_kind, raw_kind)
    source = _SOURCE_KO.get(str(turn.get("source")), str(turn.get("source")))
    ticker = turn.get("ticker") or "—"
    status = turn.get("status")
    action = turn.get("action")
    thesis = turn.get("thesis")
    reasons = turn.get("override_reasons") or []
    error = turn.get("error")
    model = turn.get("model")
    chips = [_badge(status)] if status else []
    if action:
        chips.append(f'<span class="badge badge-neutral">{_esc(_action(action))}</span>')
    chips.append(f'<span class="badge badge-neutral">{_esc(source)}</span>')
    if model:
        chips.append(f'<span class="chip-static">{_esc(model)}</span>')
    token_line = ""
    if turn.get("total_tokens") is not None:
        token_line = (
            f'<p class="hint" style="margin:0.35rem 0 0">토큰 입력 {_esc(_fmt_n(turn.get("prompt_tokens")))}'
            f' · 출력 {_esc(_fmt_n(turn.get("completion_tokens")))}'
            f' · 합계 {_esc(_fmt_n(turn.get("total_tokens")))}</p>'
        )
    reason_html = ""
    if reasons:
        reason_html = (
            '<p class="hint" style="margin:0.35rem 0 0">이유: '
            + ", ".join(_esc(_text_ko(r)) for r in reasons)
            + "</p>"
        )
    user = turn.get("user_prompt") or ""
    system = turn.get("system_prompt") or ""
    response = turn.get("response_text") or ""
    if raw_kind.startswith("legacy_b_") or raw_kind == "portfolio_judge":
        q_summary = "legacy-b와 같은 한 GPT 대화에서 앞선 답변을 이어받아 분석한 단계입니다."
    elif raw_kind == "extract":
        q_summary = f"{ticker} 공시 본문에서 가이던스·실적 방향 등을 JSON으로 추출해 달라는 요청입니다."
    elif str(turn.get("kind")) == "research":
        q_summary = f"{ticker}의 공시·뉴스·산업·매크로 근거 팩을 만들라는 리서치 요청입니다."
    elif str(turn.get("kind")) == "adversarial":
        q_summary = f"{ticker} 근거 팩을 공격하고 반대 증거를 찾으라는 요청입니다."
    else:
        q_summary = (
            f"{ticker}의 Quant 점수와 Gemini 근거 팩을 보고 "
            "GPT-5.6 Sol이 최종 액션·근거를 JSON으로 판단해 달라는 요청입니다."
        )
    a_summary = ""
    if thesis:
        a_summary = _text_ko(thesis)
    elif error:
        a_summary = _text_ko(error)
    elif action:
        a_summary = _action(action)
    elif response:
        from trading_system.llm_log import answer_summary_from_blob, _json_object

        a_summary = _text_ko(answer_summary_from_blob(_json_object(response), response_text=response))
    return f"""
    <div class="card llm-turn">
      <div class="row" style="border:0;padding:0;align-items:flex-start">
        <div>
          <p style="margin:0;font-weight:700">{_esc(ticker)}</p>
          <p class="hint" style="margin:0.15rem 0 0">{_esc(kind)}</p>
        </div>
        <div class="chips">{"".join(chips)}</div>
      </div>
      {reason_html}
      {token_line}
      <div class="bubble bubble-q">
        <div class="who">질문</div>
        <p class="llm-summary">{_esc(q_summary)}</p>
      </div>
      <div class="bubble bubble-a">
        <div class="who">답변</div>
        <p class="llm-summary">{_esc(a_summary) if a_summary else "기록된 답변 없음"}</p>
      </div>
      <details class="tech">
        <summary>질문 전문</summary>
        <pre>{_esc(user) if user else "기록된 질문 없음"}</pre>
      </details>
      <details class="tech">
        <summary>답변 원문</summary>
        <pre>{_esc(response) if response else "기록된 답변 없음"}</pre>
      </details>
      <details class="tech">
        <summary>시스템 프롬프트</summary>
        <pre>{_esc(system) if system else "없음"}</pre>
      </details>
    </div>
    """
