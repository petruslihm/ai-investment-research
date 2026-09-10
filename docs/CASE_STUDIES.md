# 문제 해결 사례

이 저장소의 재현 사례와 별도 `etf-radar`에서 관찰한 사례를 구분했습니다. ETF Radar의 최신 코드는 [별도 공개 저장소](https://github.com/petruslihm/etf-radar)에서 확인할 수 있습니다. 당시 원본 계좌·응답·DB는 포함하지 않습니다.

## 공개 코드: 판단과 화면을 같은 결과에 연결

**문제:** AI가 행동이나 배분을 변경해도 카드·요약이 이전 Quant 값으로 남으면, 설명과 실제 표시 결과가 달라집니다. 실패한 GPT 단계까지 최종 AI 판단처럼 보일 수도 있습니다.

**구현:** 구조화 결과의 종목 집합·행동·단위·순위를 검증한 뒤 제약을 적용합니다. `effective`를 카드·설명·정렬·요약의 기준으로 저장하고, 실패 대체에는 출처를 표시합니다. 동일 입력 ID의 단계 기록과 화면 snapshot을 같은 tick transaction에 저장합니다.

**재현:** 데모의 정상 시나리오에서 ALPHA는 WATCH 0, GAMMA는 요청 400에서 한도 250으로 바뀝니다. GPT 실패 시 Gemini 반영 결과가 대체로 표시됩니다. 저장 결과를 다시 읽어 같은 판단이 복원되는지 확인할 수 있습니다.

**근거:** [판단 검증 코드](../src/trading_system/final_decision.py), [tick 저장](../src/trading_system/storage/ticks.py), [데모 회귀 테스트](../tests/test_submission_demo.py).

**한계:** 합성 입력으로 소프트웨어 계약을 확인합니다. 실제 LLM이 더 좋은 투자 결정을 내렸다는 증거는 아닙니다.

## 별도 etf-radar: 불완전한 계좌를 전액 현금으로 해석

**관찰:** 과거 리서치에서 계좌 정보가 불완전한데도 LLM이 전액 현금인 것처럼 해석한 사례가 있었습니다. 당시 실행 계획의 수량 변화는 미확정으로 남아 있었습니다. 자연어 해석과 실행 가능한 상태를 구분해야 했습니다.

**현재 코드에서 확인한 통제:** 계좌 snapshot의 명시적 현금·보유 평가 정보가 충분한지 확인하고, 불완전하면 숫자로 된 매매 수량을 산출하지 않습니다. ‘모르는 값’을 0으로 채우지 않습니다.

**근거 범위:** ETF Radar의 [계좌 snapshot](https://github.com/petruslihm/etf-radar/blob/main/screener/portfolio_units.py), [단위 계산](https://github.com/petruslihm/etf-radar/blob/main/screener/decision_plan.py)과 [합성 재현](https://github.com/petruslihm/etf-radar/blob/main/docs/CASES.md)을 공개했습니다. 이 저장소에도 평가 불가 보유의 `null` 보존 데모가 있지만, 두 프로젝트가 동일 코드라는 뜻은 아닙니다.

**한계:** 코드의 통제와 테스트 확인입니다. 모든 과거 LLM 응답의 계좌 해석을 바로잡았거나 실거래 오류를 통계적으로 줄였다고 주장하지 않습니다.

## 별도 etf-radar: 최종 JSON 단계에서 판단 조건이 변경

**관찰:** 조사 결과를 최종 구조로 정리하는 과정에서 새로운 조사 없이 조건이 바뀌는 사례가 있었습니다. 형식 변환이 판단 수정으로 이어지는 문제입니다.

**이전 대응:** 앞서 확인한 운영 버전에서는 초안 검증 후 `assert_draft_preserved`로 행동, 진입 범위, 추격 한도, 이벤트 조건, 종료 조건, 목표 비중, 현금 관련 필드를 비교했습니다. 지정 필드가 달라지면 보존 검사를 통과하지 못하는 방식이었습니다.

**현재 공개 버전:** workflow 5에서는 별도 초안 단계와 `assert_draft_preserved`를 없애고 조사 뒤 구조화 최종 판단으로 연결했습니다. `validate_final`이 대상·날짜·행동·순위·숫자 범위·종료 조건을 검사합니다. [현재 소스와 한계](https://github.com/petruslihm/etf-radar/blob/main/docs/CASES.md), [검증 기록](https://github.com/petruslihm/etf-radar/blob/main/docs/VALIDATION.md)에서 직접 확인할 수 있습니다. 이전 버전의 검사 결과를 현재 방식의 증거로 사용하지 않습니다.

**한계:** 현재 검사는 구조와 지정 조건을 확인합니다. 정정 응답의 모든 판단 의미가 원래 답변과 같은지, 조사 내용이 사실인지까지 보증하지 않습니다. 판단 단계 축소만으로 과거 문제가 완전히 해결됐다고 주장하지 않습니다.

## 공개 코드: 모델 지표와 출처 라벨의 과장 수정

**발견:** 배치 학습 기록에 Ridge의 MAE가 다른 모델에도 공통으로 사용되고 있었습니다. 작은 표본의 학습·평가 중복도 성능 지표처럼 해석될 수 있었습니다. 또한 출처 목록에 URL이 있거나 SEC 도메인이라는 이유로 `VERIFIED_SOURCE`를 붙이는 코드는 원문 검증 수준을 과장했습니다.

**수정:**

- 각 회귀 모델이 실제 사용한 입력과 자체 예측으로 시간순 분리 MAE를 계산합니다. LSTM은 시퀀스 구성에 맞춘 표본과 날짜를 사용합니다.
- 학습·평가 중복, 예측 기간에 필요한 날짜 간격 부족, 유효하지 않은 값은 평가 불가로 남깁니다. 순위 모델은 별도 순위 지표를 구현하기 전까지 수익률 MAE를 붙이지 않습니다.
- 이전 세대의 다른 평가 구간과 개선율을 계산하지 않습니다. 과거 공유 MAE는 원본 기록을 보존하되 모델별 성능 표시에서 제외합니다. Ridge 대체 학습은 대체 상태를 표시합니다.
- URL 목록 일치는 `CITED_URL`, SEC 호스트 일치는 `SEC_DOMAIN_ONLY`, 나머지는 `UNVERIFIED`로 표시합니다. 어느 라벨도 페이지 존재나 주장의 사실성을 보증하지 않습니다.

**근거:** [모델 평가](../src/trading_system/ml_engine.py), [모델 화면](../src/trading_system/ui/models_view.py), [출처 분류](../src/trading_system/judge_package.py), [평가 무결성 회귀 테스트](../tests/test_evaluation_integrity.py).

**한계:** 단일 시간순 분리 진단입니다. 반복 walk-forward, 완전한 과거 재현, 거래비용을 포함한 전략 성과, 모델 승격의 우월성을 입증하지 않습니다. 원문 내용의 검증은 추가 구현이 필요합니다.

[README](../README.md) · [평가](EVALUATION.md) · [현재 한계](LIMITATIONS.md)
