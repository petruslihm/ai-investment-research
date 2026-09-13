# 문제 해결 사례

이 문서는 **두 번째 프로그램인 이 저장소에서 확인할 수 있는 구현**을 설명합니다. 첫 번째는 [초기 ETF Radar](https://github.com/petruslihm/first-etf-radar)이며, 세 번째 개인 운용 프로그램은 비공개입니다. 개발 순서와 차이는 [세 프로그램 비교](DEVELOPMENT.md)에 정리했습니다.

## 반복되는 수동 입력을 API로 연결

첫 번째 프로그램은 후보와 GPT용 프롬프트를 만들고 사용자가 직접 입력하는 방식이었습니다. 두 번째는 Quant 후보를 Gemini 조사와 GPT 검토에 연결하고, 각 단계의 입력과 결과를 남깁니다.

신규 주식 후보의 Gemini 조사 결과는 배분 보정에도 사용합니다. 최종 GPT 결과는 구조를 검증한 후 화면에 반영합니다.

**근거:** [단계 실행](../src/trading_system/v1_cycle.py), [기업 조사](../src/trading_system/research_agent.py), [GPT 판단](../src/trading_system/openai_judge.py).

## 재시도할 때 이전 대화와 새 입력이 섞이는 문제

중단된 GPT 분석을 단순히 다시 실행하면 시세·점수·후보·계좌 상태가 달라질 수 있습니다. 이어가는 대화가 참조한 입력과 새 입력이 다르면 같은 분석이라고 보기 어렵습니다.

후보·계좌·조사 결과를 입력 묶음으로 고정하고, 재개할 때 일봉·가격·보유 상태·배분 정책 등의 조건을 비교합니다. 동일 입력이면 저장한 묶음을 사용하며, 입력이 달라지거나 현재 계약으로 해석할 수 없으면 그대로 재개하지 않습니다.

**근거:** [입력 저장과 대화 재개](../src/trading_system/openai_judge.py), [재개 조건과 입력 복원](../src/trading_system/v1_cycle.py), [계약 검사](../src/trading_system/final_decision.py).

## 판단과 화면을 같은 결과에 연결

AI가 행동이나 배분을 바꿔도 카드·요약이 이전 Quant 값으로 남으면 설명과 표시 결과가 달라집니다. 실패한 GPT 단계까지 완료된 판단처럼 표시할 위험도 있습니다.

종목 집합·행동·단위·순위를 검증한 뒤 제약을 적용하고, `effective`를 카드·설명·정렬·요약의 기준으로 저장합니다. 실패 대체에는 출처를 표시합니다. 같은 입력 ID의 단계 기록과 화면 snapshot을 하나의 tick transaction으로 저장합니다.

**근거:** [판단 검증](../src/trading_system/final_decision.py), [tick 저장](../src/trading_system/storage/ticks.py), [회귀 테스트](../tests/test_submission_demo.py).

회귀 테스트는 가상 입력으로 한도 적용·실패 대체·복원을 확인합니다. 실제 API 판단의 우수성을 증명하는 결과는 아닙니다.

## 평가할 수 없는 보유 금액을 0으로 오인하지 않기

취득 기록은 있지만 현재 평가 가격이 없는 자산은 미보유나 전액 현금으로 처리할 수 없습니다. 보유 여부·취득 단위·현재 평가 단위를 구분하고, 평가 불가 금액은 `null`로 보존합니다. 전체 포트폴리오 비중도 확정할 수 없으면 미확정으로 표시합니다.

**근거:** [보유 상태 정규화](../src/trading_system/recommendations.py), [최종 판단 처리](../src/trading_system/final_decision.py), [미확정 값 회귀 검사](../tests/test_submission_demo.py).

세 번째 프로그램의 계좌 snapshot·매매 단위 계산은 별도 구현입니다. 그 비공개 코드의 검사를 이 저장소의 증거로 사용하지 않습니다.

## 모델 지표와 출처 표시의 과장 수정

배치 학습 기록에서 Ridge의 MAE가 다른 모델에도 공유되던 문제를 수정했습니다. 각 회귀 모델의 입력과 예측으로 시간순 분리 MAE를 계산하고, LSTM은 시퀀스에 맞는 표본을 사용합니다. 표본 부족·학습과 평가의 중복·예측 기간에 필요한 간격 부족은 평가 불가로 남깁니다.

순위 모델에 수익률 MAE를 붙이지 않으며, 다른 평가 기간의 모델끼리 단순 개선율을 계산하지 않습니다. 과거 공유 지표는 현재 모델별 성능에서 제외하고 Ridge 대체 학습은 대체 상태를 표시합니다.

출처 목록의 URL 일치는 `CITED_URL`, SEC 호스트 일치는 `SEC_DOMAIN_ONLY`, 나머지는 `UNVERIFIED`로 구분합니다. URL 목록에 있다는 사실만으로 원문 주장의 사실성을 검증한 것은 아닙니다.

**근거:** [모델 평가](../src/trading_system/ml_engine.py), [모델 화면](../src/trading_system/ui/models_view.py), [출처 분류](../src/trading_system/judge_package.py), [평가 무결성 검사](../tests/test_evaluation_integrity.py).

이 지표는 모델 진단이며 실제 거래비용을 반영한 전략 수익률 비교가 아닙니다.

[README](../README.md) · [평가 범위](EVALUATION.md)
