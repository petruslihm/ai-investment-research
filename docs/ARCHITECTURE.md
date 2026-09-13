# 구조와 코드 탐색

## 실행 경로

평가용 Windows 실행기는 Python과 의존성을 준비한 뒤 `scripts/launch_desktop.py`에서 실제 FastAPI/Jinja 연구 UI를 엽니다. Alpaca·Gemini·OpenAI 키가 빠져 있으면 설정 화면으로 안내하고 분석 요청을 차단합니다. 자동 스캔·학습은 시작하지 않으며, 비어 있는 로컬 포트를 사용합니다.

개발자가 직접 실행할 때는 `investassist --app`을 사용합니다. 연구 경로의 실제 실행은 `v1_cycle.run_v1_cycle`입니다. 완료 시세 → 피처 → 모델 예측 → 기술 필터 → Quant 배분 → Gemini 조사·배분 보정 → 구조화 GPT → 제약 → effective → 원자적 tick 저장 순서입니다. 개발 배경은 [개발 경위](DEVELOPMENT.md)에 설명합니다.

기존 `demo.py`와 `--demo`는 가상 입력의 회귀 검사 경로로 남아 있습니다. 평가자에게 제공하는 실행기는 이 경로를 사용하지 않습니다.

## 코드를 짧게 보는 순서

| 파일 | 확인할 설계 |
|---|---|
| `scripts/windows_bootstrap.ps1`, `scripts/launch_desktop.py` | Python 준비·실제 UI 시작·필수 API 확인 |
| `src/trading_system/market/decision_data.py` | 확정 종가와 미완성 관측·가격 출처·입력 fingerprint |
| `src/trading_system/recommendations.py` | 구조화 추천과 미보유/보유/평가 불가 의미 |
| `src/trading_system/final_decision.py` | 종목 집합·방향·단위·순위 검증, 제약 적용, canonical report |
| `src/trading_system/v1_cycle.py` | 단계별 기록과 같은 입력에 대한 재개 |
| `src/trading_system/storage/ticks.py` | transaction·idempotent tick commit |
| `tests/test_submission_demo.py` | 실패·null·상한·저장·복원·외부 호출 금지 |
| `src/trading_system/ml_engine.py` | 모델별 평가 입력·시간순 분리 지표·평가 불가 상태 |
| `src/trading_system/judge_package.py` | 출처 목록 일치와 SEC 호스트 분류; 원문 사실 검증은 수행하지 않음 |
| `tests/test_evaluation_integrity.py` | 모델별 지표·평가 중복 차단·과거 공유 지표 표시 제외·출처 라벨 |

## 기록 계약

- `quant_only`: 조사 이전 원본.
- `research_adjusted`: Gemini 반영 후 별도 추천.
- `llm_final`: 검증된 GPT 판단과 제약 결과. `requested_units`와 적용 단위를 함께 기록합니다.
- `effective`: 실제 표시할 전체 결과. GPT 미심사 종목 또는 실패 대체는 출처·상태를 별도로 표시합니다.

`decision_comparison_v1` 관측에는 입력 ID·시각·각 단계와 화면 snapshot이 같은 tick transaction으로 저장됩니다. 앱이 다시 열리면 이 snapshot을 읽어 서로 다른 실행의 수치를 섞지 않습니다. 현재 계약은 `final_decision_v2`이며 공개본은 현재 계약·동일 입력 guard만 재개합니다. 운영용 과거 패키지 마이그레이션은 포함하지 않습니다.

판단과 배분의 검증은 통계적 예측 확률 검증이 아닙니다. LLM 품질·자신감 점수는 자가평가이며 상승 확률로 보증하지 않습니다.
