# 제출 검증 기록

2026-09-10, Windows / Python 3.12.10. 기본 제출 경로는 DEMO / SYNTHETIC이며 실제 운영·유료 API·주문·배포는 검증 대상에서 제외했습니다.

## 설치·실행·패키징

- 기존 환경과 분리한 새 가상환경에서 `uv sync --extra dev --locked`로 54개 패키지 설치 구성을 확인했습니다. 저장소의 uv.lock을 사용했습니다.
- README의 `uv run investassist --demo`로 loopback 데모를 실행하고 브라우저에서 정상·호출 실패·평가 불가 시나리오를 확인했습니다. 실제 화면을 README 이미지로 저장했습니다.
- 기본 CLI는 키가 없어도 데모를 시작하며, 테스트에서 키가 설정된 환경에서도 외부 제공자 연결이 없는 것을 확인했습니다.
- `investassist --help`를 확인했고 Windows cp949 콘솔에서 출력할 수 있는 도움말을 회귀 테스트로 남겼습니다.
- `uv lock --check`, `uv build`로 잠금파일·sdist·wheel을 확인했습니다. 별도 환경에 wheel을 설치해 소스 checkout이 아닌 site-packages에서 데모 HTML·정적 CSS·저장 snapshot을 읽는 것을 확인했습니다.

## 핵심 회귀 테스트

테스트는 영향 범위별로 실행했습니다. 전체 테스트 수를 투자 성과나 외부 API 검증처럼 해석하지 않습니다.

| 범위 | 결과·의미 |
|---|---|
| `test_openai_judge.py`, `test_v1_pipeline.py` | 53개 통과. 구조화 요청·실패·기존 연구 실행·단계 기록·재개 경로; 외부 호출 대역 사용 |
| `test_submission_demo.py` | 8개 통과. 방향·순위·배분 상한·설명·카드·null·실패 대체·저장 복원·환경 키 무시·CLI 문자 인코딩 |
| `test_recommendations_page.py` | 20개 통과. 기존 추천 카드·상세 설명·요약 렌더링 호환성 |
| `test_price_provenance.py` | 2개 통과. 당일 시세 수정 시 피처 갱신, 미완성 거래량 제외, 확정 종가/관측과 수집 시각 구분 |

초기 데모 네트워크 차단 테스트가 Windows asyncio의 내부 loopback socket pair까지 차단해 실패했습니다. 외부 연결 차단은 유지하고 내부 loopback만 허용한 뒤 데모 테스트가 통과했습니다. 브라우저에서 발견한 대체 결과의 잘못된 0% 추적 안내는 effective 요약으로 통일하고 회귀 검증했습니다.

실제 제공자 호출은 한 번도 수행하지 않았습니다. 모델 학습·예측 성능·수익률 비교를 수행한 결과가 아닙니다. 기존 전체 427개 테스트 통과 기록은 이전 공개본의 기록이며 이번 변경 전체의 검증 결과로 재사용하지 않았습니다. 현재 전체 테스트 실행 방법은 `uv run pytest -q`입니다.

## 공개 범위

추적 파일과 새 제출 파일, 로컬 Git의 도달 가능한 이력 blob을 검사했습니다. 인증정보 패턴·개인 이메일·사용자 로컬 절대경로·운영 DB/로그 파일명 후보를 검사하며 비밀값은 출력하지 않았습니다. 발견된 실제 비밀값·개인 운영 자료는 없습니다. example.com과 GitHub noreply 주소는 예제/커밋 식별자로 구분합니다.

운영 DB, 실제 LLM 응답 fixture, 개인 보유 자료, 운영 대기 기록 변환·백업 문서는 이식하지 않았습니다. `data/`, `.env`, DB·모델·가상환경·빌드 산출물은 Git에서 제외합니다. 새 화면의 ALPHA/BETA/GAMMA는 가상 예제입니다. 원격 push·별도 원격 첨부자료 검사는 수행하지 않았습니다.

## 남은 한계

- 실제 API 모델·응답 형식·출처의 사실성·현재 비용은 미검증입니다.
- 기존 FastAPI/Starlette TestClient의 httpx deprecation 경고가 남아 있습니다.
- 선택적 React/Vite 의존성에는 이전에 moderate/high 경고가 보고됐습니다. 기본 Python 데모에 사용되지 않으며 이번 작업에서 React를 빌드하거나 해당 경고를 해결했다고 주장하지 않습니다.
- 완전한 과거 시점 복원, 배분·거래비용을 반영한 초과수익 검증, 공개 서비스 배포는 범위 밖입니다.

[실행 안내](RUNNING.md) · [설계](ARCHITECTURE.md) · [한계](LIMITATIONS.md)
