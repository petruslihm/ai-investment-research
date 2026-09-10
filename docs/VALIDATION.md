# 검증 기록

검증일: **2026-09-10**. Windows / Python 3.12.10 / uv 0.12.6. 공개 합성 데모와 코드·패키징을 대상으로 검사했습니다. 이 작업에서 실제 제공자 API나 주문을 실행하지 않았습니다.

## 재현 명령

저장소 루트에서 실행합니다. 스레드 설정은 테스트 실행 자원을 제한하기 위한 값이며 모델·전략 설정을 바꾸지 않습니다.

```powershell
uv sync --locked --extra dev
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
uv run --locked pytest -q
uv lock --check
uv build
```

**최종 전체 회귀: 442개 통과, 경고 1개, 229.55초.** 경고는 아래에 기록한 TestClient의 httpx deprecation입니다. 관련 검사만 통과한 결과를 전체 결과로 대신하지 않았습니다.

## 핵심 확인 범위

| 검사 | 확인한 동작 |
|---|---|
| `test_submission_demo.py` | 방향·순위·배분 상한·설명·카드·null·실패 대체·저장 복원·환경 키 무시 |
| `test_openai_judge.py`, `test_v1_pipeline.py` | 구조화 요청·실패 처리·단계 기록·동일 입력 재개; 외부 호출 대역 사용 |
| `test_price_provenance.py` | 확정 종가/장중 관측 구분, 피처 갱신, 미완성 거래량 제외 |
| `test_evaluation_integrity.py` | 모델별 자체 평가 입력·지표, LSTM 표본 수, 중복/간격 부족 시 평가 보류, 과거 공유 지표 표시 제외, 출처 라벨 |
| `test_features_extended.py` | 실험 피처의 저장·복원과 과거 데이터 prefix 기반 재계산 |
| `scripts/smoke_wheel.py` | 설치된 wheel의 데모 HTML·CSS·제약·저장 snapshot 복원 |

데모 테스트는 외부 socket 연결을 금지하고 Windows asyncio가 사용하는 내부 loopback만 허용합니다. 테스트용 DB와 모델 산출물은 임시 디렉터리를 사용합니다. 테스트 통과는 실제 API의 가용성·사실성이나 투자 수익률 검증이 아닙니다.

## 설치·패키지 검증

- 새 가상환경에서 `uv sync --locked --extra dev`로 잠금파일의 54개 패키지를 설치했습니다.
- `uv lock --check`와 `uv build`가 통과했습니다. sdist에서 wheel을 빌드합니다.
- 해당 환경의 editable 설치를 생성한 wheel로 교체한 뒤, 저장소 밖의 작업 디렉터리에서 `python -I`로 smoke 검사를 실행했습니다. import가 `site-packages`에서 오는지 검사하고 HTML·CSS·배분 한도·DB 저장 복원을 확인했습니다.
- [GitHub Actions](https://github.com/petruslihm/ai-investment-research/actions/workflows/ci.yml)에는 Windows / Python 3.12에서 잠금파일 설치, 전체 pytest, 빌드, 설치 wheel 검사를 수행하는 워크플로를 추가했습니다. 원격 실행 결과는 해당 실행의 상태와 커밋을 기준으로 확인할 수 있습니다.

## 검사 중 발견한 문제와 수정

모델 지표와 출처 라벨의 문제는 [사례 문서](CASE_STUDIES.md)에 기록했습니다. 회귀 테스트는 서로 다른 모델 예측값을 넣어 같은 Ridge MAE가 다른 모델에 복사되지 않는지 확인합니다. 소표본 학습·평가 중복과 날짜 간격 부족은 지표를 남기지 않아야 합니다.

첫 전체 실행은 441개 통과·1개 실패였습니다. 기존 확장 피처 테스트 fixture가 확정 일봉 상태를 `FINAL`로 넣었지만, 실제 enum과 조회 계약은 `final`이었습니다. fixture를 계약에 맞게 수정한 뒤 관련 검사 10개가 통과했습니다. 운영 쿼리에서 미완성 일봉을 허용하는 방식으로 우회하지 않았습니다.

## 공개 범위 검사

이번 변경·추가 파일에서 인증정보 형식과 개인 로컬 절대경로 패턴을 검사했고 후보를 발견하지 못했습니다. Markdown의 저장소 내부 파일 링크도 확인했습니다. 이는 범위가 정해진 패턴 검사이며 모든 비밀정보를 탐지하는 보증은 아닙니다.

개인 계좌, 운영 DB, 실제 LLM 응답 원본을 이식하지 않았습니다. 별도 `etf-radar` 사례는 [집계·관찰](EVALUATION.md)로만 설명하며, 그 테스트 수를 공개본의 전체 결과에 합산하지 않습니다. 기존 Git 이력을 재작성하지 않고 개선 내용을 추가합니다.

## 남아 있는 한계

- FastAPI/Starlette TestClient의 httpx deprecation 경고가 있습니다.
- 실제 API 모델·도구 지원·응답 품질·현재 비용은 이번 공개본 검사에서 미검증입니다.
- 선택적 React/Vite 의존성에는 이전에 moderate/high 경고가 보고됐습니다. 기본 Python 데모에서 사용하지 않으며 이번 작업에서 React 빌드나 해당 경고 해결을 수행했다고 주장하지 않습니다.
- 완전한 과거 시점 복원, 전략 수익률 비교, 다중 사용자 서비스 배포는 검증하지 않았습니다.

[실행 안내](RUNNING.md) · [평가](EVALUATION.md) · [구조](ARCHITECTURE.md) · [한계](LIMITATIONS.md)
