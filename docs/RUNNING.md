# 실행 안내

## 키 없는 제출 데모

Python 3.12와 uv를 설치한 뒤 저장소 루트에서:

```powershell
uv sync --extra dev --locked
uv run investassist --demo
```

localhost:8744에서 확인합니다. 기본 `uv run investassist`도 데모입니다. 포트 충돌 시 `uv run investassist --demo --port 8754`로 바꿀 수 있습니다. `--data-dir data/another-demo`로 별도 데모 기록을 만들 수 있습니다. 종료는 Ctrl+C입니다.

데모는 외부 키가 있어도 제공자를 호출하지 않습니다. 모든 단계 출력은 합성 예제이며, 실제 모델 학습·추론 결과가 아닙니다. 단위는 가상 배분단위이고 모든 추천은 실행 불가입니다. `/resume`과 `/api/demo/snapshot`은 마지막 저장 결과를 읽습니다.

## 선택적 연구 UI

```powershell
Copy-Item .env.example .env
# .env에 필요한 제공자 설정을 입력합니다. 비밀값을 Git에 추가하지 마세요.
uv run investassist --app
```

localhost:8743의 연구 UI에서 설정과 수동 스캔을 사용합니다. `.env`가 이미 있으면 복사 명령 대신 기존 파일을 편집합니다. Alpaca는 시장 데이터, `SEC_CONTACT_EMAIL`은 공시 요청의 연락처, Gemini/OpenAI 키는 조사·최종 판단에 사용됩니다. 제공자가 미설정이면 해당 기능의 미설정 상태를 표시합니다. 키 없는 합성 데이터 경로도 있지만 첫 연구 스캔은 모델 학습을 수행하므로 제출 확인에는 데모를 권장합니다.

이 경로는 외부 호출과 비용을 유발할 수 있습니다. 자동 일일 스케줄은 앱이 실행되는 동안 활성화될 수 있습니다. UTC 일일 기본 $15 soft budget은 자동 Gemini/OpenAI 호출 예상 비용 기준이며 결제 상한이 아닙니다. 수동 스캔은 자동 예산에 합산되지 않습니다. 실제 API 호출은 제출 검증에서 수행하지 않았습니다.

기존 `START_STOCK_AI.bat`·스케줄 스크립트는 연구 UI용 보조 도구입니다. 제출 데모에는 필요하지 않습니다. 운영 DB·기록을 이 저장소에 가져오지 않습니다.

## 테스트와 패키징

```powershell
uv run pytest tests/test_submission_demo.py tests/test_openai_judge.py tests/test_v1_pipeline.py -q
uv run pytest tests/test_price_provenance.py -q
uv run pytest tests/test_evaluation_integrity.py -q
uv lock --check
uv build
```

전체 회귀 검사는 `uv run pytest -q`입니다. 테스트 데이터·DB는 임시 디렉터리에 생성됩니다. 배포나 외부 주문 기능은 없습니다.

## 선택적 React client

기본 데모와 연구 UI에 Node.js는 필요하지 않습니다. React는 연구 API 조회용 최소 client이며 데모의 단계 비교 UI를 대체하지 않습니다. `ui`에서 `npm ci`, `npm run build`로 빌드할 수 있습니다. 이번 제출에서는 Python 기본 경로를 검증하며 기존 React 의존성 경고는 검증 문서에 구분합니다.
