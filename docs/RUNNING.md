# 실행 안내

## 기본 UI: API 키 없이 시작

Python 3.12와 [uv](https://docs.astral.sh/uv/)를 설치하고 저장소 루트에서 실행합니다. 아래 명령은 PowerShell 기준입니다. 기본 UI에는 Node.js가 필요하지 않습니다.

```powershell
uv sync --extra dev --locked
uv run investassist
```

[http://127.0.0.1:8743](http://127.0.0.1:8743)을 열고 **지금 스캔 실행**을 누릅니다. 첫 실행에는 모델 학습 시간이 필요합니다. 다음 스캔은 호환되는 저장 모델을 재사용합니다. 서버 종료는 터미널에서 `Ctrl+C`입니다.

시세·LLM 키가 없는 새 checkout에서는 합성 시세를 사용합니다. 기존 `.env` 또는 프로세스 환경변수에 API 키가 있으면 해당 연동이 사용될 수 있습니다. 키 없는 실행은 시세·LLM 호출 없이 정량 경로를 확인하는 방법이며, SEC 공시나 UI 폰트 등까지 포함한 완전한 오프라인 모드는 아닙니다. SEC 연락처가 없거나 조회가 실패해도 실패 상태를 표시하고 계속합니다.

## 데모 화면을 재현하는 조건

README 이미지는 2026-09-08 공개본에서 생성한 합성 시세·기본 unit과 실제 FastAPI/Jinja 화면을 캡처했습니다. 이미지 편집으로 결과를 바꾸거나 LLM 보고서를 만들어 넣지 않았습니다. 실제 계좌·API 키·LLM live 결과는 사용하지 않았습니다.

1. `.env`, 기존 `data/`, 외부 서비스 키가 없는 **새 공개 checkout**에서 시작합니다. 사용 중인 데이터 폴더를 덮어쓰거나 삭제할 필요가 없습니다.
2. 자동 작업을 제외하고 직접 확인하려면 아래 두 환경변수를 같은 터미널에 설정한 뒤 시작합니다. 이 설정은 해당 터미널의 자식 프로세스에만 적용됩니다.
3. **지금 스캔 실행**이 끝나면 `/`, `/recommendations`, `/data-health`에서 캡처합니다. 이 데모에는 매수 기준을 통과한 주식이 없으며 LLM도 미설정입니다. 그 상태 자체가 실행 결과입니다.

```powershell
$env:DAILY_SCAN_ENABLED = "false"
$env:DAILY_TRAIN_ENABLED = "false"
uv run investassist
```

데이터 생성일·모델 상태에 따라 수치는 달라집니다. 같은 날짜의 GPT 재개와 앱 재시작에는 [알려진 상태 표시 한계](LIMITATIONS.md)가 있으므로 상태 배지만으로 live 호출 성공을 판단하지 않습니다.

| 이미지 | 화면 | 확인하는 것 |
| --- | --- | --- |
| [dashboard-demo.jpg](images/dashboard-demo.jpg) | `/` | 비중 요약, API 미설정·공시 사용 불가, 앱 지속 실행 |
| [recommendations-demo.jpg](images/recommendations-demo.jpg) | `/recommendations` | 합성 시세의 종목 점수, 조건 미달 시 현금 유지 |
| [data-health-demo.jpg](images/data-health-demo.jpg) | `/data-health` | 합성 history 사용 안내와 최근 기능 상태 |

## 실제 데이터와 선택적 LLM

지원 설정은 [`.env.example`](../.env.example)에 있습니다. `.env`는 Git에서 제외됩니다. 이미 있다면 기존 파일에서 필요한 값만 수정합니다. 키는 Settings 화면에서도 저장할 수 있습니다.

| 연동 | 설정 | 범위 |
| --- | --- | --- |
| Alpaca | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | 시장 데이터 전용. Trading client 없음 |
| SEC EDGAR | `SEC_CONTACT_EMAIL` | Fair-access User-Agent 연락처. 로컬 `.env`에만 설정 |
| Gemini | `GEMINI_API_KEY`, `GEMINI_RESEARCH_MODEL` | 신규 주식 후보의 근거 조사 |
| OpenAI | `OPENAI_API_KEY`, `LLM_JUDGE_MODEL` | 포트폴리오 서술형 검토 |
| Kakao | `.env.example`의 Kakao 설정 | 선택적 개인용 ‘나에게 보내기’ 알림 |

Settings에서 저장한 뒤 앱을 재시작하면 새 설정을 일관되게 적용할 수 있습니다. Anthropic credential을 저장하는 기능은 있지만 현재 scan의 판단 경로에는 참여하지 않습니다. 외부 모델 사용 가능 여부는 계정 권한·제공자 정책에 의존합니다.

LLM을 켜면 종목·입력 근거와 포트폴리오 맥락이 외부 API로 전송됩니다. Unit 표현만으로 입력이 익명화되지는 않습니다. 회사나 고객의 내부 데이터를 투입하도록 설계된 배포가 아닙니다.

## 스케줄과 수동 실행

| 동작 | 기본값과 범위 |
| --- | --- |
| `uv run investassist` | UI와 프로세스 내부 일일 스케줄러 시작 |
| `uv run investassist --init` | 초기화·분석 한 번 후 종료. 모델이 없으면 학습 |
| `uv run investassist --live` | 180초 주기 watchlist/BTC 갱신, 900초 주기 broad scan |
| **지금 스캔 실행** | 수동 분석. 자동 LLM 예산에 합산되지 않음 |
| **모델 학습** | Model-only 재학습. SEC·Gemini·GPT 호출 없음 |

일일 스캔은 NYSE 거래일의 장 시작 60분 전부터, model-only 학습은 장 종료 60분 뒤부터 실행 가능 여부를 확인합니다. 프로세스가 실행 중이어야 하며, 정상 완료 여부와 슬롯 상태에 따라 실행을 제한합니다. 작업을 수행하지 못한 일부 실패는 슬롯을 반환해 재시도할 수 있고 일별 시도 횟수는 제한됩니다. 정확한 규칙은 [`daily_scan.py`](../src/trading_system/daily_scan.py)에 있습니다. UI의 ‘1회’ 문구만으로 모든 실패 후 재시도가 금지된다고 해석하지 않습니다.

자동 Gemini·OpenAI 호출은 UTC 날짜별 기본 `$15` 예상 비용 기준을 다음 호출 전에 검사합니다. 실제 결제액의 hard cap이 아닙니다. `--init`과 사용자가 누른 스캔은 자동 예산 집계 대상이 아닙니다.

Windows에서 [`START_STOCK_AI.bat`](../START_STOCK_AI.bat)을 사용할 수 있습니다. 이 보조 파일은 `py -3.12 -m uv`를 호출하므로 Python launcher와 해당 Python에 설치된 uv 모듈이 필요합니다(`py -3.12 -m pip install uv`). 별도 실행 파일로 설치한 uv만 있다면 위의 `uv run investassist`를 사용하세요. [`install_daily_scan_task.ps1`](../scripts/install_daily_scan_task.ps1)은 Windows 시작 시 실행을 등록하는 선택 기능입니다. 기본 설치 과정에는 작업 스케줄러 등록이 필요하지 않습니다.

## 선택적 React client

React client는 기본 Jinja 화면과 동일한 완성도의 대체 UI가 아니라, JSON API를 조회하는 간단한 별도 client입니다. Node.js/npm이 필요합니다. 기본 FastAPI 서버를 먼저 시작하고 별도 터미널에서 실행합니다.

```powershell
cd ui
npm ci
npm run dev
```

[http://localhost:5173/app/](http://localhost:5173/app/)에서 확인합니다. Vite의 base는 `/app/`이며 `/api` 요청을 `127.0.0.1:8743`에 전달합니다. 보유 내역·credential 편집은 기본 UI를 사용합니다.

```powershell
npm run build
```

빌드는 `src/trading_system/ui/static/app/`에 결과를 생성합니다. **FastAPI 서버를 재시작**한 뒤 [http://127.0.0.1:8743/app/](http://127.0.0.1:8743/app/)에서 볼 수 있습니다. 이 경로는 빌드 디렉터리가 있을 때 서버 시작 시 등록됩니다. URL 전환은 HashRouter를 사용합니다.

의존성의 알려진 경고와 노출 범위는 [검증 기록](VALIDATION.md)에 있습니다. 개발 서버를 외부에 노출하는 `--host` 옵션은 실행 안내에 포함하지 않습니다.

## 실행 확인과 문제 확인

```powershell
uv run investassist --help
Invoke-RestMethod http://127.0.0.1:8743/api/health
```

Health의 `ok: true`, `trading: false`는 앱의 응답과 주문 경계를 나타내며 데이터나 LLM의 연결 성공을 보장하지 않습니다. 데이터 상태·런타임·LLM 기록 화면을 함께 확인합니다. 포트가 사용 중이거나 DB가 다른 프로세스에서 열려 있다면 같은 checkout의 중복 실행 여부를 확인하세요.

[프로젝트 개요](../README.md) · [설계](ARCHITECTURE.md) · [한계](LIMITATIONS.md)
