# 검증 기록

2026-09-08 공개본 준비 과정에서 실행한 검사입니다. 테스트 수와 도구 버전은 이 시점의 기록이며, 이후 변경의 성공이나 투자 성과를 보장하지 않습니다.

## 확인한 결과

환경: Windows, Python 3.12.10, uv 0.12.6, Node.js 24.19.0, npm 11.17.0.

| 검사 | 결과 |
| --- | --- |
| 새 Python 가상환경에서 `uv sync --extra dev --locked` | 설치 성공 |
| `uv run pytest -q` | **427 passed**, warning 1개, 224.74초 |
| `uv lock --check` | 성공 |
| `uv build` | sdist·wheel 생성 성공 |
| React `npm ci` | lockfile 기준 clean install 성공 |
| React `npm run build` | TypeScript 검사·production build 성공 |
| `uv run investassist --help` | 정상, `--init`·`--live` 옵션 확인 |
| `GET /api/health` | HTTP 200, `ok=true`, **`trading=false`** |
| 로컬 UI GET 검사 | 아래 8개 경로 모두 HTTP 200 |

UI 확인 경로: `/`, `/portfolio`, `/recommendations`, `/models`, `/runtime`, `/data-health`, `/llm`, `/app/`.
서버 주소는 `http://127.0.0.1:8743`입니다. `/app/`은 production build 후 제공되는 선택적 React 화면입니다.

Python warning은 FastAPI/Starlette TestClient의 httpx 사용에 관한 deprecation 1건입니다. 현재 테스트 실패는 없으며, 후속 의존성 정비 때 확인할 항목입니다.

## 재현 명령

저장소 루트에서 실행합니다. 테스트와 fixture 실행에 외부 API key는 필요하지 않습니다.

```powershell
uv sync --extra dev --locked
uv lock --check
uv run pytest -q
uv build
uv run investassist --help
```

선택적 React client를 빌드합니다.

```powershell
cd ui
npm ci
npm run build
npm audit
cd ..
uv run investassist
```

서버를 실행한 상태에서 다른 터미널로 확인합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8743/api/health
```

화면 응답 검사에서 분석 실행이나 외부 모델 호출을 유발하는 endpoint는 사용하지 않았습니다. HTTP 200은 접근 가능성을 확인하는 검사이며, 모든 화면 동작을 검증하는 E2E 테스트와 같지 않습니다.

## 의존성 점검과 남은 과제

같은 날짜의 `npm audit --json` 결과는 **moderate 3개, high 1개, critical 0개**입니다. 이는 취약점이 연결된 package 집계이며, 고유 advisory 개수와 다릅니다. audit은 경고 때문에 종료 코드 1을 반환했습니다.

| 항목 | 현재 구성에 대한 검토 |
| --- | --- |
| Vite 개발 서버 | [Windows 파일 접근 우회](https://github.com/advisories/GHSA-fx2h-pf6j-xcff)는 외부 네트워크 노출 등 적용 조건이 있습니다. 현재 설정에는 `--host`나 `server.host`가 없고, 배포용 화면은 FastAPI가 빌드된 정적 파일로 제공합니다. |
| React Router navigation | [외부 경로로의 잘못된 이동](https://github.com/advisories/GHSA-wrjc-x8rr-h8h6)은 공격자가 제공한 경로의 사용과 관련됩니다. 현재 `NavLink`는 코드에 정의된 고정 경로를 사용합니다. |
| React Router hydration | [SSR 오류 복원 취약점](https://github.com/advisories/GHSA-337j-9hxr-rhxg)은 특정 SSR/hydration 흐름과 관련됩니다. 현재 React 화면은 `HashRouter`·client rendering을 사용하며 해당 SSR 흐름은 없습니다. |
| esbuild | [개발 서버의 응답 노출](https://github.com/advisories/GHSA-67mh-4wv8-2f99)은 esbuild serve 기능에 관한 경고입니다. 저장소에서 이 기능을 직접 실행하는 script는 없습니다. |

추가로 Vite의 [source map 경로 처리](https://github.com/advisories/GHSA-4w7w-66w2-5vf9), [Windows UNC 경로 처리](https://github.com/advisories/GHSA-v6wh-96g9-6wx3) 경고가 포함됩니다. 위 구성 검토는 노출 조건을 이해하기 위한 것으로, 취약점이 없거나 영향이 전혀 없다는 판정은 아닙니다.

audit이 제안한 해결 경로는 Vite와 React Router의 major version 변경입니다. 이번 공개 문서 정리에서는 자동 강제 업데이트를 적용하지 않았습니다. 후속 작업에서 호환성 확인과 UI 회귀 검증을 거쳐 해결해야 하며, 개발 서버를 외부에 노출하는 운영 방식은 검증 대상이 아닙니다.

## 공개 범위 점검

tracked 파일 **134개**, 공개 root commit **1개** 및 로컬 Git object 전체를 검사했습니다. 실제 credential·개인 이메일·개인 절대경로·금지된 내부 문서나 데이터 산출물은 발견되지 않았고, author·committer의 GitHub noreply 주소를 확인했습니다. 예제 이메일과 dependency artifact 숫자는 개인정보 후보에서 구분했습니다.

추가 문서의 상대 링크 84개와 fixture 화면 JPEG 3개를 확인했습니다. 이미지는 모두 1265×712이며 EXIF/XMP metadata와 눈에 보이는 계좌정보·개인정보·secret이 없었습니다. `git fsck --full --no-reflogs`에서도 잔여 object 관련 문제가 보고되지 않았습니다.

## 검증의 한계

- 테스트는 시점 정합성, outcome maturity, cache/resume 경계, allocation, 실패 처리, UI 입력·credential 노출 등의 회귀를 확인합니다. 모든 장애 조합이나 모든 형식의 개인정보 유출을 증명하는 검사는 아닙니다.
- fixture·mock 기반 검사와 화면은 실제 시장 성과, 외부 provider의 실시간 가용성, LLM의 사실 정확성을 입증하지 않습니다.
- 출처 구분과 outcome 지원 코드가 있어도 현행 GPT 경로는 구조화된 `llm_final` 추천을 만들지 않습니다. 순수 정량 대조군 보존과 행동·배분·비용을 반영한 평가부터 보완해야 하며, 현재 LLM 도입 효과가 입증된 것은 아닙니다.
- 이번 실행 기록에는 실거래, 실제 주문, 투자 수익률 검증, 금융기관 배포 수준의 보안 심사나 부하 테스트가 포함되지 않습니다.
