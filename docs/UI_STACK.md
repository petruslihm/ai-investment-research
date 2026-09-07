# UI stack

## 구성

**Backend:** FastAPI (MIT)가 JSON API와 별도 build 없이 사용할 수 있는 Jinja HTML shell을 제공합니다.

**선택적 frontend:** Vite + React + TypeScript를 사용하며 data grid에는 TanStack Table을 사용합니다.

설정에는 `ui_stack = "fastapi_vite_react"`로 기록됩니다.

기본 제품 화면은 Jinja shell이며, React는 JSON API 조회를 위한 간단한 선택 client입니다. 두 화면의 기능 범위는 같지 않습니다.

## License

| Component | License | 역할 |
|-----------|---------|------|
| FastAPI | MIT | Local HTTP API와 shell |
| Uvicorn | BSD-3-Clause | ASGI server |
| Jinja2 | BSD-3-Clause | Server-rendered shell template |
| Vite | MIT | Frontend bundler |
| React | MIT | UI component |
| React Router | MIT | Client route |
| TanStack Table | MIT | Permissive table/grid library |
| TypeScript | Apache-2.0 | Typed frontend |

## 위치

- Python shell: `src/trading_system/ui/` — Dashboard, Portfolio, Stock Recommendations, BTC Signal, Models/Learning, Runtime Activity, Data Health, Settings route
- 선택적 React client: `ui/` — development server가 `/api`를 FastAPI로 전달

## 실행

```powershell
uv run investassist          # FastAPI UI: http://127.0.0.1:8743
cd ui
npm ci
npm run dev                  # 선택적 Vite/React: http://127.0.0.1:5173
```

Vite base는 `/app/`입니다. 개발 시 `http://localhost:5173/app/`을 사용합니다. Production build를 만든 뒤 FastAPI를 재시작하면 `http://127.0.0.1:8743/app/`에서도 열 수 있습니다. 자세한 설정과 빌드 순서는 [실행 안내](RUNNING.md)를 참고하세요.
