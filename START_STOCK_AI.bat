@echo off
setlocal
title Stock AI
cd /d "%~dp0"

rem Probe server health without reading analysis data or triggering a scan.
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8743/api/health' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto OPEN_UI

echo Starting Stock AI on http://127.0.0.1:8743 ...
start "Stock AI Server" cmd /k "py -3.12 -m uv run investassist"

for /l %%I in (1,1,40) do (
    powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8743/api/health' -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
    if not errorlevel 1 goto OPEN_UI
    timeout /t 1 /nobreak >nul
)

echo Server did not become ready. Check the Stock AI Server window for errors.
pause
exit /b 1

:OPEN_UI
start "" http://127.0.0.1:8743
exit /b 0
