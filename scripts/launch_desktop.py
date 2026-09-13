"""Open the actual research UI after the Windows bootstrap prepares Python."""
from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check', action='store_true', help='Start, check local pages, then stop')
    args = parser.parse_args()
    # Opening the evaluator's app must not start scans, training, or provider calls.
    os.environ['INVESTASSIST_DAILY_SCAN'] = '0'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    import uvicorn
    from fastapi.responses import JSONResponse, RedirectResponse
    from trading_system.config import get_settings
    from trading_system.ui.app import create_app

    app = create_app()

    def missing_required():
        cfg = get_settings()
        return [name for name, value in (
            ('ALPACA_API_KEY', cfg.alpaca_api_key),
            ('ALPACA_SECRET_KEY', cfg.alpaca_secret_key),
            ('GEMINI_API_KEY', cfg.gemini_api_key),
            ('OPENAI_API_KEY', cfg.openai_api_key),
        ) if not (value or '').strip()]

    @app.middleware('http')
    async def require_research_keys(request, call_next):
        if request.url.path in ('/run-once', '/train-once', '/api/v1/snapshot'):
            missing = missing_required()
            if missing:
                if request.url.path.startswith('/api/'):
                    return JSONResponse({'error': 'REQUIRED_API_KEYS_MISSING', 'missing': missing}, status_code=428)
                return RedirectResponse('/settings?required=1', status_code=303)
        return await call_next(request)

    instance = uuid4().hex
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_dir / 'running.json'
    identity = {'application': 'ai-investment-research-desktop', 'instance': instance}

    @app.get('/_launcher/health', include_in_schema=False)
    def launcher_health():
        return {**identity, 'ok': True, 'automatic_scans': False}

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    errors = []

    def open_when_ready():
        try:
            for _ in range(120):
                if server.should_exit:
                    return
                try:
                    with urllib.request.urlopen(url + '/_launcher/health', timeout=1) as response:
                        status = json.load(response)
                    if status.get('instance') == instance:
                        break
                except (OSError, ValueError):
                    time.sleep(.5)
            else:
                raise RuntimeError('The app did not become ready within 60 seconds.')
            state_file.write_text(json.dumps({**identity, 'url': url}), encoding='utf-8')
            print(f'App ready: {url}', flush=True)
            print('Alpaca, Gemini and OpenAI API keys are required. Enter them in Settings.', flush=True)
            print('Keep this window open. Press Ctrl+C to stop the app.', flush=True)
            if args.check:
                for route in ('/', '/settings'):
                    with urllib.request.urlopen(url + route, timeout=30) as response:
                        page = response.read().decode('utf-8')
                    if 'AI Investment Research' not in page and '설정' not in page:
                        raise RuntimeError('The research UI did not render.')
                if missing_required():
                    try:
                        urllib.request.urlopen(url + '/api/v1/snapshot', timeout=5)
                    except urllib.error.HTTPError as exc:
                        if exc.code != 428:
                            raise
                    else:
                        raise RuntimeError('Analysis was not blocked when required keys were missing.')
                print('CHECK PASSED: real dashboard and settings responded; no scan was requested.', flush=True)
                server.should_exit = True
            elif not args.no_browser:
                webbrowser.open(url + '/settings' if missing_required() else url)
        except Exception as exc:
            errors.append(str(exc))
            server.should_exit = True

    ready = threading.Thread(target=open_when_ready, daemon=True)
    ready.start()
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()
        ready.join(timeout=2)
        if state_file.exists():
            try:
                if json.loads(state_file.read_text(encoding='utf-8')).get('instance') == instance:
                    state_file.unlink()
            except (OSError, ValueError):
                pass
    if errors:
        raise RuntimeError(errors[0])


if __name__ == '__main__':
    main()
