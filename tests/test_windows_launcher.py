"""Check that the desktop entry point starts real UI pages without provider calls."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_all_three_providers_are_required():
    from trading_system.credentials import missing_provider_keys
    keys = {'ALPACA_API_KEY': 'test', 'ALPACA_SECRET_KEY': 'test',
            'OPENAI_API_KEY': 'test', 'GEMINI_API_KEY': 'test'}
    assert missing_provider_keys(keys) == []
    assert missing_provider_keys({**keys, 'GEMINI_API_KEY': ''}) == ['Gemini research']
    assert missing_provider_keys({**keys, 'OPENAI_API_KEY': '', 'ANTHROPIC_API_KEY': 'test'}) == ['OpenAI GPT']


def test_launcher_pins_source_and_preserves_cmd_line_endings(tmp_path):
    spec = importlib.util.spec_from_file_location('launcher_builder', ROOT / 'scripts/build_windows_launcher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / 'space and 한글' / 'AI-Investment-Research.cmd'
    commit = 'a' * 40
    module.build(commit, output)
    content = output.read_bytes()
    assert content.startswith(b'@echo off\r\n')
    assert b'@SOURCE_COMMIT@' not in content
    assert f"$SourceCommit = '{commit}'".encode() in content
    assert b'ExecutionPolicy' not in content
    with pytest.raises(ValueError):
        module.build('main', output)


def test_desktop_starts_real_pages_without_scans_or_provider_network(tmp_path):
    # Patch networking in the child process, allowing only the local UI server.
    code = '''
import runpy, socket, sys
original = socket.socket.connect
def local_only(sock, address):
    if address[0] not in ('127.0.0.1', '::1'):
        raise AssertionError('Desktop launch must not call external providers')
    return original(sock, address)
socket.socket.connect = local_only
sys.argv = [sys.argv[1], '--state-dir', sys.argv[2], '--check']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONUTF8='1',
               DUCKDB_PATH=str(tmp_path / 'empty.duckdb'),
               INVESTASSIST_DAILY_SCAN='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code,
                             str(ROOT / 'scripts/launch_desktop.py'), str(tmp_path)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CHECK PASSED: real dashboard and settings responded' in result.stdout
    assert not (tmp_path / 'running.json').exists()
