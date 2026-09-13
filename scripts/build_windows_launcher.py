"""Build a self-contained CMD launcher pinned to an exact public source commit."""
import argparse
import re
from pathlib import Path

HEADER = '''@echo off
setlocal
title AI Investment Research
set "AIR_LAUNCHER_FILE=%~f0"
powershell.exe -NoLogo -NoProfile -Command "$s=[IO.File]::ReadAllText($env:AIR_LAUNCHER_FILE,[Text.Encoding]::UTF8);$m='#'+' POWERSHELL_PAYLOAD';$i=$s.IndexOf($m);if($i -lt 0){exit 1};& ([scriptblock]::Create($s.Substring($i+$m.Length)))"
exit /b %errorlevel%
# POWERSHELL_PAYLOAD
'''


def build(commit, output):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('A full source commit is required')
    script = Path(__file__).with_name('windows_bootstrap.ps1').read_text(encoding='utf-8')
    result = HEADER + script.replace('@SOURCE_COMMIT@', commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build(args.commit, args.output)
