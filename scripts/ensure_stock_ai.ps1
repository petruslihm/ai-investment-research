# Start the local UI if it is not already listening. Does not open a browser.
# Task Scheduler / logon startup call this so scan and train can run unattended.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$logDir = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "ensure_stock_ai.log"

function Write-Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line -Encoding UTF8
}

function Test-UiHealth {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8743/api/health" -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (Test-UiHealth) {
    Write-Log "already running"
    exit 0
}

Write-Log "starting UI"
$pythonw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (Test-Path $pythonw) {
    Start-Process -FilePath $pythonw -ArgumentList "-m","trading_system.ui.app" -WorkingDirectory $root -WindowStyle Hidden
} else {
    Start-Process -FilePath "py" -ArgumentList "-3.12","-m","uv","run","investassist" -WorkingDirectory $root -WindowStyle Hidden
}

for ($i = 0; $i -lt 40; $i++) {
    if (Test-UiHealth) {
        Write-Log "ready"
        exit 0
    }
    Start-Sleep -Seconds 1
}
Write-Log "start failed"
exit 1
