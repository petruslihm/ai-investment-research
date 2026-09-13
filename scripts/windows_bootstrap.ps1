# Embedded in the release's single downloadable .cmd file by build_windows_launcher.py.
param(
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'AIR'),
    [string]$SourceCommit = '@SOURCE_COMMIT@',
    [switch]$Check,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$installLock = $null

function Download-File([string]$Url, [string]$Destination) {
    $part = $Destination + '.' + [Guid]::NewGuid().ToString('N') + '.part'
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $part -TimeoutSec 300
            Move-Item -LiteralPath $part -Destination $Destination -Force
            return
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds 2
        }
    }
}

function Assert-InInstallRoot([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    $prefix = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\') + '\'
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Installation path escaped the application folder.'
    }
}

function Expand-CheckedZip([string]$Archive, [string]$Destination) {
    Assert-InInstallRoot $Destination
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($Archive)
    try {
        $prefix = [IO.Path]::GetFullPath($Destination).TrimEnd('\') + '\'
        foreach ($entry in $zip.Entries) {
            $target = [IO.Path]::GetFullPath((Join-Path $Destination $entry.FullName))
            if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw 'Unsafe archive path.'
            }
        }
    } finally { $zip.Dispose() }
    [IO.Compression.ZipFile]::ExtractToDirectory($Archive, $Destination)
}

try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows 64-bit is required.' }
    if ($SourceCommit -notmatch '^[0-9a-f]{40}$') { throw 'Use the built release launcher, not this template.' }
    $InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    $statePath = Join-Path $InstallRoot 'running.json'
    if ((Test-Path -LiteralPath $statePath) -and -not $Check) {
        try {
            $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
            $address = [Uri]$state.url
            if ($address.Scheme -eq 'http' -and $address.Host -eq '127.0.0.1') {
                $health = Invoke-RestMethod -Uri ($state.url + '/_launcher/health') -TimeoutSec 2
                if ($health.application -eq 'ai-investment-research-desktop' -and $health.instance -eq $state.instance) {
                    if (-not $NoBrowser) { Start-Process $state.url }
                    return
                }
            }
        } catch { }
    }
    try {
        $installLock = [IO.File]::Open((Join-Path $InstallRoot 'install.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    } catch { throw 'Another launch is still preparing the app. Please wait for that window.' }

    Write-Host 'AI Investment Research - automatic setup'
    Write-Host 'The first run needs internet and may take several minutes.'
    Write-Host ('Application folder: ' + $InstallRoot)
    $downloads = Join-Path $InstallRoot 'downloads'
    $toolsDir = Join-Path $InstallRoot 'tools'
    New-Item -ItemType Directory -Force -Path $downloads,$toolsDir | Out-Null
    $uvDir = Join-Path $toolsDir 'uv-0.12.6'
    $uv = Join-Path $uvDir 'uv.exe'
    if (-not (Test-Path -LiteralPath $uv)) {
        Write-Host '[1/4] Downloading the setup tool...'
        $uvArchive = Join-Path $downloads 'uv-0.12.6.zip'
        Download-File 'https://github.com/astral-sh/uv/releases/download/0.12.6/uv-x86_64-pc-windows-msvc.zip' $uvArchive
        if ((Get-FileHash -LiteralPath $uvArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'df7cb9f243eae1621400d4fcf5b1b3d90f20e264ece91b64deb3b0078abca6ef') {
            throw 'The setup tool checksum does not match. Download was not executed.'
        }
        $uvStage = Join-Path $toolsDir ([Guid]::NewGuid().ToString('N'))
        Expand-CheckedZip $uvArchive $uvStage
        Assert-InInstallRoot $uvStage
        Assert-InInstallRoot $uvDir
        Move-Item -LiteralPath $uvStage -Destination $uvDir
    }
    if (-not (Test-Path -LiteralPath $uv)) { throw 'The setup tool was not extracted correctly.' }

    $env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallRoot 'python'
    $env:UV_CACHE_DIR = Join-Path $InstallRoot 'cache'
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $InstallRoot 'venv'
    $env:UV_NO_MODIFY_PATH = '1'
    $env:PYTHONUTF8 = '1'
    Write-Host '[2/4] Preparing a private Python 3.12 installation...'
    & $uv python install 3.12 --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed. Check the message above.' }

    $appsDir = Join-Path $InstallRoot 'apps'
    New-Item -ItemType Directory -Force -Path $appsDir | Out-Null
    $appDir = Join-Path $appsDir $SourceCommit.Substring(0,12)
    if (-not (Test-Path -LiteralPath (Join-Path $appDir 'pyproject.toml'))) {
        Write-Host '[3/4] Downloading the program...'
        $sourceArchive = Join-Path $downloads ($SourceCommit + '.zip')
        Download-File ('https://github.com/petruslihm/ai-investment-research/archive/' + $SourceCommit + '.zip') $sourceArchive
        $sourceStage = Join-Path $appsDir ([Guid]::NewGuid().ToString('N'))
        Expand-CheckedZip $sourceArchive $sourceStage
        $sourceFolder = Join-Path $sourceStage ('ai-investment-research-' + $SourceCommit)
        if (-not (Test-Path -LiteralPath (Join-Path $sourceFolder 'uv.lock'))) { throw 'Downloaded program is incomplete.' }
        Assert-InInstallRoot $sourceFolder
        Assert-InInstallRoot $appDir
        Move-Item -LiteralPath $sourceFolder -Destination $appDir
    }
    Write-Host '[4/4] Installing the required packages and opening the program...'
    & $uv sync --project $appDir --locked --no-dev --managed-python --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Check the message above.' }
    $python = Join-Path $env:UV_PROJECT_ENVIRONMENT 'Scripts\python.exe'
    $runArgs = @((Join-Path $appDir 'scripts\launch_desktop.py'), '--state-dir', $InstallRoot)
    if ($Check) { $runArgs += '--check' }
    if ($NoBrowser) { $runArgs += '--no-browser' }
    Set-Location -LiteralPath $appDir
    & $python @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'The program stopped with an error. Check the message above.' }
} catch {
    Write-Host ''
    Write-Host ('Could not start: ' + $_.Exception.Message) -ForegroundColor Red
    if (-not $Check) { Read-Host 'Press Enter to close this window' | Out-Null }
    exit 1
} finally {
    if ($installLock) { $installLock.Dispose() }
}
