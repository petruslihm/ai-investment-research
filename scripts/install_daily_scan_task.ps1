# Register unattended start: Windows Startup folder + current-user scheduled task.
# The UI process then fires the daily pre-open scan and post-close model-only training.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script = Join-Path $root "scripts\ensure_stock_ai.ps1"
$taskName = "StockAI-AutoStart"

$startup = [Environment]::GetFolderPath("Startup")
$lnkPath = Join-Path $startup "StockAI.lnk"
$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut($lnkPath)
$lnk.TargetPath = "powershell.exe"
$lnk.Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$lnk.WorkingDirectory = $root
$lnk.WindowStyle = 7
$lnk.Description = "Start Stock AI in the background if it is not running"
$lnk.Save()
Write-Output "Startup shortcut: $lnkPath"

$xmlPath = Join-Path $env:TEMP "StockAI-AutoStart.xml"
$cmd = "powershell.exe"
$args = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Start Stock AI so the weekday pre-open scan and post-close model training can run without opening the browser.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
    <CalendarTrigger>
      <StartBoundary>2026-01-05T21:20:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek>
          <Monday /><Tuesday /><Wednesday /><Thursday /><Friday />
        </DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
    <CalendarTrigger>
      <StartBoundary>2026-01-03T05:30:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek>
          <Saturday />
        </DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <StartWhenAvailable>true</StartWhenAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$cmd</Command>
      <Arguments>$args</Arguments>
      <WorkingDirectory>$root</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
Set-Content -Path $xmlPath -Value $xml -Encoding Unicode
$taskOk = $false
try {
    schtasks.exe /Create /TN $taskName /XML $xmlPath /F | Out-Null
    if ($LASTEXITCODE -eq 0) { $taskOk = $true }
} catch {
    $taskOk = $false
}
if (-not $taskOk) {
    try {
        $action = New-ScheduledTaskAction -Execute $cmd -Argument $args -WorkingDirectory $root
        $weekday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 9:20PM
        $saturday = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 5:30AM
        $logon = New-ScheduledTaskTrigger -AtLogOn
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($weekday, $saturday, $logon) -Settings $settings -Force | Out-Null
        $taskOk = $true
    } catch {
        Write-Output "Scheduled task was not registered (permission). Startup shortcut still starts it at logon."
    }
}
if ($taskOk) {
    Write-Output "Scheduled task '$taskName': logon, weekdays 21:20, Saturday 05:30."
}
Write-Output "PC must be on (or awake). The browser does not need to be open."
