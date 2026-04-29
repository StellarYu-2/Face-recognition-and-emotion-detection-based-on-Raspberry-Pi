param(
  [string]$EnvName = "asdun-cloud",
  [string]$HostName = "0.0.0.0",
  [int]$Port = 9000,
  [switch]$UseVenv,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Resolve-Path (Join-Path $scriptDir "..")
$serverDir = Join-Path $rootDir "platform_server"
$venvDir = Join-Path $serverDir ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$pyDepsDir = Join-Path $serverDir ".pydeps"

Set-Location $serverDir

function Find-EnvTool {
  foreach ($name in @("mamba", "conda")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
      return $cmd.Source
    }
  }
  return $null
}

function Test-CondaEnvExists {
  param(
    [string]$EnvTool,
    [string]$Name
  )

  $envList = & $EnvTool env list
  return [bool]($envList | Select-String -Pattern ("(^|\s)" + [regex]::Escape($Name) + "(\s|$)"))
}

if (-not $UseVenv) {
  $envTool = Find-EnvTool
  if ($envTool -and (Test-CondaEnvExists $envTool $EnvName)) {
    if (-not $SkipInstall) {
      & $envTool run -n $EnvName python -m pip install -r requirements.txt
    }
    Write-Host "[platform] starting http://$HostName`:$Port with conda env: $EnvName"
    & $envTool run -n $EnvName python -m uvicorn app:app --host $HostName --port $Port
    exit $LASTEXITCODE
  }
}

if (-not (Test-Path $pythonExe)) {
  python -m venv $venvDir 2>$null
}

$useVenv = $false
if (Test-Path $pythonExe) {
  & $pythonExe -m pip --version *> $null
  $useVenv = ($LASTEXITCODE -eq 0)
}

if (-not $SkipInstall) {
  if ($useVenv) {
    & $pythonExe -m pip install -U pip
    & $pythonExe -m pip install -r requirements.txt
  } else {
    New-Item -ItemType Directory -Force $pyDepsDir | Out-Null
    python -m pip install --target $pyDepsDir -r requirements.txt
  }
}

if (-not $useVenv) {
  if (Test-Path $pyDepsDir) {
    $env:PYTHONPATH = "$pyDepsDir;$env:PYTHONPATH"
  }
  $pythonExe = "python"
}

Write-Host "[platform] starting http://$HostName`:$Port"
& $pythonExe -m uvicorn app:app --host $HostName --port $Port
