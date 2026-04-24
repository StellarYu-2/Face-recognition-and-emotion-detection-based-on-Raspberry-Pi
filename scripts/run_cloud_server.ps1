param(
    [string]$EnvName = "asdun-cloud",
    [switch]$UseVenv,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ServerDir = Join-Path $RepoRoot "cloud_server"
$EnvFile = Join-Path $ServerDir "environment.yml"
$VenvDir = Join-Path $ServerDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$BaseRequirements = Join-Path $ServerDir "requirements.txt"
$GpuRequirements = Join-Path $ServerDir "requirements-gpu.txt"
$CpuRequirements = Join-Path $ServerDir "requirements-cpu.txt"

function Find-EnvTool {
    foreach ($name in @("mamba", "conda")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    return $null
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$CommandArgs,
        [string]$ErrorMessage
    )

    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Test-CondaEnvExists {
    param(
        [string]$EnvTool,
        [string]$Name
    )

    $envList = & $EnvTool env list
    return [bool]($envList | Select-String -Pattern ("(^|\s)" + [regex]::Escape($Name) + "(\s|$)"))
}

function Install-VenvRequirementsRequired {
    param([string]$RequirementsPath)

    Invoke-Checked -Exe $VenvPython -CommandArgs @("-m", "pip", "install", "-r", $RequirementsPath) -ErrorMessage "pip install failed: $RequirementsPath"
}

function Install-VenvRequirementsOptional {
    param([string]$RequirementsPath)

    & $VenvPython -m pip install -r $RequirementsPath
    return ($LASTEXITCODE -eq 0)
}

function Start-WithConda {
    param([string]$EnvTool)

    if (-not $SkipInstall) {
        if (Test-CondaEnvExists $EnvTool $EnvName) {
            Write-Host "[cloud] updating conda environment: $EnvName"
            Invoke-Checked -Exe $EnvTool -CommandArgs @("env", "update", "-n", $EnvName, "-f", $EnvFile, "--prune") -ErrorMessage "conda env update failed"
        } else {
            Write-Host "[cloud] creating conda environment: $EnvName"
            if ($EnvName -eq "asdun-cloud") {
                Invoke-Checked -Exe $EnvTool -CommandArgs @("env", "create", "-f", $EnvFile) -ErrorMessage "conda env create failed"
            } else {
                Invoke-Checked -Exe $EnvTool -CommandArgs @("env", "create", "-n", $EnvName, "-f", $EnvFile) -ErrorMessage "conda env create failed"
            }
        }
    } else {
        Write-Host "[cloud] skipping dependency install/update"
    }

    Write-Host "[cloud] starting FastAPI server on 0.0.0.0:8000 with conda env: $EnvName"
    Push-Location $ServerDir
    try {
        & $EnvTool run -n $EnvName python -m uvicorn app:app --host 0.0.0.0 --port 8000
    } finally {
        Pop-Location
    }
}

function Start-WithVenv {
    if (-not (Test-Path $VenvPython)) {
        Write-Host "[cloud] creating virtual environment fallback..."
        Invoke-Checked -Exe "python" -CommandArgs @("-m", "venv", $VenvDir) -ErrorMessage "python venv creation failed"
    }

    if (-not $SkipInstall) {
        Write-Host "[cloud] installing/updating venv dependencies..."
        Invoke-Checked -Exe $VenvPython -CommandArgs @("-m", "pip", "install", "-U", "pip") -ErrorMessage "pip upgrade failed"
        Install-VenvRequirementsRequired $BaseRequirements

        Write-Host "[cloud] trying ONNX Runtime GPU package..."
        if (-not (Install-VenvRequirementsOptional $GpuRequirements)) {
            Write-Warning "[cloud] onnxruntime-gpu install failed; falling back to CPU runtime for Stage A."
            Install-VenvRequirementsRequired $CpuRequirements
        }
    } else {
        Write-Host "[cloud] skipping dependency install/update"
    }

    Write-Host "[cloud] starting FastAPI server on 0.0.0.0:8000 with venv fallback"
    Push-Location $ServerDir
    try {
        & $VenvPython -m uvicorn app:app --host 0.0.0.0 --port 8000
    } finally {
        Pop-Location
    }
}

if (-not $UseVenv) {
    $EnvTool = Find-EnvTool
    if ($EnvTool) {
        Start-WithConda $EnvTool
        exit $LASTEXITCODE
    }

    Write-Warning "[cloud] conda/mamba was not found; falling back to .venv."
}

Start-WithVenv
