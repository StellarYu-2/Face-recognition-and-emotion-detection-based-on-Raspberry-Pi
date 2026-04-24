param(
    [string]$PiHost = "asdun",
    [string]$PiUser = "pi",
    [string]$CloudUrl = "http://asdun-cloud:8000",
    [int]$TimeoutSeconds = 3,
    [switch]$SkipWindowsLocal
)

$ErrorActionPreference = "Stop"

function Join-HttpPath {
    param(
        [string]$BaseUrl,
        [string]$Path
    )

    $base = $BaseUrl.TrimEnd([char[]]"/")
    if (-not $Path.StartsWith("/")) {
        $Path = "/" + $Path
    }
    return "$base$Path"
}

$healthUrl = Join-HttpPath -BaseUrl $CloudUrl -Path "/health"
$target = $PiHost
if (-not [string]::IsNullOrWhiteSpace($PiUser)) {
    $target = "$PiUser@$PiHost"
}

if (-not $SkipWindowsLocal) {
    $localHealthUrl = Join-HttpPath -BaseUrl "http://127.0.0.1:8000" -Path "/health"
    Write-Host "[network] checking Windows local inference server: $localHealthUrl"
    try {
        Invoke-RestMethod -Uri $localHealthUrl -TimeoutSec $TimeoutSeconds | ConvertTo-Json -Depth 6
    } catch {
        Write-Warning "[network] Windows local health check failed. Start it with: .\scripts\run_cloud_server.ps1 -SkipInstall"
        Write-Warning $_.Exception.Message
    }
}

$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $ssh) {
    throw "OpenSSH client was not found. Install or enable the Windows OpenSSH Client first."
}

Write-Host "[network] checking from Raspberry Pi ($target) to inference server: $healthUrl"
$remoteCommand = "curl -fsS --max-time $TimeoutSeconds '$healthUrl'"
& $ssh.Source $target $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "Pi-to-cloud health check failed. Confirm Tailscale is online on both devices and Windows is named asdun-cloud."
}

Write-Host "[network] Pi-to-cloud health check passed."
