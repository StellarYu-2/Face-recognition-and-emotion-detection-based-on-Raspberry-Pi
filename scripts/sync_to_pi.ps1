param(
    [string]$PiHost = "asdun",
    [string]$PiUser = "pi",
    [string]$RemoteDir = "~/asdun_pi",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$target = $PiHost
if (-not [string]::IsNullOrWhiteSpace($PiUser)) {
    $target = "$PiUser@$PiHost"
}

$items = @(
    "CMakeLists.txt",
    "config",
    "include",
    "src",
    "models/README.md",
    "models/onnx_to_ncnn",
    "scripts/install_ncnn_rpi.sh",
    "scripts/rpi_performance_mode.sh",
    "scripts/run_rpi_turbo.sh"
)

$existingItems = @()
foreach ($item in $items) {
    if (Test-Path -LiteralPath $item) {
        $existingItems += $item
    } else {
        Write-Warning "[sync] missing local path, skipped: $item"
    }
}

if ($existingItems.Count -eq 0) {
    throw "No sync items found."
}

Write-Host "[sync] target: ${target}:${RemoteDir}"
Write-Host "[sync] items:"
foreach ($item in $existingItems) {
    Write-Host "  - $item"
}

if ($DryRun) {
    Write-Host "[sync] dry run only; no files copied."
    exit 0
}

$ssh = Get-Command ssh -ErrorAction SilentlyContinue
$tar = Get-Command tar -ErrorAction SilentlyContinue
if (-not $ssh) {
    throw "OpenSSH client was not found. Install or enable the Windows OpenSSH Client first."
}
if (-not $tar) {
    throw "tar was not found. On recent Windows versions bsdtar is available as tar.exe."
}

$remoteCommand = "mkdir -p $RemoteDir && tar -xzf - -C $RemoteDir"
Write-Host "[sync] copying files without deleting remote-only data..."
& $tar.Source -czf - @existingItems | & $ssh.Source $target $remoteCommand
if (-not $?) {
    throw "Sync failed."
}

Write-Host "[sync] done. On the Pi, build/run from: $RemoteDir"
