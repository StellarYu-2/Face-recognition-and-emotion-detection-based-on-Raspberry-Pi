param(
  [string]$PlatformUrl = "http://127.0.0.1:9000",
  [string]$DeviceId = "pi-01",
  [string]$Command = "ping",
  [string]$PayloadJson = "{}",
  [string]$AdminToken = "",
  [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

$baseUrl = $PlatformUrl.TrimEnd("/")
$headers = @{
  "Content-Type" = "application/json"
}
if ($AdminToken) {
  $headers["X-ASDUN-Admin-Token"] = $AdminToken
}

$payload = @{}
if ($PayloadJson.Trim()) {
  $payload = $PayloadJson | ConvertFrom-Json
}

$body = @{
  device_id = $DeviceId
  command = $Command
  payload = $payload
} | ConvertTo-Json -Depth 8

$created = Invoke-RestMethod `
  -Uri "$baseUrl/api/commands" `
  -Method Post `
  -Headers $headers `
  -Body $body

$commandId = $created.command.command_id
Write-Host "[command] created $commandId for $DeviceId command=$Command"

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
  Start-Sleep -Seconds 2
  $commands = Invoke-RestMethod `
    -Uri "$baseUrl/api/commands?device_id=$([uri]::EscapeDataString($DeviceId))&limit=20" `
    -Method Get

  $matched = @($commands.commands | Where-Object { $_.command_id -eq $commandId } | Select-Object -First 1)
  if ($matched.Count -gt 0 -and $matched[0].status -ne "pending") {
    $matched[0]
    exit 0
  }
  Write-Host "[command] waiting for result..."
} while ((Get-Date) -lt $deadline)

throw "Timed out waiting for command result: $commandId"
