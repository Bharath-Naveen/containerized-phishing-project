$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "archive\sample_data\challenge\*.txt"
$dst = Join-Path $root "data\raw"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path $src -Destination $dst -Force
Write-Host "Copied challenge samples to $dst"
