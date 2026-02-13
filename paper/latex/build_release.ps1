$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Build from main.tex, then publish one canonical PDF artifact.
latexmk -pdf -interaction=nonstopmode main.tex | Out-Host

$releaseName = "DiscoverThenDistill_ComputeMatched_v2_2026-02-13.pdf"
$releaseLocal = Join-Path $PSScriptRoot $releaseName
$releaseRoot = Join-Path (Split-Path $PSScriptRoot -Parent) $releaseName

Copy-Item (Join-Path $PSScriptRoot "main.pdf") $releaseLocal -Force
Copy-Item (Join-Path $PSScriptRoot "main.pdf") $releaseRoot -Force
# Keep the source directory clean.
if (Test-Path (Join-Path $PSScriptRoot "main.pdf")) {
    Remove-Item (Join-Path $PSScriptRoot "main.pdf") -Force
}

Write-Host "Release PDF ready:"
Write-Host " - $releaseLocal"
Write-Host " - $releaseRoot"
