$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"

uv run mlflow ui `
    --backend-store-uri sqlite:///mlflow.db `
    --default-artifact-root ./mlruns `
    --host 127.0.0.1 `
    --port 5000
