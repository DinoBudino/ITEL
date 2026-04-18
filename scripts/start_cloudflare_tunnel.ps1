$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot

Write-Host 'Starting production containers...' -ForegroundColor Cyan
docker compose -f "$repoRoot/docker-compose.prod.yml" up -d --build

Write-Host 'Waiting for local proxy on http://127.0.0.1:80 ...' -ForegroundColor Cyan
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1' -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            break
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
}

Write-Host 'Launching Cloudflare Tunnel...' -ForegroundColor Green
cloudflared tunnel --url http://127.0.0.1:80
