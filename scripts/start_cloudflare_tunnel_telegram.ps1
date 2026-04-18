$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $repoRoot '.env.telegram.local'

function Read-LocalTelegramConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $config = @{}

    if (-not (Test-Path $Path)) {
        return $config
    }

    foreach ($line in Get-Content -Path $Path) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith('#')) {
            continue
        }

        $parts = $line -split '=', 2
        if ($parts.Count -ne 2) {
            continue
        }

        $config[$parts[0].Trim()] = $parts[1].Trim()
    }

    return $config
}

function Save-LocalTelegramConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$BotToken,
        [Parameter(Mandatory = $true)]
        [string]$ChatId
    )

    $content = @(
        "ITEL_TELEGRAM_BOT_TOKEN=$BotToken"
        "ITEL_TELEGRAM_CHAT_ID=$ChatId"
    )

    Set-Content -Path $Path -Value $content -Encoding ascii
}

function Get-ConfigValue {
    param(
        [string]$EnvironmentValue,
        [hashtable]$LocalConfig,
        [string]$Key
    )

    if (-not [string]::IsNullOrWhiteSpace($EnvironmentValue)) {
        return $EnvironmentValue
    }

    if ($LocalConfig.ContainsKey($Key) -and -not [string]::IsNullOrWhiteSpace($LocalConfig[$Key])) {
        return $LocalConfig[$Key]
    }

    return $null
}

function Get-CloudflaredExecutable {
    $command = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $knownPaths = @(
        'C:\Program Files (x86)\cloudflared\cloudflared.exe'
        'C:\Program Files\cloudflared\cloudflared.exe'
    )

    foreach ($path in $knownPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    throw 'cloudflared executable not found. Install cloudflared or add it to PATH.'
}

$localConfig = Read-LocalTelegramConfig -Path $configPath
$botToken = Get-ConfigValue -EnvironmentValue $env:ITEL_TELEGRAM_BOT_TOKEN -LocalConfig $localConfig -Key 'ITEL_TELEGRAM_BOT_TOKEN'
$chatId = Get-ConfigValue -EnvironmentValue $env:ITEL_TELEGRAM_CHAT_ID -LocalConfig $localConfig -Key 'ITEL_TELEGRAM_CHAT_ID'
$cloudflaredPath = Get-CloudflaredExecutable

if ([string]::IsNullOrWhiteSpace($botToken) -or [string]::IsNullOrWhiteSpace($chatId)) {
    Write-Host 'Telegram configuration not found. Please enter it once and it will be saved locally.' -ForegroundColor Yellow

    if ([string]::IsNullOrWhiteSpace($botToken)) {
        $botToken = Read-Host 'Enter ITEL_TELEGRAM_BOT_TOKEN'
    }

    if ([string]::IsNullOrWhiteSpace($chatId)) {
        $chatId = Read-Host 'Enter ITEL_TELEGRAM_CHAT_ID'
    }

    if ([string]::IsNullOrWhiteSpace($botToken)) {
        throw 'Telegram bot token cannot be empty.'
    }

    if ([string]::IsNullOrWhiteSpace($chatId)) {
        throw 'Telegram chat id cannot be empty.'
    }

    Save-LocalTelegramConfig -Path $configPath -BotToken $botToken -ChatId $chatId
    Write-Host "Telegram configuration saved to $configPath" -ForegroundColor Green
}

function Send-TelegramMessage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Token,
        [Parameter(Mandatory = $true)]
        [string]$ChatId,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $uri = "https://api.telegram.org/bot$Token/sendMessage"
    $body = @{
        chat_id = $ChatId
        text = $Message
    }

    Invoke-RestMethod -Uri $uri -Method Post -Body $body | Out-Null
}

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

Write-Host 'Launching Cloudflare Tunnel with Telegram notifications...' -ForegroundColor Green
$urlSent = $false

try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $cloudflaredPath tunnel --url http://127.0.0.1:80 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line

        if (-not $urlSent -and $line -match 'https://[a-z0-9.-]+\.trycloudflare\.com') {
            $publicUrl = $Matches[0]
            try {
                Send-TelegramMessage -Token $botToken -ChatId $chatId -Message "ITEL demo online: $publicUrl"
                Write-Host 'Telegram notification sent.' -ForegroundColor Green
                $urlSent = $true
            }
            catch {
                Write-Warning "Could not send Telegram notification: $($_.Exception.Message)"
                $urlSent = $true
            }
        }
    }
    $ErrorActionPreference = $previousErrorActionPreference
}
catch {
    $ErrorActionPreference = $previousErrorActionPreference
    throw
}
