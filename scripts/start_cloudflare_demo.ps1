param(
    [string]$Pin,
    [switch]$PromptForPin
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$prodCompose = Join-Path $repoRoot 'docker-compose.prod.yml'
$demoCompose = Join-Path $repoRoot 'docker-compose.demo.yml'
$credentialPath = Join-Path $repoRoot '.demo.htpasswd'
$czechDataset = Join-Path $repoRoot 'research/LANNA_Czech_Data'
$englishDataset = Join-Path $repoRoot 'research/English_Data'
$cacheRoot = Join-Path $repoRoot 'artifacts/serving_cache'
$catalogRoot = Join-Path $repoRoot 'artifacts/demo_catalog'
$catalogPath = Join-Path $catalogRoot '.sample_audio_catalog.json'
$catalogBuilder = Join-Path $repoRoot 'research/scripts/build_demo_catalog.py'
$projectName = 'itel-demo'
$stackStarted = $false

function Get-CloudflaredExecutable {
    $command = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    foreach ($path in @(
        'C:\Program Files (x86)\cloudflared\cloudflared.exe',
        'C:\Program Files\cloudflared\cloudflared.exe'
    )) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            return $path
        }
    }

    throw 'cloudflared executable not found. Install cloudflared or add it to PATH.'
}

function New-DemoPin {
    $bytes = New-Object byte[] 4
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    }
    finally {
        $rng.Dispose()
    }
    return ([BitConverter]::ToUInt32($bytes, 0) % 100000000).ToString('D8')
}

function Write-Htpasswd {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Password
    )

    $sha1 = [Security.Cryptography.SHA1]::Create()
    try {
        $hash = $sha1.ComputeHash([Text.Encoding]::UTF8.GetBytes($Password))
    }
    finally {
        $sha1.Dispose()
    }
    $entry = 'demo:{SHA}' + [Convert]::ToBase64String($hash)
    Set-Content -LiteralPath $Path -Value $entry -Encoding Ascii
}

function Assert-DemoInputs {
    foreach ($dataset in @($czechDataset, $englishDataset)) {
        if (-not (Test-Path -LiteralPath $dataset -PathType Container)) {
            throw "Required dataset directory not found: $dataset"
        }
        if (-not (Get-ChildItem -LiteralPath $dataset -File -Recurse | Where-Object { $_.Extension -ieq '.wav' } | Select-Object -First 1)) {
            throw "No WAV files found in required dataset: $dataset"
        }
    }

    if (-not (Test-Path -LiteralPath $cacheRoot -PathType Container)) {
        throw "Serving cache directory not found: $cacheRoot"
    }
    $cacheManifests = @(Get-ChildItem -LiteralPath $cacheRoot -Filter 'cache-manifest.json' -File -Recurse)
    if ($cacheManifests.Count -eq 0) {
        throw "No cache-manifest.json found below: $cacheRoot"
    }
    $cacheSampleCounts = @{}
    foreach ($manifestPath in $cacheManifests) {
        try {
            $manifest = Get-Content -LiteralPath $manifestPath.FullName -Raw | ConvertFrom-Json
        }
        catch {
            throw "Serving cache manifest is not valid JSON: $($manifestPath.FullName)"
        }
        if ([string]::IsNullOrWhiteSpace([string]$manifest.artifact.file)) {
            throw "Serving cache manifest has no artifact file: $($manifestPath.FullName)"
        }
        $payloadPath = Join-Path $manifestPath.DirectoryName ([string]$manifest.artifact.file)
        if (-not (Test-Path -LiteralPath $payloadPath -PathType Leaf)) {
            throw "Serving cache payload not found: $payloadPath"
        }
        $modelId = [string]$manifest.model_id
        if (-not $cacheSampleCounts.ContainsKey($modelId)) {
            $cacheSampleCounts[$modelId] = 0
        }
        $cacheSampleCounts[$modelId] += [int]$manifest.source_sample_count
    }
    if ($cacheSampleCounts['multilingual-cz-en'] -ne 3023) {
        throw "Multilingual cache must cover 3023 source samples; found $($cacheSampleCounts['multilingual-cz-en'])."
    }
    if ($cacheSampleCounts['czech-layermix'] -ne 2975) {
        throw "Czech LayerMix cache must cover 2975 source samples; found $($cacheSampleCounts['czech-layermix'])."
    }

    if (-not (Test-Path -LiteralPath $catalogPath -PathType Leaf)) {
        throw "Demo catalog not found: $catalogPath"
    }

    try {
        $catalog = Get-Content -LiteralPath $catalogPath -Raw | ConvertFrom-Json
    }
    catch {
        throw "Demo catalog is not valid JSON: $catalogPath"
    }
    $subjects = @($catalog.subjects)
    if ($subjects.Count -ne 146 -or [int]$catalog.audio_count -ne 3023) {
        throw "Demo catalog must contain 146 subjects and 3023 audios: $catalogPath"
    }
    if ($subjects.Count -eq 0) {
        throw "Demo catalog is empty: $catalogPath"
    }

    $languages = @{}
    foreach ($subject in $subjects) {
        $languages[[string]$subject.language] = $true
        foreach ($audio in @($subject.audios)) {
            $relativePath = [string]$audio.relative_path
            if ($relativePath.Contains('\')) {
                throw "Catalog paths must use container-compatible forward slashes: $relativePath"
            }
            $parts = $relativePath.Replace('\', '/').Split('/', 2)
            if ($parts.Count -ne 2 -or $parts[0] -notin @('czech', 'english')) {
                throw "Catalog paths must begin with czech/ or english/: $relativePath"
            }
            if ([IO.Path]::IsPathRooted($relativePath) -or $relativePath.Split(@('/', '\')) -contains '..') {
                throw "Unsafe path in demo catalog: $relativePath"
            }

            $datasetRoot = if ($parts[0] -eq 'czech') { $czechDataset } else { $englishDataset }
            $audioPath = Join-Path $datasetRoot $parts[1]
            if (-not (Test-Path -LiteralPath $audioPath -PathType Leaf)) {
                throw "Catalog audio file not found: $relativePath"
            }
        }
    }
    if (-not $languages.cs -or -not $languages.en) {
        throw 'Demo catalog must include at least one Czech and one English sample.'
    }
}

try {
    foreach ($command in @('docker')) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "$command executable not found."
        }
    }
    $cloudflaredPath = Get-CloudflaredExecutable

    if (Test-Path -LiteralPath $catalogBuilder -PathType Leaf) {
        if (-not (Test-Path -LiteralPath $catalogRoot -PathType Container)) {
            New-Item -ItemType Directory -Path $catalogRoot | Out-Null
        }
        $venvPython = Join-Path $repoRoot '.venv-research/Scripts/python.exe'
        $python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { (Get-Command python).Source }
        Write-Host 'Building the demo catalog...' -ForegroundColor Cyan
        & $python $catalogBuilder --czech-root $czechDataset --english-root $englishDataset --output $catalogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Demo catalog builder exited with code $LASTEXITCODE."
        }
    }

    Assert-DemoInputs

    if ($PromptForPin) {
        $Pin = Read-Host 'Enter the temporary demo PIN'
    }
    if ([string]::IsNullOrWhiteSpace($Pin)) {
        $Pin = New-DemoPin
    }
    if ($Pin.Contains(':') -or $Pin -match '[^\x21-\x7e]') {
        throw 'The demo PIN must contain only printable ASCII characters other than colon.'
    }
    Write-Htpasswd -Path $credentialPath -Password $Pin

    Write-Host 'Starting authenticated demo containers...' -ForegroundColor Cyan
    $stackStarted = $true
    & docker compose --project-name $projectName -f $prodCompose -f $demoCompose up -d --build
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up exited with code $LASTEXITCODE."
    }
    $credential = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("demo:$Pin"))
    $headers = @{ Authorization = "Basic $credential" }
    $ready = $false
    Write-Host 'Waiting for the authenticated local proxy...' -ForegroundColor Cyan
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri 'http://127.0.0.1/api/ready' -Headers $headers -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    if (-not $ready) {
        throw 'Authenticated demo proxy did not become ready within 120 seconds.'
    }

    $authRequired = $false
    try {
        Invoke-WebRequest -Uri 'http://127.0.0.1/' -UseBasicParsing -TimeoutSec 3 | Out-Null
    }
    catch {
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 401) {
            $authRequired = $true
        }
    }
    if (-not $authRequired) {
        throw 'Demo proxy did not reject an unauthenticated request.'
    }

    Write-Host "Demo username: demo" -ForegroundColor Green
    Write-Host "Demo PIN:      $Pin" -ForegroundColor Green
    Write-Host 'Launching Cloudflare Tunnel. Press Ctrl+C to stop and clean up.' -ForegroundColor Green
    & $cloudflaredPath tunnel --url http://127.0.0.1:80
    if ($LASTEXITCODE -ne 0) {
        throw "cloudflared exited with code $LASTEXITCODE."
    }
}
finally {
    if ($stackStarted) {
        Write-Host 'Stopping demo containers...' -ForegroundColor Cyan
        & docker compose --project-name $projectName -f $prodCompose -f $demoCompose down
    }
    if (Test-Path -LiteralPath $credentialPath) {
        Remove-Item -LiteralPath $credentialPath -Force
    }
}
