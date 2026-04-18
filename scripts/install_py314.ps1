$ErrorActionPreference = "Stop"

Write-Host "Instalando dependencias del proyecto con Python 3.14..." -ForegroundColor Cyan
py -3.14 -m pip install --upgrade pip
py -3.14 -m pip install -r apps/api/requirements.txt

Write-Host "Instalacion completada. Puedes iniciar la API con:" -ForegroundColor Green
Write-Host "py -3.14 -m uvicorn apps.api.app.main:app --reload"
