@echo off
echo Instalando dependencias del proyecto con Python 3.14...
py -3.14 -m pip install --upgrade pip
py -3.14 -m pip install -r apps/api/requirements.txt
echo.
echo Instalacion completada. Puedes iniciar la API con:
echo py -3.14 -m uvicorn apps.api.app.main:app --reload
