@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_cloudflare_demo.ps1" %*
exit /b %errorlevel%
