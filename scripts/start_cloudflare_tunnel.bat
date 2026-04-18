@echo off
setlocal

powershell -ExecutionPolicy Bypass -File "%~dp0start_cloudflare_tunnel.ps1"
