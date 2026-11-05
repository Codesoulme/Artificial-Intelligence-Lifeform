@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing in D:\SPORA\.venv
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m spora.chat %*
if errorlevel 1 (
  pause
  exit /b 1
)
