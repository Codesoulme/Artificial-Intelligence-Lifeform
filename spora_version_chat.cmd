@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe. See docs\version-chat-local-use.md.
  if "%~1"=="" pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m scripts.spora_version_chat %*
if errorlevel 1 (
  if "%~1"=="" pause
  exit /b 1
)
