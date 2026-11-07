@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m spora.memory_chat %*
pause
