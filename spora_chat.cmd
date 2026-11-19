@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
set "SPORA_PROFILE=%~dp0runs\spora-awake-local"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. See docs\awake-local-use.md.
  pause
  exit /b 1
)
for %%F in (self-name.pt recall-intent.pt subject-intent.pt) do (
  if not exist "%SPORA_PROFILE%\%%F" (
    echo Missing local artifact %%F. See docs\awake-local-use.md.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -X utf8 scripts\deepseek_continuous_memory_chat.py ^
  --memory-file "%SPORA_PROFILE%\memory.sqlite3" ^
  --awake-state "%SPORA_PROFILE%\awake.pt" --awake-topic-recall ^
  --awake-recall-intent-head "%SPORA_PROFILE%\recall-intent.pt" ^
  --awake-subject-intent-head "%SPORA_PROFILE%\subject-intent.pt" --awake-subject-hops 3 ^
  --self-name-state "%SPORA_PROFILE%\self-name.pt" --inherited-router --retrained-gate --earlier-readout %*
if errorlevel 1 (
  pause
  exit /b 1
)
