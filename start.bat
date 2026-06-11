@echo off
REM Double-click to start Paper Vault Web UI (Windows)
cd /d "%~dp0"

REM Activate conda environment
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: conda not found on PATH.
  pause
  exit /b 1
)

call conda activate papervault 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo ERROR: conda env 'papervault' not found. Create it first:
  echo   conda create -n papervault python=3.11 -c conda-forge -y
  echo   conda activate papervault
  echo   pip install -e .
  pause
  exit /b 1
)

echo Starting Paper Vault...
python pv.py serve
pause
