@echo off
REM =====================================================================
REM  Photovoltaic module service-life calculation launcher (Windows).
REM  Just double-click this file.
REM
REM  On first run it creates a Python environment and installs all
REM  required libraries. Next runs start instantly.
REM =====================================================================

cd /d "%~dp0"

echo ======================================================================
echo   PV module service-life calculation
echo ======================================================================
echo.

REM Check Python is available
python --version >/dev/null 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install Python from https://www.python.org/downloads/
    echo During install, check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

REM Create the virtual environment on first run
if not exist ".venv" (
    echo First run: creating Python environment and installing libraries...
    echo (takes 1-2 minutes, internet required^)
    echo.
    python -m venv .venv
    .venv\Scripts\python -m pip install --quiet --upgrade pip
    .venv\Scripts\pip install --quiet -r requirements.txt
    echo Environment ready.
    echo.
)

REM Run the calculation
echo Running calculation...
echo.
.venv\Scripts\python main.py

REM Open the results folder
echo.
echo Opening the folder with charts and tables...
start "" output

echo.
echo ======================================================================
echo   Done. Charts and tables are in the output\ folder
echo ======================================================================
echo.
pause
