@echo off
REM FinSight Virtual Environment Setup Script for Windows
REM This script creates and sets up a virtual environment for FinSight

echo Setting up FinSight virtual environment...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

REM Run the Python setup script
python setup_venv.py
if %errorlevel% neq 0 (
    echo Error: Setup failed
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate
echo.
echo To run FinSight:
echo   python finsight_system.py
echo.
pause 