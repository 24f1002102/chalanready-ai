@echo off
title ChalanReady AI - Flipkart Gridlock Hackathon 2.0
color 0B

echo.
echo  ====================================================
echo   ChalanReady AI  -  Officer Command Center
echo   Flipkart Gridlock Hackathon 2.0  -  Problem Statement 3
echo   Bengaluru Traffic Police  x  ASTraM Unit
echo  ====================================================
echo.

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found. Run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Demo processing is explicit from the dashboard.
:: Use "Run Demo Pipeline" in the UI, or run the synthetic generator manually:
::   python sample_data\create_synthetic_video.py
echo [*] Startup demo processing is disabled. Use the dashboard demo button when needed.

echo.
echo [*] Starting ChalanReady AI backend server...
echo [*] Dashboard will be available at: http://127.0.0.1:8000
echo [*] API docs at: http://127.0.0.1:8000/docs
echo.

:: Open browser after short delay
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

:: Start the server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

pause
