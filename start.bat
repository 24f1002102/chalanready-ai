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

:: Regenerate the synthetic demo video so it always matches sample_data\evaluate.py.
echo [*] Generating synthetic demo video...
python sample_data\create_synthetic_video.py
echo [OK] Demo video ready.

:: Run violation pipeline on demo video
echo [*] Running AI violation detection pipeline on demo video...
python -m backend.pipeline --input sample_data/videos/synthetic_stage1.mp4 --output sample_data/outputs/stage1_annotated.mp4 --detector color --zone "Zone-A / MG Road"
echo [OK] Pipeline complete. Check sample_data\outputs\ for annotated video and snapshots.

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
