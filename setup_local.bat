@echo off
REM VisionX — Local Development Setup (Windows)
REM Run this once to install dependencies and train models

echo.
echo ===================================================
echo  VisionX — Local Dev Setup
echo ===================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Install backend dependencies
echo [1/4] Installing Python dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)
cd ..

echo [2/4] Creating directories...
mkdir backend\trained_models 2>nul
mkdir backend\logs 2>nul
mkdir backend\data\raw 2>nul
mkdir backend\data\processed 2>nul

echo [3/4] Generating dataset and training models...
cd backend
python training\generate_dataset.py
python training\train_models.py
cd ..

echo [4/4] Done!
echo.
echo ===================================================
echo  Setup complete. Start the backend with:
echo.
echo    cd backend
echo    uvicorn app.main:app --reload --port 8000
echo.
echo  Then open: login.html in VS Code Live Server
echo  (right-click login.html -> Open with Live Server)
echo ===================================================
echo.
pause
