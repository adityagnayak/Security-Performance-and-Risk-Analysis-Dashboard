@echo off
echo ================================================
echo Security Performance ^& Risk Analysis Dashboard
echo Setup Script for Windows
echo ================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

python --version
echo [OK] Python is installed
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo [OK] Pip upgraded
echo.

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed successfully
echo.

echo ================================================
echo Setup completed successfully!
echo ================================================
echo.
echo To run the dashboard:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Run the dashboard:
echo    streamlit run security_analysis_dashboard.py
echo.
echo 3. Open your browser and navigate to:
echo    http://localhost:8501
echo.
echo ================================================
pause
