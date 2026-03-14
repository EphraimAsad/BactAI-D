@echo off
title BactAID - AI Bacterial Identification System
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║   ██████╗  █████╗  ██████╗████████╗ █████╗ ██╗██████╗        ║
echo  ║   ██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║██╔══██╗       ║
echo  ║   ██████╔╝███████║██║        ██║   ███████║██║██║  ██║       ║
echo  ║   ██╔══██╗██╔══██║██║        ██║   ██╔══██║██║██║  ██║       ║
echo  ║   ██████╔╝██║  ██║╚██████╗   ██║   ██║  ██║██║██████╔╝       ║
echo  ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═════╝        ║
echo  ║                                                              ║
echo  ║        AI-Powered Bacterial Identification System            ║
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH
    echo  Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo  [OK] Python found

:: Check for Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo  [WARN] Ollama not found - RAG will use template responses
    echo  Install Ollama from https://ollama.com for AI-powered explanations
) else (
    echo  [OK] Ollama found
)

:: Change to script directory
cd /d "%~dp0"

:: Check if requirements are installed (quick check for flask)
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Installing dependencies... (this may take a few minutes on first run)
    pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo  [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo  Starting BactAID server...
echo  ═══════════════════════════════════════════════════════════════
echo.
echo  Web interface will open at: http://localhost:8000
echo  Press Ctrl+C to stop the server
echo.

:: Start the server and open browser
start "" http://localhost:8000
python backend\app.py

pause
