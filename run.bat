@echo off
chcp 65001 >nul
title WhisperBlocks - Transcribing

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto :no_venv

call ".venv\Scripts\activate.bat"
python src\transcript.py

echo.
pause
exit /b 0

:no_venv
echo.
echo  [ERROR] Python virtual environment not found in .venv\
echo.
echo  Run install.bat once to create it, then try again.
echo.
pause
exit /b 1
