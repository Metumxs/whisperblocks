@echo off
setlocal
chcp 65001 >nul
title WhisperBlocks

cd /d "%~dp0"

REM Console fallback for WhisperBlocks.exe: same checks, then starts the
REM window without keeping this console open.

if not exist ".venv\Scripts\python.exe" goto :no_venv

".venv\Scripts\python.exe" -c "import PySide6, tomlkit" >nul 2>&1
if not errorlevel 1 goto :start

echo.
echo  [i] Installing the window components (PySide6, one time, about 250 MB)...
echo.
".venv\Scripts\python.exe" -m pip install -r requirements\requirements-gui.txt
if errorlevel 1 goto :gui_failed

:start
start "" ".venv\Scripts\pythonw.exe" src\gui_main.py %*
exit /b 0

:gui_failed
echo.
echo  [ERROR] Could not install the window components.
echo  Check the internet connection and run this file again.
echo.
pause
exit /b 1

:no_venv
echo.
echo  [ERROR] Python virtual environment not found in .venv\
echo.
echo  Run install.bat once to create it, then try again.
echo.
pause
exit /b 1
