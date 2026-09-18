@echo off
chcp 65001 >nul
title WhisperBlocks — Transcribing

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe src\transcript.py
) else (
    python src\transcript.py
)

echo.
pause