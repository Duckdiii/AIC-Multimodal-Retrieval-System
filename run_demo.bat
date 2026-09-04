@echo off
chcp 65001 > nul
echo ======================================================
echo    AIC 2026 - Online Pipeline Verification Demo
echo ======================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" demo_pipeline.py
) else (
    python demo_pipeline.py
)

pause
