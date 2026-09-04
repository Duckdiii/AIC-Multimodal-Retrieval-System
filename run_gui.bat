@echo off
chcp 65001 > nul
echo ======================================================
echo    AIC 2026 - Enhanced Video Retrieval Desktop GUI
echo ======================================================
echo Dang khoi dong giao dien...

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui.py
) else (
    python gui.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Co loi xay ra khi chay GUI. Vui long kiem tra lai moi truong Python.
    pause
)
