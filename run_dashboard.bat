@echo off
title EV Battery Life & Health Prediction System
echo ========================================================
echo [INFO] Starting EV Battery Life Prediction Web Dashboard...
echo [INFO] Powered by NASA PCoE and CALCE Degredation Datasets
echo ========================================================
echo.

cd /d "%~dp0"

echo [INFO] Launching web server at http://127.0.0.1:5000 ...
start http://127.0.0.1:5000

python app.py
pause
