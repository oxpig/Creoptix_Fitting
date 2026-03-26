@echo off
REM ================================
REM Activate venv and run Creoptix Plotting script
REM ================================

REM Create/write log to file with timestamp
echo ===== Run started %date% %time% ===== >> run_creoptix_plotting.log

REM Activate the venv
call C:\Users\_USERNAME_\Documents\Git\Creoptix_Fitting\.venv\Scripts\activate.bat >> run_creoptix_plotting.log 2>&1

REM Run Python script (log output)
python run_creoptix_plotting.py >> run_creoptix_plotting.log 2>&1

REM Deactivate venv
deactivate >> run_creoptix_plotting.log 2>&1

REM Pause so you can see any output/errors before the window closes
pause