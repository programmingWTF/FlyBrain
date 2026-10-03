@echo off
rem ============================================================
rem  Stop the FlyBrain demo server (kills whatever holds port 8620).
rem
rem  Why this exists: closing the console window sometimes leaves the
rem  python child process alive, so the server keeps running invisibly.
rem  This script cleans that up.
rem
rem  Note: only plain ASCII on purpose -- cmd.exe garbles UTF-8 Chinese.
rem ============================================================
setlocal
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"TCP.*:8620 .*LISTENING"') do (
  echo Stopping process on port 8620: PID=%%p
  taskkill /F /PID %%p >nul 2>&1
  set FOUND=1
)
if "%FOUND%"=="0" (
  echo No server is running on port 8620.
) else (
  echo Stopped.
)
echo.
echo Remaining demo server processes (should be empty):
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*demo*server.py*' } | Select-Object ProcessId,CommandLine | Format-List"
pause
