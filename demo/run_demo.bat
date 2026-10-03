@echo off
rem ============================================================
rem  FlyBrain demo launcher  (fruit-fly escape reflex, whole brain)
rem  Double-click this file.  Close this window (or Ctrl+C) to stop.
rem  Then open  http://127.0.0.1:8620/
rem
rem  Note: only plain ASCII on purpose -- cmd.exe garbles UTF-8 Chinese.
rem ============================================================
setlocal
cd /d "%~dp0.."
set PY=D:\Code\FlyBrain\env\python.exe

rem Refuse to start a second copy: two servers would fight over the same brain.
netstat -ano | findstr /r /c:"TCP.*:8620 .*LISTENING" >nul 2>&1
if %errorlevel%==0 (
  echo [!] Port 8620 already has a server running.
  echo     Just open http://127.0.0.1:8620/ in your browser.
  echo     To restart, run demo\stop_demo.bat first.
  echo.
  pause
  exit /b 1
)

echo Loading frozen spiking brain (whole-brain asset, 144,837 neurons)...
echo Takes about 3-10 seconds.  The browser opens automatically when ready.
echo Close this window to stop the server.
echo.
"%PY%" demo\server.py --asset spiking_full --port 8620
echo.
echo Server stopped.
pause
