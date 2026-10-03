@echo off
rem ============================================================
rem  One-shot health check for the FlyBrain demo.
rem
rem  Covers the things that actually broke before:
rem   1. js syntax        -- a stray */ in a doc comment silently killed the
rem                          whole module; the page then sat on its initial
rem                          text and looked like "the server is down".
rem   2. pipe drawing     -- drawn pipe must match the collision test pixel
rem                          for pixel (verify_pipe_visual.js).
rem   3. real browser     -- boots the page headlessly, reports console errors
rem                          and final page state (diag_page.js).
rem   4. gameplay parity  -- page algorithm vs scripts/flappy_bench.py.
rem
rem  Requires the server to be running on port 8620 (run_demo.bat).
rem  Note: ASCII only on purpose -- cmd.exe garbles UTF-8 Chinese.
rem ============================================================
setlocal
cd /d "%~dp0.."
set PY=D:\Code\FlyBrain\env\python.exe
set FAIL=0

echo [1/5] server on port 8620 ?
netstat -ano | findstr /r /c:"TCP.*:8620 .*LISTENING" >nul 2>&1
if %errorlevel%==0 (echo       up) else (
  echo       NOT running - start it with demo\run_demo.bat first
  set FAIL=1
)

echo [2/5] js syntax
node demo\check_syntax.mjs demo\app.js demo\verify_pipe_visual.js demo\diag_page.js
if errorlevel 1 set FAIL=1

echo [3/5] pipe drawing vs collision test
node demo\verify_pipe_visual.js >nul 2>&1
if errorlevel 1 (echo       MISMATCH - see demo\verify_pipe_visual.js output & set FAIL=1) else (echo       pixel-exact)

echo [4/5] page boots in a real browser
node demo\diag_page.js
if errorlevel 1 set FAIL=1

echo [5/5] gameplay parity (page algorithm vs benchmark)
"%PY%" scripts\flappy_page_parity.py --games 20
if errorlevel 1 set FAIL=1

echo.
if "%FAIL%"=="0" (echo ALL CHECKS PASSED) else (echo SOME CHECKS FAILED - see above)
pause
