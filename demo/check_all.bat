@echo off
rem ============================================================
rem  One-shot health check for the FlyBrain Flappy demo.
rem
rem  Covers the things that actually broke before:
rem   1. js syntax        -- a stray */ in a doc comment silently killed the
rem                          whole module; the page then sat on its initial
rem                          text and looked like "the server is down".
rem                          WARNING: this does NOT catch calling a function
rem                          that no longer exists (that cost me hours once) --
rem                          only step 5 catches that.
rem   2. physics parity   -- the server's GameWorld must be tick-for-tick
rem                          identical to the offline benchmark's World,
rem                          otherwise "the score" means nothing.
rem   3. control loop     -- the server's WHOLE loop (brain + physics) must be
rem                          tick-for-tick identical to the offline benchmark
rem                          when both are given the same seeds. This is the
rem                          check that caught the global-RNG noise bug, where
rem                          "same seed" produced a membrane-potential gap of
rem                          0.674 and different flap counts.
rem   4. collision pixels -- the server's collision test must match the pipes
rem                          the browser DRAWS, pixel for pixel. The user
rem                          reported "it looked like I hit the pipe but it
rem                          did not count as a fail"; this is the guard.
rem   5. shape invariants -- pipe drawing shape, bird outer radius.
rem   6. real browser     -- boots the page headlessly, reports console errors,
rem                          render fps, and whether the SERVER sim rate is a
rem                          constant 50 steps/s (verify_page.js).
rem
rem  Requires the server to be running on port 8620 (run_demo.bat).
rem  Note: ASCII only on purpose -- cmd.exe garbles UTF-8 Chinese.
rem ============================================================
setlocal
cd /d "%~dp0.."
set PY=D:\Code\FlyBrain\env\python.exe
set FAIL=0

echo [1/7] server on port 8620 ?
netstat -ano | findstr /r /c:"TCP.*:8620 .*LISTENING" >nul 2>&1
if %errorlevel%==0 (echo       up) else (
  echo       NOT running - start it with demo\run_demo.bat first
  set FAIL=1
)

echo [2/7] js syntax
node demo\check_syntax.mjs demo\app.js demo\verify_pipe_visual.js demo\diag_page.js
if errorlevel 1 set FAIL=1

echo [3/7] server GameWorld vs benchmark World (tick-for-tick)
"%PY%" scripts\verify_server_physics.py --games 12 --ticks 1200
if errorlevel 1 set FAIL=1

echo [4/7] server control loop vs benchmark (tick-for-tick, INCLUDING the brain)
"%PY%" scripts\verify_server_vs_bench.py --games 2 --max-ticks 1000 --first-gap-extra 0
if errorlevel 1 set FAIL=1

echo [5/7] server collision test vs drawn pipes (pixel-exact)
node demo\verify_server_collision.js >nul 2>&1
if errorlevel 1 (echo       MISMATCH - run demo\verify_server_collision.js and read the table & set FAIL=1) else (echo       pixel-exact)

echo [6/7] shape invariants
node demo\verify_pipe_visual.js >nul 2>&1
if errorlevel 1 (echo       pipe shape invariant broken & set FAIL=1) else (echo       pipe ok)
node demo\verify_bird_volume.js >nul 2>&1
if errorlevel 1 (echo       bird volume mismatch & set FAIL=1) else (echo       bird ok)

echo [7/7] page boots in a real browser + server rate is constant
node demo\verify_page.js 24
if errorlevel 1 set FAIL=1

echo.
if "%FAIL%"=="0" (echo ALL CHECKS PASSED) else (echo SOME CHECKS FAILED - see above)
pause
