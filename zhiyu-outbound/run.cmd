@echo off
rem ---------------------------------------------------------------
rem  One-click entry that does NOT depend on PowerShell execution policy.
rem  Usage:
rem      run.cmd           test + eval + serve
rem      run.cmd test      unit tests + websocket e2e
rem      run.cmd eval      eval suite (real-ish latency, ~90s)
rem      run.cmd eval fast eval suite, instant
rem      run.cmd serve     start web call console (Ctrl+C to stop)
rem ---------------------------------------------------------------
setlocal enabledelayedexpansion
cd /d "%~dp0"

set MODE=%~1
if "%MODE%"=="" set MODE=all
set OPT=%~2

if /i "%MODE%"=="test"  goto do_test
if /i "%MODE%"=="eval"  goto do_eval
if /i "%MODE%"=="serve" goto do_serve
if /i "%MODE%"=="all"   goto do_all
echo [error] unknown mode "%MODE%"
echo usage: run.cmd [test^|eval^|serve^|all] [fast]
exit /b 2

:do_test
echo.
echo === unit tests + websocket e2e ===
python -m unittest discover -s tests
exit /b %ERRORLEVEL%

:do_eval
echo.
echo === eval suite ===
if /i "%OPT%"=="fast" (
    python eval/run.py --fast
) else (
    python eval/run.py
)
exit /b %ERRORLEVEL%

:do_serve
echo.
echo === starting call console ===
echo open   http://127.0.0.1:8000/?ws=8765   in your browser
echo press  Ctrl+C to stop
python server/ws_server.py
exit /b %ERRORLEVEL%

:do_all
echo.
echo === 1/3 unit tests ===
python -m unittest discover -s tests
if errorlevel 1 (
    echo [abort] tests failed
    exit /b 1
)

echo.
echo === 2/3 eval suite (fast) ===
python eval/run.py --fast
if errorlevel 1 (
    echo [warn] some eval cases failed, see table above
)

echo.
echo === 3/3 starting call console ===
echo open   http://127.0.0.1:8000/?ws=8765   in your browser
echo press  Ctrl+C to stop
python server/ws_server.py
exit /b %ERRORLEVEL%
