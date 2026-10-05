@echo off
rem speciesClassifiers app launcher (Windows): double-click to start.
rem The first start downloads Python and the app's packages into this folder (a few minutes);
rem nothing is installed system-wide apart from the small "uv" tool, if you don't have it yet.
setlocal
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  ) else (
    echo Installing uv, a small tool that sets up Python just for this app...
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
  )
)
echo Starting speciesClassifiers. The first start takes a few minutes; the app opens in your browser.
echo Keep this window open while you use the app.
uv run --project python --extra app --python 3.12 speciesclassifiers app %*
if errorlevel 1 pause
