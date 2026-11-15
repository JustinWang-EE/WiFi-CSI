@echo off
setlocal
set "PYTHONPATH=%~dp0..\src"
if not defined INSENSE_PYTHON set "INSENSE_PYTHON=python"
"%INSENSE_PYTHON%" -B -m wifi_densepose.confidence_calibration %*
exit /b %errorlevel%
