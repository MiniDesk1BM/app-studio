@echo off
rem Lance App Studio (127.0.0.1 uniquement) et ouvre le navigateur.
cd /d "%~dp0"
if "%STUDIO_PORT%"=="" set STUDIO_PORT=8080
start "" "http://127.0.0.1:%STUDIO_PORT%"
python server.py
