@echo off
setlocal
set WORKDIR=C:\Users\thoma\.lmstudio\apps\bionic\projects\d49037d8-47f4-5808-9028-c707de117f8f\workspace
cd /d "%WORKDIR%"
python TRANSLATION_OS_V1\scripts\learn_wikidata_loop_byteoffset.py > TRANSLATION_OS_V1\runtime\logs\wikidata_learn_service.log 2>&1
if errorlevel 1 (
    echo Restarting in 10 seconds...
    timeout /t 10
    start "" "%~f0"
)
