REM Set high performance power plan
echo Setting High Performance power plan...
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

REM Close common applications
echo Closing background applications...
taskkill /F /IM chrome.exe 2>nul
taskkill /F /IM firefox.exe 2>nul
taskkill /F /IM msedge.exe 2>nul
taskkill /F /IM OneDrive.exe 2>nul
taskkill /F /IM Teams.exe 2>nul
taskkill /F /IM Discord.exe 2>nul

REM Disable Windows services temporarily
echo Stopping Windows services...
net stop wuauserv >nul 2>&1
net stop WSearch >nul 2>&1

echo.
echo System prepared!
echo.
pause