Write-Host "Starting app using local virtual environment (venv) to bypass Application Control policy issues..." -ForegroundColor Green
& ".\venv\Scripts\python.exe" app.py
Read-Host "Press Enter to exit..."
