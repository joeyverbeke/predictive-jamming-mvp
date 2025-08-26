@echo off
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated!
echo.
echo To run the system:
echo   python src/main_windows.py
echo.
echo To test the system:
echo   python test_audio_windows.py
echo.
cmd /k
