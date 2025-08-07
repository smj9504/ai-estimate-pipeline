@echo off
REM Gemini 테스트를 conda 환경에서 실행

echo ============================================================
echo   Phase 0 Test with Gemini Model (Conda Environment)
echo ============================================================
echo.

REM Anaconda 경로 설정
set ANACONDA_PATH=C:\Users\user\anaconda3
set CONDA_EXE=%ANACONDA_PATH%\Scripts\conda.exe

REM conda 환경 활성화 및 실행
if exist "%CONDA_EXE%" (
    echo Activating conda environment: ai-estimate
    echo.
    
    REM conda 환경에서 Python 스크립트 실행
    call %ANACONDA_PATH%\Scripts\activate.bat ai-estimate
    
    echo Running test_gemini.py in conda environment...
    python test_gemini.py
    
    echo.
    echo Test completed.
) else (
    echo [ERROR] Conda not found at %CONDA_EXE%
    echo Please check Anaconda installation path
)

pause