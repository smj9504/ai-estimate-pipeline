@echo off
REM conda 환경에서 Python 스크립트 실행하는 배치 파일
REM UTF-8 인코딩 설정 포함

REM UTF-8 코드 페이지 설정 (인코딩 문제 해결)
chcp 65001 > nul

REM Anaconda 경로 설정
set ANACONDA_PATH=C:\Users\user\anaconda3
set CONDA_ENV=ai-estimate

REM 첫 번째 인수가 없으면 도움말 표시
if "%1"=="" goto :help

REM conda 환경 활성화 및 Python 스크립트 실행
echo [INFO] Activating conda environment: %CONDA_ENV%
call "%ANACONDA_PATH%\Scripts\activate.bat" %CONDA_ENV%

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment
    exit /b 1
)

echo [INFO] Running: python %*
python %*

goto :end

:help
echo Usage: run_conda.bat [script.py] [arguments]
echo.
echo Examples:
echo   run_conda.bat test_gemini.py
echo   run_conda.bat run.py
echo   run_conda.bat -m src.main
echo.

:end