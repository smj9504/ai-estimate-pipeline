@echo off
REM AI Estimate Pipeline - 통합 실행 배치 파일

REM 기본 설정
set DEFAULT_PYTHON=python
set ANACONDA_PATH=C:\Users\user\anaconda3
set CONDA_PYTHON=%ANACONDA_PATH%\python.exe

REM 명령줄 인수 확인
if "%1"=="" goto :run_default
if /i "%1"=="setup" goto :run_setup
if /i "%1"=="test" goto :run_test
if /i "%1"=="conda" goto :run_conda
if /i "%1"=="help" goto :show_help
goto :invalid_arg

:run_default
echo ========================================
echo   AI Estimate Pipeline Server
echo ========================================
echo.

REM Anaconda Python이 있으면 사용, 없으면 일반 Python 사용
if exist "%CONDA_PYTHON%" (
    echo Using Anaconda Python: %CONDA_PYTHON%
    echo Starting server at http://localhost:8000
    echo API Docs at http://localhost:8000/docs
    echo Press Ctrl+C to stop
    echo.
    "%CONDA_PYTHON%" "%~dp0run.py"
) else (
    echo Starting server at http://localhost:8000
    echo API Docs at http://localhost:8000/docs
    echo Press Ctrl+C to stop
    echo.
    %DEFAULT_PYTHON% "%~dp0run.py"
)
goto :end

:run_setup
echo Running setup check...
echo.
if exist "%CONDA_PYTHON%" (
    "%CONDA_PYTHON%" "%~dp0run.py" setup
) else (
    %DEFAULT_PYTHON% "%~dp0run.py" setup
)
goto :end

:run_test
echo Running tests...
echo.
if exist "%CONDA_PYTHON%" (
    "%CONDA_PYTHON%" "%~dp0run.py" test
) else (
    %DEFAULT_PYTHON% "%~dp0run.py" test
)
goto :end

:run_conda
echo ========================================
echo   AI Estimate Pipeline Server
echo   Using Anaconda Python
echo ========================================
echo.

REM Anaconda Python 존재 확인
if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Anaconda Python not found at %CONDA_PYTHON%
    echo Please install Anaconda or update the path
    goto :end
)

echo Using Python: %CONDA_PYTHON%
echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop
echo.
"%CONDA_PYTHON%" "%~dp0run.py" --conda
goto :end

:show_help
echo AI Estimate Pipeline - 통합 실행 스크립트
echo.
echo Usage:
echo   run.bat              - 서버 실행 (기본 Python)
echo   run.bat setup        - 프로젝트 설정 확인
echo   run.bat test         - 기본 테스트 실행
echo   run.bat conda        - Anaconda Python으로 실행
echo   run.bat help         - 이 도움말 표시
echo.
echo Examples:
echo   run.bat              - Start server
echo   run.bat setup        - Check project setup
echo   run.bat conda        - Run with Anaconda
goto :end

:invalid_arg
echo [ERROR] Invalid argument: %1
echo.
echo Use "run.bat help" for usage information
goto :end

:end
pause