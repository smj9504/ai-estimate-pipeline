#!/usr/bin/env python
"""
AI Estimate Pipeline - 통합 실행 스크립트
여러 서버 실행 파일들을 하나로 통합한 버전

Usage:
    python run.py              # 서버 실행 (기본)
    python run.py setup        # 프로젝트 설정 확인
    python run.py test         # 기본 테스트 실행
    python run.py --conda      # Conda 환경으로 실행
    python run.py --help       # 도움말 표시
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class AIEstimatePipelineRunner:
    """AI Estimate Pipeline 통합 실행기"""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.anaconda_path = Path(r"C:\Users\user\anaconda3")
        self.conda_python = self.anaconda_path / "python.exe"
        
    def print_header(self, title: str = "AI Estimate Pipeline Server"):
        """헤더 출력"""
        print("=" * 50)
        print(f"  {title}")
        print("=" * 50)
        
    def check_anaconda(self) -> bool:
        """Anaconda 설치 확인"""
        if self.anaconda_path.exists() and self.conda_python.exists():
            return True
        return False
        
    def check_dependencies(self) -> bool:
        """필수 패키지 확인"""
        required_packages = [
            'fastapi', 'uvicorn', 'pydantic', 'jinja2',
            'python-multipart', 'pyyaml', 'python-dotenv',
            'openai', 'anthropic', 'google-generativeai',
            'numpy', 'scipy', 'pandas'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("\n[ERROR] 누락된 패키지:")
            for package in missing_packages:
                print(f"  - {package}")
            print("\n설치 명령어:")
            print(f"  pip install {' '.join(missing_packages)}")
            return False
        else:
            print("[OK] 모든 필수 패키지가 설치되어 있습니다.")
            return True
            
    def check_config_files(self) -> bool:
        """설정 파일 존재 확인"""
        required_files = [
            "config/settings.yaml",
            "src/main.py",
            "requirements.txt"
        ]
        
        missing_files = []
        
        for file_path in required_files:
            full_path = self.base_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("\n[ERROR] 누락된 파일:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            return False
            
        # .env 파일 확인
        env_file = self.base_dir / ".env"
        if not env_file.exists():
            print("\n[WARNING] .env 파일이 없습니다.")
            print("  .env.example을 .env로 복사하고 API 키를 설정하세요:")
            print("  - OPENAI_API_KEY")
            print("  - ANTHROPIC_API_KEY")
            print("  - GOOGLE_API_KEY")
            
            # .env.example이 있으면 복사 제안
            env_example = self.base_dir / ".env.example"
            if env_example.exists():
                print("\n  복사 명령어: copy .env.example .env")
        else:
            print("[OK] .env 파일이 존재합니다.")
            
        return len(missing_files) == 0
        
    def setup_project(self):
        """프로젝트 설정 확인 및 초기화"""
        self.print_header("Project Setup Check")
        
        print("\n1. 의존성 패키지 확인...")
        deps_ok = self.check_dependencies()
        
        print("\n2. 설정 파일 확인...")
        config_ok = self.check_config_files()
        
        print("\n3. Anaconda 환경 확인...")
        if self.check_anaconda():
            print(f"[OK] Anaconda가 설치되어 있습니다: {self.anaconda_path}")
        else:
            print("[INFO] Anaconda가 설치되어 있지 않습니다.")
            print("  일반 Python 환경을 사용합니다.")
        
        # 웹 디렉토리 생성
        web_dir = self.base_dir / "web"
        templates_dir = web_dir / "templates"
        static_dir = web_dir / "static"
        
        templates_dir.mkdir(parents=True, exist_ok=True)
        static_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n4. 웹 디렉토리 확인...")
        print(f"[OK] Templates: {templates_dir}")
        print(f"[OK] Static: {static_dir}")
        
        if deps_ok and config_ok:
            print("\n[OK] 프로젝트 설정이 완료되었습니다!")
            print("\n서버를 실행하려면:")
            print("  python run.py")
            return True
        else:
            print("\n[ERROR] 설정을 완료한 후 다시 시도하세요.")
            return False
            
    def run_tests(self):
        """기본 테스트 실행"""
        self.print_header("Basic Tests")
        
        print("\n테스트 1: 샘플 데이터 로드")
        sample_file = self.base_dir / "data" / "samples" / "sample_input.json"
        
        try:
            if sample_file.exists():
                with open(sample_file, 'r') as f:
                    data = json.load(f)
                print(f"[OK] 샘플 데이터 로드 성공")
                if isinstance(data, list) and len(data) > 0:
                    print(f"   프로젝트: {data[0].get('Jobsite', 'N/A')}")
            else:
                print(f"[ERROR] 샘플 파일이 없습니다: {sample_file}")
        except Exception as e:
            print(f"[ERROR] 샘플 데이터 로드 실패: {e}")
            
        print("\n테스트 2: 설정 파일 로드")
        try:
            from src.utils.config_loader import ConfigLoader
            loader = ConfigLoader()
            config = loader.load_config()
            print("[OK] 설정 로드 성공")
            print(f"   모델 가중치: GPT-4({config.model_weights.gpt4}), "
                  f"Claude({config.model_weights.claude}), "
                  f"Gemini({config.model_weights.gemini})")
        except Exception as e:
            print(f"[ERROR] 설정 로드 실패: {e}")
            
        print("\n테스트 완료!")
        
    def run_server(self, use_conda: bool = True):
        """서버 실행"""
        self.print_header()
        
        # Conda가 있으면 우선적으로 사용, 없으면 일반 Python 사용
        if use_conda and self.check_anaconda():
            # Conda Python을 subprocess로 실행
            self.run_with_subprocess()
            return
        
        # Conda가 없거나 use_conda=False인 경우 일반 Python 사용
        if not use_conda:
            print("\n[INFO] 일반 Python 환경을 사용합니다 (--no-conda 옵션)...")
        else:
            print("\n[INFO] Anaconda를 찾을 수 없어 일반 Python 환경을 사용합니다...")
        print("  URL: http://localhost:8000")
        print("  API 문서: http://localhost:8000/docs")
        print("  종료: Ctrl+C\n")
        
        try:
            import uvicorn
            
            # 서버 실행
            uvicorn.run(
                "src.main:app",
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info"
            )
            
        except ImportError as e:
            print(f"\n[ERROR] 모듈을 찾을 수 없습니다: {e}")
            print("\n필수 패키지를 설치하세요:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
            
        except KeyboardInterrupt:
            print("\n[INFO] 서버가 중지되었습니다.")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n[ERROR] 서버 시작 실패: {e}")
            sys.exit(1)
            
    def run_with_subprocess(self):
        """subprocess를 사용한 Conda Python 실행"""
        if not self.check_anaconda():
            print("[ERROR] Anaconda가 설치되어 있지 않습니다.")
            print(f"  경로: {self.anaconda_path}")
            sys.exit(1)
            
        self.print_header()
        print(f"\n[INFO] Anaconda Python 사용: {self.conda_python}")
        print("[INFO] 서버를 시작합니다...")
        print("  URL: http://localhost:8000")
        print("  API 문서: http://localhost:8000/docs")
        print("  종료: Ctrl+C\n")
        
        # uvicorn을 subprocess로 실행
        cmd = [
            str(self.conda_python),
            "-m", "uvicorn",
            "src.main:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n[INFO] 서버가 중지되었습니다.")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] 서버 실행 실패: {e}")
            sys.exit(1)
            
    def show_help(self):
        """도움말 표시"""
        print(__doc__)
        print("\n옵션:")
        print("  (없음)        서버를 실행 (Anaconda 우선, 없으면 일반 Python)")
        print("  setup        프로젝트 설정 확인 및 초기화")
        print("  test         기본 테스트 실행")
        print("  --conda      Anaconda Python으로 강제 실행")
        print("  --no-conda   일반 Python으로 강제 실행")
        print("  --help       이 도움말 표시")
        print("\n예시:")
        print("  python run.py              # 서버 시작 (자동 환경 감지)")
        print("  python run.py setup        # 설정 확인")
        print("  python run.py test         # 테스트 실행")
        print("  python run.py --conda      # Conda로 강제 실행")
        print("  python run.py --no-conda   # 일반 Python으로 실행")


def main():
    """메인 함수"""
    runner = AIEstimatePipelineRunner()
    
    # 명령줄 인수 처리
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ["--help", "-h", "help"]:
            runner.show_help()
        elif arg == "setup":
            runner.setup_project()
        elif arg == "test":
            runner.run_tests()
        elif arg == "--conda":
            runner.run_with_subprocess()
        elif arg == "--subprocess":
            runner.run_with_subprocess()
        elif arg == "--no-conda":
            # Conda를 사용하지 않고 일반 Python 사용
            runner.run_server(use_conda=False)
        else:
            print(f"[ERROR] 알 수 없는 옵션: {arg}")
            print("도움말을 보려면: python run.py --help")
            sys.exit(1)
    else:
        # 기본 동작: Conda가 있으면 Conda 사용, 없으면 일반 Python
        runner.run_server()


if __name__ == "__main__":
    main()