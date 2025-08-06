# setup.py
"""
Reconstruction Estimator 프로젝트 setup 스크립트
"""
import os
import sys
from pathlib import Path

def create_web_directories():
    """웹 템플릿 디렉토리 생성 및 HTML 파일 저장"""
    base_dir = Path(__file__).resolve().parent
    
    # 웹 디렉토리 생성
    web_dir = base_dir / "web"
    templates_dir = web_dir / "templates"
    static_dir = web_dir / "static"
    
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # index.html 파일 생성 (위에서 만든 HTML을 여기에 저장)
    index_html = templates_dir / "index.html"
    
    # HTML 내용은 위의 web_interface artifact에서 복사
    html_content = '''<!-- 위에서 만든 HTML 내용을 여기에 복사 -->'''
    
    print(f"웹 템플릿 디렉토리 생성: {templates_dir}")
    print(f"정적 파일 디렉토리 생성: {static_dir}")
    
    return templates_dir, static_dir

def check_dependencies():
    """필수 의존성 확인"""
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'jinja2', 
        'python-multipart', 'pyyaml', 'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 누락된 패키지:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n설치 명령어: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ 모든 필수 패키지가 설치되어 있습니다.")
        return True

def check_config_files():
    """설정 파일 존재 확인"""
    base_dir = Path(__file__).resolve().parent
    
    required_files = [
        "config/settings.yaml",
        ".env.example"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print("📁 설정 파일 상태:")
    for file_path in existing_files:
        print(f"  ✅ {file_path}")
    
    for file_path in missing_files:
        print(f"  ❌ {file_path} (누락)")
    
    # .env 파일 확인
    env_file = base_dir / ".env"
    if not env_file.exists():
        print("\n⚠️  .env 파일이 없습니다.")
        print("   cp .env.example .env 명령으로 복사한 후 API 키를 설정하세요.")
    
    return len(missing_files) == 0

def run_development_server():
    """개발 서버 실행"""
    print("\n🚀 개발 서버를 시작합니다...")
    print("   URL: http://localhost:8000")
    print("   종료: Ctrl+C\n")
    
    try:
        import uvicorn
        uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n서버가 종료되었습니다.")
    except ImportError:
        print("❌ uvicorn이 설치되지 않았습니다.")
        print("   pip install uvicorn")

def main():
    """메인 setup 함수"""
    print("=" * 50)
    print("  Reconstruction Estimator Setup")
    print("=" * 50)
    
    # 1. 웹 디렉토리 생성
    print("\n1. 웹 디렉토리 설정...")
    create_web_directories()
    
    # 2. 의존성 확인
    print("\n2. 의존성 확인...")
    deps_ok = check_dependencies()
    
    # 3. 설정 파일 확인
    print("\n3. 설정 파일 확인...")
    config_ok = check_config_files()
    
    # 4. 개발 서버 실행 여부 확인
    if deps_ok and config_ok:
        print("\n✅ 프로젝트 설정이 완료되었습니다!")
        
        if len(sys.argv) > 1 and sys.argv[1] == "run":
            run_development_server()
        else:
            print("\n다음 명령어로 서버를 실행하세요:")
            print("  python setup.py run")
            print("  또는")
            print("  python -m uvicorn src.main:app --reload")
    else:
        print("\n❌ 설정을 완료한 후 다시 시도하세요.")

if __name__ == "__main__":
    main()


# run_server.py - 간단한 서버 실행 스크립트
"""
개발 서버 실행 전용 스크립트
"""

if __name__ == "__main__":
    print("🚀 Reconstruction Estimator 시작중...")
    
    try:
        import uvicorn
        uvicorn.run(
            "src.main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("❌ uvicorn이 설치되지 않았습니다.")
        print("설치 명령어: pip install uvicorn")
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")


# test_setup.py - 기본 테스트
"""
기본 기능 테스트
"""
import json
from pathlib import Path

def test_sample_data():
    """샘플 데이터 로드 테스트"""
    base_dir = Path(__file__).resolve().parent
    sample_file = base_dir / "data" / "samples" / "sample_input.json"
    
    try:
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        print("✅ 샘플 데이터 로드 성공")
        print(f"   프로젝트: {data[0].get('Jobsite', 'N/A')}")
        print(f"   방 개수: {len(data[1].get('rooms', []))}")
        
        return True
    except Exception as e:
        print(f"❌ 샘플 데이터 로드 실패: {e}")
        return False

def test_config_load():
    """설정 파일 로드 테스트"""
    try:
        from src.utils.config_loader import ConfigLoader
        
        loader = ConfigLoader()
        config = loader.load_config()
        
        print("✅ 설정 로드 성공")
        print(f"   모델 가중치: GPT-4({config.model_weights.gpt4}), Claude({config.model_weights.claude})")
        
        return True
    except Exception as e:
        print(f"❌ 설정 로드 실패: {e}")
        return False

def run_tests():
    """모든 테스트 실행"""
    print("🧪 기본 기능 테스트 시작...\n")
    
    tests = [
        ("샘플 데이터 테스트", test_sample_data),
        ("설정 로드 테스트", test_config_load),
    ]
    
    passed = 0
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print(f"🏁 테스트 결과: {passed}/{len(tests)} 통과")
    
    if passed == len(tests):
        print("✅ 모든 테스트가 통과했습니다!")
    else:
        print("⚠️  일부 테스트가 실패했습니다. 설정을 확인하세요.")

if __name__ == "__main__":
    run_tests()