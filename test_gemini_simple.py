"""
간단한 Gemini API 테스트
"""
import os
import sys

# 프로젝트 경로를 시스템 경로에 추가
sys.path.insert(0, r'C:\My projects\ai-estimate-pipeline')

print("Python 경로:", sys.executable)
print("Python 버전:", sys.version)
print()

try:
    # .env 파일 로드
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env 파일 로드 완료")
    
    # 환경 변수에서 API 키 확인
    google_key = os.getenv('GOOGLE_API_KEY')
    if google_key:
        print(f"[OK] Google API Key 설정됨 (길이: {len(google_key)})")
    else:
        print("[ERROR] Google API Key가 설정되지 않음")
        print("       .env 파일을 확인하거나 환경 변수를 설정하세요")
        sys.exit(1)
    
    # Gemini 라이브러리 임포트 테스트
    print("\n1. google-generativeai 임포트 테스트...")
    import google.generativeai as genai
    print("[OK] google-generativeai 임포트 성공")
    
    # API 설정
    print("\n2. Gemini API 설정...")
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("[OK] Gemini 1.5 Flash 모델 초기화 성공")
    
    # 간단한 테스트 프롬프트
    print("\n3. 간단한 API 호출 테스트...")
    test_prompt = "Say 'Hello, Gemini is working!' in exactly 5 words."
    
    response = model.generate_content(test_prompt)
    print(f"[OK] API 응답: {response.text}")
    
    print("\n" + "="*60)
    print("[SUCCESS] Gemini API 연결 테스트 성공!")
    print("="*60)
    
except ImportError as e:
    print(f"[ERROR] 임포트 오류: {e}")
    print("       pip install google-generativeai 실행 필요")
    
except Exception as e:
    print(f"[ERROR] 테스트 실패: {e}")
    import traceback
    traceback.print_exc()