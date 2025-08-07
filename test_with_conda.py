"""
conda 환경에서 실행 가능한 테스트 스크립트
"""
import subprocess
import sys
import os

# UTF-8 인코딩 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'

def run_in_conda():
    """conda 환경에서 test_gemini.py 실행"""
    
    conda_path = r"C:\Users\user\anaconda3"
    conda_env = "ai-estimate"
    
    # conda 환경의 Python 경로
    conda_python = os.path.join(conda_path, "envs", conda_env, "python.exe")
    
    if not os.path.exists(conda_python):
        print(f"[ERROR] conda Python not found: {conda_python}")
        return False
    
    print(f"[INFO] Using conda Python: {conda_python}")
    print(f"[INFO] Running test_gemini.py in conda environment...")
    print("=" * 60)
    
    # conda 환경에서 스크립트 실행
    try:
        result = subprocess.run(
            [conda_python, "test_gemini.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # 출력 표시
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("[STDERR]:", result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"[ERROR] Failed to run: {e}")
        return False

if __name__ == "__main__":
    success = run_in_conda()
    sys.exit(0 if success else 1)