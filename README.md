# AI Estimate Pipeline

Multi-model AI system for residential reconstruction estimates using GPT-4, Claude, and Gemini.

## 🚀 Quick Start

### 즉시 실행 (자동 환경 감지)
```bash
# Windows - Anaconda Python 자동 감지 및 사용
run.bat

# Mac/Linux/Windows - Python 스크립트 실행
python run.py

# 서버가 http://localhost:8000 에서 실행됩니다
# API 문서: http://localhost:8000/docs
```

## 📋 Prerequisites

### 필수 요구사항
- Python 3.10 이상 (Anaconda 권장)
- API Keys:
  - OpenAI API Key (GPT-4)
  - Anthropic API Key (Claude)
  - Google API Key (Gemini)

## 🔧 Setup

### 1. 프로젝트 클론 및 설정 확인
```bash
# 프로젝트 클론
git clone https://github.com/your-username/ai-estimate-pipeline.git
cd ai-estimate-pipeline

# 설정 확인 및 초기화
python run.py setup  # 자동으로 Anaconda 감지하여 사용
# 또는
run.bat setup       # Windows 배치 파일
```

### 2. Python 환경 설정

#### Option A: Anaconda 환경 (권장 - 자동 감지됨)
```bash
# Anaconda가 설치되어 있으면 run.py가 자동으로 사용합니다
# 수동으로 환경을 만들고 싶은 경우:
conda create -n ai-estimate python=3.10
conda activate ai-estimate
pip install -r requirements.txt
```

#### Option B: 일반 Python 환경
```bash
# 가상 환경 생성
python -m venv venv

# 활성화
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성 (Windows)
copy .env.example .env

# .env 파일 생성 (Mac/Linux)
cp .env.example .env

# .env 파일을 편집하여 API 키 설정:
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

## 🎯 실행 방법

### 방법 1: 자동 환경 감지 (권장) ✨
```bash
# Anaconda가 설치되어 있으면 자동으로 사용합니다
python run.py        # 서버 시작 (자동 환경 감지)
run.bat             # Windows 배치 파일 (자동 환경 감지)
```

### 방법 2: 통합 실행 스크립트 옵션
```bash
# 서버 실행 옵션
python run.py              # 자동 환경 감지 (Anaconda 우선)
python run.py --conda      # Anaconda Python 강제 사용
python run.py --no-conda   # 일반 Python 강제 사용

# 유틸리티 명령
python run.py setup        # 프로젝트 설정 확인
python run.py test         # 기본 테스트 실행
python run.py --help       # 도움말 표시
```

### 방법 3: Windows 배치 파일
```bash
run.bat              # 서버 실행 (자동 환경 감지)
run.bat setup        # 설정 확인
run.bat test         # 테스트 실행
run.bat conda        # Anaconda 강제 사용
run.bat help         # 도움말 표시
```

### 방법 4: 직접 실행 (고급 사용자)
```bash
# Anaconda Python 직접 사용
"C:\Users\user\anaconda3\python.exe" -m uvicorn src.main:app --reload

# 일반 Python 사용
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
ai-estimate-pipeline/
├── src/
│   ├── main.py                    # FastAPI 메인 애플리케이션
│   ├── models/
│   │   ├── data_models.py         # Pydantic 데이터 모델
│   │   └── model_interface.py     # AI 모델 인터페이스 (GPT-4, Claude, Gemini)
│   ├── phases/                    # 단계별 프로세서
│   │   ├── phase_manager.py       # 전체 Phase 관리
│   │   ├── phase0_processor.py    # Phase 0: Generate Scope
│   │   ├── phase1_processor.py    # Phase 1: Merge Measurement
│   │   └── phase2_processor.py    # Phase 2: Quantity Survey
│   ├── processors/
│   │   └── result_merger.py       # 멀티모델 결과 병합 로직
│   ├── validators/
│   │   └── estimation_validator.py # Remove & Replace 검증 로직
│   └── utils/
│       ├── config_loader.py       # 설정 로더
│       ├── prompt_manager.py      # 프롬프트 관리
│       ├── statistical_utils.py   # 통계 처리
│       └── validation_utils.py    # 검증 유틸리티
├── prompts/                       # AI 프롬프트 템플릿
│   ├── phase_0_generate_scope.md
│   ├── phase_1_merge_measurement.md
│   └── phase_2_quantity_survey.md
├── web/
│   ├── templates/
│   │   └── index.html            # 웹 인터페이스
│   └── static/                   # CSS, JS 파일
├── config/
│   └── settings.yaml             # 애플리케이션 설정
├── data/
│   └── samples/                  # 샘플 데이터
├── tests/                        # 테스트 파일
├── requirements.txt              # Python 의존성
├── .env.example                  # 환경변수 템플릿
├── run.py                        # 통합 실행 스크립트 (자동 환경 감지)
└── run.bat                       # Windows 배치 파일 (자동 환경 감지)
```

## 🌟 Features

### 현재 구현된 기능 (Phase 0-2)
- ✅ **Phase 0**: Generate Scope of Work - 단일 모델로 초기 작업 범위 생성
- ✅ **Phase 1**: Merge Measurement & Work Scope - 멀티모델 병렬 처리
- ✅ **Phase 2**: Quantity Survey - 정량적 견적 생성 (멀티모델)
- ✅ **Multi-Model Consensus**: GPT-4, Claude, Gemini 3개 모델 합의 도출
- ✅ **Remove & Replace Logic**: 철거 및 교체 로직 자동 적용
- ✅ **Statistical Merging**: IQR 기반 이상치 제거 및 가중평균
- ✅ **Web Interface**: 드래그 앤 드롭 JSON 업로드 인터페이스
- ✅ **Auto Environment Detection**: Anaconda Python 자동 감지 및 사용

### 개발 예정 기능 (Phase 3-6)
- ⏳ **Phase 3**: Market Research - DMV 지역 시장가격 조사
- ⏳ **Phase 4**: Timeline & Disposal - 작업 일정 및 폐기물 처리
- ⏳ **Phase 5**: Final Estimate - 최종 견적 완성
- ⏳ **Phase 6**: JSON Formatting - 클라이언트 형식 출력

## 🧪 Testing

```bash
# 기본 테스트 실행
python run.py test

# pytest 사용
pytest

# 특정 테스트 파일 실행
pytest tests/test_model_interface.py

# 상세 출력
pytest -v
```

## 🛠️ Development

### 코드 포맷팅
```bash
# Black으로 코드 포맷팅
black src/ tests/

# Flake8으로 린팅
flake8 src/ tests/
```

### 환경 변수 확인
```bash
# 설정된 API 키 확인
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
python -c "import os; print('Anthropic:', bool(os.getenv('ANTHROPIC_API_KEY')))"
python -c "import os; print('Google:', bool(os.getenv('GOOGLE_API_KEY')))"
```

## 📝 API Documentation

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 주요 엔드포인트
- `POST /api/phase/execute` - Phase 실행
- `POST /api/phase/approve` - Phase 결과 승인
- `GET /api/phase/status/{session_id}` - Phase 상태 조회
- `POST /api/estimate/merge` - 레거시 멀티모델 병합 API
- `GET /api/health` - 시스템 상태 확인

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.
