# AI Estimate Pipeline

Multi-model AI system for residential reconstruction estimates using GPT-4, Claude, and Gemini.

## 🚀 Quick Start

### 즉시 실행 (자동 환경 감지)
```bash
# Windows - Anaconda Python 자동 감지 및 사용
run.bat
# 또는
python scripts/run.py

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
python scripts/run.py setup  # 자동으로 Anaconda 감지하여 사용
# 또는
run.bat setup       # Windows 배치 파일 (하위 호환성)
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
python scripts/run.py  # 서버 시작 (자동 환경 감지)
run.bat               # Windows 배치 파일 (루트에서 실행 가능)
```

### 방법 2: 통합 실행 스크립트 옵션
```bash
# 서버 실행 옵션
python scripts/run.py              # 자동 환경 감지 (Anaconda 우선)
python scripts/run.py --conda      # Anaconda Python 강제 사용
python scripts/run.py --no-conda   # 일반 Python 강제 사용

# 유틸리티 명령
python scripts/run.py setup        # 프로젝트 설정 확인
python scripts/run.py test         # 기본 테스트 실행
python scripts/run.py --help       # 도움말 표시
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
├── scripts/                       # 실행 스크립트
│   ├── run.py                     # 메인 실행 스크립트
│   ├── run.bat                    # Windows 배치 파일
│   └── run_conda.bat              # Conda 환경 실행
├── tools/                         # 개발 도구
│   ├── test_fixes.py              # 테스트 수정 도구
│   ├── compare_test_results.py    # 결과 비교 도구
│   └── install_tracking.py        # 설치 추적 도구
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
- ✅ **API Token Tracking**: 토큰 사용량 및 비용 실시간 추적
- ✅ **Comprehensive Testing**: 21가지 AI 모델 조합 테스트 시스템

### 개발 예정 기능 (Phase 3-6)
- ⏳ **Phase 3**: Market Research - DMV 지역 시장가격 조사
- ⏳ **Phase 4**: Timeline & Disposal - 작업 일정 및 폐기물 처리
- ⏳ **Phase 5**: Final Estimate - 최종 견적 완성
- ⏳ **Phase 6**: JSON Formatting - 클라이언트 형식 출력

## 🧪 Testing

### 기본 테스트
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

### Phase별 테스트 시스템
프로젝트는 체계적인 Phase 테스트 구조를 제공합니다:

```bash
# Phase 1 단독 테스트 (기본: 전체 모델)
python run_phase_tests.py single --phase 1

# 전체 파이프라인 테스트 (Phase 0→1→2)
python run_phase_tests.py pipeline --phases 0 1 2

# 모델 조합 비교 테스트
python run_phase_tests.py compare --phase 1 --compare-type models
```

#### AI 모델 선택 기능 ✨

각 테스트 명령어에서 `--models` 플래그를 사용하여 원하는 AI 모델 조합을 선택할 수 있습니다:

```bash
# 단일 모델 테스트
python run_phase_tests.py single --phase 1 --models gpt4
python run_phase_tests.py single --phase 1 --models claude
python run_phase_tests.py single --phase 1 --models gemini

# 두 모델 조합 테스트
python run_phase_tests.py single --phase 1 --models gpt4 claude
python run_phase_tests.py single --phase 1 --models claude gemini
python run_phase_tests.py single --phase 1 --models gpt4 gemini

# 전체 모델 테스트 (명시적 지정)
python run_phase_tests.py single --phase 1 --models gpt4 claude gemini

# 파이프라인에서 모델 선택
python run_phase_tests.py pipeline --phases 0 1 2 --models gpt4 claude

# 모델 비교 테스트에서 특정 모델들만 비교
python run_phase_tests.py compare --phase 1 --models gpt4 claude --compare-type models
```

**사용 가능한 모델 조합**:

| 선택 | 명령어 예시 | 설명 |
|------|------------|------|
| **GPT-4 단독** | `--models gpt4` | OpenAI GPT-4만 사용 |
| **Claude 단독** | `--models claude` | Anthropic Claude만 사용 |
| **Gemini 단독** | `--models gemini` | Google Gemini만 사용 |
| **GPT+Claude** | `--models gpt4 claude` | 2모델 조합 |
| **Claude+Gemini** | `--models claude gemini` | 2모델 조합 |
| **GPT+Gemini** | `--models gpt4 gemini` | 2모델 조합 |
| **전체 모델** | `--models gpt4 claude gemini` | 3모델 조합 (기본값) |

**도움말 확인**:
```bash
python run_phase_tests.py single --help  # 단일 Phase 테스트 옵션
python run_phase_tests.py pipeline --help  # 파이프라인 테스트 옵션
python run_phase_tests.py compare --help   # 비교 테스트 옵션
```

#### 테스트 데이터 구조
```
test_data/
├── sample_demo.json          # 철거 범위 데이터
├── sample_measurement.json   # 측정 데이터
└── sample_intake_form.txt    # 작업 범위 입력 양식
```

각 프로젝트는 위 3가지 데이터를 필요로 하며, 테스트 시스템은 실제 프로젝트 데이터를 사용합니다.

### AI 모델 조합 테스트
`docs/AI_SYSTEM_ENHANCEMENT_STRATEGIES.md`에 따라 모든 가능한 AI 모델 조합을 체계적으로 테스트할 수 있습니다:

#### 빠른 실행
```bash
# 필수 테스트 (7개 구성)
python -m tests.model_combinations.test_runner --test-type essential

# 성능 비교 테스트 (10개 구성)
python -m tests.model_combinations.test_runner --test-type performance

# 전체 포괄적 테스트 (21개 구성)
python -m tests.model_combinations.test_runner --test-type comprehensive
```

#### 테스트 매트릭스
- **단일 모델**: GPT-4, Claude, Gemini 개별 테스트
- **모델 쌍**: GPT-4+Claude, GPT-4+Gemini, Claude+Gemini
- **전체 조합**: GPT-4+Claude+Gemini
- **검증 모드**: Strict, Balanced, Lenient
- **처리 방식**: Parallel, Sequential

#### 테스트 결과 비교
```bash
# 테스트 결과 비교 도구
python compare_test_results.py

# 출력 파일 형식
output/
├── phase1_GCM_BAL_ROOM_SAMPLE_20250808_120000.json
│   └── G=GPT-4, C=Claude, M=Gemini, BAL=Balanced, ROOM=방별처리
├── phase1_G_STR_BATCH_081132_20250808_121500.json
│   └── G=GPT-4만, STR=Strict, BATCH=일괄처리
└── comparison_reports/
    ├── report_20250808.html    # HTML 시각화 리포트
    └── report_20250808.xlsx    # Excel 분석 리포트
```

### 대화형 테스트 데모
```bash
# 대화형 모델 조합 테스트
python tests/demo_model_testing.py
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

## 📊 API Token Usage Tracking

프로젝트는 모든 AI API 호출의 토큰 사용량과 비용을 자동으로 추적합니다:

### 설치 및 초기화
```bash
# 토큰 추적 시스템 설치
python install_tracking.py
```

### 웹 대시보드
서버 실행 후 `http://localhost:8000/usage`에서 실시간 사용량을 확인할 수 있습니다:
- 실시간 토큰 사용량 통계
- 모델별 비용 분석
- 일별/주별/월별 리포트
- CSV/Excel 데이터 내보내기

### CLI 명령어
```bash
# 현재 사용량 통계
python -m src.tracking.cli stats

# 일일 리포트 생성
python -m src.tracking.cli report daily

# 모델별 가격 정보
python -m src.tracking.cli pricing

# 최근 30일 데이터 CSV 내보내기
python -m src.tracking.cli export csv --days 30

# 실시간 모니터링
python -m src.tracking.cli live
```

### 지원 모델 및 가격
- **OpenAI**: GPT-4o ($5.00/$15.00 per 1M), GPT-4o-mini ($0.15/$0.60 per 1M)
- **Anthropic**: Claude-3.5-Sonnet ($3.00/$15.00 per 1M), Claude-3-Sonnet ($3.00/$15.00 per 1M)
- **Google**: Gemini-1.5-Pro ($3.50/$10.50 per 1M), Gemini-1.5-Flash ($0.075/$0.30 per 1M)

## 📝 API Documentation

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Usage Dashboard: http://localhost:8000/usage

### 주요 엔드포인트
- `POST /api/phase/execute` - Phase 실행
- `POST /api/phase/approve` - Phase 결과 승인
- `GET /api/phase/status/{session_id}` - Phase 상태 조회
- `POST /api/estimate/merge` - 레거시 멀티모델 병합 API
- `GET /api/health` - 시스템 상태 확인

### 토큰 추적 API
- `GET /api/tracking/stats` - 사용량 통계
- `GET /api/tracking/reports/daily` - 일일 리포트
- `GET /api/tracking/dashboard/summary` - 대시보드 데이터
- `GET /api/tracking/export/csv` - CSV 내보내기
- `GET /api/tracking/projections` - 비용 예측

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.
