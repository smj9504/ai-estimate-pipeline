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

## 🏗️ 데이터 파이프라인 아키텍처

### ELT 패턴 기반 데이터 흐름

프로젝트는 **Extract-Load-Transform (ELT)** 패턴을 채택하여 대용량 AI 처리에 최적화된 데이터 파이프라인을 구축합니다:

```
📥 Extract (추출)     📦 Load (적재)      🔄 Transform (변환)
    ↓                    ↓                   ↓
JSON 입력 데이터  →  Raw Data Store  →  AI Model Processing
    │                    │                   │
    ├─ 측정값            ├─ 캐시 레이어        ├─ Phase 1: 작업 범위
    ├─ 철거 범위         ├─ 중간 결과         ├─ Phase 2: 수량 산정
    └─ 작업 명세         └─ 메타데이터        └─ 결과 병합
```

### 데이터 플로우 최적화 전략

**1. 스트리밍 처리 (Streaming)**
```python
# Phase별 점진적 처리로 메모리 효율성 확보
async def process_pipeline(data):
    phase1_result = await process_phase1_stream(data)
    phase2_result = await process_phase2_stream(phase1_result)
    return merge_results(phase2_result)
```

**2. 배치 처리 (Batching)**
```python
# AI 모델 호출 배치화로 API 효율성 향상
async def batch_ai_calls(prompts, batch_size=3):
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        results = await asyncio.gather(*[call_model(p) for p in batch])
        yield results
```

**3. 캐시 기반 최적화**
```python
# 다층 캐싱으로 중복 계산 방지
@cache_result(ttl=3600)  # 1시간 캐시
async def get_ai_response(model, prompt, data_hash):
    return await model.generate(prompt, data_hash)
```

### 품질 게이트웨이 (Quality Gateway)

각 데이터 변환 단계마다 품질 검증 게이트웨이를 설치:

```python
# 4단계 품질 검증 파이프라인
quality_gates = [
    ("입력 검증", validate_input_schema),      # 스키마 준수 확인
    ("비즈니스 규칙", validate_business_rules), # Remove & Replace 로직
    ("일관성 검사", validate_consistency),     # 모델 간 응답 일관성
    ("출력 검증", validate_output_format)     # 최종 형식 검증
]

for gate_name, validator in quality_gates:
    if not validator(data):
        raise QualityGateError(f"{gate_name} 검증 실패")
```

### 모니터링 및 관찰 가능성 (Observability)

**실시간 메트릭 수집**:
```python
# 파이프라인 성능 지표 실시간 모니터링
pipeline_metrics = {
    "처리량": "건/시간",
    "지연시간": "평균 응답시간",
    "오류율": "실패/전체 요청",
    "비용 효율성": "$/건"
}
```

**분산 추적 (Distributed Tracing)**:
```python
# 요청별 전체 파이프라인 추적
@trace_request
async def process_estimate(request_id, data):
    with tracer.start_span("phase1") as span1:
        phase1_result = await phase1_processor(data)
        span1.set_attribute("confidence", phase1_result.confidence)
    
    with tracer.start_span("phase2") as span2:
        phase2_result = await phase2_processor(phase1_result)
        span2.set_attribute("items_count", len(phase2_result.items))
    
    return merge_results([phase1_result, phase2_result])
```

### 확장성 및 성능 최적화

**수평 확장 (Horizontal Scaling)**:
```yaml
# Docker Compose로 멀티 인스턴스 배포
version: '3.8'
services:
  ai-estimator:
    image: ai-estimate-pipeline:latest
    replicas: 3
    environment:
      - LOAD_BALANCER_ENABLED=true
      - CACHE_REDIS_URL=redis://cache:6379
```

**비동기 처리 최적화**:
```python
# 3개 AI 모델 병렬 호출로 처리 시간 1/3 단축
async def parallel_ai_processing(data):
    tasks = [
        gpt4_model.process(data),
        claude_model.process(data),
        gemini_model.process(data)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return merge_consensus(results)
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

## 🧪 Testing Strategy & Framework

### 테스트 전략 개요

프로젝트는 **이중 트랙 접근법(Fast Track vs Full Track)**을 통해 개발 속도와 품질을 모두 확보합니다:

- **Fast Track**: 빠른 피드백을 위한 경량화된 테스트
- **Full Track**: 품질 보증을 위한 포괄적 테스트
- **테스트 피라미드**: 유닛 → 통합 → E2E → Phase별 → 파이프라인 전체

### 테스트 아키텍처

```
테스트 피라미드 구조:
          ┌─────────────────┐
          │ 파이프라인 통합   │ ← 전체 Phase 연계 테스트
          │   (Phase 0-2)   │
          └─────────────────┘
         ┌──────────────────────┐
         │    Phase별 독립     │ ← Phase 1, 2 개별 테스트
         │   (단위 기능 검증)   │
         └──────────────────────┘
        ┌────────────────────────────┐
        │      통합 테스트 (E2E)      │ ← API → AI Models → 병합
        │   (ModelInterface + Merger) │
        └────────────────────────────┘
       ┌──────────────────────────────────┐
       │         유닛 테스트              │ ← 개별 모듈/클래스
       │  (단일 클래스/메서드 검증)       │
       └──────────────────────────────────┘
```

### 기본 테스트 실행

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

### Phase별 독립 테스트 (Fast Track)

각 Phase를 독립적으로 테스트하여 빠른 개발 사이클 지원:

```bash
# Phase 1 단독 테스트 (기본: 전체 모델)
python run_phase_tests.py single --phase 1

# Phase 2 단독 테스트 (특정 모델 조합)
python run_phase_tests.py single --phase 2 --models gpt4 claude

# 캐시된 Phase 1 결과로 Phase 2 테스트 (빠른 반복)
python run_phase_tests.py single --phase 2 --use-cache --models gpt4 claude
```

### 파이프라인 통합 테스트 (Full Track)

전체 워크플로우를 검증하여 품질 보증:

```bash
# 전체 파이프라인 테스트 (Phase 0→1→2)
python run_phase_tests.py pipeline --phases 0 1 2

# 중간 Phase부터 파이프라인 테스트
python run_phase_tests.py pipeline --phases 1 2

# 실제 Phase 1 → Phase 2 데이터 흐름 검증
python run_phase_tests.py pipeline --phases 1 2 --force-fresh
```

### 이중 트랙 테스트 전략

프로젝트는 **개발 속도**와 **품질 보증**을 모두 확보하는 이중 트랙 접근법을 사용합니다:

**Fast Track (캐시 활용)** - CI/CD 및 빠른 개발 피드백용
- Phase 1 실행 결과를 `intermediate/` 폴더에 캐시
- Phase 2 테스트 시 캐시된 결과 재사용
- 실행 시간: 30초~2분
- 사용 케이스: 개발 중 빠른 반복, CI/CD 파이프라인

**Full Track (전체 실행)** - 정확도 검증 및 최종 품질 보증용  
- Phase 1 → Phase 2 전체 파이프라인 실제 실행
- AI 모델의 실시간 응답으로 정확도 검증
- 실행 시간: 2분~5분
- 사용 케이스: PR 머지 전, 배포 전 검증

### CI/CD 최적화 테스트

지속적 통합을 위한 빠른 검증:

```bash
# CI용 필수 테스트 (7개 핵심 구성, ~5분)
python -m tests.model_combinations.test_runner --test-type essential

# 성능 벤치마크 테스트 (10개 구성, ~10분)
python -m tests.model_combinations.test_runner --test-type performance

# 전체 포괄적 테스트 (21개 구성, ~30분)
python -m tests.model_combinations.test_runner --test-type comprehensive
```

## 📊 테스트 데이터 관리 및 품질 보증

### 테스트 데이터 계층 구조

프로젝트는 ELT(Extract-Load-Transform) 패턴을 기반으로 한 체계적인 데이터 관리 시스템을 제공합니다:

```
test_data/
├── golden/                    # Golden Standard (검증된 예상 결과)
│   ├── phase1_expected/       # Phase 1 검증된 출력 결과
│   ├── phase2_expected/       # Phase 2 검증된 출력 결과
│   └── pipeline_expected/     # 전체 파이프라인 예상 결과
├── intermediate/              # 중간 결과물 (캐시 및 재사용) ⭐
│   ├── phase1_outputs/        # Phase 1 실제 출력 (Phase 2 입력으로 재사용)
│   ├── model_responses/       # 개별 AI 모델 응답 캐시
│   └── validation_cache/      # 검증 결과 캐시
├── synthetic/                 # 생성된 테스트 케이스
│   ├── edge_cases/           # 엣지 케이스 (빈 데이터, 극값 등)
│   ├── regression_tests/     # 회귀 테스트용 데이터
│   └── stress_tests/         # 성능/부하 테스트용 데이터
└── real_samples/             # 실제 프로젝트 데이터 (익명화)
    ├── residential/          # 주거용 건축물 케이스
    ├── commercial/           # 상업용 건축물 케이스
    └── historical/           # 과거 프로젝트 아카이브
```

### Phase 1 → Phase 2 데이터 파이프라인 전략

**핵심 질문과 답변:**

**Q1: Phase 1 결과를 pipeline_test에서 재사용해야 하나?**
- ✅ **권장**: 이중 트랙 접근법으로 상황에 맞게 선택
- Fast Track: 캐시된 결과 재사용 (개발/CI용)
- Full Track: 전체 재실행 (품질 검증용)

**Q2: 각 Phase 독립 테스트 vs 통합 테스트 균형점은?**
- Phase 개발 중: 독립 테스트로 빠른 반복
- 기능 완성 후: 통합 테스트로 데이터 흐름 검증
- 배포 전: 전체 파이프라인 테스트로 최종 품질 보증

### 다층 캐싱 전략

**L1 Cache (인메모리)**: 실행 중 AI 모델 응답 캐싱
```python
# 세션 내 AI 응답 재사용으로 API 비용 절약
cache_key = f"{model_name}_{prompt_hash}_{data_hash}"
if cache_key in session_cache:
    return session_cache[cache_key]
```

**L2 Cache (파일시스템)**: Phase별 중간 결과 저장
```bash
# Phase 1 출력을 Phase 2 테스트에 재사용
intermediate/
├── phase1_outputs/
│   ├── sample_demo_gpt4_claude.json  # 특정 모델 조합 결과
│   └── sample_demo_all_models.json   # 전체 모델 결과
```

**L3 Cache (데이터베이스)**: 성능 메트릭 및 벤치마크 저장
```sql
-- 테스트 실행 이력 및 성능 추적
CREATE TABLE test_performance (
    test_id TEXT PRIMARY KEY,
    model_combination TEXT,
    execution_time REAL,
    confidence_score REAL,
    created_at TIMESTAMP
);
```

### 데이터 품질 관리 (6차원 품질 평가)

**1. 완전성 (Completeness)**: 누락된 필수 필드 검증
```python
completeness_score = (채워진_필드_수 / 전체_필수_필드_수) * 100
```

**2. 정확성 (Accuracy)**: 예상 결과 대비 실제 결과 일치도
```python
accuracy_score = (일치하는_값_수 / 전체_비교_값_수) * 100
```

**3. 일관성 (Consistency)**: 모델 간 응답 일관성 평가
```python
consistency_score = 1 - (표준편차 / 평균값)  # 변이계수 기반
```

**4. 유효성 (Validity)**: 비즈니스 규칙 준수 여부
```python
validity_checks = [
    "Remove & Replace 로직 적용",
    "측정값 정확 사용",
    "demo_scope 중복 방지"
]
```

**5. 적시성 (Timeliness)**: 처리 시간 효율성
```python
# 성능 벤치마크 (단위: 초)
acceptable_times = {
    "phase1_single_model": 30,
    "phase1_all_models": 90,
    "pipeline_full": 180
}
```

**6. 유용성 (Usefulness)**: 실제 업무 활용 가능성
```python
usefulness_metrics = [
    "신뢰도 점수 ≥ 85%",
    "비즈니스 로직 정확도 ≥ 90%",
    "사용자 승인률 ≥ 80%"
]
```

### 데이터 버전 관리 및 계보 추적

**Git 기반 데이터 버전 관리**:
```bash
# 테스트 데이터 변경사항 추적
git log --oneline test_data/golden/
git diff HEAD~1 test_data/golden/phase1_expected/

# Phase 1 → Phase 2 데이터 의존성 추적
git log --follow test_data/intermediate/phase1_outputs/
```

**데이터 계보 (Data Lineage) 추적**:
```yaml
# lineage.yaml - 데이터 생성 이력
phase2_test_input.json:
  source: "phase1_outputs/sample_demo_gpt4_claude.json"
  phase1_metadata:
    models: ["gpt4", "claude"]
    execution_time: "2024-01-25 14:30:00"
    confidence_score: 88.5
  transformations:
    - "phase1_execution (2024-01-25)"
    - "quality_validation (2024-01-25)"
    - "cache_storage (2024-01-25)"
  usage_context: "Phase 2 독립 테스트 입력"
  
sample_demo_result.json:
  source: "real_samples/residential/project_001.json"
  transformations:
    - "anonymization (2024-01-15)"
    - "validation by domain expert (2024-01-20)"
    - "golden standard approval (2024-01-25)"
  validators: ["김전문가", "이건축사"]
  confidence_level: "high"
```

### 테스트 데이터 관리 전략 

**Smart Test Data Manager 구현 예정**:
```python
class TestDataManager:
    """Phase 간 데이터 의존성 관리"""
    
    def get_phase2_input(self, mode="cached"):
        if mode == "cached":
            # Fast Track: 캐시된 Phase 1 결과 사용
            return self.load_cached_phase1_output()
        elif mode == "fresh":
            # Full Track: Phase 1 실시간 실행
            return self.execute_phase1_fresh()
        elif mode == "golden":
            # 결정적 테스트: 검증된 골든 데이터 사용
            return self.load_golden_dataset()
```

## 🚀 테스트 실행 방법 가이드

### 빠른 시작 (Quick Start Testing)

```bash
# 1. 기본 기능 검증 (30초)
python run.py test

# 2. Phase 1 단독 테스트 (2분)
python run_phase_tests.py single --phase 1 --models gpt4

# 3. Phase 2 캐시 활용 테스트 (30초) ⭐ NEW
python run_phase_tests.py single --phase 2 --use-cache

# 4. 전체 파이프라인 검증 (5분)
python run_phase_tests.py pipeline --phases 0 1 2 --models gpt4 claude
```

### 개발 단계별 테스트 전략

**🏃‍♂️ 개발 중 (빠른 피드백)**
```bash
# 단일 모델로 빠른 검증
python run_phase_tests.py single --phase 1 --models gpt4 --prompt-version fast

# Phase 2 개발 시 캐시 활용 (Phase 1 재실행 없이) ⭐
python run_phase_tests.py single --phase 2 --use-cache --models claude

# 특정 Phase만 집중 테스트
python run_phase_tests.py single --phase 2 --models claude --prompt-version fast
```

**🔍 통합 테스트 (품질 확인)**
```bash
# 2개 모델 조합 검증
python run_phase_tests.py single --phase 1 --models gpt4 claude

# Phase 1→2 데이터 흐름 검증 ⭐
python run_phase_tests.py pipeline --phases 1 2 --validate-flow

# 전체 파이프라인 통합 테스트
python run_phase_tests.py pipeline --phases 0 1 2 --models gpt4 claude gemini
```

**✅ 배포 전 (완전한 검증)**
```bash
# 모든 모델 조합 포괄적 테스트
python -m tests.model_combinations.test_runner --test-type comprehensive

# 전체 파이프라인 Fresh 실행 (캐시 무시) ⭐
python run_phase_tests.py pipeline --phases 0 1 2 --force-fresh

# 고품질 프롬프트로 최종 검증
python run_phase_tests.py pipeline --phases 0 1 2 --prompt-version improved
```

### 테스트 옵션 매트릭스

| 목적 | 명령어 | 소요시간 | 품질 수준 |
|------|---------|----------|-----------|
| **개발 중 빠른 검증** | `--models gpt4 --prompt-version fast` | 30-60초 | 85% |
| **일반적인 검증** | `--models gpt4 claude` | 60-120초 | 90% |
| **배포 전 완전 검증** | `--models gpt4 claude gemini --prompt-version improved` | 120-240초 | 95%+ |
| **CI/CD 자동화** | `--test-type essential` | ~5분 | 88% |
| **성능 벤치마크** | `--test-type performance` | ~10분 | 92% |
| **완전한 품질 보증** | `--test-type comprehensive` | ~30분 | 95%+ |

### 모델별 테스트 가이드

**GPT-4 중심 개발**
```bash
# GPT-4의 안정성을 활용한 기준선 설정
python run_phase_tests.py single --phase 1 --models gpt4
python run_phase_tests.py single --phase 2 --models gpt4
```

**Claude 품질 검증**
```bash
# Claude의 세밀한 분석 능력 검증
python run_phase_tests.py single --phase 1 --models claude --prompt-version improved
```

**Gemini 비용 효율성**
```bash
# Gemini의 비용 대비 성능 평가
python run_phase_tests.py single --phase 1 --models gemini --prompt-version fast
```

**멀티모델 합의 검증**
```bash
# 3개 모델 합의 도출 과정 검증
python run_phase_tests.py compare --phase 1 --models gpt4 claude gemini --compare-type models
```

### 고급 테스트 옵션

#### 프롬프트 버전 최적화 🚀

`--prompt-version` 플래그로 시나리오별 최적화된 프롬프트 사용:

```bash
# 빠른 개발용 (30-60초, 85% 품질)
python run_phase_tests.py single --phase 1 --prompt-version fast

# 기본 운영용 (60-120초, 90% 품질) - 기본값
python run_phase_tests.py single --phase 1

# 고품질 검토용 (120-240초, 95%+ 품질)
python run_phase_tests.py single --phase 1 --prompt-version improved
```

#### 도움말 및 옵션 확인

```bash
python run_phase_tests.py single --help     # 단일 Phase 테스트 옵션
python run_phase_tests.py pipeline --help   # 파이프라인 테스트 옵션  
python run_phase_tests.py compare --help    # 비교 테스트 옵션
```

### 테스트 결과 분석 및 비교

#### 자동화된 결과 비교
```bash
# 테스트 결과 비교 도구
python compare_test_results.py

# 대화형 모델 조합 테스트
python tests/demo_model_testing.py
```

#### 출력 파일 명명 규칙
```
output/
├── phase1_GCM_BAL_ROOM_SAMPLE_20250808_120000.json
│   └── G=GPT-4, C=Claude, M=Gemini, BAL=Balanced, ROOM=방별처리
├── phase1_G_STR_BATCH_081132_20250808_121500.json  
│   └── G=GPT-4만, STR=Strict, BATCH=일괄처리
└── comparison_reports/
    ├── report_20250808.html    # HTML 시각화 리포트
    └── report_20250808.xlsx    # Excel 분석 리포트
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
