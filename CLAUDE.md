# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Multi-model AI system for residential reconstruction estimates** that integrates GPT-4, Claude, and Gemini to generate consensus-based construction estimates. The system processes JSON data containing room specifications, materials, and work scopes to produce detailed reconstruction estimates with confidence scoring.

## Project Development Status

### Phase Implementation Status
- **Phase 1: Merge Measurement & Work Scope** ✅ **완료**
- **Phase 2: Quantity Survey (멀티모델 적용)** ✅ **완료**
- **Phase 3: Market Research** ⏳ **다음 구현 예정**
- **Phase 4: Timeline & Disposal Calculation** ⏳ **다음 구현 예정**
- **Phase 5: Final Estimate Completion** ⏳ **다음 구현 예정**
- **Phase 6: Formatting to JSON** ⏳ **다음 구현 예정**

### Current System Performance
- **Processing Time**: 2-5 seconds (3 models parallel execution)
- **Reliability**: 85%+ average confidence (test data based)
- **Scalability**: Easy model addition via interface structure
- **Stability**: Comprehensive error handling and fallback mechanisms

## Common Development Tasks

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup project structure and check dependencies
python setup.py

# Copy environment template and configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

### Running the Application
```bash
# Start development server
python -m uvicorn src.main:app --reload

# Alternative using setup script
python setup.py run

# Quick start (ready to use)
python run_server.py

# Access web interface at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Testing
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_model_interface.py

# Basic setup validation
python setup.py  # Check project configuration
```

### Code Quality
```bash
# Format code (mentioned in README)
black src/ tests/

# Lint code (flake8 is in requirements.txt)
flake8 src/ tests/

# Update requirements file
pipreqs . --force --encoding utf8
```

## Architecture Overview

### Core Components

**Multi-Model AI Pipeline**: The system follows a 4-stage pipeline:
1. **Model Orchestration** (`src/models/model_interface.py`) - Parallel AI model execution
   - GPT-4, Claude-3-Sonnet, Gemini-Pro 동시 호출
   - AsyncIO 기반 병렬 처리로 효율성 극대화
   - API 키 자동 검증 및 에러 핸들링
   - 표준화된 응답 데이터 추출 (JSON/텍스트 모두 지원)

2. **Result Merging** (`src/processors/result_merger.py`) - Consensus-based result combination
   - **QualitativeMerger**: 작업 항목 수집, 유사 작업 그룹핑, 2/3 합의 규칙 적용
   - **QuantitativeMerger**: IQR 아웃라이어 제거, 모델별 가중평균 (GPT-4: 35%, Claude: 35%, Gemini: 30%), 동적 안전마진 적용

3. **Validation** (`src/validators/estimation_validator.py`) - Business logic validation
   - **RemoveReplaceValidator**: Remove & Replace 로직 검증, demo_scope 중복 처리 방지
   - **MeasurementValidator**: 면적 vs 작업량 일관성, 높은 천장 할증 적용 정확성
   - **ComprehensiveValidator**: 종합 검증, 신뢰도 레벨 자동 계산

4. **Web Interface** (`src/main.py`) - FastAPI REST API and web frontend
   - JSON 파일 드래그앤드롭 업로드
   - AI 모델 선택 (GPT-4/Claude/Gemini)
   - 실시간 처리 상태 표시
   - 병합 결과 및 신뢰도 점수 시각화

### Key Design Patterns

**Strategy Pattern**: Each AI model (GPT-4, Claude, Gemini) implements `AIModelInterface` with specific API handling.

**Merger Pattern**: Two-phase merging system:
- `QualitativeMerger` - Consensus-based task identification and grouping
- `QuantitativeMerger` - Statistical merging of quantities with weighted averaging

**Validation Chain**: Comprehensive validation system using specialized validators:
- `RemoveReplaceValidator` - Validates "Remove & Replace" business logic
- `MeasurementValidator` - Ensures measurement data is properly used
- `ComprehensiveValidator` - Orchestrates all validation checks

### Data Flow Architecture

```
JSON Input → ProjectData (Pydantic) → ModelOrchestrator → Parallel AI Calls
    ↓
MergedEstimate ← ResultMerger ← [ModelResponse, ModelResponse, ModelResponse]
    ↓
ValidationResult ← ComprehensiveValidator ← ProjectData + MergedEstimate
    ↓
JSON Response (via FastAPI)
```

### Critical Business Logic

**"Remove & Replace" Strategy**: Core reconstruction logic where:
- `demo_scope` quantities represent already completed demolition
- Remaining materials require explicit removal tasks
- New installation required for entire calculated area
- High ceiling premium (>9 feet) applies to wall/ceiling work

**Consensus Rules**: 
- Minimum 2 models must agree for task inclusion (2/3 합의 규칙)
- Safety-critical tasks included with single model support (안전 관련은 1/3도 포함)
- Statistical outlier detection using IQR method
- Weighted averaging: GPT-4 (35%) + Claude (35%) + Gemini (30%)

**핵심 알고리즘 구현**:
```python
# 질적 병합 (작업 범위)
1. 3개 모델 결과 수집 → 편차 확인
2. 유사 작업 그룹핑 (텍스트 유사도 기반)
3. 합의 규칙 적용 (2/3 합의 → 채택, 안전 관련 1/3도 고려)
4. 아웃라이어 작업 플래그

# 정량적 병합 (수량)  
1. IQR 방식 아웃라이어 제거 (3개 샘플에 최적)
2. 가중평균 = (GPT×0.35 + Claude×0.35 + Gemini×0.30)
3. 분산도에 따른 동적 안전마진 (낮은분산 5%, 높은분산 10%)
4. 최종값 = 병합값 × (1 + 안전마진)
```

**검증 체크리스트 (프롬프트 요구사항 구현)**:
- ✅ Remove & Replace 로직 정확 적용
- ✅ 제공된 측정값 정확 사용  
- ✅ 특수 작업 및 할증 누락 없음
- ✅ demo_scope 중복 계산 방지
- ✅ 데이터 일관성 검증

## Configuration System

**Settings**: `config/settings.yaml` contains model weights, deviation thresholds, and consensus rules.

**Environment Variables**: API keys stored in `.env` file (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY).

**Configuration Loading**: `ConfigLoader` class handles YAML + environment variable combination with fallback defaults.

## API Endpoints

**Main Endpoint**: `POST /api/estimate/merge` - Primary estimation API
**Health Check**: `GET /api/health` - System status and model availability  
**Model Status**: `GET /api/models/status` - Individual model API key validation
**File Upload**: `POST /api/upload-json` - JSON file processing endpoint

## Data Models

**Input**: JSON array format with jobsite info + floor data containing rooms with materials, work_scope, measurements, demo_scope, and additional_notes.

**Output**: `MergedEstimate` with rooms, confidence scores, metadata including consensus levels, processing time, and validation results.

## Testing Strategy

**Test Structure**: Comprehensive test suite with mocking for AI model calls:
- **Unit tests**: 각 모듈별 기능 검증 (model interfaces, mergers, validators)
- **Integration tests**: 전체 파이프라인 테스트 with mocked responses
- **Mocking tests**: API 호출 없이 로직 검증
- **Edge cases**: 빈 데이터, 오류 상황 처리
- Fixture-based test data using realistic reconstruction scenarios

**Key Test Files**:
- `test_model_interface.py` - Model orchestration and API interface tests
- Validation tests embedded in same file for RemoveReplaceValidator, MeasurementValidator
- Integration tests for full pipeline flow

## Technology Stack

**Backend**: FastAPI, Pydantic, AsyncIO
**AI APIs**: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini)  
**Data Processing**: NumPy, Pandas, SciPy
**Testing**: Pytest, AsyncIO
**Frontend**: HTML/CSS/JavaScript (Vanilla)
**Configuration**: YAML, python-dotenv

## Future Development (Phase 3-6)

### Phase 3: Market Research (우선순위 1)
```python
# 시장 가격 조사 모듈 구현 필요
- DMV 지역 재료비/인건비 데이터 수집
- 멀티 모델로 가격 정보 교차 검증
- 인플레이션, 계절성 등 동적 요소 반영
- 가격 데이터베이스 구축 또는 실시간 API 연동
```

### Phase 4: Timeline & Disposal
```python
- 작업 순서 의존성 계산
- 공기 산정 (멀티 모델 적용)
- 폐기물 처리비용 산정
```

### Phase 5-6: 최종 견적 완성 및 JSON 포맷팅
```python
- 모든 단계 결과 통합
- 최종 견적서 생성
- 클라이언트 요구 형식으로 출력
```

## Development Notes

**Async Architecture**: Uses `asyncio` for parallel AI model execution with proper exception handling and timeout management.

**Error Handling**: Graceful degradation - system continues with available models if some fail, includes detailed error reporting in API responses.

**Bilingual Support**: Korean comments and messages throughout codebase, indicating Korean development team.

**Performance Considerations**: Statistical processing using numpy/scipy, caching for repeated operations, parallel model execution for speed optimization.

**Ready to Use**: 즉시 실행 가능 - API 키 설정 후 바로 실행 가능 (`python run_server.py`)

**Next Meeting Agenda**: Phase 3 Market Research 모듈 설계 및 구현 방향 논의