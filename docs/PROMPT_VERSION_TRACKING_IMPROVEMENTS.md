# Prompt Version Tracking Improvements

## Overview
프롬프트 버전 관리 시스템이 전체 파이프라인에 걸쳐 적절하게 추적되도록 코드를 개선했습니다.

## 개선 사항

### 1. Data Models 업데이트
**파일**: `src/models/data_models.py`

- `ModelResponse` 클래스에 `prompt_version` 필드 추가
- `MergeMetadata` 클래스에 `prompt_version` 필드 추가

```python
class ModelResponse(BaseModel):
    # ... existing fields ...
    prompt_version: Optional[str] = None  # 프롬프트 버전 추가

class MergeMetadata(BaseModel):
    # ... existing fields ...
    prompt_version: Optional[str] = None  # 프롬프트 버전 추가
```

### 2. Model Interface 개선
**파일**: `src/models/model_interface.py`

- `ModelOrchestrator.run_single_model()`에 `prompt_version` 매개변수 추가
- `ModelOrchestrator.run_parallel()`에 `prompt_version` 매개변수 추가
- 모델 응답에 프롬프트 버전 자동 할당

```python
async def run_single_model(self, model_name: str, prompt: str, json_data: Dict[str, Any], 
                          prompt_version: Optional[str] = None) -> Optional[ModelResponse]:
    # ... existing code ...
    if result and prompt_version:
        result.prompt_version = prompt_version
    return result

async def run_parallel(self, prompt: str, json_data: Dict[str, Any], 
                      model_names: List[str] = None,
                      enable_validation: Optional[bool] = None,
                      min_quality_threshold: float = 30.0,
                      prompt_version: Optional[str] = None) -> List[ModelResponse]:
    # ... existing code ...
```

### 3. Phase Processors 업데이트

#### Phase 1 Processor
**파일**: `src/phases/phase1_processor.py`

- 결과에 `prompt_version` 필드 추가
- orchestrator 호출 시 `prompt_version` 전달

```python
result = {
    'phase': 1,
    'phase_name': 'Merge Measurement & Work Scope',
    # ... other fields ...
    'prompt_version': effective_version or 'default',  # 프롬프트 버전 추가
    # ... rest of fields ...
}
```

#### Phase 2 Processor
**파일**: `src/phases/phase2_processor.py`

- `process()` 메소드에 `prompt_version` 매개변수 추가
- 프롬프트 버전에 따른 프롬프트 로드 로직 구현
- 결과에 `prompt_version` 필드 추가

```python
async def process(self,
                 phase1_output: Dict[str, Any],
                 models_to_use: List[str] = None,
                 project_id: Optional[str] = None,
                 prompt_version: Optional[str] = None) -> Dict[str, Any]:
    # ... implementation ...
```

### 4. Result Merger 개선
**파일**: `src/processors/result_merger.py`

- `_create_merged_estimate()` 메소드에서 prompt_version 추출 및 메타데이터에 포함

```python
# 프롬프트 버전 추출 (모든 모델이 동일한 버전 사용)
prompt_version = None
for response in model_responses:
    if hasattr(response, 'prompt_version') and response.prompt_version:
        prompt_version = response.prompt_version
        break

# 메타데이터에 포함
metadata = MergeMetadata(
    # ... other fields ...
    prompt_version=prompt_version,  # 프롬프트 버전 추가
)
```

### 5. 테스트 스크립트 작성
**파일**: `tests/test_prompt_version_tracking.py`

프롬프트 버전 추적을 검증하는 테스트 스크립트를 작성했습니다.

## 사용 방법

### Phase 1에서 프롬프트 버전 지정
```python
phase1 = Phase1Processor()
result = await phase1.process(
    input_data=data,
    models_to_use=['gpt4', 'claude', 'gemini'],
    prompt_version='improved'  # 또는 'fast', None(기본)
)
```

### Phase 2에서 프롬프트 버전 지정
```python
phase2 = Phase2Processor()
result = await phase2.process(
    phase1_output=phase1_result,
    models_to_use=['gpt4', 'claude', 'gemini'],
    prompt_version='improved'
)
```

### 결과에서 프롬프트 버전 확인
```python
# Phase 결과에서 직접 확인
prompt_version = result.get('prompt_version')
print(f"사용된 프롬프트 버전: {prompt_version}")

# 병합된 데이터의 메타데이터에서 확인
if 'data' in result and 'metadata' in result['data']:
    metadata_version = result['data']['metadata'].get('prompt_version')
    print(f"메타데이터의 프롬프트 버전: {metadata_version}")
```

## 테스트 실행

```bash
# 프롬프트 버전 추적 테스트 실행
python tests/test_prompt_version_tracking.py
```

## 장점

1. **추적성**: 각 추정 결과가 어떤 프롬프트 버전으로 생성되었는지 명확히 알 수 있음
2. **일관성**: 모든 모델이 동일한 프롬프트 버전을 사용하도록 보장
3. **디버깅**: 문제 발생 시 프롬프트 버전별로 원인 파악 가능
4. **A/B 테스팅**: 프롬프트 버전별 성능 비교 가능

## 향후 개선 사항

1. **Phase 0 지원**: Phase 0 프로세서에도 prompt_version 지원 추가 고려
2. **버전 히스토리**: 프롬프트 버전별 성능 메트릭 추적 시스템 구축
3. **자동 버전 선택**: 컨텍스트에 따른 최적 프롬프트 버전 자동 선택 기능
4. **UI 통합**: 웹 인터페이스에서 프롬프트 버전 선택 옵션 추가