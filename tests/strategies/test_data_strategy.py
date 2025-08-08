"""
AI Estimation Pipeline - 테스트 데이터 관리 전략 구현
"""
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from ..phases.base import PhaseTestResult, PhaseTestConfig


class TestDataMode(Enum):
    """테스트 데이터 모드"""
    DETERMINISTIC = "deterministic"  # 고정 데이터
    LIVE = "live"                   # 실시간 데이터
    HYBRID = "hybrid"               # 혼합 모드


@dataclass
class GoldenDataset:
    """골든 데이터셋 정의"""
    name: str
    description: str
    phase_inputs: Dict[int, Any]  # Phase별 입력 데이터
    expected_outputs: Dict[int, Any]  # 기대 출력 데이터
    metadata: Dict[str, Any]
    created_at: datetime
    data_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldenDataset':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class TestDataManager:
    """테스트 데이터 관리자"""
    
    def __init__(self, data_dir: str = "tests/fixtures/golden_datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
    
    # ============ 골든 데이터셋 관리 ============
    
    def create_golden_dataset(self, 
                            name: str,
                            description: str,
                            live_results: List[PhaseTestResult],
                            metadata: Dict[str, Any] = None) -> GoldenDataset:
        """실시간 결과로부터 골든 데이터셋 생성"""
        phase_inputs = {}
        expected_outputs = {}
        
        # Phase 결과들을 입력/출력으로 변환
        for i, result in enumerate(live_results):
            phase_num = result.phase_number
            expected_outputs[phase_num] = result.output_data
            
            # 다음 Phase의 입력으로 사용
            if i + 1 < len(live_results):
                next_phase_num = live_results[i + 1].phase_number
                phase_inputs[next_phase_num] = result.output_data
        
        # 데이터 해시 생성 (일관성 검증용)
        data_content = json.dumps({
            'inputs': phase_inputs,
            'outputs': expected_outputs
        }, sort_keys=True)
        data_hash = hashlib.sha256(data_content.encode()).hexdigest()[:16]
        
        golden_dataset = GoldenDataset(
            name=name,
            description=description,
            phase_inputs=phase_inputs,
            expected_outputs=expected_outputs,
            metadata=metadata or {},
            created_at=datetime.now(),
            data_hash=data_hash
        )
        
        # 저장
        self.save_golden_dataset(golden_dataset)
        return golden_dataset
    
    def save_golden_dataset(self, dataset: GoldenDataset):
        """골든 데이터셋 저장"""
        filename = f"{dataset.name}_{dataset.data_hash}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, indent=2, default=str, ensure_ascii=False)
    
    def load_golden_dataset(self, name: str) -> Optional[GoldenDataset]:
        """골든 데이터셋 로드"""
        if name in self.cache:
            return self.cache[name]
        
        # 파일 패턴으로 찾기 (해시 포함)
        pattern = f"{name}_*.json"
        matching_files = list(self.data_dir.glob(pattern))
        
        if not matching_files:
            return None
        
        # 최신 파일 사용
        latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dataset = GoldenDataset.from_dict(data)
        self.cache[name] = dataset
        return dataset
    
    def list_golden_datasets(self) -> List[str]:
        """사용 가능한 골든 데이터셋 목록"""
        datasets = []
        for file in self.data_dir.glob("*.json"):
            name = file.stem.split('_')[0]  # Remove hash
            if name not in datasets:
                datasets.append(name)
        return datasets
    
    # ============ 테스트 실행 전략 ============
    
    async def run_deterministic_test(self, 
                                   phase_number: int,
                                   golden_dataset_name: str,
                                   test_config: PhaseTestConfig) -> Tuple[PhaseTestResult, Dict[str, Any]]:
        """결정적 테스트 실행 (고정 데이터 사용)"""
        dataset = self.load_golden_dataset(golden_dataset_name)
        if not dataset:
            raise ValueError(f"Golden dataset not found: {golden_dataset_name}")
        
        # Phase 입력 데이터 준비
        if phase_number in dataset.phase_inputs:
            input_data = dataset.phase_inputs[phase_number]
        else:
            raise ValueError(f"No input data for Phase {phase_number} in dataset {golden_dataset_name}")
        
        # 기대 출력
        expected_output = dataset.expected_outputs.get(phase_number)
        
        # 실제 Phase 실행 (모킹 또는 실제 실행)
        from ..phases.individual import get_phase_test_class
        phase_test = get_phase_test_class(phase_number)()
        phase_test.set_input_data(input_data)
        
        result = await phase_test.run_test(test_config)
        
        # 결과 비교 분석
        comparison = self._compare_with_expected(result, expected_output) if expected_output else {}
        
        return result, comparison
    
    async def run_live_integration_test(self, 
                                      phases: List[int],
                                      test_config: PhaseTestConfig,
                                      save_as_golden: str = None) -> List[PhaseTestResult]:
        """실시간 통합 테스트 실행"""
        from ..phases.orchestrator import PhaseTestOrchestrator
        from ..phases.integration.test_phase_pipeline import PipelineTestConfig
        
        # Pipeline 설정
        pipeline_config = PipelineTestConfig(
            phases=phases,
            models=test_config.models,
            test_name=f"live_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            continue_on_failure=False,
            save_intermediate=True
        )
        
        # Orchestrator로 실행
        orchestrator = PhaseTestOrchestrator("test_outputs/live_integration")
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        
        # 골든 데이터셋으로 저장 (선택적)
        if save_as_golden and session.overall_success:
            self.create_golden_dataset(
                name=save_as_golden,
                description=f"Live integration test results for phases {phases}",
                live_results=session.phase_results,
                metadata={
                    'models_used': test_config.models,
                    'validation_mode': test_config.validation_mode,
                    'session_id': session.session_id
                }
            )
        
        return session.phase_results
    
    async def run_hybrid_test(self,
                            phases: List[int], 
                            test_config: PhaseTestConfig,
                            use_golden_for: List[int] = None) -> Dict[str, Any]:
        """하이브리드 테스트 - 일부는 골든데이터, 일부는 실시간"""
        use_golden_for = use_golden_for or []
        results = {}
        
        for phase in phases:
            if phase in use_golden_for:
                # 결정적 테스트
                golden_name = f"phase{phase}_baseline"
                result, comparison = await self.run_deterministic_test(
                    phase, golden_name, test_config
                )
                results[f'phase{phase}'] = {
                    'mode': 'deterministic',
                    'result': result,
                    'comparison': comparison
                }
            else:
                # 실시간 테스트
                live_results = await self.run_live_integration_test([phase], test_config)
                results[f'phase{phase}'] = {
                    'mode': 'live',
                    'result': live_results[0] if live_results else None
                }
        
        return results
    
    # ============ 결과 분석 ============
    
    def _compare_with_expected(self, 
                             actual_result: PhaseTestResult,
                             expected_output: Dict[str, Any]) -> Dict[str, Any]:
        """실제 결과와 기대 결과 비교"""
        comparison = {
            'confidence_score_diff': None,
            'structure_match': False,
            'key_differences': [],
            'similarity_score': 0.0
        }
        
        if not expected_output or not actual_result.output_data:
            return comparison
        
        actual_data = actual_result.output_data
        
        # 신뢰도 점수 비교
        expected_confidence = expected_output.get('confidence_score', 0)
        actual_confidence = actual_result.confidence_score
        comparison['confidence_score_diff'] = actual_confidence - expected_confidence
        
        # 구조 일치 검사
        comparison['structure_match'] = self._check_structure_match(
            actual_data, expected_output
        )
        
        # 핵심 차이점 식별
        comparison['key_differences'] = self._find_key_differences(
            actual_data, expected_output
        )
        
        # 유사도 점수 계산
        comparison['similarity_score'] = self._calculate_similarity(
            actual_data, expected_output
        )
        
        return comparison
    
    def _check_structure_match(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """구조 일치 확인"""
        def get_structure(data):
            if isinstance(data, dict):
                return {k: get_structure(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [get_structure(item) for item in data[:1]]  # 첫 항목만
            else:
                return type(data).__name__
        
        return get_structure(actual) == get_structure(expected)
    
    def _find_key_differences(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
        """주요 차이점 찾기"""
        differences = []
        
        def compare_recursive(a, e, path=""):
            if isinstance(a, dict) and isinstance(e, dict):
                for key in set(a.keys()) | set(e.keys()):
                    new_path = f"{path}.{key}" if path else key
                    if key not in a:
                        differences.append(f"Missing key: {new_path}")
                    elif key not in e:
                        differences.append(f"Extra key: {new_path}")
                    else:
                        compare_recursive(a[key], e[key], new_path)
            elif a != e:
                differences.append(f"Value difference at {path}: {a} != {e}")
        
        compare_recursive(actual, expected)
        return differences[:10]  # 처음 10개만
    
    def _calculate_similarity(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """유사도 점수 계산 (0-1)"""
        # 간단한 구조 기반 유사도
        def flatten_dict(d, prefix=""):
            items = []
            for k, v in d.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key))
                else:
                    items.append((new_key, str(v)))
            return set(items)
        
        try:
            actual_items = flatten_dict(actual)
            expected_items = flatten_dict(expected)
            
            intersection = len(actual_items & expected_items)
            union = len(actual_items | expected_items)
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    # ============ 성능 메트릭 ============
    
    def generate_performance_report(self, 
                                  results: List[PhaseTestResult]) -> Dict[str, Any]:
        """성능 리포트 생성"""
        return {
            'execution_times': [r.execution_time for r in results],
            'avg_execution_time': sum(r.execution_time for r in results) / len(results),
            'confidence_scores': [r.confidence_score for r in results],
            'avg_confidence': sum(r.confidence_score for r in results) / len(results),
            'success_rate': sum(1 for r in results if r.success) / len(results),
            'model_success_rates': [r.model_success_rate for r in results],
            'consensus_levels': [r.consensus_level for r in results]
        }