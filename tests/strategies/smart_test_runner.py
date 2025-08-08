"""
Smart Test Runner - 지능형 테스트 실행 엔진
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..phases.base import PhaseTestConfig, PhaseTestResult, TestSession
from .test_data_strategy import TestDataManager, TestDataMode


class TestPriority(Enum):
    """테스트 우선순위"""
    CRITICAL = "critical"      # CI/CD 필수 테스트
    HIGH = "high"             # 중요 기능 테스트  
    MEDIUM = "medium"         # 일반 회귀 테스트
    LOW = "low"               # 성능/품질 테스트


class TestType(Enum):
    """테스트 유형"""
    SMOKE = "smoke"           # 기본 동작 검증
    REGRESSION = "regression" # 회귀 방지
    INTEGRATION = "integration" # 통합 테스트
    PERFORMANCE = "performance" # 성능 테스트
    EXPLORATORY = "exploratory" # 탐색적 테스트


@dataclass
class SmartTestConfig:
    """지능형 테스트 설정"""
    name: str
    phases: List[int]
    test_mode: TestDataMode
    test_type: TestType
    priority: TestPriority
    models: List[str]
    timeout_seconds: int = 300
    retry_attempts: int = 2
    
    # 성능 기준
    max_execution_time: float = 60.0
    min_confidence_score: float = 0.7
    min_consensus_level: float = 0.6
    
    # 골든 데이터 설정
    golden_dataset_name: Optional[str] = None
    save_as_golden: Optional[str] = None
    
    # 고급 설정
    parallel_execution: bool = False
    failure_tolerance: float = 0.1  # 10% 실패 허용
    performance_baseline: Optional[Dict[str, float]] = None


class SmartTestRunner:
    """지능형 테스트 실행기"""
    
    def __init__(self, output_dir: str = "test_outputs/smart_runner"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_manager = TestDataManager()
        
        # 성능 히스토리 (간단한 메모리 캐시)
        self.performance_history = {}
    
    # ============ 테스트 실행 전략 ============
    
    async def run_smart_test(self, config: SmartTestConfig) -> Dict[str, Any]:
        """지능형 테스트 실행"""
        start_time = time.time()
        test_session_id = f"smart_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 테스트 전 검증
            self._validate_test_config(config)
            
            # 실행 전략 결정
            execution_strategy = self._determine_execution_strategy(config)
            
            # 테스트 실행
            results = await self._execute_with_strategy(config, execution_strategy)
            
            # 결과 분석
            analysis = self._analyze_test_results(config, results)
            
            # 성능 히스토리 업데이트
            self._update_performance_history(config.name, analysis)
            
            execution_time = time.time() - start_time
            
            return {
                'session_id': test_session_id,
                'config': config.__dict__,
                'execution_strategy': execution_strategy,
                'results': results,
                'analysis': analysis,
                'execution_time': execution_time,
                'success': analysis['overall_success']
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'session_id': test_session_id,
                'config': config.__dict__,
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def _validate_test_config(self, config: SmartTestConfig):
        """테스트 설정 검증"""
        if not config.phases:
            raise ValueError("At least one phase must be specified")
        
        if not config.models:
            raise ValueError("At least one model must be specified")
        
        if config.test_mode == TestDataMode.DETERMINISTIC:
            if not config.golden_dataset_name:
                raise ValueError("Golden dataset name required for deterministic mode")
        
        # 골든 데이터셋 존재 확인
        if config.golden_dataset_name:
            dataset = self.data_manager.load_golden_dataset(config.golden_dataset_name)
            if not dataset:
                raise ValueError(f"Golden dataset not found: {config.golden_dataset_name}")
    
    def _determine_execution_strategy(self, config: SmartTestConfig) -> Dict[str, Any]:
        """최적 실행 전략 결정"""
        strategy = {
            'mode': config.test_mode.value,
            'parallel': config.parallel_execution,
            'fast_fail': config.priority == TestPriority.CRITICAL,
            'extensive_validation': config.test_type == TestType.REGRESSION,
            'performance_monitoring': config.test_type == TestType.PERFORMANCE
        }
        
        # 우선순위 기반 최적화
        if config.priority == TestPriority.CRITICAL:
            strategy.update({
                'models': config.models[:1],  # 단일 모델로 빠른 실행
                'timeout_reduction': 0.5,     # 타임아웃 50% 단축
                'skip_detailed_analysis': True
            })
        elif config.priority == TestPriority.LOW:
            strategy.update({
                'comprehensive_analysis': True,
                'performance_profiling': True,
                'save_detailed_logs': True
            })
        
        # 테스트 타입 기반 조정
        if config.test_type == TestType.PERFORMANCE:
            strategy.update({
                'warmup_runs': 2,
                'measurement_runs': 5,
                'resource_monitoring': True
            })
        
        return strategy
    
    async def _execute_with_strategy(self, 
                                   config: SmartTestConfig,
                                   strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """전략 기반 테스트 실행"""
        
        if config.test_mode == TestDataMode.DETERMINISTIC:
            return await self._run_deterministic_tests(config, strategy)
        elif config.test_mode == TestDataMode.LIVE:
            return await self._run_live_tests(config, strategy)
        elif config.test_mode == TestDataMode.HYBRID:
            return await self._run_hybrid_tests(config, strategy)
        else:
            raise ValueError(f"Unsupported test mode: {config.test_mode}")
    
    async def _run_deterministic_tests(self,
                                     config: SmartTestConfig,
                                     strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """결정적 테스트 실행"""
        results = []
        
        test_config = PhaseTestConfig(
            phase_numbers=config.phases,
            models=strategy.get('models', config.models),
            timeout_seconds=int(config.timeout_seconds * strategy.get('timeout_reduction', 1.0)),
            validation_mode="strict" if strategy.get('extensive_validation') else "balanced"
        )
        
        for phase in config.phases:
            try:
                result, comparison = await self.data_manager.run_deterministic_test(
                    phase, config.golden_dataset_name, test_config
                )
                
                results.append({
                    'phase': phase,
                    'mode': 'deterministic',
                    'result': result.to_dict(),
                    'comparison': comparison,
                    'success': result.success
                })
                
                # 빠른 실패 (Critical 테스트용)
                if strategy.get('fast_fail') and not result.success:
                    break
                    
            except Exception as e:
                results.append({
                    'phase': phase,
                    'mode': 'deterministic',
                    'error': str(e),
                    'success': False
                })
                
                if strategy.get('fast_fail'):
                    break
        
        return results
    
    async def _run_live_tests(self,
                            config: SmartTestConfig,
                            strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """실시간 테스트 실행"""
        test_config = PhaseTestConfig(
            phase_numbers=config.phases,
            models=strategy.get('models', config.models),
            timeout_seconds=config.timeout_seconds,
            save_outputs=True
        )
        
        # 성능 모니터링이 필요한 경우 여러 번 실행
        runs = strategy.get('measurement_runs', 1)
        if strategy.get('warmup_runs', 0) > 0:
            # 워밍업 실행 (결과 무시)
            await self.data_manager.run_live_integration_test(
                config.phases, test_config
            )
        
        all_results = []
        for run in range(runs):
            live_results = await self.data_manager.run_live_integration_test(
                config.phases, 
                test_config,
                save_as_golden=config.save_as_golden if run == 0 else None
            )
            
            all_results.append({
                'run': run,
                'mode': 'live',
                'results': [r.to_dict() for r in live_results],
                'success': all(r.success for r in live_results)
            })
        
        return all_results
    
    async def _run_hybrid_tests(self,
                              config: SmartTestConfig,
                              strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """하이브리드 테스트 실행"""
        # 초기 Phase들은 골든 데이터, 후반부는 실시간
        mid_point = len(config.phases) // 2
        use_golden_for = config.phases[:mid_point]
        
        test_config = PhaseTestConfig(
            phase_numbers=config.phases,
            models=config.models,
            timeout_seconds=config.timeout_seconds
        )
        
        hybrid_results = await self.data_manager.run_hybrid_test(
            config.phases, test_config, use_golden_for
        )
        
        return [
            {
                'mode': 'hybrid',
                'phase_results': hybrid_results,
                'success': all(
                    result.get('result', {}).get('success', False)
                    for result in hybrid_results.values()
                )
            }
        ]
    
    # ============ 결과 분석 ============
    
    def _analyze_test_results(self,
                            config: SmartTestConfig,
                            results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """테스트 결과 종합 분석"""
        analysis = {
            'overall_success': True,
            'phase_success_rate': {},
            'performance_metrics': {},
            'quality_metrics': {},
            'recommendations': [],
            'trend_analysis': {}
        }
        
        # 성공률 분석
        total_phases = len(config.phases)
        successful_phases = 0
        
        for result in results:
            if result.get('success', False):
                successful_phases += 1
        
        analysis['overall_success'] = (successful_phases / total_phases) >= (1 - config.failure_tolerance)
        analysis['phase_success_rate']['total'] = successful_phases / total_phases
        
        # 성능 메트릭 추출
        execution_times = []
        confidence_scores = []
        
        for result in results:
            if 'result' in result:
                result_data = result['result']
                if isinstance(result_data, dict):
                    execution_times.append(result_data.get('execution_time', 0))
                    confidence_scores.append(result_data.get('confidence_score', 0))
        
        if execution_times:
            analysis['performance_metrics'] = {
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'max_execution_time': max(execution_times),
                'performance_within_budget': all(t <= config.max_execution_time for t in execution_times)
            }
        
        if confidence_scores:
            analysis['quality_metrics'] = {
                'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                'min_confidence': min(confidence_scores),
                'confidence_within_budget': all(c >= config.min_confidence_score for c in confidence_scores)
            }
        
        # 권장사항 생성
        if analysis['phase_success_rate']['total'] < 0.9:
            analysis['recommendations'].append("Review phase implementations - success rate below 90%")
        
        if analysis['performance_metrics'].get('avg_execution_time', 0) > config.max_execution_time:
            analysis['recommendations'].append("Consider performance optimization - execution time exceeds budget")
        
        if analysis['quality_metrics'].get('avg_confidence', 0) < config.min_confidence_score:
            analysis['recommendations'].append("Review model performance - confidence scores below threshold")
        
        # 트렌드 분석 (히스토리 대비)
        if config.name in self.performance_history:
            previous = self.performance_history[config.name]
            current_avg_time = analysis['performance_metrics'].get('avg_execution_time', 0)
            previous_avg_time = previous.get('performance_metrics', {}).get('avg_execution_time', 0)
            
            if previous_avg_time > 0:
                time_change = (current_avg_time - previous_avg_time) / previous_avg_time
                analysis['trend_analysis']['execution_time_change'] = time_change
                
                if time_change > 0.1:  # 10% 증가
                    analysis['recommendations'].append("Performance regression detected - execution time increased")
        
        return analysis
    
    def _update_performance_history(self, test_name: str, analysis: Dict[str, Any]):
        """성능 히스토리 업데이트"""
        self.performance_history[test_name] = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': analysis.get('performance_metrics', {}),
            'quality_metrics': analysis.get('quality_metrics', {}),
            'success_rate': analysis.get('phase_success_rate', {}).get('total', 0)
        }
    
    # ============ 테스트 스위트 실행 ============
    
    async def run_test_suite(self, configs: List[SmartTestConfig]) -> Dict[str, Any]:
        """테스트 스위트 실행"""
        suite_start = time.time()
        suite_results = []
        
        # 우선순위별 정렬
        sorted_configs = sorted(configs, key=lambda c: c.priority.value)
        
        for config in sorted_configs:
            result = await self.run_smart_test(config)
            suite_results.append(result)
            
            # Critical 테스트 실패시 중단
            if config.priority == TestPriority.CRITICAL and not result['success']:
                break
        
        suite_time = time.time() - suite_start
        
        return {
            'suite_execution_time': suite_time,
            'total_tests': len(configs),
            'executed_tests': len(suite_results),
            'overall_success': all(r['success'] for r in suite_results),
            'results': suite_results
        }
    
    # ============ 사전 정의된 테스트 스위트 ============
    
    def get_ci_cd_test_suite(self) -> List[SmartTestConfig]:
        """CI/CD용 빠른 테스트 스위트"""
        return [
            SmartTestConfig(
                name="smoke_test",
                phases=[1],
                test_mode=TestDataMode.DETERMINISTIC,
                test_type=TestType.SMOKE,
                priority=TestPriority.CRITICAL,
                models=["gpt4"],
                golden_dataset_name="phase1_baseline",
                timeout_seconds=60,
                max_execution_time=30.0
            ),
            SmartTestConfig(
                name="integration_check",
                phases=[0, 1, 2],
                test_mode=TestDataMode.DETERMINISTIC,
                test_type=TestType.INTEGRATION,
                priority=TestPriority.HIGH,
                models=["gpt4"],
                golden_dataset_name="pipeline_baseline",
                timeout_seconds=180,
                max_execution_time=120.0,
                parallel_execution=True
            )
        ]
    
    def get_regression_test_suite(self) -> List[SmartTestConfig]:
        """회귀 테스트 스위트"""
        return [
            SmartTestConfig(
                name="full_regression",
                phases=[0, 1, 2],
                test_mode=TestDataMode.HYBRID,
                test_type=TestType.REGRESSION,
                priority=TestPriority.HIGH,
                models=["gpt4", "claude", "gemini"],
                timeout_seconds=600,
                max_execution_time=300.0,
                min_confidence_score=0.75
            )
        ]
    
    def get_performance_test_suite(self) -> List[SmartTestConfig]:
        """성능 테스트 스위트"""
        return [
            SmartTestConfig(
                name="performance_benchmark",
                phases=[1, 2],
                test_mode=TestDataMode.LIVE,
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.MEDIUM,
                models=["gpt4", "claude"],
                timeout_seconds=300,
                max_execution_time=60.0,
                performance_baseline={
                    'avg_execution_time': 45.0,
                    'avg_confidence': 0.8
                }
            )
        ]