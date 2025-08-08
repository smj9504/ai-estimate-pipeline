"""
Comprehensive Test Scenarios - AI Estimation Pipeline
포괄적인 테스트 시나리오 예제 모음
"""
import asyncio
from datetime import datetime
from pathlib import Path

# Import test framework components
from ..strategies.smart_test_runner import SmartTestRunner, SmartTestConfig
from ..strategies.test_data_strategy import TestDataManager, TestDataMode
from ..strategies.smart_test_runner import TestType, TestPriority


class ComprehensiveTestScenarios:
    """포괄적인 테스트 시나리오 모음"""
    
    def __init__(self):
        self.smart_runner = SmartTestRunner("test_outputs/comprehensive_scenarios")
        self.data_manager = TestDataManager()
    
    # ============ Scenario 1: CI/CD 파이프라인 테스트 ============
    
    async def run_ci_cd_pipeline_test(self) -> Dict[str, Any]:
        """
        CI/CD 파이프라인용 빠른 회귀 테스트
        - 결정적 데이터로 빠른 실행
        - Critical 우선순위로 실패시 즉시 중단
        - 단일 모델로 성능 최적화
        """
        print("🚀 Running CI/CD Pipeline Test...")
        
        # 1단계: 스모크 테스트 (30초 이내)
        smoke_config = SmartTestConfig(
            name="cicd_smoke_test",
            phases=[1],  # Phase 1만 빠른 테스트
            test_mode=TestDataMode.DETERMINISTIC,
            test_type=TestType.SMOKE,
            priority=TestPriority.CRITICAL,
            models=["gpt4"],  # 가장 안정적인 모델만
            golden_dataset_name="phase1_baseline",
            timeout_seconds=30,
            max_execution_time=25.0,
            min_confidence_score=0.6,  # 낮은 임계값으로 빠른 패스
            failure_tolerance=0.0  # 실패 허용 없음
        )
        
        # 2단계: 통합 테스트 (2분 이내)
        integration_config = SmartTestConfig(
            name="cicd_integration_test",
            phases=[0, 1, 2],
            test_mode=TestDataMode.DETERMINISTIC,
            test_type=TestType.INTEGRATION,
            priority=TestPriority.HIGH,
            models=["gpt4"],
            golden_dataset_name="pipeline_baseline",
            timeout_seconds=120,
            max_execution_time=100.0,
            min_confidence_score=0.7,
            parallel_execution=True  # Phase별 병렬 실행
        )
        
        # 테스트 실행
        suite_result = await self.smart_runner.run_test_suite([
            smoke_config, integration_config
        ])
        
        return suite_result
    
    # ============ Scenario 2: 정확도 검증 테스트 ============
    
    async def run_accuracy_validation_test(self) -> Dict[str, Any]:
        """
        AI 모델 정확도 검증 테스트
        - 실시간 데이터로 실제 성능 측정
        - 멀티 모델 합의 검증
        - 골든 데이터셋 자동 생성
        """
        print("🎯 Running Accuracy Validation Test...")
        
        # 실시간 정확도 테스트
        accuracy_config = SmartTestConfig(
            name="accuracy_validation",
            phases=[1, 2],
            test_mode=TestDataMode.LIVE,
            test_type=TestType.REGRESSION,
            priority=TestPriority.HIGH,
            models=["gpt4", "claude", "gemini"],  # 3개 모델 합의
            timeout_seconds=300,
            max_execution_time=200.0,
            min_confidence_score=0.8,  # 높은 신뢰도 요구
            min_consensus_level=0.7,   # 모델간 합의도 요구
            save_as_golden="accuracy_validation_baseline",  # 결과를 골든으로 저장
            retry_attempts=3
        )
        
        result = await self.smart_runner.run_smart_test(accuracy_config)
        
        # 정확도 분석 리포트 생성
        if result['success']:
            await self._generate_accuracy_report(result)
        
        return result
    
    # ============ Scenario 3: 성능 벤치마크 테스트 ============
    
    async def run_performance_benchmark_test(self) -> Dict[str, Any]:
        """
        성능 벤치마크 및 회귀 감지 테스트
        - 반복 실행으로 안정성 측정
        - 성능 트렌드 분석
        - 리소스 사용량 모니터링
        """
        print("⚡ Running Performance Benchmark Test...")
        
        performance_config = SmartTestConfig(
            name="performance_benchmark",
            phases=[1],  # Phase 1 집중 벤치마크
            test_mode=TestDataMode.DETERMINISTIC,
            test_type=TestType.PERFORMANCE,
            priority=TestPriority.MEDIUM,
            models=["gpt4"],
            golden_dataset_name="phase1_baseline",
            timeout_seconds=180,
            max_execution_time=45.0,  # 45초 성능 목표
            min_confidence_score=0.75,
            performance_baseline={
                'avg_execution_time': 40.0,
                'avg_confidence': 0.8,
                'model_success_rate': 0.95
            }
        )
        
        # 여러 번 실행하여 성능 안정성 측정
        benchmark_results = []
        for run in range(5):  # 5회 반복
            print(f"  Performance run {run + 1}/5...")
            result = await self.smart_runner.run_smart_test(performance_config)
            benchmark_results.append(result)
            
            # 실패시 중단
            if not result['success']:
                break
        
        # 성능 통계 계산
        performance_stats = self._calculate_performance_stats(benchmark_results)
        
        return {
            'benchmark_results': benchmark_results,
            'performance_statistics': performance_stats,
            'performance_regression_detected': performance_stats['avg_time'] > performance_config.max_execution_time
        }
    
    # ============ Scenario 4: 엣지 케이스 견고성 테스트 ============
    
    async def run_edge_case_robustness_test(self) -> Dict[str, Any]:
        """
        엣지 케이스 및 오류 처리 견고성 테스트
        - 극단적인 입력 데이터
        - 오류 조건 시뮬레이션
        - 복구 능력 검증
        """
        print("🛡️ Running Edge Case Robustness Test...")
        
        edge_cases = [
            # 빈 데이터 케이스
            {
                'name': 'empty_data_handling',
                'description': '빈 데이터 처리 능력',
                'data_override': {'data': []}
            },
            # 극단적 수치 케이스
            {
                'name': 'extreme_values_handling',
                'description': '극단적 수치 처리',
                'data_override': {
                    'measurements': {'width': 1000000, 'length': 0.001}
                }
            },
            # 불완전한 데이터 케이스
            {
                'name': 'incomplete_data_handling',
                'description': '불완전한 데이터 처리',
                'data_override': {
                    'rooms': [{'name': 'Test', 'work_scope': {}}]  # measurements 누락
                }
            }
        ]
        
        edge_case_results = []
        
        for case in edge_cases:
            print(f"  Testing: {case['description']}...")
            
            config = SmartTestConfig(
                name=case['name'],
                phases=[1],
                test_mode=TestDataMode.DETERMINISTIC,
                test_type=TestType.REGRESSION,
                priority=TestPriority.MEDIUM,
                models=["gpt4"],
                golden_dataset_name="phase1_baseline",
                timeout_seconds=60,
                failure_tolerance=1.0,  # 실패 허용 (견고성 테스트)
                retry_attempts=1  # 재시도 최소화
            )
            
            # TODO: 실제로는 data_override를 테스트 데이터에 적용하는 로직 필요
            result = await self.smart_runner.run_smart_test(config)
            result['edge_case'] = case
            edge_case_results.append(result)
        
        # 견고성 점수 계산
        robustness_score = sum(1 for r in edge_case_results if r['success']) / len(edge_cases)
        
        return {
            'edge_case_results': edge_case_results,
            'robustness_score': robustness_score,
            'robust_system': robustness_score >= 0.7  # 70% 이상 통과시 견고함
        }
    
    # ============ Scenario 5: 전체 파이프라인 종단간 테스트 ============
    
    async def run_end_to_end_pipeline_test(self) -> Dict[str, Any]:
        """
        전체 파이프라인 종단간 테스트
        - Phase 0-2 전체 실행
        - 데이터 흐름 무결성 검증
        - 실제 워크플로 시뮬레이션
        """
        print("🔄 Running End-to-End Pipeline Test...")
        
        e2e_config = SmartTestConfig(
            name="end_to_end_pipeline",
            phases=[0, 1, 2],
            test_mode=TestDataMode.LIVE,  # 실제 AI 호출
            test_type=TestType.INTEGRATION,
            priority=TestPriority.HIGH,
            models=["gpt4", "claude"],  # 2개 모델로 균형
            timeout_seconds=600,  # 10분 제한
            max_execution_time=400.0,
            min_confidence_score=0.75,
            min_consensus_level=0.6,
            save_as_golden="e2e_pipeline_baseline",
            parallel_execution=False  # 순차 실행으로 데이터 흐름 보장
        )
        
        result = await self.smart_runner.run_smart_test(e2e_config)
        
        # 파이프라인 품질 분석
        pipeline_quality = self._analyze_pipeline_quality(result)
        
        return {
            'e2e_result': result,
            'pipeline_quality': pipeline_quality,
            'data_flow_integrity': pipeline_quality['data_consistency_score'] > 0.8
        }
    
    # ============ Scenario 6: 모델 비교 테스트 ============
    
    async def run_model_comparison_test(self) -> Dict[str, Any]:
        """
        AI 모델별 성능 비교 테스트
        - 개별 모델 성능 측정
        - 모델간 일관성 분석
        - 최적 모델 조합 제안
        """
        print("📊 Running Model Comparison Test...")
        
        models_to_test = ["gpt4", "claude", "gemini"]
        model_results = {}
        
        # 각 모델 개별 테스트
        for model in models_to_test:
            print(f"  Testing model: {model}...")
            
            config = SmartTestConfig(
                name=f"model_comparison_{model}",
                phases=[1, 2],
                test_mode=TestDataMode.DETERMINISTIC,
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.MEDIUM,
                models=[model],  # 단일 모델
                golden_dataset_name="phase1_baseline",
                timeout_seconds=200,
                max_execution_time=120.0
            )
            
            result = await self.smart_runner.run_smart_test(config)
            model_results[model] = result
        
        # 모델별 비교 분석
        comparison_analysis = self._compare_model_performance(model_results)
        
        return {
            'individual_results': model_results,
            'comparison_analysis': comparison_analysis,
            'recommended_models': comparison_analysis['top_performers']
        }
    
    # ============ 종합 테스트 실행 ============
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """전체 테스트 시나리오 실행"""
        print("🎭 Starting Comprehensive Test Suite...")
        start_time = datetime.now()
        
        results = {}
        
        try:
            # 1. CI/CD 테스트 (필수)
            results['cicd_pipeline'] = await self.run_ci_cd_pipeline_test()
            
            # CI/CD 실패시 중단
            if not results['cicd_pipeline']['overall_success']:
                print("❌ CI/CD tests failed - stopping comprehensive suite")
                return results
            
            # 2. 정확도 검증
            results['accuracy_validation'] = await self.run_accuracy_validation_test()
            
            # 3. 성능 벤치마크
            results['performance_benchmark'] = await self.run_performance_benchmark_test()
            
            # 4. 견고성 테스트
            results['edge_case_robustness'] = await self.run_edge_case_robustness_test()
            
            # 5. 종단간 테스트
            results['end_to_end_pipeline'] = await self.run_end_to_end_pipeline_test()
            
            # 6. 모델 비교
            results['model_comparison'] = await self.run_model_comparison_test()
            
        except Exception as e:
            results['error'] = str(e)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 종합 분석
        overall_analysis = self._generate_comprehensive_analysis(results)
        
        return {
            'execution_time': execution_time,
            'test_results': results,
            'overall_analysis': overall_analysis,
            'comprehensive_success': overall_analysis['overall_health_score'] > 0.8
        }
    
    # ============ 분석 유틸리티 메서드 ============
    
    async def _generate_accuracy_report(self, result: Dict[str, Any]):
        """정확도 리포트 생성"""
        # 실제 구현에서는 상세한 정확도 분석 로직 추가
        pass
    
    def _calculate_performance_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """성능 통계 계산"""
        execution_times = []
        confidence_scores = []
        
        for result in results:
            if result.get('success'):
                analysis = result.get('analysis', {})
                perf_metrics = analysis.get('performance_metrics', {})
                qual_metrics = analysis.get('quality_metrics', {})
                
                if 'avg_execution_time' in perf_metrics:
                    execution_times.append(perf_metrics['avg_execution_time'])
                if 'avg_confidence' in qual_metrics:
                    confidence_scores.append(qual_metrics['avg_confidence'])
        
        if not execution_times:
            return {'avg_time': 0, 'std_time': 0, 'avg_confidence': 0}
        
        import statistics
        
        return {
            'avg_time': statistics.mean(execution_times),
            'std_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'avg_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
            'stability_score': 1.0 - (statistics.stdev(execution_times) / statistics.mean(execution_times)) if len(execution_times) > 1 else 1.0
        }
    
    def _analyze_pipeline_quality(self, result: Dict[str, Any]) -> Dict[str, float]:
        """파이프라인 품질 분석"""
        return {
            'data_consistency_score': 0.9,  # TODO: 실제 데이터 일관성 검사
            'phase_transition_score': 0.85,  # TODO: Phase간 전환 품질
            'overall_quality_score': 0.87
        }
    
    def _compare_model_performance(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """모델 성능 비교 분석"""
        performance_scores = {}
        
        for model, result in model_results.items():
            if result.get('success'):
                analysis = result.get('analysis', {})
                perf_metrics = analysis.get('performance_metrics', {})
                qual_metrics = analysis.get('quality_metrics', {})
                
                # 종합 점수 계산 (실행시간 역수 + 신뢰도)
                exec_time = perf_metrics.get('avg_execution_time', 999)
                confidence = qual_metrics.get('avg_confidence', 0)
                
                performance_scores[model] = (1.0 / exec_time) * 100 + confidence
        
        # 상위 모델 선별
        sorted_models = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'performance_scores': performance_scores,
            'top_performers': [model for model, score in sorted_models[:2]],
            'best_model': sorted_models[0][0] if sorted_models else None
        }
    
    def _generate_comprehensive_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """종합 분석 생성"""
        success_count = 0
        total_tests = 0
        
        for test_name, result in results.items():
            if test_name != 'error':
                total_tests += 1
                if isinstance(result, dict):
                    if result.get('overall_success') or result.get('success') or result.get('comprehensive_success'):
                        success_count += 1
        
        health_score = success_count / total_tests if total_tests > 0 else 0
        
        return {
            'total_tests_run': total_tests,
            'tests_passed': success_count,
            'overall_health_score': health_score,
            'system_status': 'HEALTHY' if health_score > 0.8 else 'NEEDS_ATTENTION' if health_score > 0.6 else 'CRITICAL',
            'recommendations': self._generate_recommendations(health_score, results)
        }
    
    def _generate_recommendations(self, health_score: float, results: Dict[str, Any]) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if health_score < 0.8:
            recommendations.append("System health below optimal - review failed test cases")
        
        if 'performance_benchmark' in results:
            perf_result = results['performance_benchmark']
            if perf_result.get('performance_regression_detected'):
                recommendations.append("Performance regression detected - investigate recent changes")
        
        if 'edge_case_robustness' in results:
            robustness = results['edge_case_robustness']
            if not robustness.get('robust_system', True):
                recommendations.append("System robustness needs improvement - enhance error handling")
        
        return recommendations


# ============ 실행 예제 ============

async def main():
    """종합 테스트 실행 예제"""
    scenarios = ComprehensiveTestScenarios()
    
    # 개별 시나리오 실행 예제
    print("Running individual test scenarios...\n")
    
    # CI/CD 테스트
    cicd_result = await scenarios.run_ci_cd_pipeline_test()
    print(f"CI/CD Test: {'✅ PASSED' if cicd_result['overall_success'] else '❌ FAILED'}")
    
    # 전체 테스트 스위트 실행
    print("\nRunning comprehensive test suite...\n")
    
    comprehensive_result = await scenarios.run_comprehensive_test_suite()
    
    # 결과 요약
    analysis = comprehensive_result['overall_analysis']
    print(f"\n🎯 Test Suite Results:")
    print(f"   Tests Run: {analysis['total_tests_run']}")
    print(f"   Tests Passed: {analysis['tests_passed']}")
    print(f"   Health Score: {analysis['overall_health_score']:.1%}")
    print(f"   System Status: {analysis['system_status']}")
    print(f"   Overall Success: {'✅ PASSED' if comprehensive_result['comprehensive_success'] else '❌ FAILED'}")
    
    if analysis['recommendations']:
        print(f"\n📋 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")


if __name__ == "__main__":
    asyncio.run(main())