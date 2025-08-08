"""
CLI Runner for Smart Test Framework
지능형 테스트 프레임워크를 위한 명령행 인터페이스
"""
import asyncio
import argparse
import json
from typing import List, Dict, Any
from pathlib import Path

from .smart_test_runner import (
    SmartTestRunner, SmartTestConfig, 
    TestDataMode, TestType, TestPriority
)
from ..examples.comprehensive_test_scenarios import ComprehensiveTestScenarios


class SmartTestCLI:
    """스마트 테스트 CLI"""
    
    def __init__(self):
        self.runner = SmartTestRunner()
        self.scenarios = ComprehensiveTestScenarios()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """CLI 파서 생성"""
        parser = argparse.ArgumentParser(
            description='AI Estimation Pipeline Smart Test Runner',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run CI/CD test suite
  python -m tests.strategies.cli_runner cicd
  
  # Run single phase with specific configuration  
  python -m tests.strategies.cli_runner single --phase 1 --models gpt4 --mode deterministic
  
  # Run performance benchmark
  python -m tests.strategies.cli_runner benchmark --phases 1 2 --runs 5
  
  # Run comprehensive test suite
  python -m tests.strategies.cli_runner comprehensive
  
  # Run custom configuration
  python -m tests.strategies.cli_runner custom --config my_test.json
            """)
        
        subparsers = parser.add_subparsers(dest='command', help='Test commands')
        
        # CI/CD 테스트
        cicd_parser = subparsers.add_parser('cicd', help='Run CI/CD test suite')
        cicd_parser.add_argument('--fast', action='store_true',
                               help='Run only smoke tests for fastest execution')
        
        # 단일 테스트
        single_parser = subparsers.add_parser('single', help='Run single test')
        single_parser.add_argument('--phase', type=int, required=True,
                                 choices=[0, 1, 2], help='Phase to test')
        single_parser.add_argument('--models', nargs='+',
                                 default=['gpt4'],
                                 choices=['gpt4', 'claude', 'gemini'],
                                 help='AI models to use')
        single_parser.add_argument('--mode', choices=['deterministic', 'live', 'hybrid'],
                                 default='deterministic',
                                 help='Test data mode')
        single_parser.add_argument('--timeout', type=int, default=300,
                                 help='Timeout in seconds')
        single_parser.add_argument('--golden-dataset',
                                 help='Golden dataset name for deterministic mode')
        
        # 성능 벤치마크
        bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
        bench_parser.add_argument('--phases', type=int, nargs='+',
                                default=[1], choices=[0, 1, 2],
                                help='Phases to benchmark')
        bench_parser.add_argument('--runs', type=int, default=3,
                                help='Number of benchmark runs')
        bench_parser.add_argument('--models', nargs='+',
                                default=['gpt4'],
                                choices=['gpt4', 'claude', 'gemini'],
                                help='Models to benchmark')
        
        # 종합 테스트
        comp_parser = subparsers.add_parser('comprehensive', help='Run comprehensive test suite')
        comp_parser.add_argument('--skip-slow', action='store_true',
                               help='Skip slow tests')
        comp_parser.add_argument('--save-results', 
                               help='Save results to specified file')
        
        # 커스텀 테스트
        custom_parser = subparsers.add_parser('custom', help='Run custom test configuration')
        custom_parser.add_argument('--config', required=True,
                                 help='JSON configuration file path')
        
        # 시나리오별 테스트
        scenario_parser = subparsers.add_parser('scenario', help='Run specific test scenario')
        scenario_parser.add_argument('scenario_name',
                                   choices=['accuracy', 'performance', 'robustness', 'e2e', 'model_comparison'],
                                   help='Scenario to run')
        
        # 골든 데이터 관리
        golden_parser = subparsers.add_parser('golden', help='Golden dataset management')
        golden_subparsers = golden_parser.add_subparsers(dest='golden_command')
        
        # 골든 데이터 생성
        create_parser = golden_subparsers.add_parser('create', help='Create golden dataset')
        create_parser.add_argument('--name', required=True, help='Dataset name')
        create_parser.add_argument('--phases', type=int, nargs='+', required=True,
                                 help='Phases to include')
        create_parser.add_argument('--models', nargs='+', default=['gpt4', 'claude', 'gemini'],
                                 help='Models to use for generation')
        
        # 골든 데이터 목록
        golden_subparsers.add_parser('list', help='List available golden datasets')
        
        # 일반 옵션
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Quiet mode (minimal output)')
        parser.add_argument('--output-dir', default='test_outputs/cli_runs',
                          help='Output directory for results')
        
        return parser
    
    async def run_cicd_tests(self, args) -> Dict[str, Any]:
        """CI/CD 테스트 실행"""
        if not args.quiet:
            print("🚀 Running CI/CD Test Suite...")
        
        if args.fast:
            # 스모크 테스트만
            config = SmartTestConfig(
                name="cicd_smoke",
                phases=[1],
                test_mode=TestDataMode.DETERMINISTIC,
                test_type=TestType.SMOKE,
                priority=TestPriority.CRITICAL,
                models=["gpt4"],
                golden_dataset_name="phase1_baseline",
                timeout_seconds=30,
                max_execution_time=25.0
            )
            return await self.runner.run_smart_test(config)
        else:
            # 전체 CI/CD 스위트
            return await self.scenarios.run_ci_cd_pipeline_test()
    
    async def run_single_test(self, args) -> Dict[str, Any]:
        """단일 테스트 실행"""
        if not args.quiet:
            print(f"🎯 Running single test: Phase {args.phase}")
        
        # 골든 데이터셋 결정
        golden_dataset = args.golden_dataset
        if not golden_dataset and args.mode == 'deterministic':
            golden_dataset = f"phase{args.phase}_baseline"
        
        config = SmartTestConfig(
            name=f"single_phase{args.phase}",
            phases=[args.phase],
            test_mode=TestDataMode[args.mode.upper()],
            test_type=TestType.REGRESSION,
            priority=TestPriority.HIGH,
            models=args.models,
            timeout_seconds=args.timeout,
            golden_dataset_name=golden_dataset
        )
        
        return await self.runner.run_smart_test(config)
    
    async def run_benchmark_test(self, args) -> Dict[str, Any]:
        """성능 벤치마크 실행"""
        if not args.quiet:
            print(f"⚡ Running performance benchmark: {args.runs} runs")
        
        results = []
        for run in range(args.runs):
            if not args.quiet:
                print(f"  Benchmark run {run + 1}/{args.runs}...")
            
            config = SmartTestConfig(
                name=f"benchmark_run_{run}",
                phases=args.phases,
                test_mode=TestDataMode.DETERMINISTIC,
                test_type=TestType.PERFORMANCE,
                priority=TestPriority.MEDIUM,
                models=args.models,
                golden_dataset_name="phase1_baseline" if 1 in args.phases else "phase0_baseline",
                timeout_seconds=180
            )
            
            result = await self.runner.run_smart_test(config)
            results.append(result)
            
            if not result['success']:
                break
        
        # 벤치마크 통계 계산
        stats = self._calculate_benchmark_stats(results)
        
        return {
            'benchmark_runs': results,
            'statistics': stats,
            'performance_stable': stats.get('coefficient_of_variation', 1.0) < 0.1
        }
    
    async def run_comprehensive_tests(self, args) -> Dict[str, Any]:
        """종합 테스트 실행"""
        if not args.quiet:
            print("🎭 Running Comprehensive Test Suite...")
        
        # TODO: skip_slow 옵션 처리
        result = await self.scenarios.run_comprehensive_test_suite()
        
        # 결과 저장
        if args.save_results:
            output_file = Path(args.save_results)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            
            if not args.quiet:
                print(f"Results saved to: {output_file}")
        
        return result
    
    async def run_custom_test(self, args) -> Dict[str, Any]:
        """커스텀 테스트 실행"""
        config_file = Path(args.config)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # JSON을 SmartTestConfig로 변환
        config = SmartTestConfig(**config_data)
        
        if not args.quiet:
            print(f"🔧 Running custom test: {config.name}")
        
        return await self.runner.run_smart_test(config)
    
    async def run_scenario_test(self, args) -> Dict[str, Any]:
        """시나리오 테스트 실행"""
        scenario_map = {
            'accuracy': self.scenarios.run_accuracy_validation_test,
            'performance': self.scenarios.run_performance_benchmark_test,
            'robustness': self.scenarios.run_edge_case_robustness_test,
            'e2e': self.scenarios.run_end_to_end_pipeline_test,
            'model_comparison': self.scenarios.run_model_comparison_test
        }
        
        scenario_func = scenario_map[args.scenario_name]
        
        if not args.quiet:
            print(f"🎬 Running scenario: {args.scenario_name}")
        
        return await scenario_func()
    
    async def manage_golden_data(self, args) -> Dict[str, Any]:
        """골든 데이터 관리"""
        if args.golden_command == 'create':
            return await self._create_golden_dataset(args)
        elif args.golden_command == 'list':
            return await self._list_golden_datasets(args)
    
    async def _create_golden_dataset(self, args) -> Dict[str, Any]:
        """골든 데이터셋 생성"""
        if not args.quiet:
            print(f"📦 Creating golden dataset: {args.name}")
        
        # 실시간 테스트 실행하여 결과 수집
        from ..phases.base import PhaseTestConfig
        from ..strategies.test_data_strategy import TestDataManager
        
        test_config = PhaseTestConfig(
            phase_numbers=args.phases,
            models=args.models,
            save_outputs=True
        )
        
        data_manager = TestDataManager()
        live_results = await data_manager.run_live_integration_test(
            args.phases, test_config, save_as_golden=args.name
        )
        
        return {
            'dataset_name': args.name,
            'phases_included': args.phases,
            'models_used': args.models,
            'success': all(r.success for r in live_results)
        }
    
    async def _list_golden_datasets(self, args) -> Dict[str, Any]:
        """골든 데이터셋 목록"""
        from ..strategies.test_data_strategy import TestDataManager
        
        data_manager = TestDataManager()
        datasets = data_manager.list_golden_datasets()
        
        if not args.quiet:
            print("📋 Available Golden Datasets:")
            for dataset in datasets:
                print(f"  • {dataset}")
        
        return {'datasets': datasets}
    
    def _calculate_benchmark_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """벤치마크 통계 계산"""
        execution_times = []
        
        for result in results:
            if result.get('success'):
                analysis = result.get('analysis', {})
                perf_metrics = analysis.get('performance_metrics', {})
                if 'avg_execution_time' in perf_metrics:
                    execution_times.append(perf_metrics['avg_execution_time'])
        
        if not execution_times:
            return {}
        
        import statistics
        
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        return {
            'mean_execution_time': mean_time,
            'std_execution_time': std_time,
            'coefficient_of_variation': std_time / mean_time if mean_time > 0 else 0,
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'runs_count': len(execution_times)
        }
    
    def format_results(self, results: Dict[str, Any], verbose: bool = False) -> str:
        """결과 포매팅"""
        if not results:
            return "No results to display"
        
        # 기본 결과
        success = results.get('success', False)
        status_emoji = "✅" if success else "❌"
        
        output = [f"{status_emoji} Test Result: {'PASSED' if success else 'FAILED'}"]
        
        # 실행 시간
        exec_time = results.get('execution_time', 0)
        if exec_time > 0:
            output.append(f"⏱️  Execution Time: {exec_time:.2f}s")
        
        # 분석 결과
        analysis = results.get('analysis', {})
        if analysis:
            perf_metrics = analysis.get('performance_metrics', {})
            qual_metrics = analysis.get('quality_metrics', {})
            
            if 'avg_execution_time' in perf_metrics:
                output.append(f"📊 Average Execution Time: {perf_metrics['avg_execution_time']:.2f}s")
            
            if 'avg_confidence' in qual_metrics:
                output.append(f"🎯 Average Confidence: {qual_metrics['avg_confidence']:.1%}")
        
        # Verbose 모드에서 추가 정보
        if verbose:
            if 'config' in results:
                config = results['config']
                output.append(f"🔧 Configuration: {config.get('name', 'N/A')}")
                output.append(f"   Models: {config.get('models', [])}")
                output.append(f"   Phases: {config.get('phases', [])}")
            
            # 권장사항
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                output.append("💡 Recommendations:")
                for rec in recommendations:
                    output.append(f"   • {rec}")
        
        return "\n".join(output)
    
    async def run(self):
        """CLI 실행"""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            # 명령 실행
            if args.command == 'cicd':
                results = await self.run_cicd_tests(args)
            elif args.command == 'single':
                results = await self.run_single_test(args)
            elif args.command == 'benchmark':
                results = await self.run_benchmark_test(args)
            elif args.command == 'comprehensive':
                results = await self.run_comprehensive_tests(args)
            elif args.command == 'custom':
                results = await self.run_custom_test(args)
            elif args.command == 'scenario':
                results = await self.run_scenario_test(args)
            elif args.command == 'golden':
                results = await self.manage_golden_data(args)
            else:
                parser.print_help()
                return
            
            # 결과 출력
            if not args.quiet:
                print("\n" + "="*50)
                print(self.format_results(results, args.verbose))
                print("="*50)
            
            # JSON 출력 옵션
            if hasattr(args, 'json_output') and args.json_output:
                print(json.dumps(results, indent=2, default=str))
                
        except Exception as e:
            if args.verbose:
                import traceback
                print(f"❌ Error: {e}")
                print(traceback.format_exc())
            else:
                print(f"❌ Error: {e}")


def main():
    """메인 진입점"""
    cli = SmartTestCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()