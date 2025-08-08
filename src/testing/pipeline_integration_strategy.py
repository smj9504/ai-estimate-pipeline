# src/testing/pipeline_integration_strategy.py
"""
Pipeline Integration Strategy for AI Estimation Testing
Integrates all data pipeline components with existing codebase
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import yaml

from src.testing.data_pipeline_orchestrator import DataPipelineOrchestrator, ProcessingMode
from src.testing.data_quality_framework import DataQualityFramework, DataQualityReport
from src.testing.intelligent_cache_manager import IntelligentCacheManager, CacheLevel
from src.phases.phase1_processor import Phase1Processor
from src.phases.phase2_processor import Phase2Processor
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


@dataclass
class TestConfiguration:
    """Test execution configuration"""
    name: str
    description: str
    phases: List[int]
    model_combinations: List[List[str]]
    processing_mode: ProcessingMode
    quality_gates_enabled: bool = True
    caching_enabled: bool = True
    parallel_execution: bool = True
    max_concurrent_tests: int = 5
    cache_warming_count: int = 50


class PipelineIntegrationStrategy:
    """
    Comprehensive integration strategy for AI estimation testing pipeline
    
    Integrates:
    - Data pipeline orchestration
    - Quality framework
    - Intelligent caching  
    - Existing phase processors
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = ConfigLoader(config or {})
        
        # Initialize core components
        self.orchestrator = DataPipelineOrchestrator(self.config.data)
        self.quality_framework = DataQualityFramework(self.config.data)
        self.cache_manager = IntelligentCacheManager(self.config.data)
        
        # Initialize phase processors
        self.phase1_processor = Phase1Processor(self.config.data)
        self.phase2_processor = Phase2Processor(self.config.data)
        
        # Test results storage
        self.results_path = Path("test_outputs/integrated_results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.performance_metrics = {
            'test_execution_times': [],
            'cache_hit_rates': [],
            'quality_scores': [],
            'processing_throughput': []
        }
    
    async def execute_comprehensive_test_suite(self, 
                                               test_config: TestConfiguration,
                                               test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute comprehensive test suite with full pipeline integration
        
        Args:
            test_config: Test execution configuration
            test_scenarios: List of test scenarios to execute
        
        Returns:
            Comprehensive test results with analytics
        """
        logger.info(f"Starting comprehensive test suite: {test_config.name}")
        start_time = datetime.now()
        
        # Phase 1: Cache warming (if enabled)
        if test_config.caching_enabled:
            await self._warm_caches(test_scenarios, test_config.cache_warming_count)
        
        # Phase 2: Execute test scenarios
        scenario_results = await self._execute_test_scenarios(test_config, test_scenarios)
        
        # Phase 3: Quality assessment
        quality_reports = await self._assess_scenario_quality(scenario_results, test_config)
        
        # Phase 4: Performance analysis
        performance_analysis = await self._analyze_performance_metrics()
        
        # Phase 5: Generate comprehensive report
        comprehensive_report = await self._generate_comprehensive_report(
            test_config, scenario_results, quality_reports, performance_analysis, start_time
        )
        
        # Store results
        await self._store_comprehensive_results(comprehensive_report)
        
        logger.info(f"Test suite completed: {test_config.name} - {len(test_scenarios)} scenarios")
        return comprehensive_report
    
    async def _warm_caches(self, test_scenarios: List[Dict], warming_count: int):
        """Warm caches with representative data"""
        logger.info(f"Warming caches with {warming_count} scenarios")
        
        # Select representative scenarios for warming
        warming_scenarios = test_scenarios[:warming_count] if len(test_scenarios) >= warming_count else test_scenarios
        
        # Warm with common patterns
        async def cache_warming_generator(count: int) -> List[Dict]:
            return warming_scenarios[:count]
        
        warming_result = await self.cache_manager.warm_cache(
            dataset_generator=cache_warming_generator,
            warm_count=len(warming_scenarios),
            priority=3
        )
        
        logger.info(f"Cache warming completed: {warming_result}")
    
    async def _execute_test_scenarios(self, 
                                      test_config: TestConfiguration,
                                      test_scenarios: List[Dict]) -> List[Dict]:
        """Execute all test scenarios with configured parameters"""
        scenario_results = []
        
        # Create execution tasks
        tasks = []
        for scenario_idx, scenario in enumerate(test_scenarios):
            for combo_idx, model_combo in enumerate(test_config.model_combinations):
                task = self._execute_single_scenario(
                    scenario=scenario,
                    scenario_idx=scenario_idx,
                    model_combo=model_combo,
                    combo_idx=combo_idx,
                    test_config=test_config
                )
                tasks.append(task)
        
        # Execute with controlled concurrency
        if test_config.parallel_execution:
            # Execute in batches to control resource usage
            batch_size = test_config.max_concurrent_tests
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Scenario execution error: {result}")
                    else:
                        scenario_results.append(result)
        else:
            # Sequential execution
            for task in tasks:
                try:
                    result = await task
                    scenario_results.append(result)
                except Exception as e:
                    logger.error(f"Scenario execution error: {e}")
        
        return scenario_results
    
    async def _execute_single_scenario(self,
                                       scenario: Dict,
                                       scenario_idx: int,
                                       model_combo: List[str],
                                       combo_idx: int,
                                       test_config: TestConfiguration) -> Dict:
        """Execute single test scenario"""
        scenario_id = f"scenario_{scenario_idx}_combo_{combo_idx}"
        execution_start = datetime.now()
        
        try:
            results = {}
            
            # Phase 1 execution (if configured)
            if 1 in test_config.phases:
                phase1_result = await self.orchestrator.process_single_phase(
                    phase_num=1,
                    input_data=scenario,
                    models=model_combo,
                    processing_mode=test_config.processing_mode,
                    cache_intermediate=test_config.caching_enabled
                )
                results['phase1'] = phase1_result
                
                # Phase 2 execution (if configured and Phase 1 successful)
                if 2 in test_config.phases and phase1_result.get('success', False):
                    phase2_result = await self.orchestrator.process_single_phase(
                        phase_num=2,
                        input_data=self.phase1_processor.prepare_for_phase2(phase1_result),
                        models=model_combo,
                        processing_mode=test_config.processing_mode,
                        cache_intermediate=test_config.caching_enabled
                    )
                    results['phase2'] = phase2_result
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            self.performance_metrics['test_execution_times'].append(execution_time)
            
            return {
                'scenario_id': scenario_id,
                'scenario_idx': scenario_idx,
                'combo_idx': combo_idx,
                'model_combo': model_combo,
                'execution_time_seconds': execution_time,
                'success': True,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing scenario {scenario_id}: {e}")
            return {
                'scenario_id': scenario_id,
                'scenario_idx': scenario_idx,
                'combo_idx': combo_idx,
                'model_combo': model_combo,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _assess_scenario_quality(self, 
                                       scenario_results: List[Dict],
                                       test_config: TestConfiguration) -> List[DataQualityReport]:
        """Assess quality for all scenario results"""
        if not test_config.quality_gates_enabled:
            return []
        
        quality_reports = []
        
        for phase in test_config.phases:
            # Extract results for this phase
            phase_results = []
            for scenario_result in scenario_results:
                if scenario_result.get('success') and f'phase{phase}' in scenario_result.get('results', {}):
                    phase_results.append(scenario_result['results'][f'phase{phase}'])
            
            if phase_results:
                quality_report = await self.quality_framework.assess_data_quality(
                    dataset=phase_results,
                    phase=f'phase{phase}'
                )
                quality_reports.append(quality_report)
                
                # Store quality score for performance metrics
                self.performance_metrics['quality_scores'].append(quality_report.overall_score)
        
        return quality_reports
    
    async def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics collected during execution"""
        import numpy as np
        
        metrics = self.performance_metrics
        cache_analytics = await self.cache_manager.get_cache_analytics()
        
        analysis = {
            'execution_performance': {
                'mean_execution_time': np.mean(metrics['test_execution_times']) if metrics['test_execution_times'] else 0,
                'median_execution_time': np.median(metrics['test_execution_times']) if metrics['test_execution_times'] else 0,
                'max_execution_time': np.max(metrics['test_execution_times']) if metrics['test_execution_times'] else 0,
                'min_execution_time': np.min(metrics['test_execution_times']) if metrics['test_execution_times'] else 0,
                'execution_time_std': np.std(metrics['test_execution_times']) if metrics['test_execution_times'] else 0,
                'total_tests': len(metrics['test_execution_times'])
            },
            'quality_performance': {
                'mean_quality_score': np.mean(metrics['quality_scores']) if metrics['quality_scores'] else 0,
                'median_quality_score': np.median(metrics['quality_scores']) if metrics['quality_scores'] else 0,
                'quality_score_std': np.std(metrics['quality_scores']) if metrics['quality_scores'] else 0,
                'quality_trend': self._calculate_quality_trend(metrics['quality_scores'])
            },
            'cache_performance': cache_analytics,
            'throughput_analysis': {
                'tests_per_minute': len(metrics['test_execution_times']) / (sum(metrics['test_execution_times']) / 60) if metrics['test_execution_times'] else 0,
                'average_processing_rate': np.mean(metrics['processing_throughput']) if metrics['processing_throughput'] else 0
            }
        }
        
        return analysis
    
    def _calculate_quality_trend(self, quality_scores: List[float]) -> str:
        """Calculate quality trend over time"""
        if len(quality_scores) < 2:
            return 'insufficient_data'
        
        import numpy as np
        
        # Simple linear trend calculation
        x = np.arange(len(quality_scores))
        slope = np.polyfit(x, quality_scores, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'
    
    async def _generate_comprehensive_report(self,
                                             test_config: TestConfiguration,
                                             scenario_results: List[Dict],
                                             quality_reports: List[DataQualityReport],
                                             performance_analysis: Dict,
                                             start_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate success rates
        successful_scenarios = sum(1 for result in scenario_results if result.get('success', False))
        total_scenarios = len(scenario_results)
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Analyze model performance
        model_performance = self._analyze_model_performance(scenario_results)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            scenario_results, quality_reports, performance_analysis
        )
        
        comprehensive_report = {
            'test_suite_metadata': {
                'name': test_config.name,
                'description': test_config.description,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'configuration': {
                    'phases': test_config.phases,
                    'model_combinations': test_config.model_combinations,
                    'processing_mode': test_config.processing_mode.value,
                    'quality_gates_enabled': test_config.quality_gates_enabled,
                    'caching_enabled': test_config.caching_enabled,
                    'parallel_execution': test_config.parallel_execution
                }
            },
            'execution_summary': {
                'total_scenarios': total_scenarios,
                'successful_scenarios': successful_scenarios,
                'success_rate': success_rate,
                'failed_scenarios': total_scenarios - successful_scenarios,
                'average_execution_time': performance_analysis['execution_performance']['mean_execution_time']
            },
            'quality_assessment': {
                'overall_quality_score': performance_analysis['quality_performance']['mean_quality_score'],
                'quality_trend': performance_analysis['quality_performance']['quality_trend'],
                'quality_reports_summary': [
                    {
                        'phase': report.processing_phase,
                        'overall_score': report.overall_score,
                        'passed_quality_gate': report.passed_quality_gate,
                        'failed_dimensions': [
                            m.dimension.value for m in report.metrics if not m.passed
                        ]
                    }
                    for report in quality_reports
                ]
            },
            'performance_analysis': performance_analysis,
            'model_performance': model_performance,
            'scenario_results': scenario_results,
            'quality_reports': [report.to_dict() for report in quality_reports],
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        return comprehensive_report
    
    def _analyze_model_performance(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by model combination"""
        model_stats = {}
        
        for result in scenario_results:
            if not result.get('success'):
                continue
            
            model_combo = tuple(result['model_combo'])  # Convert to tuple for hashing
            model_combo_str = '_'.join(result['model_combo'])
            
            if model_combo_str not in model_stats:
                model_stats[model_combo_str] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'execution_times': [],
                    'confidence_scores': []
                }
            
            stats = model_stats[model_combo_str]
            stats['total_tests'] += 1
            stats['successful_tests'] += 1
            stats['execution_times'].append(result['execution_time_seconds'])
            
            # Extract confidence scores from phase results
            for phase_key, phase_result in result.get('results', {}).items():
                if isinstance(phase_result, dict) and 'confidence_score' in phase_result:
                    stats['confidence_scores'].append(phase_result['confidence_score'])
        
        # Calculate summary statistics
        import numpy as np
        
        for model_combo, stats in model_stats.items():
            stats['success_rate'] = stats['successful_tests'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
            stats['mean_execution_time'] = np.mean(stats['execution_times']) if stats['execution_times'] else 0
            stats['mean_confidence_score'] = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0
            stats['execution_time_std'] = np.std(stats['execution_times']) if stats['execution_times'] else 0
        
        # Rank models by performance
        ranked_models = sorted(
            model_stats.items(),
            key=lambda x: (x[1]['success_rate'], x[1]['mean_confidence_score'], -x[1]['mean_execution_time']),
            reverse=True
        )
        
        return {
            'model_statistics': model_stats,
            'ranked_performance': [
                {
                    'model_combination': combo,
                    'rank': idx + 1,
                    'success_rate': stats['success_rate'],
                    'mean_confidence': stats['mean_confidence_score'],
                    'mean_execution_time': stats['mean_execution_time']
                }
                for idx, (combo, stats) in enumerate(ranked_models)
            ]
        }
    
    async def _generate_recommendations(self,
                                        scenario_results: List[Dict],
                                        quality_reports: List[DataQualityReport],
                                        performance_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on test results"""
        recommendations = []
        
        # Performance-based recommendations
        exec_perf = performance_analysis['execution_performance']
        if exec_perf['mean_execution_time'] > 30:  # > 30 seconds average
            recommendations.append("Consider optimizing execution time - average execution exceeds 30 seconds")
        
        if exec_perf['execution_time_std'] > exec_perf['mean_execution_time'] * 0.5:
            recommendations.append("High execution time variability detected - investigate performance inconsistencies")
        
        # Quality-based recommendations
        quality_perf = performance_analysis['quality_performance']
        if quality_perf['mean_quality_score'] < 0.8:
            recommendations.append("Overall quality score below threshold (0.8) - review data generation process")
        
        if quality_perf['quality_trend'] == 'declining':
            recommendations.append("Quality trend is declining - implement quality monitoring alerts")
        
        # Cache performance recommendations
        cache_perf = performance_analysis['cache_performance']
        if cache_perf.get('hit_rate', 0) < 0.6:
            recommendations.append("Low cache hit rate (<60%) - consider improving cache warming strategy")
        
        # Model-specific recommendations
        model_perf = performance_analysis.get('model_performance', {})
        ranked_models = model_perf.get('ranked_performance', [])
        
        if ranked_models:
            best_model = ranked_models[0]
            worst_model = ranked_models[-1]
            
            if best_model['success_rate'] - worst_model['success_rate'] > 0.2:
                recommendations.append(f"Significant performance gap between models - consider focusing on {best_model['model_combination']}")
        
        # Quality report recommendations
        for report in quality_reports:
            recommendations.extend(report.recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _store_comprehensive_results(self, report: Dict[str, Any]):
        """Store comprehensive test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_path / f"comprehensive_test_report_{timestamp}.json"
        
        # Store main report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Store summary for quick access
        summary_file = self.results_path / f"test_summary_{timestamp}.json"
        summary = {
            'test_name': report['test_suite_metadata']['name'],
            'total_duration': report['test_suite_metadata']['total_duration_seconds'],
            'success_rate': report['execution_summary']['success_rate'],
            'overall_quality_score': report['quality_assessment']['overall_quality_score'],
            'total_scenarios': report['execution_summary']['total_scenarios'],
            'best_model_combination': report['model_performance']['ranked_performance'][0] if report['model_performance']['ranked_performance'] else None,
            'key_recommendations': report['recommendations'][:5],  # Top 5 recommendations
            'generated_at': report['generated_at']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Test results stored: {report_file}")
    
    async def create_test_configuration_from_yaml(self, config_path: str) -> TestConfiguration:
        """Create test configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return TestConfiguration(
            name=config_data['name'],
            description=config_data['description'],
            phases=config_data['phases'],
            model_combinations=config_data['model_combinations'],
            processing_mode=ProcessingMode(config_data.get('processing_mode', 'streaming')),
            quality_gates_enabled=config_data.get('quality_gates_enabled', True),
            caching_enabled=config_data.get('caching_enabled', True),
            parallel_execution=config_data.get('parallel_execution', True),
            max_concurrent_tests=config_data.get('max_concurrent_tests', 5),
            cache_warming_count=config_data.get('cache_warming_count', 50)
        )
    
    async def load_test_scenarios_from_directory(self, scenarios_dir: str) -> List[Dict[str, Any]]:
        """Load test scenarios from directory"""
        scenarios_path = Path(scenarios_dir)
        if not scenarios_path.exists():
            raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
        
        scenarios = []
        
        for scenario_file in scenarios_path.glob("*.json"):
            try:
                with open(scenario_file, 'r') as f:
                    scenario_data = json.load(f)
                    scenario_data['_source_file'] = scenario_file.name
                    scenarios.append(scenario_data)
            except Exception as e:
                logger.error(f"Error loading scenario {scenario_file}: {e}")
        
        logger.info(f"Loaded {len(scenarios)} test scenarios from {scenarios_dir}")
        return scenarios


# Example configuration template
EXAMPLE_CONFIG = """
name: "Comprehensive Phase 1-2 Testing"
description: "Full pipeline testing with quality gates and performance monitoring"
phases: [1, 2]
model_combinations:
  - ["gpt4", "claude"]
  - ["gpt4", "gemini"]
  - ["claude", "gemini"]
  - ["gpt4", "claude", "gemini"]
processing_mode: "streaming"
quality_gates_enabled: true
caching_enabled: true
parallel_execution: true
max_concurrent_tests: 3
cache_warming_count: 25
"""

# Example usage
async def main():
    """Example usage of pipeline integration strategy"""
    strategy = PipelineIntegrationStrategy()
    
    # Create test configuration
    test_config = TestConfiguration(
        name="Example Test Suite",
        description="Testing Phase 1-2 integration",
        phases=[1, 2],
        model_combinations=[["gpt4", "claude"], ["claude", "gemini"]],
        processing_mode=ProcessingMode.STREAMING,
        quality_gates_enabled=True,
        caching_enabled=True,
        parallel_execution=True
    )
    
    # Sample test scenarios
    test_scenarios = [
        {
            "rooms": [
                {
                    "name": "living_room",
                    "room_type": "Living Room",
                    "measurements": {"sqft": 200, "height": 9},
                    "materials": ["drywall", "paint", "carpet"],
                    "work_scope": {"drywall": "Remove & Replace"},
                    "demo_scope(already demo'd)": {"drywall": 0}
                }
            ]
        }
    ]
    
    # Execute comprehensive test suite
    results = await strategy.execute_comprehensive_test_suite(test_config, test_scenarios)
    
    print(f"Test suite completed: {results['execution_summary']['success_rate']:.2%} success rate")
    print(f"Overall quality score: {results['quality_assessment']['overall_quality_score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())