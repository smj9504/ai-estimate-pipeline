# src/testing/automated_testing_pipeline.py
"""
Automated Testing Pipeline for Multi-Model Construction Estimation
Provides continuous integration, regression testing, and performance monitoring
"""
import asyncio
import json
import yaml
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from enum import Enum

from src.testing.ab_testing_framework import ABTestingFramework
from src.testing.benchmark_generator import BenchmarkDatasetGenerator
from src.testing.performance_metrics import ConstructionEstimationMetrics, PerformanceReport
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


class TestType(Enum):
    """Types of automated tests"""
    REGRESSION = "regression"
    PERFORMANCE = "performance" 
    A_B_COMPARISON = "ab_comparison"
    STRESS_TEST = "stress_test"
    SMOKE_TEST = "smoke_test"
    BENCHMARK_VALIDATION = "benchmark_validation"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestConfiguration:
    """Configuration for automated tests"""
    test_id: str
    test_type: TestType
    test_name: str
    description: str
    model_combinations: List[List[str]]
    test_data_path: Optional[str]
    schedule_cron: Optional[str]  # Cron-like schedule
    timeout_minutes: int = 60
    retry_count: int = 3
    alert_on_failure: bool = True
    performance_thresholds: Dict[str, float] = None
    tags: List[str] = None
    enabled: bool = True


@dataclass
class TestResult:
    """Results from automated test execution"""
    test_id: str
    test_type: TestType
    status: TestStatus
    start_time: str
    end_time: str
    duration_seconds: float
    model_combinations: List[List[str]]
    results_summary: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]]
    error_message: Optional[str] = None
    detailed_results: Optional[Dict[str, Any]] = None
    artifacts: List[str] = None  # Paths to generated artifacts


class AutomatedTestingPipeline:
    """
    Comprehensive automated testing pipeline for construction estimation AI
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/testing_pipeline.yaml"
        self.logger = get_logger('automated_testing_pipeline')
        
        # Load configuration
        self.config = self._load_pipeline_config()
        
        # Initialize components
        self.orchestrator = ModelOrchestrator()
        self.ab_framework = ABTestingFramework()
        self.benchmark_generator = BenchmarkDatasetGenerator()
        self.metrics_calculator = ConstructionEstimationMetrics()
        self.merger = ResultMerger()
        
        # Test management
        self.test_configurations = {}
        self.test_history = []
        self.active_tests = {}
        
        # Scheduling
        self.scheduler_running = False
        
        # Results storage
        self.results_dir = Path(self.config.get('results_directory', 'automated_test_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.baseline_metrics = {}
        
        self.logger.info("Automated Testing Pipeline initialized")
    
    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load pipeline configuration from YAML file"""
        
        config_path = Path(self.config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded pipeline configuration from {config_path}")
        else:
            # Default configuration
            config = {
                'results_directory': 'automated_test_results',
                'max_concurrent_tests': 3,
                'default_timeout_minutes': 60,
                'alert_settings': {
                    'email_notifications': False,
                    'slack_webhook': None,
                    'critical_failure_threshold': 2
                },
                'performance_monitoring': {
                    'enabled': True,
                    'regression_threshold': 0.05,  # 5% performance drop
                    'baseline_update_frequency': 'weekly'
                },
                'retention_policy': {
                    'keep_results_days': 30,
                    'archive_after_days': 90
                }
            }
            
            # Save default configuration
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.logger.info(f"Created default configuration at {config_path}")
        
        return config
    
    def register_test(self, test_config: TestConfiguration):
        """Register a new test configuration"""
        
        self.test_configurations[test_config.test_id] = test_config
        self.logger.info(f"Registered test: {test_config.test_id} - {test_config.test_name}")
        
        # Schedule if cron expression provided
        if test_config.schedule_cron and test_config.enabled:
            self._schedule_test(test_config)
    
    def _schedule_test(self, test_config: TestConfiguration):
        """Schedule a test using cron-like expression"""
        
        # Simple scheduling implementation (could be extended with proper cron parsing)
        cron = test_config.schedule_cron
        
        if cron == 'daily':
            schedule.every().day.at("02:00").do(
                lambda: asyncio.create_task(self.execute_test(test_config.test_id))
            )
        elif cron == 'weekly':
            schedule.every().sunday.at("01:00").do(
                lambda: asyncio.create_task(self.execute_test(test_config.test_id))
            )
        elif cron == 'hourly':
            schedule.every().hour.do(
                lambda: asyncio.create_task(self.execute_test(test_config.test_id))
            )
        elif cron.startswith('every_'):
            # Parse "every_Xm" for X minutes
            try:
                minutes = int(cron.split('_')[1].replace('m', ''))
                schedule.every(minutes).minutes.do(
                    lambda: asyncio.create_task(self.execute_test(test_config.test_id))
                )
            except:
                self.logger.warning(f"Invalid cron expression: {cron}")
        
        self.logger.info(f"Scheduled test {test_config.test_id} with cron: {cron}")
    
    async def execute_test(self, test_id: str) -> TestResult:
        """Execute a specific test by ID"""
        
        if test_id not in self.test_configurations:
            raise ValueError(f"Test {test_id} not found in configurations")
        
        test_config = self.test_configurations[test_id]
        
        if not test_config.enabled:
            self.logger.info(f"Test {test_id} is disabled, skipping")
            return self._create_skipped_result(test_config)
        
        self.logger.info(f"Starting test execution: {test_id}")
        start_time = datetime.now()
        
        try:
            # Mark test as running
            self.active_tests[test_id] = {
                'start_time': start_time,
                'status': TestStatus.RUNNING,
                'config': test_config
            }
            
            # Execute based on test type
            if test_config.test_type == TestType.REGRESSION:
                result = await self._execute_regression_test(test_config)
            elif test_config.test_type == TestType.PERFORMANCE:
                result = await self._execute_performance_test(test_config)
            elif test_config.test_type == TestType.A_B_COMPARISON:
                result = await self._execute_ab_test(test_config)
            elif test_config.test_type == TestType.STRESS_TEST:
                result = await self._execute_stress_test(test_config)
            elif test_config.test_type == TestType.SMOKE_TEST:
                result = await self._execute_smoke_test(test_config)
            elif test_config.test_type == TestType.BENCHMARK_VALIDATION:
                result = await self._execute_benchmark_test(test_config)
            else:
                raise ValueError(f"Unknown test type: {test_config.test_type}")
            
            # Mark as completed
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            # Store result
            self.test_history.append(result)
            await self._save_test_result(result)
            
            # Update performance tracking
            if result.performance_metrics:
                self._update_performance_history(result)
            
            # Check for alerts
            await self._check_alert_conditions(result)
            
            self.logger.info(f"Test {test_id} completed: {result.status.value}")
            
        except Exception as e:
            # Handle test execution error
            end_time = datetime.now()
            
            result = TestResult(
                test_id=test_id,
                test_type=test_config.test_type,
                status=TestStatus.ERROR,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                model_combinations=test_config.model_combinations,
                results_summary={'error': str(e)},
                performance_metrics=None,
                error_message=str(e)
            )
            
            self.test_history.append(result)
            await self._save_test_result(result)
            
            self.logger.error(f"Test {test_id} failed with error: {e}")
        
        finally:
            # Remove from active tests
            if test_id in self.active_tests:
                del self.active_tests[test_id]
        
        return result
    
    async def _execute_regression_test(self, test_config: TestConfiguration) -> TestResult:
        """Execute regression test to detect performance degradation"""
        
        # Load test data
        test_data = await self._load_test_data(test_config.test_data_path)
        
        results_by_combination = {}
        
        for combination in test_config.model_combinations:
            self.logger.info(f"Running regression test for models: {combination}")
            
            # Run test cases
            test_results = []
            for test_case in test_data[:20]:  # Limit for regression testing
                try:
                    # Execute model combination
                    model_responses = await self.orchestrator.run_parallel(
                        prompt=test_case.get('prompt', 'Generate construction work scope'),
                        json_data=test_case.get('data', {}),
                        model_names=combination
                    )
                    
                    # Merge results
                    merged_result = self.merger.merge_results(model_responses)
                    
                    # Convert to test result format
                    test_result = {
                        'success': bool(merged_result.total_work_items > 0),
                        'total_work_items': merged_result.total_work_items,
                        'confidence_score': merged_result.overall_confidence,
                        'consensus_level': merged_result.metadata.consensus_level,
                        'processing_time': sum(r.processing_time for r in model_responses if hasattr(r, 'processing_time')),
                        'models_used': combination,
                        'models_responded': len(model_responses)
                    }
                    
                    test_results.append(test_result)
                    
                except Exception as e:
                    self.logger.warning(f"Test case failed: {e}")
                    test_results.append({
                        'success': False,
                        'error': str(e),
                        'models_used': combination
                    })
            
            # Calculate metrics
            performance_report = self.metrics_calculator.calculate_comprehensive_metrics(
                test_results, None, combination
            )
            
            results_by_combination['+'.join(combination)] = {
                'performance_report': performance_report,
                'test_results': test_results
            }
        
        # Compare with baseline if available
        regression_detected = False
        baseline_comparison = {}
        
        if self.baseline_metrics:
            for combo_name, results in results_by_combination.items():
                current_score = results['performance_report'].overall_score
                baseline_score = self.baseline_metrics.get(combo_name, current_score)
                
                regression_threshold = self.config['performance_monitoring']['regression_threshold']
                regression = (baseline_score - current_score) / baseline_score
                
                baseline_comparison[combo_name] = {
                    'current_score': current_score,
                    'baseline_score': baseline_score,
                    'regression': regression,
                    'regression_detected': regression > regression_threshold
                }
                
                if regression > regression_threshold:
                    regression_detected = True
        
        # Determine overall status
        status = TestStatus.FAILED if regression_detected else TestStatus.PASSED
        
        # Create result
        result = TestResult(
            test_id=test_config.test_id,
            test_type=TestType.REGRESSION,
            status=status,
            start_time=datetime.now().isoformat(),
            end_time="",  # Will be set by caller
            duration_seconds=0,  # Will be set by caller
            model_combinations=test_config.model_combinations,
            results_summary={
                'regression_detected': regression_detected,
                'combinations_tested': len(test_config.model_combinations),
                'test_cases_executed': len(test_data),
                'baseline_comparison': baseline_comparison
            },
            performance_metrics={
                combo: results['performance_report'].overall_score 
                for combo, results in results_by_combination.items()
            },
            detailed_results=results_by_combination
        )
        
        return result
    
    async def _execute_performance_test(self, test_config: TestConfiguration) -> TestResult:
        """Execute performance benchmarking test"""
        
        # Load test data
        test_data = await self._load_test_data(test_config.test_data_path)
        
        performance_results = {}
        
        for combination in test_config.model_combinations:
            self.logger.info(f"Performance testing models: {combination}")
            
            # Performance metrics tracking
            processing_times = []
            success_rates = []
            task_counts = []
            confidence_scores = []
            
            # Run performance test cases
            for i, test_case in enumerate(test_data[:50]):  # More cases for performance testing
                start_time = time.time()
                
                try:
                    model_responses = await self.orchestrator.run_parallel(
                        prompt=test_case.get('prompt', 'Generate construction work scope'),
                        json_data=test_case.get('data', {}),
                        model_names=combination
                    )
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    if model_responses:
                        merged_result = self.merger.merge_results(model_responses)
                        success_rates.append(1.0)
                        task_counts.append(merged_result.total_work_items)
                        confidence_scores.append(merged_result.overall_confidence)
                    else:
                        success_rates.append(0.0)
                        task_counts.append(0)
                        confidence_scores.append(0.0)
                
                except Exception as e:
                    processing_times.append(time.time() - start_time)
                    success_rates.append(0.0)
                    task_counts.append(0)
                    confidence_scores.append(0.0)
                    self.logger.warning(f"Performance test case {i} failed: {e}")
            
            # Calculate performance metrics
            performance_results['+'.join(combination)] = {
                'avg_processing_time': np.mean(processing_times),
                'max_processing_time': np.max(processing_times),
                'min_processing_time': np.min(processing_times),
                'processing_time_std': np.std(processing_times),
                'success_rate': np.mean(success_rates),
                'avg_task_count': np.mean(task_counts),
                'avg_confidence': np.mean(confidence_scores),
                'throughput_tasks_per_second': np.mean(task_counts) / np.mean(processing_times) if np.mean(processing_times) > 0 else 0
            }
        
        # Check performance thresholds
        performance_passed = True
        threshold_failures = []
        
        if test_config.performance_thresholds:
            for combo, metrics in performance_results.items():
                for threshold_name, threshold_value in test_config.performance_thresholds.items():
                    if threshold_name in metrics:
                        if threshold_name.endswith('_time') or threshold_name == 'processing_time':
                            # Lower is better for time metrics
                            if metrics[threshold_name] > threshold_value:
                                performance_passed = False
                                threshold_failures.append(f"{combo}: {threshold_name} exceeded ({metrics[threshold_name]:.2f} > {threshold_value})")
                        else:
                            # Higher is better for other metrics
                            if metrics[threshold_name] < threshold_value:
                                performance_passed = False
                                threshold_failures.append(f"{combo}: {threshold_name} below threshold ({metrics[threshold_name]:.2f} < {threshold_value})")
        
        status = TestStatus.PASSED if performance_passed else TestStatus.FAILED
        
        result = TestResult(
            test_id=test_config.test_id,
            test_type=TestType.PERFORMANCE,
            status=status,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_seconds=0,
            model_combinations=test_config.model_combinations,
            results_summary={
                'performance_passed': performance_passed,
                'threshold_failures': threshold_failures,
                'combinations_tested': len(test_config.model_combinations)
            },
            performance_metrics={
                combo: metrics['avg_processing_time'] 
                for combo, metrics in performance_results.items()
            },
            detailed_results=performance_results
        )
        
        return result
    
    async def _execute_ab_test(self, test_config: TestConfiguration) -> TestResult:
        """Execute A/B comparison test"""
        
        # Load test data
        test_data = await self._load_test_data(test_config.test_data_path)
        
        # Run A/B test using the framework
        ab_results = await self.ab_framework.compare_model_combinations(
            test_data=test_data,
            combinations_to_test=test_config.model_combinations,
            test_name=f"automated_ab_{test_config.test_id}"
        )
        
        # Determine status based on statistical significance
        significant_results = [
            result for result in ab_results['ab_test_results'] 
            if result.get('statistical_significance', False)
        ]
        
        status = TestStatus.PASSED if significant_results else TestStatus.FAILED
        
        # Extract key metrics
        best_combination = ab_results['rankings'][0] if ab_results['rankings'] else None
        
        result = TestResult(
            test_id=test_config.test_id,
            test_type=TestType.A_B_COMPARISON,
            status=status,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_seconds=0,
            model_combinations=test_config.model_combinations,
            results_summary={
                'best_combination': best_combination['combination'] if best_combination else None,
                'best_score': best_combination['composite_score'] if best_combination else 0,
                'significant_comparisons': len(significant_results),
                'total_comparisons': len(ab_results['ab_test_results'])
            },
            performance_metrics={
                combo['combination']: combo['composite_score'] 
                for combo in ab_results['rankings']
            },
            detailed_results=ab_results
        )
        
        return result
    
    async def _execute_stress_test(self, test_config: TestConfiguration) -> TestResult:
        """Execute stress test with high load"""
        
        # Load test data
        test_data = await self._load_test_data(test_config.test_data_path)
        
        stress_results = {}
        
        # Stress test parameters
        concurrent_requests = [1, 5, 10, 20]  # Gradually increase load
        requests_per_level = 20
        
        for combination in test_config.model_combinations:
            combo_name = '+'.join(combination)
            stress_results[combo_name] = {}
            
            for concurrency in concurrent_requests:
                self.logger.info(f"Stress testing {combo_name} with {concurrency} concurrent requests")
                
                # Prepare test tasks
                test_tasks = []
                for i in range(min(requests_per_level, len(test_data))):
                    test_case = test_data[i]
                    task = self._run_single_estimation(
                        combination, 
                        test_case.get('prompt', 'Generate work scope'),
                        test_case.get('data', {})
                    )
                    test_tasks.append(task)
                
                # Execute with limited concurrency
                start_time = time.time()
                
                # Use semaphore to limit concurrency
                semaphore = asyncio.Semaphore(concurrency)
                
                async def limited_task(task):
                    async with semaphore:
                        return await task
                
                limited_tasks = [limited_task(task) for task in test_tasks]
                results = await asyncio.gather(*limited_tasks, return_exceptions=True)
                
                end_time = time.time()
                
                # Analyze results
                successful_results = [r for r in results if not isinstance(r, Exception)]
                failed_results = [r for r in results if isinstance(r, Exception)]
                
                stress_results[combo_name][f'concurrency_{concurrency}'] = {
                    'total_requests': len(test_tasks),
                    'successful_requests': len(successful_results),
                    'failed_requests': len(failed_results),
                    'success_rate': len(successful_results) / len(test_tasks) if test_tasks else 0,
                    'total_time': end_time - start_time,
                    'requests_per_second': len(test_tasks) / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                    'avg_response_time': (end_time - start_time) / len(test_tasks) if test_tasks else 0
                }
                
                # Break if system is failing under load
                if len(failed_results) > len(successful_results):
                    self.logger.warning(f"High failure rate at concurrency {concurrency}, stopping stress test")
                    break
        
        # Determine overall status
        overall_success = True
        for combo_results in stress_results.values():
            for level_results in combo_results.values():
                if level_results['success_rate'] < 0.8:  # 80% success rate threshold
                    overall_success = False
                    break
        
        status = TestStatus.PASSED if overall_success else TestStatus.FAILED
        
        result = TestResult(
            test_id=test_config.test_id,
            test_type=TestType.STRESS_TEST,
            status=status,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_seconds=0,
            model_combinations=test_config.model_combinations,
            results_summary={
                'max_successful_concurrency': self._get_max_successful_concurrency(stress_results),
                'overall_system_stable': overall_success,
                'combinations_tested': len(test_config.model_combinations)
            },
            performance_metrics={
                combo: self._calculate_stress_score(results)
                for combo, results in stress_results.items()
            },
            detailed_results=stress_results
        )
        
        return result
    
    async def _execute_smoke_test(self, test_config: TestConfiguration) -> TestResult:
        """Execute basic smoke test for system health"""
        
        # Simple test data for smoke testing
        smoke_test_data = [
            {
                'prompt': 'Generate work scope for basic bedroom renovation',
                'data': {
                    'name': 'Test Bedroom',
                    'materials': {'Paint': 'Existing'},
                    'work_scope': {'Paint': 'Remove & Replace'},
                    'measurements': {'width': 12, 'length': 14, 'height': 9}
                }
            }
        ]
        
        smoke_results = {}
        
        for combination in test_config.model_combinations:
            combo_name = '+'.join(combination)
            
            try:
                # Quick test
                start_time = time.time()
                
                model_responses = await self.orchestrator.run_parallel(
                    prompt=smoke_test_data[0]['prompt'],
                    json_data=smoke_test_data[0]['data'],
                    model_names=combination
                )
                
                processing_time = time.time() - start_time
                
                success = bool(model_responses and len(model_responses) > 0)
                
                if success:
                    merged_result = self.merger.merge_results(model_responses)
                    task_count = merged_result.total_work_items
                    confidence = merged_result.overall_confidence
                else:
                    task_count = 0
                    confidence = 0
                
                smoke_results[combo_name] = {
                    'success': success,
                    'processing_time': processing_time,
                    'task_count': task_count,
                    'confidence': confidence,
                    'models_responded': len(model_responses) if model_responses else 0,
                    'error': None
                }
                
            except Exception as e:
                smoke_results[combo_name] = {
                    'success': False,
                    'processing_time': 0,
                    'task_count': 0,
                    'confidence': 0,
                    'models_responded': 0,
                    'error': str(e)
                }
        
        # Determine status
        all_successful = all(result['success'] for result in smoke_results.values())
        status = TestStatus.PASSED if all_successful else TestStatus.FAILED
        
        result = TestResult(
            test_id=test_config.test_id,
            test_type=TestType.SMOKE_TEST,
            status=status,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_seconds=0,
            model_combinations=test_config.model_combinations,
            results_summary={
                'all_combinations_working': all_successful,
                'successful_combinations': sum(1 for r in smoke_results.values() if r['success']),
                'total_combinations': len(smoke_results)
            },
            performance_metrics={
                combo: 1.0 if results['success'] else 0.0
                for combo, results in smoke_results.items()
            },
            detailed_results=smoke_results
        )
        
        return result
    
    async def _execute_benchmark_test(self, test_config: TestConfiguration) -> TestResult:
        """Execute benchmark validation test"""
        
        # Generate benchmark data if not provided
        if not test_config.test_data_path:
            benchmark_data, _ = await self._generate_benchmark_data()
        else:
            benchmark_data = await self._load_test_data(test_config.test_data_path)
        
        benchmark_results = {}
        
        for combination in test_config.model_combinations:
            combo_name = '+'.join(combination)
            
            # Run benchmark tests
            test_results = []
            ground_truth_data = []
            
            for test_case in benchmark_data[:30]:  # Limit for benchmark testing
                try:
                    model_responses = await self.orchestrator.run_parallel(
                        prompt=test_case.get('prompt', 'Generate work scope'),
                        json_data=test_case.get('input_data', {}).get('data', {}),
                        model_names=combination
                    )
                    
                    merged_result = self.merger.merge_results(model_responses)
                    
                    test_result = {
                        'success': True,
                        'total_work_items': merged_result.total_work_items,
                        'confidence_score': merged_result.overall_confidence,
                        'consensus_level': merged_result.metadata.consensus_level,
                        'processing_time': sum(r.processing_time for r in model_responses if hasattr(r, 'processing_time')),
                        'models_used': combination,
                        'models_responded': len(model_responses)
                    }
                    
                    test_results.append(test_result)
                    
                    # Add ground truth data if available
                    if 'ground_truth' in test_case:
                        ground_truth_data.append(test_case['ground_truth'])
                
                except Exception as e:
                    test_results.append({
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate comprehensive metrics
            performance_report = self.metrics_calculator.calculate_comprehensive_metrics(
                test_results, 
                ground_truth_data if ground_truth_data else None, 
                combination
            )
            
            benchmark_results[combo_name] = {
                'performance_report': performance_report,
                'meets_industry_standards': performance_report.overall_score >= 0.75,
                'category_scores': {
                    category.value: score 
                    for category, score in performance_report.category_scores.items()
                }
            }
        
        # Determine overall status
        all_meet_standards = all(
            results['meets_industry_standards'] 
            for results in benchmark_results.values()
        )
        
        status = TestStatus.PASSED if all_meet_standards else TestStatus.FAILED
        
        result = TestResult(
            test_id=test_config.test_id,
            test_type=TestType.BENCHMARK_VALIDATION,
            status=status,
            start_time=datetime.now().isoformat(),
            end_time="",
            duration_seconds=0,
            model_combinations=test_config.model_combinations,
            results_summary={
                'combinations_meeting_standards': sum(
                    1 for r in benchmark_results.values() if r['meets_industry_standards']
                ),
                'total_combinations': len(benchmark_results),
                'all_meet_standards': all_meet_standards
            },
            performance_metrics={
                combo: results['performance_report'].overall_score
                for combo, results in benchmark_results.items()
            },
            detailed_results=benchmark_results
        )
        
        return result
    
    async def _run_single_estimation(self, models: List[str], prompt: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single estimation for stress testing"""
        
        try:
            model_responses = await self.orchestrator.run_parallel(
                prompt=prompt,
                json_data=data,
                model_names=models
            )
            
            if model_responses:
                merged_result = self.merger.merge_results(model_responses)
                return {
                    'success': True,
                    'total_work_items': merged_result.total_work_items,
                    'confidence_score': merged_result.overall_confidence,
                    'models_responded': len(model_responses)
                }
            else:
                return {'success': False, 'error': 'No model responses'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_max_successful_concurrency(self, stress_results: Dict[str, Any]) -> Dict[str, int]:
        """Get maximum successful concurrency level for each model combination"""
        
        max_concurrency = {}
        
        for combo, results in stress_results.items():
            max_level = 0
            for level_key, level_results in results.items():
                if level_results['success_rate'] >= 0.8:  # 80% threshold
                    concurrency = int(level_key.split('_')[1])
                    max_level = max(max_level, concurrency)
            
            max_concurrency[combo] = max_level
        
        return max_concurrency
    
    def _calculate_stress_score(self, combo_results: Dict[str, Any]) -> float:
        """Calculate stress test score for a combination"""
        
        total_score = 0
        level_count = 0
        
        for level_results in combo_results.values():
            # Weight by success rate and throughput
            success_rate = level_results['success_rate']
            throughput = level_results.get('requests_per_second', 0)
            
            # Normalize throughput (assume 10 req/s is excellent)
            normalized_throughput = min(1.0, throughput / 10.0)
            
            level_score = (success_rate * 0.7) + (normalized_throughput * 0.3)
            total_score += level_score
            level_count += 1
        
        return total_score / level_count if level_count > 0 else 0
    
    async def _load_test_data(self, test_data_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load test data from file or generate default"""
        
        if test_data_path and Path(test_data_path).exists():
            with open(test_data_path, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, dict) and 'test_cases' in data:
                    return data['test_cases']
                elif isinstance(data, list):
                    return data
                else:
                    return [data]
        else:
            # Generate default test data
            return await self._generate_default_test_data()
    
    async def _generate_default_test_data(self) -> List[Dict[str, Any]]:
        """Generate default test data for automated testing"""
        
        default_cases = [
            {
                'prompt': 'Generate work scope for bedroom renovation',
                'data': {
                    'name': 'Master Bedroom',
                    'materials': {
                        'Paint - Walls': 'Existing',
                        'Paint - Ceiling': 'Existing',
                        'Carpet': 'Existing',
                        'Baseboards': 'Existing'
                    },
                    'work_scope': {
                        'Paint - Walls': 'Remove & Replace',
                        'Paint - Ceiling': 'Remove & Replace',
                        'Carpet': 'Remove & Replace',
                        'Baseboards': 'Remove & Replace'
                    },
                    'measurements': {
                        'width': 14,
                        'length': 16,
                        'height': 9,
                        'area_sqft': 224
                    }
                },
                'expected_task_count': 10
            },
            {
                'prompt': 'Generate work scope for kitchen renovation',
                'data': {
                    'name': 'Main Kitchen',
                    'materials': {
                        'Cabinets - Upper': 'Existing',
                        'Cabinets - Lower': 'Existing',
                        'Countertop': 'Granite',
                        'Flooring': 'Tile',
                        'Paint - Walls': 'Existing'
                    },
                    'work_scope': {
                        'Cabinets - Upper': 'Remove & Replace',
                        'Cabinets - Lower': 'Remove & Replace',
                        'Countertop': 'Remove & Replace',
                        'Flooring': 'Remove & Replace',
                        'Paint - Walls': 'Remove & Replace'
                    },
                    'measurements': {
                        'width': 12,
                        'length': 16,
                        'height': 10,
                        'area_sqft': 192
                    }
                },
                'expected_task_count': 15
            }
        ]
        
        return default_cases
    
    async def _generate_benchmark_data(self) -> Tuple[List[Dict[str, Any]], str]:
        """Generate benchmark test data using the benchmark generator"""
        
        # Generate a small benchmark dataset for testing
        dataset = self.benchmark_generator.generate_comprehensive_dataset(
            num_cases_per_category=5
        )
        
        # Convert to test format
        test_data = []
        for case in dataset:
            test_data.append({
                'test_id': case.test_id,
                'prompt': f"Generate work scope for {case.description}",
                'input_data': case.input_data,
                'ground_truth': case.ground_truth,
                'expected_task_count': len(case.ground_truth.expected_tasks)
            })
        
        # Save for reference
        output_path = self.results_dir / f"generated_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)
        
        return test_data, str(output_path)
    
    def _create_skipped_result(self, test_config: TestConfiguration) -> TestResult:
        """Create a skipped test result"""
        
        return TestResult(
            test_id=test_config.test_id,
            test_type=test_config.test_type,
            status=TestStatus.SKIPPED,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_seconds=0,
            model_combinations=test_config.model_combinations,
            results_summary={'reason': 'Test disabled'},
            performance_metrics=None
        )
    
    async def _save_test_result(self, result: TestResult):
        """Save test result to file"""
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.test_id}_{result.test_type.value}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to dictionary
        result_dict = asdict(result)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Add to artifacts list
        if result.artifacts is None:
            result.artifacts = []
        result.artifacts.append(str(filepath))
        
        self.logger.info(f"Test result saved to {filepath}")
    
    def _update_performance_history(self, result: TestResult):
        """Update performance history for trend analysis"""
        
        history_entry = {
            'timestamp': result.start_time,
            'test_id': result.test_id,
            'test_type': result.test_type.value,
            'status': result.status.value,
            'performance_metrics': result.performance_metrics or {},
            'model_combinations': result.model_combinations
        }
        
        self.performance_history.append(history_entry)
        
        # Keep only recent history (configurable retention)
        retention_days = self.config.get('retention_policy', {}).get('keep_results_days', 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        self.performance_history = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) > cutoff_date
        ]
    
    async def _check_alert_conditions(self, result: TestResult):
        """Check if alert conditions are met and send notifications"""
        
        alert_settings = self.config.get('alert_settings', {})
        
        if not alert_settings or not result.test_config.alert_on_failure:
            return
        
        # Check for failure conditions
        should_alert = False
        alert_message = ""
        
        if result.status == TestStatus.FAILED:
            should_alert = True
            alert_message = f"Test {result.test_id} failed: {result.error_message or 'See detailed results'}"
        
        elif result.status == TestStatus.ERROR:
            should_alert = True
            alert_message = f"Test {result.test_id} encountered an error: {result.error_message}"
        
        # Check for performance degradation
        elif result.test_type == TestType.REGRESSION:
            regression_detected = result.results_summary.get('regression_detected', False)
            if regression_detected:
                should_alert = True
                alert_message = f"Performance regression detected in test {result.test_id}"
        
        if should_alert:
            await self._send_alert(alert_message, result, alert_settings)
    
    async def _send_alert(self, message: str, result: TestResult, alert_settings: Dict[str, Any]):
        """Send alert notification"""
        
        self.logger.warning(f"ALERT: {message}")
        
        # Log alert (always done)
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'test_id': result.test_id,
            'message': message,
            'result_status': result.status.value,
            'test_type': result.test_type.value
        }
        
        # Save alert log
        alert_log_path = self.results_dir / "alerts.json"
        
        if alert_log_path.exists():
            with open(alert_log_path, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []
        
        alerts.append(alert_entry)
        
        with open(alert_log_path, 'w') as f:
            json.dump(alerts, f, indent=2, default=str)
        
        # Additional notification methods can be implemented here
        # - Email notifications
        # - Slack/Teams webhooks
        # - SMS alerts
        # - Dashboard updates
    
    def start_scheduler(self):
        """Start the automated test scheduler"""
        
        if self.scheduler_running:
            self.logger.warning("Scheduler already running")
            return
        
        self.scheduler_running = True
        self.logger.info("Starting automated test scheduler")
        
        # Run scheduler in background
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        # Start scheduler thread
        from threading import Thread
        scheduler_thread = Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the automated test scheduler"""
        
        self.scheduler_running = False
        schedule.clear()
        self.logger.info("Automated test scheduler stopped")
    
    def get_test_status_summary(self) -> Dict[str, Any]:
        """Get summary of current test status"""
        
        recent_results = [
            result for result in self.test_history[-50:]  # Last 50 results
        ]
        
        status_counts = {}
        for status in TestStatus:
            status_counts[status.value] = sum(
                1 for result in recent_results 
                if result.status == status
            )
        
        return {
            'total_tests_configured': len(self.test_configurations),
            'enabled_tests': sum(1 for config in self.test_configurations.values() if config.enabled),
            'currently_running': len(self.active_tests),
            'recent_results': status_counts,
            'scheduler_running': self.scheduler_running,
            'last_test_time': max([result.start_time for result in recent_results]) if recent_results else None
        }
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends over specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_history = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) > cutoff_date
        ]
        
        if not recent_history:
            return {'message': 'No recent performance data available'}
        
        # Group by model combination
        trends_by_combo = {}
        
        for entry in recent_history:
            for combo, score in entry.get('performance_metrics', {}).items():
                if combo not in trends_by_combo:
                    trends_by_combo[combo] = []
                
                trends_by_combo[combo].append({
                    'timestamp': entry['timestamp'],
                    'score': score,
                    'test_type': entry['test_type']
                })
        
        # Calculate trends
        trend_analysis = {}
        
        for combo, data_points in trends_by_combo.items():
            if len(data_points) >= 2:
                scores = [dp['score'] for dp in data_points]
                
                # Simple linear trend
                x = list(range(len(scores)))
                trend_slope = np.polyfit(x, scores, 1)[0] if len(scores) > 1 else 0
                
                trend_analysis[combo] = {
                    'current_score': scores[-1],
                    'avg_score': np.mean(scores),
                    'trend_slope': trend_slope,
                    'trend_direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
                    'data_points': len(scores),
                    'score_variance': np.var(scores)
                }
        
        return {
            'analysis_period_days': days,
            'combinations_tracked': len(trends_by_combo),
            'total_data_points': len(recent_history),
            'trends': trend_analysis
        }


# Factory functions for common test configurations
def create_regression_test_config(test_id: str, 
                                 model_combinations: List[List[str]], 
                                 schedule: str = 'daily') -> TestConfiguration:
    """Create a regression test configuration"""
    
    return TestConfiguration(
        test_id=test_id,
        test_type=TestType.REGRESSION,
        test_name=f"Regression Test - {test_id}",
        description="Automated regression testing to detect performance degradation",
        model_combinations=model_combinations,
        test_data_path=None,  # Will use default data
        schedule_cron=schedule,
        timeout_minutes=30,
        retry_count=2,
        alert_on_failure=True,
        tags=['regression', 'automated']
    )


def create_performance_test_config(test_id: str, 
                                 model_combinations: List[List[str]],
                                 performance_thresholds: Dict[str, float],
                                 schedule: str = 'weekly') -> TestConfiguration:
    """Create a performance test configuration"""
    
    return TestConfiguration(
        test_id=test_id,
        test_type=TestType.PERFORMANCE,
        test_name=f"Performance Test - {test_id}",
        description="Automated performance benchmarking",
        model_combinations=model_combinations,
        test_data_path=None,
        schedule_cron=schedule,
        timeout_minutes=60,
        retry_count=1,
        alert_on_failure=True,
        performance_thresholds=performance_thresholds,
        tags=['performance', 'benchmarking', 'automated']
    )


def create_smoke_test_config(test_id: str, 
                           model_combinations: List[List[str]], 
                           schedule: str = 'hourly') -> TestConfiguration:
    """Create a smoke test configuration"""
    
    return TestConfiguration(
        test_id=test_id,
        test_type=TestType.SMOKE_TEST,
        test_name=f"Smoke Test - {test_id}",
        description="Basic health check for system availability",
        model_combinations=model_combinations,
        test_data_path=None,
        schedule_cron=schedule,
        timeout_minutes=10,
        retry_count=3,
        alert_on_failure=True,
        tags=['smoke', 'health_check', 'automated']
    )


# Main pipeline initialization
async def initialize_automated_pipeline(config_path: Optional[str] = None) -> AutomatedTestingPipeline:
    """Initialize and configure automated testing pipeline"""
    
    pipeline = AutomatedTestingPipeline(config_path)
    
    # Register default test configurations
    default_model_combinations = [
        ['gpt4'],
        ['claude'], 
        ['gemini'],
        ['gpt4', 'claude'],
        ['gpt4', 'claude', 'gemini']
    ]
    
    # Smoke test (runs every hour)
    smoke_config = create_smoke_test_config(
        'system_health_check',
        default_model_combinations,
        'hourly'
    )
    pipeline.register_test(smoke_config)
    
    # Regression test (runs daily)
    regression_config = create_regression_test_config(
        'daily_regression',
        default_model_combinations,
        'daily'
    )
    pipeline.register_test(regression_config)
    
    # Performance test (runs weekly)
    performance_thresholds = {
        'avg_processing_time': 30.0,  # Max 30 seconds
        'success_rate': 0.95,         # Min 95% success rate
        'avg_confidence': 0.80        # Min 80% confidence
    }
    
    performance_config = create_performance_test_config(
        'weekly_performance',
        default_model_combinations,
        performance_thresholds,
        'weekly'
    )
    pipeline.register_test(performance_config)
    
    return pipeline