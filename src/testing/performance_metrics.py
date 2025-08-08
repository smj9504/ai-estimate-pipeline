# src/testing/performance_metrics.py
"""
Comprehensive Performance Metrics and KPIs for Construction Estimation
Defines domain-specific metrics for evaluating multi-model AI performance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from src.utils.logger import get_logger


class MetricCategory(Enum):
    """Categories of performance metrics"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness" 
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    COST_EFFECTIVENESS = "cost_effectiveness"
    BUSINESS_VALUE = "business_value"


class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 75-89%
    ACCEPTABLE = "acceptable"  # 60-74%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40-59%
    POOR = "poor"          # 0-39%


@dataclass
class MetricResult:
    """Result of a single metric calculation"""
    name: str
    category: MetricCategory
    value: float
    max_value: float
    percentage: float
    performance_level: PerformanceLevel
    description: str
    calculation_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PerformanceReport:
    """Comprehensive performance report"""
    test_id: str
    model_combination: List[str]
    overall_score: float
    metrics: Dict[str, MetricResult]
    category_scores: Dict[MetricCategory, float]
    recommendations: List[str]
    timestamp: str
    execution_metadata: Dict[str, Any]
    comparison_benchmarks: Dict[str, float] = field(default_factory=dict)


class ConstructionEstimationMetrics:
    """
    Comprehensive metrics calculator for construction estimation AI systems
    """
    
    def __init__(self):
        self.logger = get_logger('performance_metrics')
        
        # Industry standard benchmarks (DMV area construction)
        self.industry_benchmarks = {
            'cost_accuracy_threshold': 0.85,  # Within 15% of actual costs
            'task_completeness_threshold': 0.90,  # 90% of required tasks identified
            'consensus_threshold': 0.75,  # 75% model agreement
            'processing_time_threshold': 30.0,  # Max 30 seconds per estimation
            'error_rate_threshold': 0.05,  # Max 5% error rate
            'confidence_threshold': 0.80,  # Min 80% confidence
            'remove_replace_accuracy': 0.95  # 95% accuracy in Remove & Replace logic
        }
        
        # Metric weights for overall score calculation
        self.metric_weights = {
            MetricCategory.ACCURACY: 0.30,
            MetricCategory.COMPLETENESS: 0.25,
            MetricCategory.CONSISTENCY: 0.15,
            MetricCategory.EFFICIENCY: 0.15,
            MetricCategory.RELIABILITY: 0.10,
            MetricCategory.COST_EFFECTIVENESS: 0.05
        }
    
    def calculate_comprehensive_metrics(self, 
                                      test_results: List[Dict[str, Any]], 
                                      ground_truth_data: Optional[List[Dict[str, Any]]] = None,
                                      model_combination: List[str] = None) -> PerformanceReport:
        """
        Calculate comprehensive performance metrics for construction estimation
        
        Args:
            test_results: List of test case results from AI models
            ground_truth_data: Optional ground truth data for validation
            model_combination: Models used in this test
        
        Returns:
            Comprehensive performance report
        """
        self.logger.info(f"Calculating performance metrics for {len(test_results)} test results")
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Category 1: Accuracy Metrics
        accuracy_metrics = self._calculate_accuracy_metrics(test_results, ground_truth_data)
        metrics.update(accuracy_metrics)
        
        # Category 2: Completeness Metrics
        completeness_metrics = self._calculate_completeness_metrics(test_results, ground_truth_data)
        metrics.update(completeness_metrics)
        
        # Category 3: Consistency Metrics  
        consistency_metrics = self._calculate_consistency_metrics(test_results)
        metrics.update(consistency_metrics)
        
        # Category 4: Efficiency Metrics
        efficiency_metrics = self._calculate_efficiency_metrics(test_results)
        metrics.update(efficiency_metrics)
        
        # Category 5: Reliability Metrics
        reliability_metrics = self._calculate_reliability_metrics(test_results)
        metrics.update(reliability_metrics)
        
        # Category 6: Cost Effectiveness Metrics
        cost_effectiveness_metrics = self._calculate_cost_effectiveness_metrics(test_results, ground_truth_data)
        metrics.update(cost_effectiveness_metrics)
        
        # Calculate category scores
        category_scores = self._calculate_category_scores(metrics)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, category_scores)
        
        # Create performance report
        report = PerformanceReport(
            test_id=f"perf_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_combination=model_combination or [],
            overall_score=overall_score,
            metrics=metrics,
            category_scores=category_scores,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            execution_metadata={
                'test_cases_evaluated': len(test_results),
                'ground_truth_available': bool(ground_truth_data),
                'benchmark_standards': self.industry_benchmarks
            }
        )
        
        self.logger.info(f"Performance evaluation complete. Overall score: {overall_score:.2f}")
        return report
    
    def _calculate_accuracy_metrics(self, 
                                  test_results: List[Dict[str, Any]], 
                                  ground_truth_data: Optional[List[Dict[str, Any]]]) -> Dict[str, MetricResult]:
        """Calculate accuracy-related metrics"""
        
        metrics = {}
        
        # 1. Cost Estimation Accuracy
        if ground_truth_data:
            cost_accuracies = []
            for i, result in enumerate(test_results):
                if i < len(ground_truth_data):
                    actual_cost = ground_truth_data[i].get('expected_cost_range', [0, 0])
                    estimated_cost = result.get('estimated_cost', 0)
                    
                    if actual_cost[0] > 0 and estimated_cost > 0:
                        # Calculate accuracy as 1 - (error / expected)
                        avg_expected = (actual_cost[0] + actual_cost[1]) / 2
                        error_rate = abs(estimated_cost - avg_expected) / avg_expected
                        accuracy = max(0, 1 - error_rate)
                        cost_accuracies.append(accuracy)
            
            avg_cost_accuracy = np.mean(cost_accuracies) if cost_accuracies else 0
            
            metrics['cost_estimation_accuracy'] = MetricResult(
                name='Cost Estimation Accuracy',
                category=MetricCategory.ACCURACY,
                value=avg_cost_accuracy,
                max_value=1.0,
                percentage=avg_cost_accuracy * 100,
                performance_level=self._get_performance_level(avg_cost_accuracy * 100),
                description='Accuracy of cost estimates compared to ground truth',
                calculation_method='1 - (|estimated - actual| / actual)',
                metadata={'sample_size': len(cost_accuracies), 'benchmark': self.industry_benchmarks['cost_accuracy_threshold']}
            )
        
        # 2. Task Identification Accuracy  
        task_accuracies = []
        for result in test_results:
            total_tasks = result.get('total_work_items', 0)
            expected_tasks = result.get('expected_task_count', 10)  # Default expectation
            
            if expected_tasks > 0:
                accuracy = min(1.0, total_tasks / expected_tasks)
                task_accuracies.append(accuracy)
        
        avg_task_accuracy = np.mean(task_accuracies) if task_accuracies else 0
        
        metrics['task_identification_accuracy'] = MetricResult(
            name='Task Identification Accuracy',
            category=MetricCategory.ACCURACY,
            value=avg_task_accuracy,
            max_value=1.0,
            percentage=avg_task_accuracy * 100,
            performance_level=self._get_performance_level(avg_task_accuracy * 100),
            description='Accuracy of identifying required construction tasks',
            calculation_method='min(1.0, identified_tasks / expected_tasks)',
            metadata={'sample_size': len(task_accuracies)}
        )
        
        # 3. Remove & Replace Logic Accuracy
        remove_replace_accuracies = []
        for result in test_results:
            validation_result = result.get('validation', {})
            rr_logic = validation_result.get('remove_replace_logic', {})
            
            if rr_logic:
                accuracy = 1.0 if rr_logic.get('valid', False) else 0.0
                remove_replace_accuracies.append(accuracy)
        
        avg_rr_accuracy = np.mean(remove_replace_accuracies) if remove_replace_accuracies else 0
        
        metrics['remove_replace_logic_accuracy'] = MetricResult(
            name='Remove & Replace Logic Accuracy', 
            category=MetricCategory.ACCURACY,
            value=avg_rr_accuracy,
            max_value=1.0,
            percentage=avg_rr_accuracy * 100,
            performance_level=self._get_performance_level(avg_rr_accuracy * 100),
            description='Accuracy of applying Remove & Replace business logic',
            calculation_method='Percentage of cases with valid Remove & Replace logic',
            metadata={
                'sample_size': len(remove_replace_accuracies), 
                'benchmark': self.industry_benchmarks['remove_replace_accuracy']
            }
        )
        
        # 4. Measurement Utilization Accuracy
        measurement_accuracies = []
        for result in test_results:
            validation_result = result.get('validation', {})
            measurement_accuracy = validation_result.get('measurements_accuracy', {})
            
            if measurement_accuracy:
                accuracy = 1.0 if measurement_accuracy.get('valid', False) else 0.0
                measurement_accuracies.append(accuracy)
        
        avg_measurement_accuracy = np.mean(measurement_accuracies) if measurement_accuracies else 0
        
        metrics['measurement_utilization_accuracy'] = MetricResult(
            name='Measurement Utilization Accuracy',
            category=MetricCategory.ACCURACY, 
            value=avg_measurement_accuracy,
            max_value=1.0,
            percentage=avg_measurement_accuracy * 100,
            performance_level=self._get_performance_level(avg_measurement_accuracy * 100),
            description='Accuracy of using provided measurements in calculations',
            calculation_method='Percentage of cases with valid measurement usage',
            metadata={'sample_size': len(measurement_accuracies)}
        )
        
        return metrics
    
    def _calculate_completeness_metrics(self, 
                                      test_results: List[Dict[str, Any]], 
                                      ground_truth_data: Optional[List[Dict[str, Any]]]) -> Dict[str, MetricResult]:
        """Calculate completeness-related metrics"""
        
        metrics = {}
        
        # 1. Task Coverage Completeness
        task_completeness_scores = []
        for result in test_results:
            total_tasks = result.get('total_work_items', 0)
            expected_minimum = result.get('expected_task_count', 10)
            
            if expected_minimum > 0:
                completeness = min(1.0, total_tasks / expected_minimum)
                task_completeness_scores.append(completeness)
        
        avg_task_completeness = np.mean(task_completeness_scores) if task_completeness_scores else 0
        
        metrics['task_coverage_completeness'] = MetricResult(
            name='Task Coverage Completeness',
            category=MetricCategory.COMPLETENESS,
            value=avg_task_completeness,
            max_value=1.0,
            percentage=avg_task_completeness * 100,
            performance_level=self._get_performance_level(avg_task_completeness * 100),
            description='Completeness of task identification coverage',
            calculation_method='min(1.0, identified_tasks / minimum_expected_tasks)',
            metadata={
                'sample_size': len(task_completeness_scores),
                'benchmark': self.industry_benchmarks['task_completeness_threshold']
            }
        )
        
        # 2. Work Scope Coverage
        work_scope_completeness = []
        for result in test_results:
            # Check if all major work categories are covered
            rooms = result.get('data', {}).get('rooms', [])
            if not rooms and 'rooms' in result.get('data', {}):
                rooms = result['data']['rooms']
            
            coverage_score = 0
            for room in rooms:
                tasks = room.get('tasks', [])
                task_types = set()
                
                for task in tasks:
                    task_type = task.get('task_type', '').lower()
                    if task_type:
                        task_types.add(task_type)
                
                # Essential task types for construction
                required_types = {'removal', 'installation', 'preparation'}
                covered_types = required_types.intersection(task_types)
                
                if required_types:
                    room_coverage = len(covered_types) / len(required_types)
                    coverage_score += room_coverage
            
            if rooms:
                work_scope_completeness.append(coverage_score / len(rooms))
        
        avg_work_scope_completeness = np.mean(work_scope_completeness) if work_scope_completeness else 0
        
        metrics['work_scope_coverage'] = MetricResult(
            name='Work Scope Coverage',
            category=MetricCategory.COMPLETENESS,
            value=avg_work_scope_completeness,
            max_value=1.0,
            percentage=avg_work_scope_completeness * 100,
            performance_level=self._get_performance_level(avg_work_scope_completeness * 100),
            description='Coverage of essential work scope categories',
            calculation_method='Average coverage of removal/installation/preparation tasks per room',
            metadata={'sample_size': len(work_scope_completeness)}
        )
        
        # 3. Special Requirements Coverage  
        special_requirements_coverage = []
        for result in test_results:
            validation_result = result.get('validation', {})
            special_tasks = validation_result.get('special_tasks', {})
            
            if special_tasks:
                coverage = 1.0 if special_tasks.get('valid', False) else 0.0
                special_requirements_coverage.append(coverage)
        
        avg_special_coverage = np.mean(special_requirements_coverage) if special_requirements_coverage else 0
        
        metrics['special_requirements_coverage'] = MetricResult(
            name='Special Requirements Coverage',
            category=MetricCategory.COMPLETENESS,
            value=avg_special_coverage,
            max_value=1.0,
            percentage=avg_special_coverage * 100,
            performance_level=self._get_performance_level(avg_special_coverage * 100),
            description='Coverage of special construction requirements',
            calculation_method='Percentage of cases with valid special requirements handling',
            metadata={'sample_size': len(special_requirements_coverage)}
        )
        
        return metrics
    
    def _calculate_consistency_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, MetricResult]:
        """Calculate consistency-related metrics"""
        
        metrics = {}
        
        # 1. Model Consensus Level
        consensus_levels = []
        for result in test_results:
            consensus = result.get('consensus_level', 0)
            consensus_levels.append(consensus)
        
        avg_consensus = np.mean(consensus_levels) if consensus_levels else 0
        consensus_std = np.std(consensus_levels) if consensus_levels else 0
        
        metrics['model_consensus_level'] = MetricResult(
            name='Model Consensus Level',
            category=MetricCategory.CONSISTENCY,
            value=avg_consensus,
            max_value=1.0,
            percentage=avg_consensus * 100,
            performance_level=self._get_performance_level(avg_consensus * 100),
            description='Level of agreement between AI models',
            calculation_method='Average consensus level across all test cases',
            metadata={
                'std_deviation': consensus_std,
                'sample_size': len(consensus_levels),
                'benchmark': self.industry_benchmarks['consensus_threshold']
            }
        )
        
        # 2. Output Format Consistency
        format_consistency_scores = []
        for result in test_results:
            # Check if output follows expected format
            score = 1.0
            
            # Check for required fields
            required_fields = ['total_work_items', 'confidence_score', 'data']
            for field in required_fields:
                if field not in result:
                    score -= 0.2
            
            # Check data structure consistency
            data = result.get('data', {})
            if isinstance(data, dict) and 'rooms' in data:
                rooms = data['rooms']
                if isinstance(rooms, list):
                    for room in rooms:
                        if not isinstance(room, dict) or 'tasks' not in room:
                            score -= 0.1
            
            format_consistency_scores.append(max(0, score))
        
        avg_format_consistency = np.mean(format_consistency_scores) if format_consistency_scores else 0
        
        metrics['output_format_consistency'] = MetricResult(
            name='Output Format Consistency',
            category=MetricCategory.CONSISTENCY,
            value=avg_format_consistency,
            max_value=1.0,
            percentage=avg_format_consistency * 100,
            performance_level=self._get_performance_level(avg_format_consistency * 100),
            description='Consistency of output data format across tests',
            calculation_method='Deduction-based scoring for missing/malformed fields',
            metadata={'sample_size': len(format_consistency_scores)}
        )
        
        # 3. Quality Variance
        confidence_scores = [result.get('confidence_score', 0) for result in test_results]
        quality_variance = np.std(confidence_scores) if confidence_scores else 1.0
        
        # Lower variance is better (more consistent quality)
        variance_score = max(0, 1 - (quality_variance * 2))  # Scale variance to 0-1
        
        metrics['quality_variance'] = MetricResult(
            name='Quality Variance',
            category=MetricCategory.CONSISTENCY,
            value=variance_score,
            max_value=1.0,
            percentage=variance_score * 100,
            performance_level=self._get_performance_level(variance_score * 100),
            description='Consistency of output quality across test cases',
            calculation_method='1 - (std_deviation * 2), where lower variance = higher score',
            metadata={
                'quality_std_deviation': quality_variance,
                'sample_size': len(confidence_scores)
            }
        )
        
        return metrics
    
    def _calculate_efficiency_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, MetricResult]:
        """Calculate efficiency-related metrics"""
        
        metrics = {}
        
        # 1. Processing Speed
        processing_times = []
        for result in test_results:
            processing_time = result.get('processing_time', 0)
            if processing_time > 0:
                processing_times.append(processing_time)
        
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Convert to efficiency score (lower time = higher score)
        max_acceptable_time = self.industry_benchmarks['processing_time_threshold']
        efficiency_score = max(0, 1 - (avg_processing_time / max_acceptable_time))
        
        metrics['processing_efficiency'] = MetricResult(
            name='Processing Efficiency',
            category=MetricCategory.EFFICIENCY,
            value=efficiency_score,
            max_value=1.0,
            percentage=efficiency_score * 100,
            performance_level=self._get_performance_level(efficiency_score * 100),
            description='Efficiency of processing time relative to benchmark',
            calculation_method=f'max(0, 1 - (avg_time / {max_acceptable_time}s))',
            metadata={
                'avg_processing_time_seconds': avg_processing_time,
                'benchmark_threshold': max_acceptable_time,
                'sample_size': len(processing_times)
            }
        )
        
        # 2. Resource Utilization Efficiency
        models_responded = []
        models_used = []
        
        for result in test_results:
            responded = result.get('models_responded', 0)
            used = len(result.get('models_used', []))
            
            if used > 0:
                models_responded.append(responded)
                models_used.append(used)
        
        resource_efficiency_scores = []
        for i in range(len(models_responded)):
            if models_used[i] > 0:
                efficiency = models_responded[i] / models_used[i]
                resource_efficiency_scores.append(efficiency)
        
        avg_resource_efficiency = np.mean(resource_efficiency_scores) if resource_efficiency_scores else 0
        
        metrics['resource_utilization_efficiency'] = MetricResult(
            name='Resource Utilization Efficiency',
            category=MetricCategory.EFFICIENCY,
            value=avg_resource_efficiency,
            max_value=1.0,
            percentage=avg_resource_efficiency * 100,
            performance_level=self._get_performance_level(avg_resource_efficiency * 100),
            description='Efficiency of model resource utilization',
            calculation_method='models_responded / models_used',
            metadata={'sample_size': len(resource_efficiency_scores)}
        )
        
        # 3. Output Generation Rate
        output_rates = []
        for result in test_results:
            total_tasks = result.get('total_work_items', 0)
            processing_time = result.get('processing_time', 1)  # Avoid division by zero
            
            if processing_time > 0:
                rate = total_tasks / processing_time  # Tasks per second
                output_rates.append(rate)
        
        avg_output_rate = np.mean(output_rates) if output_rates else 0
        
        # Normalize to 0-1 scale (assume 1 task/second is excellent)
        output_efficiency = min(1.0, avg_output_rate / 1.0)
        
        metrics['output_generation_rate'] = MetricResult(
            name='Output Generation Rate',
            category=MetricCategory.EFFICIENCY,
            value=output_efficiency,
            max_value=1.0,
            percentage=output_efficiency * 100,
            performance_level=self._get_performance_level(output_efficiency * 100),
            description='Rate of output generation (tasks per second)',
            calculation_method='tasks_generated / processing_time_seconds',
            metadata={
                'avg_tasks_per_second': avg_output_rate,
                'sample_size': len(output_rates)
            }
        )
        
        return metrics
    
    def _calculate_reliability_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, MetricResult]:
        """Calculate reliability-related metrics"""
        
        metrics = {}
        
        # 1. Success Rate
        success_count = sum(1 for result in test_results if result.get('success', False))
        total_tests = len(test_results)
        success_rate = success_count / total_tests if total_tests > 0 else 0
        
        metrics['success_rate'] = MetricResult(
            name='Success Rate',
            category=MetricCategory.RELIABILITY,
            value=success_rate,
            max_value=1.0,
            percentage=success_rate * 100,
            performance_level=self._get_performance_level(success_rate * 100),
            description='Percentage of successful test case completions',
            calculation_method='successful_tests / total_tests',
            metadata={
                'successful_tests': success_count,
                'total_tests': total_tests
            }
        )
        
        # 2. Error Rate
        error_count = sum(1 for result in test_results if 'error' in result or not result.get('success', True))
        error_rate = error_count / total_tests if total_tests > 0 else 0
        
        # Convert error rate to reliability score (lower error = higher reliability)
        reliability_score = 1 - error_rate
        
        metrics['error_rate'] = MetricResult(
            name='Error Rate',
            category=MetricCategory.RELIABILITY,
            value=reliability_score,
            max_value=1.0,
            percentage=reliability_score * 100,
            performance_level=self._get_performance_level(reliability_score * 100),
            description='Reliability measured as 1 - error_rate',
            calculation_method='1 - (error_cases / total_cases)',
            metadata={
                'error_cases': error_count,
                'total_cases': total_tests,
                'actual_error_rate': error_rate,
                'benchmark_threshold': self.industry_benchmarks['error_rate_threshold']
            }
        )
        
        # 3. Confidence Consistency
        confidence_scores = [result.get('confidence_score', 0) for result in test_results if 'confidence_score' in result]
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            confidence_std = np.std(confidence_scores)
            
            # Higher average confidence + lower std dev = better consistency
            consistency_score = avg_confidence * (1 - min(1, confidence_std))
        else:
            consistency_score = 0
        
        metrics['confidence_consistency'] = MetricResult(
            name='Confidence Consistency',
            category=MetricCategory.RELIABILITY,
            value=consistency_score,
            max_value=1.0,
            percentage=consistency_score * 100,
            performance_level=self._get_performance_level(consistency_score * 100),
            description='Consistency and level of confidence scores',
            calculation_method='avg_confidence * (1 - min(1, std_confidence))',
            metadata={
                'avg_confidence': avg_confidence if confidence_scores else 0,
                'confidence_std': confidence_std if confidence_scores else 0,
                'sample_size': len(confidence_scores),
                'benchmark': self.industry_benchmarks['confidence_threshold']
            }
        )
        
        return metrics
    
    def _calculate_cost_effectiveness_metrics(self, 
                                            test_results: List[Dict[str, Any]], 
                                            ground_truth_data: Optional[List[Dict[str, Any]]]) -> Dict[str, MetricResult]:
        """Calculate cost-effectiveness metrics"""
        
        metrics = {}
        
        # 1. Processing Cost Efficiency
        processing_times = [result.get('processing_time', 0) for result in test_results if result.get('processing_time', 0) > 0]
        total_tasks_generated = sum(result.get('total_work_items', 0) for result in test_results)
        
        if processing_times and total_tasks_generated > 0:
            total_processing_time = sum(processing_times)
            cost_per_task = total_processing_time / total_tasks_generated  # Time cost per task
            
            # Assume optimal is 1 second per task, scale accordingly
            efficiency = max(0, 1 - (cost_per_task - 1) / 10)  # Penalty for times over 1s/task
            efficiency = max(0, min(1, efficiency))
        else:
            efficiency = 0
        
        metrics['processing_cost_efficiency'] = MetricResult(
            name='Processing Cost Efficiency',
            category=MetricCategory.COST_EFFECTIVENESS,
            value=efficiency,
            max_value=1.0,
            percentage=efficiency * 100,
            performance_level=self._get_performance_level(efficiency * 100),
            description='Efficiency of processing time per generated task',
            calculation_method='Scaled efficiency based on time per task generated',
            metadata={
                'avg_time_per_task': cost_per_task if processing_times else 0,
                'total_tasks': total_tasks_generated,
                'total_time': sum(processing_times) if processing_times else 0
            }
        )
        
        # 2. Model Resource Efficiency
        total_models_used = 0
        total_successful_responses = 0
        
        for result in test_results:
            models_used = len(result.get('models_used', []))
            models_responded = result.get('models_responded', 0)
            
            total_models_used += models_used
            total_successful_responses += models_responded
        
        resource_efficiency = total_successful_responses / total_models_used if total_models_used > 0 else 0
        
        metrics['model_resource_efficiency'] = MetricResult(
            name='Model Resource Efficiency',
            category=MetricCategory.COST_EFFECTIVENESS,
            value=resource_efficiency,
            max_value=1.0,
            percentage=resource_efficiency * 100,
            performance_level=self._get_performance_level(resource_efficiency * 100),
            description='Efficiency of model resource usage',
            calculation_method='successful_responses / total_models_called',
            metadata={
                'total_models_called': total_models_used,
                'successful_responses': total_successful_responses
            }
        )
        
        # 3. Value-to-Effort Ratio
        if ground_truth_data:
            value_scores = []
            
            for i, result in enumerate(test_results):
                if i < len(ground_truth_data):
                    # Calculate value based on accuracy and completeness
                    accuracy = result.get('confidence_score', 0)
                    completeness = min(1.0, result.get('total_work_items', 0) / ground_truth_data[i].get('expected_task_count', 10))
                    
                    value = (accuracy + completeness) / 2
                    
                    # Calculate effort (normalized processing time)
                    effort = min(1.0, result.get('processing_time', 0) / 60)  # Normalize to 1 minute
                    
                    if effort > 0:
                        value_to_effort = value / effort
                        value_scores.append(min(2.0, value_to_effort))  # Cap at 2.0
            
            avg_value_to_effort = np.mean(value_scores) if value_scores else 0
            normalized_vte = min(1.0, avg_value_to_effort / 2.0)  # Normalize to 0-1
        else:
            normalized_vte = 0.5  # Default when no ground truth available
        
        metrics['value_to_effort_ratio'] = MetricResult(
            name='Value-to-Effort Ratio',
            category=MetricCategory.COST_EFFECTIVENESS,
            value=normalized_vte,
            max_value=1.0,
            percentage=normalized_vte * 100,
            performance_level=self._get_performance_level(normalized_vte * 100),
            description='Ratio of value delivered to effort expended',
            calculation_method='(accuracy + completeness) / 2 / normalized_processing_time',
            metadata={
                'ground_truth_available': bool(ground_truth_data),
                'sample_size': len(value_scores) if ground_truth_data else 0
            }
        )
        
        return metrics
    
    def _calculate_category_scores(self, metrics: Dict[str, MetricResult]) -> Dict[MetricCategory, float]:
        """Calculate average scores for each metric category"""
        
        category_scores = {}
        
        for category in MetricCategory:
            category_metrics = [metric for metric in metrics.values() if metric.category == category]
            
            if category_metrics:
                avg_score = np.mean([metric.value for metric in category_metrics])
                category_scores[category] = avg_score
            else:
                category_scores[category] = 0.0
        
        return category_scores
    
    def _calculate_overall_score(self, category_scores: Dict[MetricCategory, float]) -> float:
        """Calculate weighted overall performance score"""
        
        weighted_sum = 0
        total_weight = 0
        
        for category, score in category_scores.items():
            weight = self.metric_weights.get(category, 0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _get_performance_level(self, percentage: float) -> PerformanceLevel:
        """Determine performance level based on percentage score"""
        
        if percentage >= 90:
            return PerformanceLevel.EXCELLENT
        elif percentage >= 75:
            return PerformanceLevel.GOOD
        elif percentage >= 60:
            return PerformanceLevel.ACCEPTABLE
        elif percentage >= 40:
            return PerformanceLevel.NEEDS_IMPROVEMENT
        else:
            return PerformanceLevel.POOR
    
    def _generate_recommendations(self, 
                                metrics: Dict[str, MetricResult], 
                                category_scores: Dict[MetricCategory, float]) -> List[str]:
        """Generate actionable recommendations based on performance analysis"""
        
        recommendations = []
        
        # Identify the lowest performing categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1])
        
        for category, score in sorted_categories[:3]:  # Focus on top 3 issues
            if score < 0.75:  # Below "good" threshold
                if category == MetricCategory.ACCURACY:
                    recommendations.append(
                        "Improve accuracy by enhancing prompt engineering, adding validation steps, "
                        "or fine-tuning models on construction-specific data"
                    )
                elif category == MetricCategory.COMPLETENESS:
                    recommendations.append(
                        "Increase task completeness by expanding prompt requirements, adding checklists, "
                        "or implementing multi-pass generation strategies"
                    )
                elif category == MetricCategory.CONSISTENCY:
                    recommendations.append(
                        "Improve consistency by standardizing output formats, implementing validation schemas, "
                        "or using ensemble methods to reduce variance"
                    )
                elif category == MetricCategory.EFFICIENCY:
                    recommendations.append(
                        "Optimize efficiency by implementing caching, parallel processing, "
                        "or selecting faster model combinations for time-sensitive operations"
                    )
                elif category == MetricCategory.RELIABILITY:
                    recommendations.append(
                        "Enhance reliability by implementing retry logic, error handling improvements, "
                        "and fallback mechanisms for failed model calls"
                    )
        
        # Specific metric-based recommendations
        for metric_name, metric in metrics.items():
            if metric.percentage < 60:  # Needs improvement
                if 'cost_estimation' in metric_name.lower():
                    recommendations.append(
                        "Consider adding cost database integration or regional cost adjustments "
                        "to improve cost estimation accuracy"
                    )
                elif 'remove_replace' in metric_name.lower():
                    recommendations.append(
                        "Review and enhance Remove & Replace business logic implementation "
                        "in prompts and validation rules"
                    )
                elif 'consensus' in metric_name.lower():
                    recommendations.append(
                        "Evaluate model combination strategy - consider using models with "
                        "more complementary strengths or adjusting consensus thresholds"
                    )
        
        # Overall performance recommendations
        overall_score = self._calculate_overall_score(category_scores)
        if overall_score < 0.70:
            recommendations.append(
                "Overall performance is below optimal. Consider comprehensive review of "
                "model selection, prompt engineering, and validation processes"
            )
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def export_performance_report(self, report: PerformanceReport, output_path: str):
        """Export performance report to JSON file"""
        
        # Convert dataclasses to dictionaries for JSON serialization
        report_dict = {
            'test_id': report.test_id,
            'model_combination': report.model_combination,
            'overall_score': report.overall_score,
            'metrics': {},
            'category_scores': {},
            'recommendations': report.recommendations,
            'timestamp': report.timestamp,
            'execution_metadata': report.execution_metadata,
            'comparison_benchmarks': report.comparison_benchmarks
        }
        
        # Convert metrics
        for name, metric in report.metrics.items():
            report_dict['metrics'][name] = {
                'name': metric.name,
                'category': metric.category.value,
                'value': metric.value,
                'max_value': metric.max_value,
                'percentage': metric.percentage,
                'performance_level': metric.performance_level.value,
                'description': metric.description,
                'calculation_method': metric.calculation_method,
                'timestamp': metric.timestamp,
                'metadata': metric.metadata
            }
        
        # Convert category scores
        for category, score in report.category_scores.items():
            report_dict['category_scores'][category.value] = score
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Performance report exported to {output_path}")
    
    def create_performance_dashboard_data(self, report: PerformanceReport) -> Dict[str, Any]:
        """Create data structure optimized for dashboard visualization"""
        
        dashboard_data = {
            'summary': {
                'overall_score': report.overall_score,
                'performance_level': self._get_performance_level(report.overall_score * 100).value,
                'test_id': report.test_id,
                'timestamp': report.timestamp,
                'model_combination': report.model_combination
            },
            'category_breakdown': [],
            'metric_details': [],
            'recommendations': report.recommendations,
            'benchmark_comparison': {},
            'trends': {}  # Placeholder for trend analysis
        }
        
        # Category breakdown for radar chart
        for category, score in report.category_scores.items():
            dashboard_data['category_breakdown'].append({
                'category': category.value.replace('_', ' ').title(),
                'score': score,
                'percentage': score * 100,
                'status': self._get_performance_level(score * 100).value
            })
        
        # Metric details for detailed view
        for name, metric in report.metrics.items():
            dashboard_data['metric_details'].append({
                'name': metric.name,
                'category': metric.category.value,
                'value': metric.value,
                'percentage': metric.percentage,
                'status': metric.performance_level.value,
                'description': metric.description,
                'benchmark_met': metric.percentage >= 75  # Good threshold
            })
        
        # Benchmark comparison
        for metric_name, metric in report.metrics.items():
            if 'benchmark' in metric.metadata:
                benchmark_value = metric.metadata['benchmark']
                dashboard_data['benchmark_comparison'][metric_name] = {
                    'actual': metric.value,
                    'benchmark': benchmark_value,
                    'meets_benchmark': metric.value >= benchmark_value,
                    'gap': metric.value - benchmark_value
                }
        
        return dashboard_data


# Convenience functions for common use cases
def evaluate_model_performance(test_results: List[Dict[str, Any]], 
                             ground_truth: Optional[List[Dict[str, Any]]] = None,
                             models: List[str] = None) -> PerformanceReport:
    """Quick performance evaluation function"""
    
    evaluator = ConstructionEstimationMetrics()
    return evaluator.calculate_comprehensive_metrics(test_results, ground_truth, models)


def benchmark_against_industry_standards(report: PerformanceReport) -> Dict[str, Any]:
    """Compare performance report against industry benchmarks"""
    
    evaluator = ConstructionEstimationMetrics()
    benchmarks = evaluator.industry_benchmarks
    
    comparison = {
        'meets_industry_standards': report.overall_score >= 0.75,
        'benchmark_details': {},
        'improvement_priority': []
    }
    
    for metric_name, metric in report.metrics.items():
        # Find corresponding benchmark
        benchmark_key = None
        for key in benchmarks.keys():
            if key.replace('_', '') in metric_name.replace('_', '').lower():
                benchmark_key = key
                break
        
        if benchmark_key:
            benchmark_value = benchmarks[benchmark_key]
            meets_benchmark = metric.value >= benchmark_value
            
            comparison['benchmark_details'][metric_name] = {
                'actual': metric.value,
                'benchmark': benchmark_value,
                'meets_standard': meets_benchmark,
                'gap': metric.value - benchmark_value
            }
            
            if not meets_benchmark:
                priority = 'high' if metric.value < benchmark_value * 0.8 else 'medium'
                comparison['improvement_priority'].append({
                    'metric': metric_name,
                    'priority': priority,
                    'gap': benchmark_value - metric.value
                })
    
    return comparison