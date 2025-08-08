"""
Performance Analyzer for Model Combinations
Analyzes and compares performance metrics across different model combinations.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path
import json

from .combination_tester import CombinationTestResult
from .test_matrix import TestConfiguration, ModelType, ValidationMode, ProcessingMode


logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a single model or combination"""
    model_combination: str
    configuration: TestConfiguration
    
    # Success metrics
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_consensus_level: float = 0.0
    avg_validation_score: float = 0.0
    avg_quality_score: float = 0.0
    
    # Performance metrics  
    avg_execution_time: float = 0.0
    avg_processing_time_per_model: float = 0.0
    total_work_items_avg: float = 0.0
    unique_rooms_avg: float = 0.0
    
    # Reliability metrics
    model_response_rate: float = 0.0
    error_rate: float = 0.0
    warning_count: int = 0
    
    # Cost efficiency (estimated)
    estimated_cost_per_test: float = 0.0
    cost_efficiency_ratio: float = 0.0  # quality / cost
    
    # Statistical measures
    confidence_std: float = 0.0
    execution_time_std: float = 0.0
    quality_score_std: float = 0.0
    
    # Sample size
    total_tests: int = 0
    successful_tests: int = 0
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.avg_quality_score > 0 and self.estimated_cost_per_test > 0:
            self.cost_efficiency_ratio = self.avg_quality_score / self.estimated_cost_per_test


class PerformanceAnalyzer:
    """Analyzes performance metrics for model combination test results"""
    
    def __init__(self, cost_per_model_call: Optional[Dict[str, float]] = None):
        """Initialize performance analyzer
        
        Args:
            cost_per_model_call: Estimated cost per API call for each model type
        """
        # Default estimated costs (in USD per API call)
        self.cost_per_model_call = cost_per_model_call or {
            "gpt4": 0.06,      # GPT-4 estimated cost
            "claude": 0.05,    # Claude-3 estimated cost  
            "gemini": 0.02     # Gemini Pro estimated cost
        }
    
    def analyze_single_configuration(self, 
                                   results: List[CombinationTestResult]) -> ModelPerformanceMetrics:
        """Analyze performance for a single configuration across multiple runs
        
        Args:
            results: List of test results for the same configuration
            
        Returns:
            Performance metrics for this configuration
        """
        if not results:
            raise ValueError("No results provided for analysis")
        
        # Validate all results are for same configuration
        config = results[0].configuration
        if not all(r.configuration.test_name == config.test_name for r in results):
            raise ValueError("All results must be for the same configuration")
        
        successful_results = [r for r in results if r.success]
        
        metrics = ModelPerformanceMetrics(
            model_combination=" + ".join(config.model_names),
            configuration=config,
            total_tests=len(results),
            successful_tests=len(successful_results)
        )
        
        if successful_results:
            # Success metrics
            metrics.success_rate = len(successful_results) / len(results)
            metrics.avg_confidence = np.mean([r.overall_confidence for r in successful_results])
            metrics.avg_consensus_level = np.mean([r.consensus_level for r in successful_results])
            metrics.avg_validation_score = np.mean([r.validation_score for r in successful_results])
            metrics.avg_quality_score = np.mean([r.quality_score for r in successful_results])
            
            # Performance metrics
            metrics.avg_execution_time = np.mean([r.execution_time for r in successful_results])
            metrics.avg_processing_time_per_model = np.mean([r.average_processing_time for r in successful_results])
            metrics.total_work_items_avg = np.mean([r.total_work_items for r in successful_results])
            metrics.unique_rooms_avg = np.mean([r.unique_rooms_processed for r in successful_results])
            
            # Reliability metrics
            metrics.model_response_rate = np.mean([r.model_success_rate for r in successful_results])
            metrics.warning_count = sum(len(r.warnings) for r in successful_results)
            
            # Statistical measures
            if len(successful_results) > 1:
                metrics.confidence_std = np.std([r.overall_confidence for r in successful_results])
                metrics.execution_time_std = np.std([r.execution_time for r in successful_results])
                metrics.quality_score_std = np.std([r.quality_score for r in successful_results])
        
        # Error rate
        metrics.error_rate = (len(results) - len(successful_results)) / len(results)
        
        # Cost estimation
        metrics.estimated_cost_per_test = self._estimate_cost_per_test(config)
        
        return metrics
    
    def analyze_multiple_configurations(self, 
                                      results_by_config: Dict[str, List[CombinationTestResult]]) -> List[ModelPerformanceMetrics]:
        """Analyze performance for multiple configurations
        
        Args:
            results_by_config: Dictionary mapping config names to lists of results
            
        Returns:
            List of performance metrics for each configuration
        """
        metrics_list = []
        
        for config_name, results in results_by_config.items():
            try:
                metrics = self.analyze_single_configuration(results)
                metrics_list.append(metrics)
            except Exception as e:
                logger.error(f"Failed to analyze configuration {config_name}: {e}")
        
        return metrics_list
    
    def compare_model_types(self, 
                          metrics_list: List[ModelPerformanceMetrics]) -> Dict[str, Dict[str, float]]:
        """Compare performance across different model types
        
        Args:
            metrics_list: List of performance metrics to compare
            
        Returns:
            Comparison dictionary with metrics by model type
        """
        model_type_metrics = {}
        
        for metrics in metrics_list:
            models = metrics.configuration.model_names
            
            # Categorize by model type pattern
            if len(models) == 1:
                category = f"single_{models[0]}"
            elif len(models) == 2:
                category = f"pair_{'+'.join(sorted(models))}"
            elif len(models) == 3:
                category = "triple_all"
            else:
                category = f"multi_{len(models)}_models"
            
            if category not in model_type_metrics:
                model_type_metrics[category] = []
            
            model_type_metrics[category].append(metrics)
        
        # Calculate averages for each category
        comparison = {}
        for category, metrics_group in model_type_metrics.items():
            comparison[category] = {
                "avg_quality_score": np.mean([m.avg_quality_score for m in metrics_group]),
                "avg_confidence": np.mean([m.avg_confidence for m in metrics_group]),
                "avg_execution_time": np.mean([m.avg_execution_time for m in metrics_group]),
                "avg_success_rate": np.mean([m.success_rate for m in metrics_group]),
                "avg_consensus_level": np.mean([m.avg_consensus_level for m in metrics_group]),
                "avg_cost_per_test": np.mean([m.estimated_cost_per_test for m in metrics_group]),
                "avg_cost_efficiency": np.mean([m.cost_efficiency_ratio for m in metrics_group]),
                "sample_size": len(metrics_group)
            }
        
        return comparison
    
    def find_optimal_configurations(self, 
                                  metrics_list: List[ModelPerformanceMetrics],
                                  optimization_criteria: str = "quality") -> List[Tuple[ModelPerformanceMetrics, float]]:
        """Find optimal configurations based on criteria
        
        Args:
            metrics_list: List of performance metrics
            optimization_criteria: Optimization criteria ("quality", "speed", "cost_efficiency", "reliability")
            
        Returns:
            List of (metrics, score) tuples sorted by optimization score
        """
        scored_metrics = []
        
        for metrics in metrics_list:
            if optimization_criteria == "quality":
                score = metrics.avg_quality_score
            elif optimization_criteria == "speed":
                # Lower execution time is better, so invert
                score = 1.0 / (metrics.avg_execution_time + 0.01) if metrics.avg_execution_time > 0 else 0
            elif optimization_criteria == "cost_efficiency":
                score = metrics.cost_efficiency_ratio
            elif optimization_criteria == "reliability":
                # Composite reliability score
                score = (metrics.success_rate * 0.4 + 
                        metrics.model_response_rate * 0.3 +
                        (1.0 - metrics.error_rate) * 0.3)
            elif optimization_criteria == "consensus":
                score = metrics.avg_consensus_level
            else:
                raise ValueError(f"Unknown optimization criteria: {optimization_criteria}")
            
            scored_metrics.append((metrics, score))
        
        # Sort by score (highest first)
        scored_metrics.sort(key=lambda x: x[1], reverse=True)
        
        return scored_metrics
    
    def generate_performance_report(self, 
                                  metrics_list: List[ModelPerformanceMetrics],
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report
        
        Args:
            metrics_list: List of performance metrics
            output_path: Optional path to save report as JSON
            
        Returns:
            Performance report dictionary
        """
        if not metrics_list:
            return {"error": "No metrics provided"}
        
        # Overall statistics
        overall_stats = {
            "total_configurations": len(metrics_list),
            "avg_success_rate": np.mean([m.success_rate for m in metrics_list]),
            "avg_quality_score": np.mean([m.avg_quality_score for m in metrics_list]),
            "avg_execution_time": np.mean([m.avg_execution_time for m in metrics_list]),
            "avg_confidence": np.mean([m.avg_confidence for m in metrics_list]),
            "total_tests_run": sum(m.total_tests for m in metrics_list)
        }
        
        # Model type comparison
        model_comparison = self.compare_model_types(metrics_list)
        
        # Top performers by different criteria
        top_performers = {
            "quality": self.find_optimal_configurations(metrics_list, "quality")[:5],
            "speed": self.find_optimal_configurations(metrics_list, "speed")[:5],
            "cost_efficiency": self.find_optimal_configurations(metrics_list, "cost_efficiency")[:5],
            "reliability": self.find_optimal_configurations(metrics_list, "reliability")[:5]
        }
        
        # Convert to serializable format
        top_performers_serializable = {}
        for criteria, performers in top_performers.items():
            top_performers_serializable[criteria] = [
                {
                    "model_combination": metrics.model_combination,
                    "configuration_name": metrics.configuration.test_name,
                    "score": score,
                    "quality_score": metrics.avg_quality_score,
                    "execution_time": metrics.avg_execution_time,
                    "success_rate": metrics.success_rate
                }
                for metrics, score in performers
            ]
        
        # Detailed metrics for each configuration
        detailed_metrics = []
        for metrics in metrics_list:
            detailed_metrics.append({
                "model_combination": metrics.model_combination,
                "configuration_name": metrics.configuration.test_name,
                "success_rate": metrics.success_rate,
                "avg_quality_score": metrics.avg_quality_score,
                "avg_confidence": metrics.avg_confidence,
                "avg_execution_time": metrics.avg_execution_time,
                "avg_consensus_level": metrics.avg_consensus_level,
                "estimated_cost_per_test": metrics.estimated_cost_per_test,
                "cost_efficiency_ratio": metrics.cost_efficiency_ratio,
                "total_tests": metrics.total_tests,
                "successful_tests": metrics.successful_tests
            })
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "overall_statistics": overall_stats,
            "model_type_comparison": model_comparison,
            "top_performers": top_performers_serializable,
            "detailed_metrics": detailed_metrics,
            "recommendations": self._generate_recommendations(metrics_list)
        }
        
        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Performance report saved to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def _estimate_cost_per_test(self, config: TestConfiguration) -> float:
        """Estimate cost per test for a configuration"""
        total_cost = 0.0
        
        for model_name in config.model_names:
            if model_name in self.cost_per_model_call:
                total_cost += self.cost_per_model_call[model_name]
        
        # Add small processing overhead
        total_cost *= 1.1
        
        return total_cost
    
    def _generate_recommendations(self, metrics_list: List[ModelPerformanceMetrics]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not metrics_list:
            return ["No data available for recommendations"]
        
        # Find best overall performer
        best_quality = max(metrics_list, key=lambda m: m.avg_quality_score)
        recommendations.append(
            f"Best overall quality: {best_quality.model_combination} "
            f"(quality score: {best_quality.avg_quality_score:.3f})"
        )
        
        # Find fastest performer
        fastest = min([m for m in metrics_list if m.avg_execution_time > 0], 
                     key=lambda m: m.avg_execution_time, default=None)
        if fastest:
            recommendations.append(
                f"Fastest execution: {fastest.model_combination} "
                f"({fastest.avg_execution_time:.2f}s avg)"
            )
        
        # Find most cost-effective
        most_efficient = max([m for m in metrics_list if m.cost_efficiency_ratio > 0], 
                           key=lambda m: m.cost_efficiency_ratio, default=None)
        if most_efficient:
            recommendations.append(
                f"Most cost-efficient: {most_efficient.model_combination} "
                f"(ratio: {most_efficient.cost_efficiency_ratio:.3f})"
            )
        
        # Compare single vs multi-model
        single_models = [m for m in metrics_list if len(m.configuration.model_names) == 1]
        multi_models = [m for m in metrics_list if len(m.configuration.model_names) > 1]
        
        if single_models and multi_models:
            avg_single_quality = np.mean([m.avg_quality_score for m in single_models])
            avg_multi_quality = np.mean([m.avg_quality_score for m in multi_models])
            
            if avg_multi_quality > avg_single_quality:
                improvement = ((avg_multi_quality - avg_single_quality) / avg_single_quality) * 100
                recommendations.append(
                    f"Multi-model combinations improve quality by {improvement:.1f}% "
                    f"over single models on average"
                )
            else:
                recommendations.append(
                    "Single models perform competitively with multi-model combinations"
                )
        
        # Reliability recommendation
        reliable_configs = [m for m in metrics_list if m.success_rate >= 0.9 and m.error_rate <= 0.1]
        if reliable_configs:
            best_reliable = max(reliable_configs, key=lambda m: m.avg_quality_score)
            recommendations.append(
                f"Most reliable high-quality option: {best_reliable.model_combination} "
                f"({best_reliable.success_rate:.1%} success rate)"
            )
        
        return recommendations
    
    def export_to_csv(self, 
                     metrics_list: List[ModelPerformanceMetrics], 
                     output_path: str) -> None:
        """Export metrics to CSV for external analysis"""
        data = []
        
        for metrics in metrics_list:
            row = {
                "model_combination": metrics.model_combination,
                "configuration_name": metrics.configuration.test_name,
                "validation_mode": metrics.configuration.validation_mode.value,
                "processing_mode": metrics.configuration.processing_mode.value,
                "success_rate": metrics.success_rate,
                "avg_quality_score": metrics.avg_quality_score,
                "avg_confidence": metrics.avg_confidence,
                "avg_consensus_level": metrics.avg_consensus_level,
                "avg_validation_score": metrics.avg_validation_score,
                "avg_execution_time": metrics.avg_execution_time,
                "avg_processing_time_per_model": metrics.avg_processing_time_per_model,
                "model_response_rate": metrics.model_response_rate,
                "error_rate": metrics.error_rate,
                "estimated_cost_per_test": metrics.estimated_cost_per_test,
                "cost_efficiency_ratio": metrics.cost_efficiency_ratio,
                "total_tests": metrics.total_tests,
                "successful_tests": metrics.successful_tests
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Metrics exported to CSV: {output_path}")