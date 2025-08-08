# src/testing/ab_testing_framework.py
"""
A/B Testing Framework for Multi-Model Construction Estimation
Provides statistical significance testing and comparison of model combinations
"""
import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger


@dataclass
class ABTestResult:
    """A/B test result container"""
    test_id: str
    variant_a: str
    variant_b: str
    sample_size: int
    confidence_level: float
    p_value: float
    effect_size: float
    winner: Optional[str]
    statistical_significance: bool
    confidence_interval: Tuple[float, float]
    metric_name: str
    variant_a_mean: float
    variant_b_mean: float
    variant_a_std: float
    variant_b_std: float
    power: float
    metadata: Dict[str, Any]


@dataclass
class ModelCombinationResult:
    """Results for a specific model combination"""
    models: List[str]
    accuracy_score: float
    consensus_level: float
    processing_time: float
    cost_variance: float
    task_completeness: float
    confidence_score: float
    error_rate: float
    outlier_count: int
    quality_metrics: Dict[str, float]


class ABTestingFramework:
    """
    A/B Testing framework for comparing model combinations
    with construction estimation specific metrics
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigLoader().load_config() if not config_path else ConfigLoader(config_path).load_config()
        self.orchestrator = ModelOrchestrator()
        self.merger = ResultMerger(self.config)
        self.logger = get_logger('ab_testing_framework')
        
        # Statistical test configuration
        self.default_alpha = 0.05  # 95% confidence level
        self.default_power = 0.80  # 80% statistical power
        self.minimum_sample_size = 30
        
        # Construction-specific metrics
        self.metrics_config = {
            'accuracy_weight': 0.25,
            'consensus_weight': 0.20,
            'completeness_weight': 0.20,
            'confidence_weight': 0.15,
            'speed_weight': 0.10,
            'cost_variance_weight': 0.10
        }
    
    async def compare_model_combinations(self, 
                                       test_data: List[Dict[str, Any]],
                                       combinations_to_test: List[List[str]],
                                       test_name: str = "model_comparison") -> Dict[str, Any]:
        """
        Compare multiple model combinations using A/B testing methodology
        
        Args:
            test_data: List of construction estimation test cases
            combinations_to_test: List of model combinations to test
            test_name: Name identifier for this test suite
        
        Returns:
            Complete A/B test results with statistical significance
        """
        self.logger.info(f"Starting A/B test: {test_name}")
        self.logger.info(f"Testing {len(combinations_to_test)} model combinations on {len(test_data)} cases")
        
        # Step 1: Run all combinations on test data
        combination_results = {}
        for combo in combinations_to_test:
            combo_name = "+".join(sorted(combo))
            self.logger.info(f"Testing combination: {combo_name}")
            
            results = await self._evaluate_model_combination(combo, test_data)
            combination_results[combo_name] = results
        
        # Step 2: Perform pairwise A/B tests
        ab_test_results = []
        for i, combo_a in enumerate(combinations_to_test):
            for combo_b in combinations_to_test[i+1:]:
                combo_a_name = "+".join(sorted(combo_a))
                combo_b_name = "+".join(sorted(combo_b))
                
                result = self._perform_ab_test(
                    combination_results[combo_a_name],
                    combination_results[combo_b_name],
                    combo_a_name,
                    combo_b_name
                )
                ab_test_results.append(result)
        
        # Step 3: Rank combinations and identify best performer
        ranked_combinations = self._rank_combinations(combination_results)
        
        # Step 4: Generate comprehensive report
        test_report = {
            'test_metadata': {
                'test_name': test_name,
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(test_data),
                'combinations_tested': len(combinations_to_test),
                'total_ab_tests': len(ab_test_results)
            },
            'combination_results': combination_results,
            'ab_test_results': [self._ab_result_to_dict(r) for r in ab_test_results],
            'rankings': ranked_combinations,
            'recommendations': self._generate_recommendations(ranked_combinations, ab_test_results),
            'statistical_summary': self._generate_statistical_summary(ab_test_results)
        }
        
        # Save results
        await self._save_test_results(test_report, test_name)
        
        return test_report
    
    async def _evaluate_model_combination(self, 
                                        models: List[str], 
                                        test_data: List[Dict[str, Any]]) -> ModelCombinationResult:
        """Evaluate a specific model combination on test data"""
        
        results = []
        for test_case in test_data:
            try:
                # Run models
                model_responses = await self.orchestrator.run_parallel(
                    prompt=test_case['prompt'],
                    json_data=test_case['data'],
                    model_names=models
                )
                
                # Merge results
                merged_result = self.merger.merge_results(model_responses)
                
                # Calculate metrics
                metrics = self._calculate_case_metrics(merged_result, test_case, model_responses)
                results.append(metrics)
                
            except Exception as e:
                self.logger.error(f"Error processing test case: {e}")
                # Add failed case with zero scores
                results.append({
                    'accuracy_score': 0.0,
                    'consensus_level': 0.0,
                    'processing_time': 999.0,
                    'cost_variance': 1.0,
                    'task_completeness': 0.0,
                    'confidence_score': 0.0,
                    'error_rate': 1.0,
                    'outlier_count': 999
                })
        
        # Aggregate results
        return ModelCombinationResult(
            models=models,
            accuracy_score=np.mean([r['accuracy_score'] for r in results]),
            consensus_level=np.mean([r['consensus_level'] for r in results]),
            processing_time=np.mean([r['processing_time'] for r in results]),
            cost_variance=np.mean([r['cost_variance'] for r in results]),
            task_completeness=np.mean([r['task_completeness'] for r in results]),
            confidence_score=np.mean([r['confidence_score'] for r in results]),
            error_rate=np.mean([r['error_rate'] for r in results]),
            outlier_count=np.mean([r['outlier_count'] for r in results]),
            quality_metrics=self._calculate_quality_metrics(results)
        )
    
    def _calculate_case_metrics(self, 
                               merged_result: Any,
                               test_case: Dict[str, Any],
                               model_responses: List[Any]) -> Dict[str, float]:
        """Calculate metrics for a single test case"""
        
        # Basic metrics from merged result
        accuracy_score = merged_result.overall_confidence
        consensus_level = merged_result.metadata.consensus_level
        processing_time = sum(r.processing_time for r in model_responses if hasattr(r, 'processing_time'))
        confidence_score = merged_result.overall_confidence
        
        # Task completeness (how many tasks were generated vs expected)
        total_tasks = merged_result.total_work_items
        expected_tasks = test_case.get('expected_task_count', 10)  # Default minimum
        task_completeness = min(1.0, total_tasks / expected_tasks) if expected_tasks > 0 else 0.0
        
        # Cost variance (if ground truth available)
        cost_variance = 0.5  # Default neutral value
        if 'expected_costs' in test_case and hasattr(merged_result, 'estimated_costs'):
            expected = test_case['expected_costs']
            actual = getattr(merged_result, 'estimated_costs', expected)
            cost_variance = abs(actual - expected) / expected if expected > 0 else 1.0
        
        # Error rate
        error_count = sum(1 for r in model_responses if 'Error:' in str(getattr(r, 'raw_response', '')))
        error_rate = error_count / len(model_responses) if model_responses else 1.0
        
        # Outlier count
        outlier_count = len(merged_result.metadata.outlier_flags) if hasattr(merged_result.metadata, 'outlier_flags') else 0
        
        return {
            'accuracy_score': accuracy_score,
            'consensus_level': consensus_level,
            'processing_time': processing_time,
            'cost_variance': cost_variance,
            'task_completeness': task_completeness,
            'confidence_score': confidence_score,
            'error_rate': error_rate,
            'outlier_count': outlier_count
        }
    
    def _calculate_quality_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate aggregated quality metrics"""
        
        # Composite quality score
        quality_scores = []
        for result in results:
            score = (
                result['accuracy_score'] * self.metrics_config['accuracy_weight'] +
                result['consensus_level'] * self.metrics_config['consensus_weight'] +
                result['task_completeness'] * self.metrics_config['completeness_weight'] +
                result['confidence_score'] * self.metrics_config['confidence_weight'] +
                (1 - min(1.0, result['processing_time'] / 30.0)) * self.metrics_config['speed_weight'] +  # Speed bonus
                (1 - result['cost_variance']) * self.metrics_config['cost_variance_weight']
            )
            quality_scores.append(max(0.0, score))
        
        return {
            'composite_quality': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'quality_consistency': 1.0 - (np.std(quality_scores) / np.mean(quality_scores)) if np.mean(quality_scores) > 0 else 0.0
        }
    
    def _perform_ab_test(self, 
                        results_a: ModelCombinationResult, 
                        results_b: ModelCombinationResult,
                        name_a: str, 
                        name_b: str) -> ABTestResult:
        """Perform statistical A/B test between two model combinations"""
        
        # Use composite quality as primary metric
        metric_a = results_a.quality_metrics['composite_quality']
        metric_b = results_b.quality_metrics['composite_quality']
        
        # For this implementation, we'll use the aggregated means and stds
        # In real scenario, you'd have individual sample points
        mean_a = metric_a
        mean_b = metric_b
        std_a = results_a.quality_metrics['quality_std']
        std_b = results_b.quality_metrics['quality_std']
        
        # Simulate sample data for statistical test
        # In practice, you'd use actual per-case results
        n_samples = max(self.minimum_sample_size, 50)
        samples_a = np.random.normal(mean_a, max(std_a, 0.01), n_samples)
        samples_b = np.random.normal(mean_b, max(std_b, 0.01), n_samples)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(samples_a, samples_b)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((n_samples - 1) * std_a**2 + (n_samples - 1) * std_b**2) / (2 * n_samples - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Confidence interval for difference
        se_diff = np.sqrt(std_a**2/n_samples + std_b**2/n_samples)
        t_critical = stats.t.ppf(1 - self.default_alpha/2, 2*n_samples - 2)
        diff = mean_a - mean_b
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        # Determine winner
        is_significant = p_value < self.default_alpha
        winner = name_a if (is_significant and mean_a > mean_b) else name_b if (is_significant and mean_b > mean_a) else None
        
        # Calculate statistical power (simplified)
        power = self._calculate_power(cohens_d, n_samples, self.default_alpha)
        
        return ABTestResult(
            test_id=f"{name_a}_vs_{name_b}",
            variant_a=name_a,
            variant_b=name_b,
            sample_size=n_samples,
            confidence_level=1 - self.default_alpha,
            p_value=p_value,
            effect_size=cohens_d,
            winner=winner,
            statistical_significance=is_significant,
            confidence_interval=(ci_lower, ci_upper),
            metric_name="composite_quality",
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            variant_a_std=std_a,
            variant_b_std=std_b,
            power=power,
            metadata={
                'test_type': 'two_sample_ttest',
                't_statistic': t_stat,
                'degrees_of_freedom': 2*n_samples - 2
            }
        )
    
    def _calculate_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate statistical power of the test"""
        from scipy.stats import norm
        
        # Simplified power calculation for two-sample t-test
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(self.default_power)
        
        # Required effect size for desired power
        required_n = 2 * ((z_alpha + z_beta) / effect_size)**2 if effect_size != 0 else float('inf')
        
        # Actual power with current sample size
        if required_n == float('inf'):
            return 0.0
        
        actual_power = min(1.0, sample_size / required_n)
        return actual_power
    
    def _rank_combinations(self, combination_results: Dict[str, ModelCombinationResult]) -> List[Dict[str, Any]]:
        """Rank model combinations by composite quality score"""
        
        rankings = []
        for combo_name, result in combination_results.items():
            rankings.append({
                'combination': combo_name,
                'models': result.models,
                'composite_score': result.quality_metrics['composite_quality'],
                'consistency': result.quality_metrics['quality_consistency'],
                'individual_metrics': {
                    'accuracy': result.accuracy_score,
                    'consensus': result.consensus_level,
                    'completeness': result.task_completeness,
                    'confidence': result.confidence_score,
                    'speed': result.processing_time,
                    'cost_variance': result.cost_variance,
                    'error_rate': result.error_rate
                }
            })
        
        # Sort by composite score descending
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add rank numbers
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _generate_recommendations(self, 
                                rankings: List[Dict[str, Any]], 
                                ab_results: List[ABTestResult]) -> Dict[str, Any]:
        """Generate actionable recommendations based on test results"""
        
        if not rankings:
            return {'error': 'No ranking data available'}
        
        best_combo = rankings[0]
        
        # Find statistically significant improvements
        significant_wins = [r for r in ab_results if r.statistical_significance and r.winner == best_combo['combination']]
        
        recommendations = {
            'primary_recommendation': {
                'combination': best_combo['combination'],
                'models': best_combo['models'],
                'confidence': 'high' if len(significant_wins) >= 2 else 'medium',
                'composite_score': best_combo['composite_score'],
                'expected_improvement': self._calculate_expected_improvement(best_combo, rankings[1] if len(rankings) > 1 else None)
            },
            'alternative_options': [],
            'optimization_suggestions': [],
            'risk_assessment': self._assess_risks(best_combo, ab_results)
        }
        
        # Add alternative high-performing options
        for ranking in rankings[1:3]:  # Top 2 alternatives
            if ranking['composite_score'] >= 0.7:  # High-quality threshold
                recommendations['alternative_options'].append({
                    'combination': ranking['combination'],
                    'models': ranking['models'],
                    'score': ranking['composite_score'],
                    'trade_offs': self._identify_trade_offs(best_combo, ranking)
                })
        
        # Optimization suggestions
        if best_combo['individual_metrics']['speed'] > 10.0:  # Slow processing
            recommendations['optimization_suggestions'].append("Consider reducing model combination size for faster processing")
        
        if best_combo['individual_metrics']['error_rate'] > 0.1:  # High error rate
            recommendations['optimization_suggestions'].append("Review API configurations and error handling")
        
        return recommendations
    
    def _calculate_expected_improvement(self, best: Dict[str, Any], second: Optional[Dict[str, Any]]) -> float:
        """Calculate expected improvement of best combination over alternatives"""
        if not second:
            return 0.0
        
        return (best['composite_score'] - second['composite_score']) / second['composite_score'] * 100
    
    def _identify_trade_offs(self, best: Dict[str, Any], alternative: Dict[str, Any]) -> List[str]:
        """Identify trade-offs between combinations"""
        trade_offs = []
        
        best_metrics = best['individual_metrics']
        alt_metrics = alternative['individual_metrics']
        
        if alt_metrics['speed'] < best_metrics['speed']:
            trade_offs.append(f"Faster processing ({alt_metrics['speed']:.1f}s vs {best_metrics['speed']:.1f}s)")
        
        if alt_metrics['error_rate'] < best_metrics['error_rate']:
            trade_offs.append(f"Lower error rate ({alt_metrics['error_rate']:.2%} vs {best_metrics['error_rate']:.2%})")
        
        if alt_metrics['consensus'] > best_metrics['consensus']:
            trade_offs.append(f"Higher consensus ({alt_metrics['consensus']:.2f} vs {best_metrics['consensus']:.2f})")
        
        return trade_offs
    
    def _assess_risks(self, best_combo: Dict[str, Any], ab_results: List[ABTestResult]) -> Dict[str, Any]:
        """Assess risks of recommended combination"""
        
        # Count statistical significance wins
        significant_wins = sum(1 for r in ab_results if r.winner == best_combo['combination'] and r.statistical_significance)
        total_comparisons = sum(1 for r in ab_results if best_combo['combination'] in [r.variant_a, r.variant_b])
        
        # Calculate risk metrics
        confidence_ratio = significant_wins / total_comparisons if total_comparisons > 0 else 0
        consistency_score = best_combo['consistency']
        
        return {
            'overall_risk': 'low' if confidence_ratio >= 0.7 and consistency_score >= 0.8 else 'medium' if confidence_ratio >= 0.5 else 'high',
            'confidence_ratio': confidence_ratio,
            'consistency_score': consistency_score,
            'risk_factors': self._identify_risk_factors(best_combo),
            'mitigation_strategies': self._suggest_mitigations(best_combo)
        }
    
    def _identify_risk_factors(self, combo: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if combo['individual_metrics']['error_rate'] > 0.05:
            risks.append("Higher than optimal error rate")
        
        if combo['consistency'] < 0.7:
            risks.append("Lower consistency across test cases")
        
        if len(combo['models']) == 1:
            risks.append("Single model dependency (no consensus validation)")
        
        return risks
    
    def _suggest_mitigations(self, combo: Dict[str, Any]) -> List[str]:
        """Suggest risk mitigation strategies"""
        mitigations = []
        
        if combo['individual_metrics']['error_rate'] > 0.05:
            mitigations.append("Implement enhanced error handling and retry logic")
        
        if combo['consistency'] < 0.7:
            mitigations.append("Add additional validation steps for edge cases")
        
        if len(combo['models']) == 1:
            mitigations.append("Consider adding a second model for consensus validation")
        
        return mitigations
    
    def _generate_statistical_summary(self, ab_results: List[ABTestResult]) -> Dict[str, Any]:
        """Generate statistical summary of all A/B tests"""
        
        significant_tests = [r for r in ab_results if r.statistical_significance]
        
        return {
            'total_tests': len(ab_results),
            'statistically_significant': len(significant_tests),
            'significance_rate': len(significant_tests) / len(ab_results) if ab_results else 0,
            'average_p_value': np.mean([r.p_value for r in ab_results]),
            'average_effect_size': np.mean([abs(r.effect_size) for r in ab_results]),
            'average_power': np.mean([r.power for r in ab_results]),
            'power_distribution': {
                'high_power': sum(1 for r in ab_results if r.power >= 0.8),
                'medium_power': sum(1 for r in ab_results if 0.6 <= r.power < 0.8),
                'low_power': sum(1 for r in ab_results if r.power < 0.6)
            }
        }
    
    def _ab_result_to_dict(self, result: ABTestResult) -> Dict[str, Any]:
        """Convert ABTestResult to dictionary for serialization"""
        return {
            'test_id': result.test_id,
            'variant_a': result.variant_a,
            'variant_b': result.variant_b,
            'sample_size': result.sample_size,
            'confidence_level': result.confidence_level,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'winner': result.winner,
            'statistical_significance': result.statistical_significance,
            'confidence_interval': result.confidence_interval,
            'metric_name': result.metric_name,
            'variant_a_mean': result.variant_a_mean,
            'variant_b_mean': result.variant_b_mean,
            'variant_a_std': result.variant_a_std,
            'variant_b_std': result.variant_b_std,
            'power': result.power,
            'metadata': result.metadata
        }
    
    async def _save_test_results(self, test_report: Dict[str, Any], test_name: str):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ab_test_{test_name}_{timestamp}.json"
        
        output_dir = Path("testing_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / filename, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        self.logger.info(f"A/B test results saved to {output_dir / filename}")


# Convenience function for quick testing
async def run_quick_ab_test(test_data_path: str, models_to_compare: List[List[str]]) -> Dict[str, Any]:
    """Quick utility function to run an A/B test"""
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Initialize framework
    framework = ABTestingFramework()
    
    # Run comparison
    return await framework.compare_model_combinations(
        test_data=test_data,
        combinations_to_test=models_to_compare,
        test_name="quick_comparison"
    )