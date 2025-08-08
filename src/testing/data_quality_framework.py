# src/testing/data_quality_framework.py
"""
Data Quality Framework for AI Estimation Pipeline
Implements comprehensive data quality monitoring and validation
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import json
import hashlib
from enum import Enum

from src.utils.logger import get_logger
from src.utils.statistical_utils import StatisticalAnalyzer

logger = get_logger(__name__)


class QualityDimension(Enum):
    """Data quality dimensions based on DAMA framework"""
    COMPLETENESS = "completeness"       # Missing/null values
    ACCURACY = "accuracy"               # Correctness vs ground truth
    CONSISTENCY = "consistency"         # Internal consistency rules
    VALIDITY = "validity"               # Schema/format compliance
    UNIQUENESS = "uniqueness"           # Duplicate detection
    TIMELINESS = "timeliness"           # Data freshness
    RELIABILITY = "reliability"         # Processing success rate


@dataclass
class QualityMetric:
    """Individual quality metric measurement"""
    dimension: QualityDimension
    score: float                        # 0.0 to 1.0
    threshold: float                    # Minimum acceptable score
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    measurement_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment"""
    dataset_id: str
    processing_phase: str
    overall_score: float
    passed_quality_gate: bool
    metrics: List[QualityMetric]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'dataset_id': self.dataset_id,
            'processing_phase': self.processing_phase,
            'overall_score': self.overall_score,
            'passed_quality_gate': self.passed_quality_gate,
            'metrics': [
                {
                    'dimension': m.dimension.value,
                    'score': m.score,
                    'threshold': m.threshold,
                    'passed': m.passed,
                    'details': m.details
                }
                for m in self.metrics
            ],
            'recommendations': self.recommendations,
            'generated_at': self.generated_at.isoformat()
        }


class DataQualityFramework:
    """
    Comprehensive data quality monitoring for AI estimation pipeline
    
    Key Features:
    - Multi-dimensional quality assessment
    - Configurable quality gates
    - Trend analysis for quality degradation detection
    - Automated recommendations for quality improvement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.quality_thresholds = self._load_quality_thresholds()
        self.stats_analyzer = StatisticalAnalyzer()
        self.quality_history = {}
        
    def _load_quality_thresholds(self) -> Dict[QualityDimension, float]:
        """Load quality thresholds from configuration"""
        default_thresholds = {
            QualityDimension.COMPLETENESS: 0.95,    # 95% complete data
            QualityDimension.ACCURACY: 0.85,        # 85% accuracy vs ground truth
            QualityDimension.CONSISTENCY: 0.90,     # 90% consistency
            QualityDimension.VALIDITY: 0.98,        # 98% valid format
            QualityDimension.UNIQUENESS: 0.99,      # 99% unique records
            QualityDimension.TIMELINESS: 0.90,      # 90% fresh data
            QualityDimension.RELIABILITY: 0.95      # 95% processing success
        }
        
        config_thresholds = self.config.get('quality_thresholds', {})
        for dimension_name, threshold in config_thresholds.items():
            if hasattr(QualityDimension, dimension_name.upper()):
                dimension = QualityDimension(dimension_name.lower())
                default_thresholds[dimension] = threshold
        
        return default_thresholds
    
    async def assess_data_quality(self,
                                  dataset: List[Dict[str, Any]],
                                  phase: str,
                                  ground_truth: Optional[List[Dict]] = None) -> DataQualityReport:
        """
        Comprehensive data quality assessment
        
        Args:
            dataset: Dataset to assess
            phase: Processing phase (phase1, phase2, etc.)
            ground_truth: Reference data for accuracy assessment
        """
        logger.info(f"Assessing data quality for {len(dataset)} records in {phase}")
        
        # Generate dataset identifier
        dataset_content = json.dumps(dataset, sort_keys=True)
        dataset_id = hashlib.md5(dataset_content.encode()).hexdigest()[:12]
        
        metrics = []
        
        # 1. Completeness Assessment
        completeness_metric = await self._assess_completeness(dataset)
        metrics.append(completeness_metric)
        
        # 2. Validity Assessment
        validity_metric = await self._assess_validity(dataset, phase)
        metrics.append(validity_metric)
        
        # 3. Consistency Assessment
        consistency_metric = await self._assess_consistency(dataset, phase)
        metrics.append(consistency_metric)
        
        # 4. Uniqueness Assessment
        uniqueness_metric = await self._assess_uniqueness(dataset)
        metrics.append(uniqueness_metric)
        
        # 5. Accuracy Assessment (if ground truth available)
        if ground_truth:
            accuracy_metric = await self._assess_accuracy(dataset, ground_truth, phase)
            metrics.append(accuracy_metric)
        
        # 6. Reliability Assessment (from processing metadata)
        reliability_metric = await self._assess_reliability(dataset)
        metrics.append(reliability_metric)
        
        # Calculate overall quality score
        overall_score = np.mean([m.score for m in metrics])
        passed_quality_gate = all(m.passed for m in metrics) and overall_score >= 0.80
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(metrics, dataset, phase)
        
        # Create quality report
        report = DataQualityReport(
            dataset_id=dataset_id,
            processing_phase=phase,
            overall_score=overall_score,
            passed_quality_gate=passed_quality_gate,
            metrics=metrics,
            recommendations=recommendations
        )
        
        # Store quality history for trend analysis
        await self._store_quality_history(report)
        
        logger.info(f"Quality assessment complete: Overall score {overall_score:.3f}, Gate passed: {passed_quality_gate}")
        
        return report
    
    async def _assess_completeness(self, dataset: List[Dict]) -> QualityMetric:
        """Assess data completeness (missing values)"""
        if not dataset:
            return QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                score=0.0,
                threshold=self.quality_thresholds[QualityDimension.COMPLETENESS],
                passed=False,
                details={'error': 'Empty dataset'}
            )
        
        total_fields = 0
        missing_fields = 0
        field_completeness = {}
        
        for record in dataset:
            for field, value in record.items():
                if not field.startswith('_'):  # Skip metadata fields
                    total_fields += 1
                    
                    if field not in field_completeness:
                        field_completeness[field] = {'total': 0, 'missing': 0}
                    
                    field_completeness[field]['total'] += 1
                    
                    if value is None or value == "" or value == {}:
                        missing_fields += 1
                        field_completeness[field]['missing'] += 1
        
        completeness_score = 1.0 - (missing_fields / total_fields) if total_fields > 0 else 0.0
        threshold = self.quality_thresholds[QualityDimension.COMPLETENESS]
        
        # Calculate field-level completeness
        field_scores = {}
        for field, stats in field_completeness.items():
            field_scores[field] = 1.0 - (stats['missing'] / stats['total'])
        
        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            threshold=threshold,
            passed=completeness_score >= threshold,
            details={
                'total_fields': total_fields,
                'missing_fields': missing_fields,
                'field_completeness': field_scores,
                'worst_fields': sorted(field_scores.items(), key=lambda x: x[1])[:3]
            }
        )
    
    async def _assess_validity(self, dataset: List[Dict], phase: str) -> QualityMetric:
        """Assess data validity (schema compliance)"""
        valid_records = 0
        total_records = len(dataset)
        validation_errors = []
        
        # Phase-specific validation rules
        validation_rules = self._get_phase_validation_rules(phase)
        
        for i, record in enumerate(dataset):
            record_valid = True
            record_errors = []
            
            for rule_name, validation_func in validation_rules.items():
                try:
                    if not validation_func(record):
                        record_valid = False
                        record_errors.append(f"Failed rule: {rule_name}")
                except Exception as e:
                    record_valid = False
                    record_errors.append(f"Rule {rule_name} error: {str(e)}")
            
            if record_valid:
                valid_records += 1
            else:
                validation_errors.append({
                    'record_index': i,
                    'errors': record_errors
                })
        
        validity_score = valid_records / total_records if total_records > 0 else 0.0
        threshold = self.quality_thresholds[QualityDimension.VALIDITY]
        
        return QualityMetric(
            dimension=QualityDimension.VALIDITY,
            score=validity_score,
            threshold=threshold,
            passed=validity_score >= threshold,
            details={
                'valid_records': valid_records,
                'total_records': total_records,
                'validation_errors': validation_errors[:10],  # Keep only first 10 errors
                'error_count': len(validation_errors)
            }
        )
    
    def _get_phase_validation_rules(self, phase: str) -> Dict[str, callable]:
        """Get validation rules for specific phase"""
        common_rules = {
            'has_success_field': lambda r: 'success' in r,
            'has_timestamp': lambda r: 'timestamp' in r,
            'has_data_field': lambda r: 'data' in r if r.get('success') else True
        }
        
        if phase == 'phase1':
            phase1_rules = {
                'has_rooms': lambda r: isinstance(r.get('data', {}).get('rooms'), list),
                'rooms_not_empty': lambda r: len(r.get('data', {}).get('rooms', [])) > 0,
                'waste_factors_applied': lambda r: r.get('waste_factors_applied', False),
                'has_confidence_score': lambda r: 'confidence_score' in r
            }
            return {**common_rules, **phase1_rules}
        
        elif phase == 'phase2':
            phase2_rules = {
                'has_cost_summary': lambda r: 'cost_summary' in r.get('data', {}),
                'has_line_items': lambda r: 'line_items' in r.get('data', {}),
                'has_tax_calculation': lambda r: 'tax_calculation' in r.get('data', {}),
                'positive_total_cost': lambda r: r.get('data', {}).get('cost_summary', {}).get('grand_total', 0) > 0
            }
            return {**common_rules, **phase2_rules}
        
        return common_rules
    
    async def _assess_consistency(self, dataset: List[Dict], phase: str) -> QualityMetric:
        """Assess internal data consistency"""
        consistent_records = 0
        total_records = len(dataset)
        consistency_errors = []
        
        # Phase-specific consistency checks
        consistency_rules = self._get_phase_consistency_rules(phase)
        
        for i, record in enumerate(dataset):
            record_consistent = True
            record_errors = []
            
            for rule_name, check_func in consistency_rules.items():
                try:
                    if not check_func(record):
                        record_consistent = False
                        record_errors.append(f"Inconsistent: {rule_name}")
                except Exception as e:
                    record_consistent = False
                    record_errors.append(f"Consistency check {rule_name} error: {str(e)}")
            
            if record_consistent:
                consistent_records += 1
            else:
                consistency_errors.append({
                    'record_index': i,
                    'errors': record_errors
                })
        
        consistency_score = consistent_records / total_records if total_records > 0 else 0.0
        threshold = self.quality_thresholds[QualityDimension.CONSISTENCY]
        
        return QualityMetric(
            dimension=QualityDimension.CONSISTENCY,
            score=consistency_score,
            threshold=threshold,
            passed=consistency_score >= threshold,
            details={
                'consistent_records': consistent_records,
                'total_records': total_records,
                'consistency_errors': consistency_errors[:10],
                'error_count': len(consistency_errors)
            }
        )
    
    def _get_phase_consistency_rules(self, phase: str) -> Dict[str, callable]:
        """Get consistency rules for specific phase"""
        common_rules = {
            'success_matches_data': lambda r: bool(r.get('success')) == bool(r.get('data')),
            'error_when_failed': lambda r: bool(r.get('error')) == (not r.get('success', True))
        }
        
        if phase == 'phase1':
            phase1_rules = {
                'waste_quantity_consistency': self._check_waste_quantity_consistency,
                'room_task_consistency': self._check_room_task_consistency,
                'confidence_validation_consistency': self._check_confidence_validation_consistency
            }
            return {**common_rules, **phase1_rules}
        
        elif phase == 'phase2':
            phase2_rules = {
                'cost_calculation_consistency': self._check_cost_calculation_consistency,
                'tax_calculation_consistency': self._check_tax_calculation_consistency,
                'overhead_profit_consistency': self._check_overhead_profit_consistency
            }
            return {**common_rules, **phase2_rules}
        
        return common_rules
    
    def _check_waste_quantity_consistency(self, record: Dict) -> bool:
        """Check if waste factor calculations are consistent"""
        try:
            rooms = record.get('data', {}).get('rooms', [])
            for room in rooms:
                for task in room.get('tasks', []):
                    if 'waste_factor' in task and 'quantity' in task and 'quantity_with_waste' in task:
                        expected = task['quantity'] * (1 + task['waste_factor'] / 100)
                        actual = task['quantity_with_waste']
                        if abs(expected - actual) / expected > 0.01:  # 1% tolerance
                            return False
            return True
        except:
            return False
    
    def _check_room_task_consistency(self, record: Dict) -> bool:
        """Check if room measurements are consistent with tasks"""
        # Implementation would check if task quantities make sense given room measurements
        return True  # Simplified for example
    
    def _check_confidence_validation_consistency(self, record: Dict) -> bool:
        """Check if confidence score aligns with validation results"""
        try:
            confidence = record.get('confidence_score', 0)
            validation = record.get('validation', {})
            overall_valid = validation.get('overall_valid', True)
            
            # High confidence should have valid validation
            if confidence > 0.8 and not overall_valid:
                return False
            
            # Low confidence with perfect validation seems inconsistent
            if confidence < 0.5 and overall_valid and not validation.get('issues', []):
                return False
            
            return True
        except:
            return False
    
    def _check_cost_calculation_consistency(self, record: Dict) -> bool:
        """Check if cost calculations are mathematically consistent"""
        try:
            cost_summary = record.get('data', {}).get('cost_summary', {})
            material = cost_summary.get('total_material', 0)
            labor = cost_summary.get('total_labor', 0)
            tax = cost_summary.get('total_tax', 0)
            overhead = cost_summary.get('overhead', 0)
            profit = cost_summary.get('profit', 0)
            total = cost_summary.get('grand_total', 0)
            
            expected_total = material + labor + tax + overhead + profit
            return abs(expected_total - total) / max(expected_total, 1) < 0.01  # 1% tolerance
        except:
            return False
    
    def _check_tax_calculation_consistency(self, record: Dict) -> bool:
        """Check if tax calculations are correct"""
        # Implementation would verify tax calculations against material costs
        return True  # Simplified for example
    
    def _check_overhead_profit_consistency(self, record: Dict) -> bool:
        """Check if overhead and profit calculations are within reasonable bounds"""
        try:
            cost_summary = record.get('data', {}).get('cost_summary', {})
            overhead_profit_pct = record.get('data', {}).get('overhead_profit_percentage', 0)
            
            # Should be between 10% and 25% for construction
            return 10 <= overhead_profit_pct <= 25
        except:
            return False
    
    async def _assess_uniqueness(self, dataset: List[Dict]) -> QualityMetric:
        """Assess data uniqueness (duplicate detection)"""
        if not dataset:
            return QualityMetric(
                dimension=QualityDimension.UNIQUENESS,
                score=1.0,
                threshold=self.quality_thresholds[QualityDimension.UNIQUENESS],
                passed=True,
                details={'note': 'Empty dataset - no duplicates'}
            )
        
        # Create content hashes for duplicate detection
        content_hashes = []
        duplicates = []
        
        for i, record in enumerate(dataset):
            # Remove metadata fields for duplicate detection
            content = {k: v for k, v in record.items() if not k.startswith('_')}
            content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
            
            if content_hash in content_hashes:
                duplicates.append({
                    'record_index': i,
                    'duplicate_of': content_hashes.index(content_hash),
                    'content_hash': content_hash
                })
            else:
                content_hashes.append(content_hash)
        
        unique_records = len(content_hashes)
        total_records = len(dataset)
        uniqueness_score = unique_records / total_records if total_records > 0 else 1.0
        threshold = self.quality_thresholds[QualityDimension.UNIQUENESS]
        
        return QualityMetric(
            dimension=QualityDimension.UNIQUENESS,
            score=uniqueness_score,
            threshold=threshold,
            passed=uniqueness_score >= threshold,
            details={
                'unique_records': unique_records,
                'total_records': total_records,
                'duplicates': duplicates,
                'duplicate_count': len(duplicates)
            }
        )
    
    async def _assess_accuracy(self, dataset: List[Dict], ground_truth: List[Dict], phase: str) -> QualityMetric:
        """Assess accuracy against ground truth data"""
        if len(dataset) != len(ground_truth):
            return QualityMetric(
                dimension=QualityDimension.ACCURACY,
                score=0.0,
                threshold=self.quality_thresholds[QualityDimension.ACCURACY],
                passed=False,
                details={'error': 'Dataset and ground truth length mismatch'}
            )
        
        accuracy_scores = []
        detailed_comparisons = []
        
        for i, (actual, expected) in enumerate(zip(dataset, ground_truth)):
            if phase == 'phase1':
                score, details = await self._compare_phase1_accuracy(actual, expected)
            elif phase == 'phase2':
                score, details = await self._compare_phase2_accuracy(actual, expected)
            else:
                score, details = 0.5, {'note': f'No accuracy assessment for {phase}'}
            
            accuracy_scores.append(score)
            detailed_comparisons.append({
                'record_index': i,
                'accuracy_score': score,
                'comparison_details': details
            })
        
        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
        threshold = self.quality_thresholds[QualityDimension.ACCURACY]
        
        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=overall_accuracy,
            threshold=threshold,
            passed=overall_accuracy >= threshold,
            details={
                'individual_scores': accuracy_scores,
                'mean_accuracy': overall_accuracy,
                'std_accuracy': np.std(accuracy_scores) if len(accuracy_scores) > 1 else 0.0,
                'detailed_comparisons': detailed_comparisons[:5]  # Keep only first 5
            }
        )
    
    async def _compare_phase1_accuracy(self, actual: Dict, expected: Dict) -> Tuple[float, Dict]:
        """Compare Phase 1 results for accuracy"""
        # Simplified accuracy comparison - would need more sophisticated logic
        score_components = []
        
        # Compare room count
        actual_rooms = len(actual.get('data', {}).get('rooms', []))
        expected_rooms = len(expected.get('data', {}).get('rooms', []))
        room_score = 1.0 if actual_rooms == expected_rooms else 0.5
        score_components.append(room_score)
        
        # Compare confidence scores
        actual_confidence = actual.get('confidence_score', 0)
        expected_confidence = expected.get('confidence_score', 0)
        confidence_score = 1.0 - abs(actual_confidence - expected_confidence)
        score_components.append(confidence_score)
        
        # Compare waste factors application
        waste_applied_score = 1.0 if actual.get('waste_factors_applied') == expected.get('waste_factors_applied') else 0.0
        score_components.append(waste_applied_score)
        
        overall_score = np.mean(score_components)
        
        return overall_score, {
            'room_count_score': room_score,
            'confidence_score': confidence_score,
            'waste_factors_score': waste_applied_score,
            'actual_rooms': actual_rooms,
            'expected_rooms': expected_rooms
        }
    
    async def _compare_phase2_accuracy(self, actual: Dict, expected: Dict) -> Tuple[float, Dict]:
        """Compare Phase 2 results for accuracy"""
        # Simplified accuracy comparison for Phase 2
        score_components = []
        
        # Compare total costs
        actual_total = actual.get('data', {}).get('cost_summary', {}).get('grand_total', 0)
        expected_total = expected.get('data', {}).get('cost_summary', {}).get('grand_total', 0)
        
        if expected_total > 0:
            cost_difference = abs(actual_total - expected_total) / expected_total
            cost_score = max(0.0, 1.0 - cost_difference)
        else:
            cost_score = 1.0 if actual_total == 0 else 0.0
        
        score_components.append(cost_score)
        
        overall_score = np.mean(score_components)
        
        return overall_score, {
            'cost_accuracy_score': cost_score,
            'actual_total': actual_total,
            'expected_total': expected_total,
            'cost_difference_percent': cost_difference * 100 if expected_total > 0 else 0
        }
    
    async def _assess_reliability(self, dataset: List[Dict]) -> QualityMetric:
        """Assess processing reliability from metadata"""
        if not dataset:
            return QualityMetric(
                dimension=QualityDimension.RELIABILITY,
                score=0.0,
                threshold=self.quality_thresholds[QualityDimension.RELIABILITY],
                passed=False,
                details={'error': 'Empty dataset'}
            )
        
        successful_records = sum(1 for record in dataset if record.get('success', False))
        total_records = len(dataset)
        reliability_score = successful_records / total_records if total_records > 0 else 0.0
        threshold = self.quality_thresholds[QualityDimension.RELIABILITY]
        
        # Analyze error patterns
        error_types = {}
        for record in dataset:
            if not record.get('success', False) and 'error' in record:
                error_msg = str(record['error'])[:100]  # First 100 chars
                error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        return QualityMetric(
            dimension=QualityDimension.RELIABILITY,
            score=reliability_score,
            threshold=threshold,
            passed=reliability_score >= threshold,
            details={
                'successful_records': successful_records,
                'total_records': total_records,
                'failure_rate': 1.0 - reliability_score,
                'error_types': error_types,
                'most_common_errors': sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        )
    
    async def _generate_recommendations(self, metrics: List[QualityMetric], dataset: List[Dict], phase: str) -> List[str]:
        """Generate actionable recommendations based on quality metrics"""
        recommendations = []
        
        for metric in metrics:
            if not metric.passed:
                recommendations.extend(self._get_dimension_recommendations(metric, dataset, phase))
        
        # Overall recommendations
        if len([m for m in metrics if not m.passed]) > len(metrics) / 2:
            recommendations.append("Consider reviewing the entire data generation process - multiple quality dimensions failed")
        
        return recommendations
    
    def _get_dimension_recommendations(self, metric: QualityMetric, dataset: List[Dict], phase: str) -> List[str]:
        """Get recommendations for specific quality dimension"""
        recommendations = []
        dimension = metric.dimension
        
        if dimension == QualityDimension.COMPLETENESS:
            worst_fields = metric.details.get('worst_fields', [])
            if worst_fields:
                field_name = worst_fields[0][0]
                recommendations.append(f"Improve completeness for field '{field_name}' - consider default values or validation")
        
        elif dimension == QualityDimension.VALIDITY:
            error_count = metric.details.get('error_count', 0)
            if error_count > 0:
                recommendations.append(f"Fix {error_count} validation errors - review schema compliance")
        
        elif dimension == QualityDimension.CONSISTENCY:
            error_count = metric.details.get('error_count', 0)
            if error_count > 0:
                recommendations.append(f"Address {error_count} consistency issues - review business logic")
        
        elif dimension == QualityDimension.RELIABILITY:
            failure_rate = metric.details.get('failure_rate', 0)
            if failure_rate > 0.1:  # > 10% failure rate
                recommendations.append(f"High failure rate ({failure_rate:.1%}) - investigate error patterns and improve error handling")
        
        elif dimension == QualityDimension.UNIQUENESS:
            duplicate_count = metric.details.get('duplicate_count', 0)
            if duplicate_count > 0:
                recommendations.append(f"Remove {duplicate_count} duplicate records to improve data uniqueness")
        
        return recommendations
    
    async def _store_quality_history(self, report: DataQualityReport):
        """Store quality report for trend analysis"""
        history_key = f"{report.processing_phase}_{report.dataset_id}"
        
        if history_key not in self.quality_history:
            self.quality_history[history_key] = []
        
        self.quality_history[history_key].append(report)
        
        # Keep only last 100 reports per key
        if len(self.quality_history[history_key]) > 100:
            self.quality_history[history_key] = self.quality_history[history_key][-100:]
    
    async def analyze_quality_trends(self, phase: str, time_window_days: int = 30) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        # Collect recent reports for the phase
        recent_reports = []
        for key, reports in self.quality_history.items():
            if key.startswith(phase):
                recent_reports.extend([
                    r for r in reports 
                    if r.generated_at >= cutoff_date
                ])
        
        if not recent_reports:
            return {'error': f'No quality history found for {phase} in last {time_window_days} days'}
        
        # Analyze trends
        trends = {}
        for dimension in QualityDimension:
            scores = []
            timestamps = []
            
            for report in recent_reports:
                dimension_metric = next((m for m in report.metrics if m.dimension == dimension), None)
                if dimension_metric:
                    scores.append(dimension_metric.score)
                    timestamps.append(report.generated_at)
            
            if scores:
                trends[dimension.value] = {
                    'current_score': scores[-1] if scores else 0,
                    'average_score': np.mean(scores),
                    'trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'declining' if len(scores) > 1 and scores[-1] < scores[0] else 'stable',
                    'score_history': scores[-10:],  # Last 10 scores
                    'volatility': np.std(scores) if len(scores) > 1 else 0
                }
        
        return {
            'phase': phase,
            'analysis_period_days': time_window_days,
            'reports_analyzed': len(recent_reports),
            'dimension_trends': trends,
            'overall_quality_trend': self._calculate_overall_trend(recent_reports)
        }
    
    def _calculate_overall_trend(self, reports: List[DataQualityReport]) -> Dict[str, Any]:
        """Calculate overall quality trend"""
        if len(reports) < 2:
            return {'trend': 'insufficient_data'}
        
        scores = [r.overall_score for r in reports]
        
        # Simple linear trend
        x = range(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        trend_direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'current_score': scores[-1],
            'initial_score': scores[0],
            'change': scores[-1] - scores[0],
            'volatility': np.std(scores)
        }


# Example usage
async def main():
    """Example usage of the data quality framework"""
    framework = DataQualityFramework()
    
    # Sample dataset
    sample_data = [
        {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': {
                'rooms': [{'name': 'living_room', 'sqft': 200}]
            },
            'confidence_score': 0.85,
            'waste_factors_applied': True
        }
    ]
    
    # Assess quality
    report = await framework.assess_data_quality(sample_data, 'phase1')
    
    print(f"Quality Assessment Report:")
    print(f"Overall Score: {report.overall_score:.3f}")
    print(f"Quality Gate Passed: {report.passed_quality_gate}")
    print(f"Recommendations: {report.recommendations}")


if __name__ == "__main__":
    asyncio.run(main())