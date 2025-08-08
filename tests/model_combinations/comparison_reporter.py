"""
Comparison Reporter for Model Combination Testing
Generates comprehensive comparison reports with visualizations and insights.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from .combination_tester import CombinationTestResult
from .performance_analyzer import PerformanceAnalyzer, ModelPerformanceMetrics
from .test_matrix import TestConfiguration


logger = logging.getLogger(__name__)


@dataclass  
class ComparisonReport:
    """Comprehensive comparison report for model combinations"""
    title: str
    generated_at: datetime
    summary: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    detailed_results: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization"""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "performance_analysis": self.performance_analysis,
            "detailed_results": self.detailed_results,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class ComparisonReporter:
    """Generates comprehensive comparison reports for model combination testing"""
    
    def __init__(self, output_directory: Optional[str] = None):
        """Initialize comparison reporter
        
        Args:
            output_directory: Directory for saving reports
        """
        self.output_directory = Path(output_directory) if output_directory else Path("test_outputs/reports")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.analyzer = PerformanceAnalyzer()
    
    def generate_comprehensive_report(self,
                                    test_results: List[CombinationTestResult],
                                    report_title: str = "Model Combination Analysis Report") -> ComparisonReport:
        """Generate comprehensive comparison report
        
        Args:
            test_results: List of all test results
            report_title: Title for the report
            
        Returns:
            Comprehensive comparison report
        """
        logger.info(f"Generating comprehensive report for {len(test_results)} test results")
        
        # Group results by configuration
        results_by_config = {}
        for result in test_results:
            config_name = result.configuration.test_name
            if config_name not in results_by_config:
                results_by_config[config_name] = []
            results_by_config[config_name].append(result)
        
        # Analyze performance metrics
        metrics_list = self.analyzer.analyze_multiple_configurations(results_by_config)
        
        # Generate summary
        summary = self._generate_summary(test_results, metrics_list)
        
        # Generate performance analysis
        performance_analysis = self.analyzer.generate_performance_report(metrics_list)
        
        # Generate detailed results
        detailed_results = self._generate_detailed_results(test_results)
        
        # Generate recommendations
        recommendations = self._generate_comprehensive_recommendations(metrics_list, test_results)
        
        # Generate metadata
        metadata = self._generate_metadata(test_results)
        
        report = ComparisonReport(
            title=report_title,
            generated_at=datetime.now(),
            summary=summary,
            performance_analysis=performance_analysis,
            detailed_results=detailed_results,
            recommendations=recommendations,
            metadata=metadata
        )
        
        return report
    
    def _generate_summary(self, 
                         test_results: List[CombinationTestResult],
                         metrics_list: List[ModelPerformanceMetrics]) -> Dict[str, Any]:
        """Generate executive summary"""
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r.success])
        
        # Configuration breakdown
        config_types = {
            "single_model": 0,
            "two_model": 0, 
            "three_model": 0
        }
        
        for result in test_results:
            model_count = len(result.configuration.models)
            if model_count == 1:
                config_types["single_model"] += 1
            elif model_count == 2:
                config_types["two_model"] += 1
            elif model_count == 3:
                config_types["three_model"] += 1
        
        # Performance highlights
        if metrics_list:
            best_quality = max(metrics_list, key=lambda m: m.avg_quality_score)
            fastest = min([m for m in metrics_list if m.avg_execution_time > 0], 
                         key=lambda m: m.avg_execution_time, default=metrics_list[0])
            
            performance_highlights = {
                "best_quality": {
                    "configuration": best_quality.model_combination,
                    "score": best_quality.avg_quality_score,
                    "confidence": best_quality.avg_confidence
                },
                "fastest": {
                    "configuration": fastest.model_combination,
                    "execution_time": fastest.avg_execution_time,
                    "quality_score": fastest.avg_quality_score
                }
            }
        else:
            performance_highlights = {}
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "configuration_breakdown": config_types,
            "unique_configurations": len(set(r.configuration.test_name for r in test_results)),
            "performance_highlights": performance_highlights,
            "test_duration": {
                "total_execution_time": sum(r.execution_time for r in test_results),
                "average_per_test": sum(r.execution_time for r in test_results) / total_tests if total_tests > 0 else 0
            }
        }
    
    def _generate_detailed_results(self, test_results: List[CombinationTestResult]) -> List[Dict[str, Any]]:
        """Generate detailed results for each test"""
        detailed_results = []
        
        for result in test_results:
            detailed_result = {
                "configuration_name": result.configuration.test_name,
                "model_combination": " + ".join(result.configuration.model_names),
                "validation_mode": result.configuration.validation_mode.value,
                "processing_mode": result.configuration.processing_mode.value,
                "success": result.success,
                "execution_time": result.execution_time,
                "quality_score": result.quality_score,
                "overall_confidence": result.overall_confidence,
                "consensus_level": result.consensus_level,
                "validation_score": result.validation_score,
                "model_success_rate": result.model_success_rate,
                "total_work_items": result.total_work_items,
                "unique_rooms_processed": result.unique_rooms_processed,
                "error_message": result.error_message,
                "warnings_count": len(result.warnings),
                "timestamp": result.timestamp.isoformat()
            }
            detailed_results.append(detailed_result)
        
        # Sort by quality score descending
        detailed_results.sort(key=lambda x: x["quality_score"], reverse=True)
        
        return detailed_results
    
    def _generate_comprehensive_recommendations(self,
                                              metrics_list: List[ModelPerformanceMetrics],
                                              test_results: List[CombinationTestResult]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        if not metrics_list:
            return ["No performance metrics available for recommendations"]
        
        # 1. Best overall performer
        best_overall = max(metrics_list, key=lambda m: m.avg_quality_score)
        recommendations.append(
            f"🏆 BEST OVERALL: {best_overall.model_combination} delivers the highest quality "
            f"(score: {best_overall.avg_quality_score:.3f}, confidence: {best_overall.avg_confidence:.3f})"
        )
        
        # 2. Speed vs Quality trade-off
        fast_configs = sorted([m for m in metrics_list if m.avg_execution_time > 0], 
                             key=lambda m: m.avg_execution_time)[:3]
        if fast_configs:
            fastest = fast_configs[0]
            recommendations.append(
                f"⚡ FASTEST: {fastest.model_combination} for speed-critical applications "
                f"({fastest.avg_execution_time:.2f}s avg, quality: {fastest.avg_quality_score:.3f})"
            )
        
        # 3. Cost efficiency
        cost_efficient = [m for m in metrics_list if m.cost_efficiency_ratio > 0]
        if cost_efficient:
            most_efficient = max(cost_efficient, key=lambda m: m.cost_efficiency_ratio)
            recommendations.append(
                f"💰 MOST COST-EFFICIENT: {most_efficient.model_combination} "
                f"(${most_efficient.estimated_cost_per_test:.3f} per test, "
                f"efficiency ratio: {most_efficient.cost_efficiency_ratio:.3f})"
            )
        
        # 4. Reliability analysis
        reliable_configs = [m for m in metrics_list if m.success_rate >= 0.9]
        if reliable_configs:
            most_reliable = max(reliable_configs, key=lambda m: m.success_rate)
            recommendations.append(
                f"🔒 MOST RELIABLE: {most_reliable.model_combination} "
                f"({most_reliable.success_rate:.1%} success rate, "
                f"quality: {most_reliable.avg_quality_score:.3f})"
            )
        
        # 5. Single vs Multi-model analysis
        single_models = [m for m in metrics_list if len(m.configuration.model_names) == 1]
        multi_models = [m for m in metrics_list if len(m.configuration.model_names) > 1]
        
        if single_models and multi_models:
            avg_single_quality = sum(m.avg_quality_score for m in single_models) / len(single_models)
            avg_multi_quality = sum(m.avg_quality_score for m in multi_models) / len(multi_models)
            avg_single_time = sum(m.avg_execution_time for m in single_models) / len(single_models)
            avg_multi_time = sum(m.avg_execution_time for m in multi_models) / len(multi_models)
            
            if avg_multi_quality > avg_single_quality * 1.05:  # 5% improvement threshold
                quality_improvement = ((avg_multi_quality - avg_single_quality) / avg_single_quality) * 100
                time_increase = ((avg_multi_time - avg_single_time) / avg_single_time) * 100 if avg_single_time > 0 else 0
                recommendations.append(
                    f"🔀 MULTI-MODEL ADVANTAGE: Multi-model combinations improve quality by "
                    f"{quality_improvement:.1f}% but increase execution time by {time_increase:.1f}%"
                )
            else:
                recommendations.append(
                    "🎯 SINGLE MODEL EFFICIENCY: Single models provide competitive quality "
                    "with faster execution times"
                )
        
        # 6. Validation mode impact
        validation_impact = {}
        for metrics in metrics_list:
            val_mode = metrics.configuration.validation_mode.value
            if val_mode not in validation_impact:
                validation_impact[val_mode] = []
            validation_impact[val_mode].append(metrics.avg_quality_score)
        
        if len(validation_impact) > 1:
            val_averages = {mode: sum(scores)/len(scores) 
                          for mode, scores in validation_impact.items()}
            best_val_mode = max(val_averages, key=val_averages.get)
            recommendations.append(
                f"✅ VALIDATION OPTIMIZATION: '{best_val_mode}' validation mode "
                f"produces highest average quality ({val_averages[best_val_mode]:.3f})"
            )
        
        # 7. Model-specific insights
        model_performance = {}
        for metrics in metrics_list:
            for model_name in metrics.configuration.model_names:
                if model_name not in model_performance:
                    model_performance[model_name] = []
                model_performance[model_name].append(metrics.avg_quality_score)
        
        model_averages = {model: sum(scores)/len(scores) 
                         for model, scores in model_performance.items()}
        if model_averages:
            best_individual_model = max(model_averages, key=model_averages.get)
            recommendations.append(
                f"🎖️ BEST INDIVIDUAL MODEL: {best_individual_model} shows strongest "
                f"individual performance (avg quality: {model_averages[best_individual_model]:.3f})"
            )
        
        # 8. Production deployment recommendations
        production_candidates = [
            m for m in metrics_list 
            if m.success_rate >= 0.95 and m.avg_quality_score >= 0.8 and m.error_rate <= 0.05
        ]
        
        if production_candidates:
            best_production = max(production_candidates, key=lambda m: m.avg_quality_score)
            recommendations.append(
                f"🚀 PRODUCTION READY: {best_production.model_combination} recommended "
                f"for production deployment (quality: {best_production.avg_quality_score:.3f}, "
                f"reliability: {best_production.success_rate:.1%})"
            )
        else:
            recommendations.append(
                "⚠️ PRODUCTION CAUTION: No configurations meet strict production criteria "
                "(95%+ success rate, 80%+ quality, <5% error rate). Consider additional testing."
            )
        
        return recommendations
    
    def _generate_metadata(self, test_results: List[CombinationTestResult]) -> Dict[str, Any]:
        """Generate report metadata"""
        models_tested = set()
        validation_modes = set()
        processing_modes = set()
        
        for result in test_results:
            models_tested.update(result.configuration.model_names)
            validation_modes.add(result.configuration.validation_mode.value)
            processing_modes.add(result.configuration.processing_mode.value)
        
        return {
            "total_test_results": len(test_results),
            "models_tested": sorted(list(models_tested)),
            "validation_modes_tested": sorted(list(validation_modes)),
            "processing_modes_tested": sorted(list(processing_modes)),
            "test_date_range": {
                "earliest": min(r.timestamp for r in test_results).isoformat(),
                "latest": max(r.timestamp for r in test_results).isoformat()
            },
            "test_environment": {
                "framework_version": "1.0",
                "analyzer_version": "1.0"
            }
        }
    
    def save_report(self, report: ComparisonReport, filename: Optional[str] = None) -> str:
        """Save comparison report to file
        
        Args:
            report: Comparison report to save
            filename: Optional filename (will generate if not provided)
            
        Returns:
            Path where report was saved
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_combination_report_{timestamp}.json"
        
        report_path = self.output_directory / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Comparison report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            raise
    
    def generate_html_report(self, report: ComparisonReport, filename: Optional[str] = None) -> str:
        """Generate HTML version of the report
        
        Args:
            report: Comparison report
            filename: Optional HTML filename
            
        Returns:
            Path where HTML report was saved
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_combination_report_{timestamp}.html"
        
        html_path = self.output_directory / filename
        
        # Generate HTML content
        html_content = self._generate_html_content(report)
        
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to: {html_path}")
            return str(html_path)
            
        except Exception as e:
            logger.error(f"Failed to save HTML report: {e}")
            raise
    
    def _generate_html_content(self, report: ComparisonReport) -> str:
        """Generate HTML content for the report"""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .metric {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendation {{ background: #f0f8f0; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .warning {{ background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .quality-high {{ color: #4CAF50; font-weight: bold; }}
        .quality-medium {{ color: #ff9800; font-weight: bold; }}
        .quality-low {{ color: #f44336; font-weight: bold; }}
        .summary-stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.title}</h1>
        <p><strong>Generated:</strong> {report.generated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Total Tests:</strong> {report.summary.get('total_tests', 0)} | 
           <strong>Success Rate:</strong> {report.summary.get('success_rate', 0):.1%}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-stats">
            <div class="stat-box">
                <div class="stat-number">{report.summary.get('total_tests', 0)}</div>
                <div>Total Tests</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{report.summary.get('success_rate', 0):.1%}</div>
                <div>Success Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{report.summary.get('unique_configurations', 0)}</div>
                <div>Configurations</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Key Recommendations</h2>
        """
        
        for rec in report.recommendations:
            html += f'<div class="recommendation">{rec}</div>\n'
        
        html += """
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Configuration</th>
                <th>Models</th>
                <th>Success</th>
                <th>Quality Score</th>
                <th>Confidence</th>
                <th>Execution Time</th>
                <th>Work Items</th>
            </tr>
        """
        
        for result in report.detailed_results[:20]:  # Show top 20
            success_icon = "✅" if result['success'] else "❌"
            quality_class = ("quality-high" if result['quality_score'] >= 0.8 
                           else "quality-medium" if result['quality_score'] >= 0.6 
                           else "quality-low")
            
            html += f"""
            <tr>
                <td>{result['configuration_name']}</td>
                <td>{result['model_combination']}</td>
                <td>{success_icon}</td>
                <td class="{quality_class}">{result['quality_score']:.3f}</td>
                <td>{result['overall_confidence']:.3f}</td>
                <td>{result['execution_time']:.2f}s</td>
                <td>{result['total_work_items']}</td>
            </tr>
            """
        
        html += """
        </table>
    </div>

    <div class="section">
        <h2>Performance Analysis</h2>
        """
        
        if 'overall_statistics' in report.performance_analysis:
            stats = report.performance_analysis['overall_statistics']
            html += f"""
            <div class="metric">
                <strong>Overall Statistics:</strong><br>
                Average Quality Score: {stats.get('avg_quality_score', 0):.3f}<br>
                Average Execution Time: {stats.get('avg_execution_time', 0):.2f}s<br>
                Average Confidence: {stats.get('avg_confidence', 0):.3f}
            </div>
            """
        
        html += """
    </div>

    <div class="section">
        <h2>Testing Metadata</h2>
        """
        
        metadata = report.metadata
        html += f"""
        <div class="metric">
            <strong>Models Tested:</strong> {', '.join(metadata.get('models_tested', []))}<br>
            <strong>Validation Modes:</strong> {', '.join(metadata.get('validation_modes_tested', []))}<br>
            <strong>Processing Modes:</strong> {', '.join(metadata.get('processing_modes_tested', []))}<br>
            <strong>Test Period:</strong> {metadata.get('test_date_range', {}).get('earliest', 'N/A')} to {metadata.get('test_date_range', {}).get('latest', 'N/A')}
        </div>
        """
        
        html += """
    </div>
</body>
</html>
        """
        
        return html
    
    def export_to_excel(self, report: ComparisonReport, filename: Optional[str] = None) -> str:
        """Export report data to Excel file
        
        Args:
            report: Comparison report
            filename: Optional Excel filename
            
        Returns:
            Path where Excel file was saved
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_combination_analysis_{timestamp}.xlsx"
        
        excel_path = self.output_directory / filename
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([report.summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed results sheet
                results_df = pd.DataFrame(report.detailed_results)
                results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
                
                # Performance analysis sheet
                if 'detailed_metrics' in report.performance_analysis:
                    metrics_df = pd.DataFrame(report.performance_analysis['detailed_metrics'])
                    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                
                # Recommendations sheet
                recommendations_df = pd.DataFrame([{'Recommendation': rec} for rec in report.recommendations])
                recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            logger.info(f"Excel report saved to: {excel_path}")
            return str(excel_path)
            
        except Exception as e:
            logger.error(f"Failed to save Excel report: {e}")
            raise