#!/usr/bin/env python
"""
Demo Script for AI Model Combination Testing Framework
Demonstrates the new comprehensive testing capabilities with actual test data.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.model_combinations import (
    ModelCombinationTester,
    ModelTestMatrix,
    TestConfiguration,
    ModelType,
    ValidationMode,
    ProcessingMode,
    PerformanceAnalyzer,
    ComparisonReporter
)
from tests.utils.test_data_loader import get_test_data_loader


async def demo_basic_functionality():
    """Demonstrate basic framework functionality"""
    print("🚀 AI Model Combination Testing Framework Demo")
    print("=" * 60)
    
    # 1. Initialize Test Data Loader
    print("\n📂 Loading Test Data...")
    test_data_loader = get_test_data_loader()
    
    # Show test data information
    demo_data = test_data_loader.load_demo_data()
    measurement_data = test_data_loader.load_measurement_data()
    intake_form = test_data_loader.load_intake_form()
    
    print(f"✅ Demo data loaded: {len(demo_data)} floors")
    print(f"✅ Measurement data loaded: {len(measurement_data)} floors")
    print(f"✅ Intake form loaded: {len(intake_form)} characters")
    
    # Create combined project data
    combined_data = test_data_loader.create_combined_project_data()
    print(f"✅ Combined project data: {len(combined_data)} items")
    
    # Validate project data
    is_valid = test_data_loader.validate_project_data(combined_data)
    print(f"✅ Project data validation: {'PASSED' if is_valid else 'FAILED'}")
    
    return test_data_loader


def demo_test_matrix():
    """Demonstrate test matrix generation"""
    print("\n🔬 Test Matrix Generation Demo...")
    
    # Create test matrix
    matrix = ModelTestMatrix(
        available_models=[ModelType.GPT4, ModelType.CLAUDE, ModelType.GEMINI],
        validation_modes=[ValidationMode.BALANCED],
        processing_modes=[ProcessingMode.PARALLEL],
        include_single_model=True,
        include_multi_model=True
    )
    
    # Generate different types of configurations
    all_configs = matrix.generate_all_configurations()
    essential_configs = matrix.generate_essential_configurations()
    performance_configs = matrix.generate_performance_comparison_matrix()
    
    print(f"📊 All configurations: {len(all_configs)}")
    print(f"⚡ Essential configurations: {len(essential_configs)}")
    print(f"🏆 Performance configurations: {len(performance_configs)}")
    
    # Show configuration summary
    summary = matrix.get_configuration_summary(essential_configs)
    print(f"\n📋 Essential Configuration Summary:")
    print(f"  - Total: {summary['total_configurations']}")
    print(f"  - Single model: {summary['single_model_tests']}")
    print(f"  - Multi-model: {summary['multi_model_tests']}")
    print(f"  - Model usage: {summary['model_usage']}")
    
    return essential_configs[:3]  # Return first 3 for demo


def demo_mock_testing():
    """Demonstrate mock testing functionality"""
    print("\n🧪 Mock Testing Demo...")
    print("(This demonstrates the testing framework without API calls)")
    
    # Create mock model responses based on real test data
    from src.models.data_models import ModelResponse
    
    mock_responses = [
        ModelResponse(
            model_name="gpt-4",
            room_estimates=[
                {
                    "name": "Living Room",
                    "tasks": [
                        {
                            "task_name": "Remove existing carpet flooring",
                            "description": "Remove carpet and padding from living room",
                            "necessity": "required",
                            "quantity": 150.0,
                            "unit": "sq_ft"
                        },
                        {
                            "task_name": "Install laminate wood flooring",
                            "description": "Install new laminate wood flooring",
                            "necessity": "required",
                            "quantity": 150.0,
                            "unit": "sq_ft"
                        }
                    ]
                }
            ],
            processing_time=2.3,
            total_work_items=2,
            confidence_self_assessment=0.87
        ),
        ModelResponse(
            model_name="claude-3-sonnet",
            room_estimates=[
                {
                    "name": "Living Room",
                    "tasks": [
                        {
                            "task_name": "Flooring removal and replacement",
                            "description": "Complete flooring replacement in living room",
                            "necessity": "required",
                            "quantity": 148.5,
                            "unit": "sq_ft"
                        },
                        {
                            "task_name": "Baseboard removal and reinstallation",
                            "description": "Remove and reinstall baseboards",
                            "necessity": "required",
                            "quantity": 60.0,
                            "unit": "lf"
                        }
                    ]
                }
            ],
            processing_time=2.1,
            total_work_items=2,
            confidence_self_assessment=0.84
        )
    ]
    
    print("✅ Created mock model responses:")
    for response in mock_responses:
        print(f"  - {response.model_name}: {response.total_work_items} tasks, "
              f"{response.confidence_self_assessment:.2f} confidence")
    
    return mock_responses


def demo_performance_analysis(mock_responses):
    """Demonstrate performance analysis"""
    print("\n📊 Performance Analysis Demo...")
    
    from tests.model_combinations.combination_tester import CombinationTestResult
    from datetime import datetime
    
    # Create mock test results
    config1 = TestConfiguration(
        models=[ModelType.GPT4],
        validation_mode=ValidationMode.BALANCED,
        processing_mode=ProcessingMode.SINGLE_MODEL,
        test_name="demo_gpt4"
    )
    
    config2 = TestConfiguration(
        models=[ModelType.GPT4, ModelType.CLAUDE],
        validation_mode=ValidationMode.BALANCED,
        processing_mode=ProcessingMode.PARALLEL,
        test_name="demo_gpt4_claude"
    )
    
    mock_results = [
        CombinationTestResult(
            configuration=config1,
            success=True,
            execution_time=2.5,
            model_responses=[mock_responses[0]],
            models_responded=1,
            overall_confidence=0.87,
            consensus_level=1.0,
            validation_score=0.85,
            total_work_items=2,
            unique_rooms_processed=1,
            timestamp=datetime.now()
        ),
        CombinationTestResult(
            configuration=config2,
            success=True,
            execution_time=3.2,
            model_responses=mock_responses,
            models_responded=2,
            overall_confidence=0.86,
            consensus_level=0.92,
            validation_score=0.88,
            total_work_items=4,
            unique_rooms_processed=1,
            timestamp=datetime.now()
        )
    ]
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    
    # Group results by configuration
    results_by_config = {
        "demo_gpt4": [mock_results[0]],
        "demo_gpt4_claude": [mock_results[1]]
    }
    
    metrics_list = analyzer.analyze_multiple_configurations(results_by_config)
    
    print("📈 Performance Metrics:")
    for metrics in metrics_list:
        print(f"\n🔍 {metrics.model_combination}:")
        print(f"  - Success Rate: {metrics.success_rate:.1%}")
        print(f"  - Avg Quality Score: {metrics.avg_quality_score:.3f}")
        print(f"  - Avg Execution Time: {metrics.avg_execution_time:.2f}s")
        print(f"  - Avg Confidence: {metrics.avg_confidence:.3f}")
        print(f"  - Cost per Test: ${metrics.estimated_cost_per_test:.3f}")
    
    # Generate performance report
    report_data = analyzer.generate_performance_report(metrics_list)
    
    print(f"\n📋 Performance Report Summary:")
    overall_stats = report_data.get('overall_statistics', {})
    print(f"  - Average Quality: {overall_stats.get('avg_quality_score', 0):.3f}")
    print(f"  - Average Speed: {overall_stats.get('avg_execution_time', 0):.2f}s")
    
    recommendations = report_data.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec}")
    
    return mock_results


def demo_reporting(test_results):
    """Demonstrate report generation"""
    print("\n📄 Report Generation Demo...")
    
    # Initialize reporter
    reporter = ComparisonReporter("test_outputs/demo_reports")
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report(
        test_results=test_results,
        report_title="Demo Model Combination Report"
    )
    
    print("✅ Generated comprehensive report:")
    print(f"  - Title: {report.title}")
    print(f"  - Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Detailed Results: {len(report.detailed_results)}")
    print(f"  - Recommendations: {len(report.recommendations)}")
    
    # Save reports
    try:
        json_path = reporter.save_report(report, "demo_report.json")
        print(f"  - JSON Report: {json_path}")
        
        html_path = reporter.generate_html_report(report, "demo_report.html")
        print(f"  - HTML Report: {html_path}")
        
        excel_path = reporter.export_to_excel(report, "demo_report.xlsx")
        print(f"  - Excel Report: {excel_path}")
        
    except Exception as e:
        print(f"⚠️ Report saving failed: {e}")
    
    # Show sample recommendations
    if report.recommendations:
        print(f"\n🎯 Sample Recommendations:")
        for i, rec in enumerate(report.recommendations[:2], 1):
            print(f"  {i}. {rec}")
    
    return report


async def main():
    """Run the complete demo"""
    print("🎭 Starting AI Model Combination Testing Framework Demo")
    print("This demo shows the framework capabilities without making actual API calls.")
    print()
    
    try:
        # 1. Test data loading
        test_data_loader = await demo_basic_functionality()
        
        # 2. Test matrix generation
        sample_configs = demo_test_matrix()
        
        # 3. Mock testing
        mock_responses = demo_mock_testing()
        
        # 4. Performance analysis
        test_results = demo_performance_analysis(mock_responses)
        
        # 5. Report generation
        report = demo_reporting(test_results)
        
        # Summary
        print("\n" + "="*60)
        print("🎉 Demo Complete!")
        print("="*60)
        print("✅ Test data loading and validation")
        print("✅ Test configuration matrix generation")
        print("✅ Mock model testing simulation") 
        print("✅ Performance analysis and metrics")
        print("✅ Comprehensive report generation")
        print()
        print("📁 Check 'test_outputs/demo_reports/' for generated reports")
        print()
        print("🚀 To run actual tests with API calls:")
        print("   python -m tests.model_combinations.test_runner --test-type quick")
        print()
        print("📖 See tests/README_COMPREHENSIVE_TESTING.md for full documentation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)