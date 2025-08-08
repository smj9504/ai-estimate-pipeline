"""
Model Combination Test Runner
Main entry point for running comprehensive model combination tests.
"""
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import sys

from .test_matrix import ModelTestMatrix, TestConfiguration, ModelType, ValidationMode, ProcessingMode
from .combination_tester import ModelCombinationTester, CombinationTestResult
from .performance_analyzer import PerformanceAnalyzer
from .comparison_reporter import ComparisonReporter
from ..utils.test_data_loader import TestDataLoader


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelCombinationTestRunner:
    """Main test runner for model combination testing"""
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 output_directory: Optional[str] = None):
        """Initialize test runner
        
        Args:
            config_file: Optional configuration file path
            output_directory: Directory for test outputs
        """
        self.config = self._load_config(config_file)
        self.output_directory = Path(output_directory) if output_directory else Path("test_outputs")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.test_data_loader = TestDataLoader()
        self.tester = ModelCombinationTester(
            test_data_loader=self.test_data_loader,
            output_directory=str(self.output_directory / "combinations")
        )
        self.analyzer = PerformanceAnalyzer()
        self.reporter = ComparisonReporter(str(self.output_directory / "reports"))
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "models": ["gpt4", "claude", "gemini"],
            "validation_modes": ["balanced"],
            "processing_modes": ["parallel"],
            "include_single_model": True,
            "include_multi_model": True,
            "max_concurrent_tests": 3,
            "timeout_seconds": 300,
            "retry_attempts": 2,
            "save_outputs": True
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                default_config.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def create_test_matrix(self, test_type: str = "comprehensive") -> List[TestConfiguration]:
        """Create test matrix based on configuration
        
        Args:
            test_type: Type of test matrix ("comprehensive", "essential", "performance", "quick")
            
        Returns:
            List of test configurations
        """
        # Convert config strings to enums
        available_models = [ModelType(m) for m in self.config["models"]]
        validation_modes = [ValidationMode(m) for m in self.config["validation_modes"]]  
        processing_modes = [ProcessingMode(m) for m in self.config["processing_modes"]]
        
        # Create test matrix
        matrix = ModelTestMatrix(
            available_models=available_models,
            validation_modes=validation_modes,
            processing_modes=processing_modes,
            include_single_model=self.config["include_single_model"],
            include_multi_model=self.config["include_multi_model"]
        )
        
        # Generate configurations based on test type
        if test_type == "comprehensive":
            configurations = matrix.generate_all_configurations()
        elif test_type == "essential":
            configurations = matrix.generate_essential_configurations()
        elif test_type == "performance":
            configurations = matrix.generate_performance_comparison_matrix()
        elif test_type == "quick":
            # Quick test with just essential configs
            essential_configs = matrix.generate_essential_configurations()
            configurations = essential_configs[:5]  # Limit to 5 configs
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Apply timeout and retry settings from config
        for config in configurations:
            config.timeout_seconds = self.config["timeout_seconds"]
            config.retry_attempts = self.config["retry_attempts"]
            config.save_outputs = self.config["save_outputs"]
        
        logger.info(f"Created {len(configurations)} test configurations for '{test_type}' testing")
        return configurations
    
    async def run_tests(self, test_type: str = "comprehensive") -> List[CombinationTestResult]:
        """Run model combination tests
        
        Args:
            test_type: Type of tests to run
            
        Returns:
            List of test results
        """
        logger.info(f"Starting {test_type} model combination testing")
        
        # Create test configurations
        configurations = self.create_test_matrix(test_type)
        
        if not configurations:
            logger.error("No test configurations generated")
            return []
        
        # Load test data
        test_dataset = self.test_data_loader.get_test_dataset()
        logger.info(f"Loaded test data: {len(test_dataset.combined_project_data)} project data items")
        
        # Run tests
        results = await self.tester.run_test_batch(
            configurations=configurations,
            test_dataset=test_dataset,
            max_concurrent=self.config["max_concurrent_tests"]
        )
        
        logger.info(f"Completed {len(results)} tests")
        return results
    
    def analyze_results(self, results: List[CombinationTestResult]) -> Dict[str, Any]:
        """Analyze test results and generate comprehensive report
        
        Args:
            results: List of test results
            
        Returns:
            Analysis report
        """
        logger.info("Analyzing test results...")
        
        # Generate comprehensive report
        report = self.reporter.generate_comprehensive_report(
            test_results=results,
            report_title="AI Model Combination Analysis Report"
        )
        
        # Save reports in multiple formats
        json_path = self.reporter.save_report(report)
        html_path = self.reporter.generate_html_report(report)
        excel_path = self.reporter.export_to_excel(report)
        
        logger.info(f"Reports saved:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  HTML: {html_path}")
        logger.info(f"  Excel: {excel_path}")
        
        return report.to_dict()
    
    def get_test_statistics(self, results: List[CombinationTestResult]) -> None:
        """Print test statistics to console"""
        stats = self.tester.get_test_statistics(results)
        
        print("\n" + "="*60)
        print("MODEL COMBINATION TEST STATISTICS")
        print("="*60)
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Successful Tests: {stats['successful_tests']}")
        print(f"Failed Tests: {stats['failed_tests']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Average Execution Time: {stats['average_execution_time']:.2f}s")
        print(f"Average Quality Score: {stats['average_quality_score']:.3f}")
        
        if stats.get('best_performing_config'):
            print(f"Best Performing Config: {stats['best_performing_config']}")
        if stats.get('fastest_config'):
            print(f"Fastest Config: {stats['fastest_config']}")
        
        print("\nModel Usage:")
        for model, usage in stats['model_usage'].items():
            success_rate = usage['successful'] / usage['total'] if usage['total'] > 0 else 0
            print(f"  {model}: {usage['successful']}/{usage['total']} ({success_rate:.1%})")
        
        print("="*60)
    
    async def run_and_analyze(self, test_type: str = "comprehensive") -> Dict[str, Any]:
        """Run tests and analyze results in one call
        
        Args:
            test_type: Type of tests to run
            
        Returns:
            Complete analysis report
        """
        # Run tests
        results = await self.run_tests(test_type)
        
        if not results:
            logger.error("No test results to analyze")
            return {}
        
        # Print statistics
        self.get_test_statistics(results)
        
        # Generate comprehensive analysis
        analysis = self.analyze_results(results)
        
        return analysis


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description="Run AI Model Combination Tests")
    parser.add_argument(
        "--test-type", 
        choices=["comprehensive", "essential", "performance", "quick"],
        default="essential",
        help="Type of test to run (default: essential)"
    )
    parser.add_argument(
        "--config", 
        help="Configuration file path"
    )
    parser.add_argument(
        "--output-dir", 
        default="test_outputs",
        help="Output directory for test results (default: test_outputs)"
    )
    parser.add_argument(
        "--models",
        nargs="+", 
        choices=["gpt4", "claude", "gemini"],
        help="Models to test (overrides config file)"
    )
    parser.add_argument(
        "--validation-mode",
        choices=["strict", "balanced", "lenient"],
        help="Validation mode to use (overrides config file)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent tests (default: 3)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test runner
    runner = ModelCombinationTestRunner(
        config_file=args.config,
        output_directory=args.output_dir
    )
    
    # Override config with command line args
    if args.models:
        runner.config["models"] = args.models
    if args.validation_mode:
        runner.config["validation_modes"] = [args.validation_mode]
    if args.max_concurrent:
        runner.config["max_concurrent_tests"] = args.max_concurrent
    
    # Run tests
    try:
        analysis = asyncio.run(runner.run_and_analyze(args.test_type))
        
        if analysis:
            print(f"\nAnalysis complete! Check output directory: {args.output_dir}")
            return 0
        else:
            print("Test execution failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())