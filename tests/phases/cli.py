#!/usr/bin/env python3
"""
Phase Testing CLI Tool
Command-line interface for running phase tests with various configurations
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.phases.orchestrator import PhaseTestOrchestrator, PipelineTestConfig, PhaseTestPresets
from tests.phases.base import PhaseTestConfig
from tests.phases.individual.test_phase0 import Phase0Test  
from tests.phases.individual.test_phase1 import Phase1Test
from tests.phases.individual.test_phase2 import Phase2Test


class PhaseTestCLI:
    """Command-line interface for phase testing"""
    
    def __init__(self):
        self.orchestrator = PhaseTestOrchestrator()
        self._register_phase_tests()
    
    def _register_phase_tests(self):
        """Register all available phase tests"""
        self.orchestrator.register_phase_test(Phase0Test())
        self.orchestrator.register_phase_test(Phase1Test())
        self.orchestrator.register_phase_test(Phase2Test())
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser"""
        parser = argparse.ArgumentParser(
            description="Phase Testing CLI - Run AI Estimate Pipeline phase tests",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run single phase with default config
  python -m tests.phases.cli single --phase 1
  
  # Run pipeline with specific models
  python -m tests.phases.cli pipeline --phases 0 1 2 --models gpt4 claude
  
  # Run comparison test
  python -m tests.phases.cli compare --phase 1 --models gpt4 claude gemini
  
  # Run with custom config
  python -m tests.phases.cli single --config tests/phases/configs/fast_test.yaml
  
  # List available configurations
  python -m tests.phases.cli list-configs
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Single phase command
        single_parser = subparsers.add_parser('single', help='Run a single phase test')
        single_parser.add_argument('--phase', type=int, required=True, 
                                 choices=[0, 1, 2], help='Phase number to test')
        single_parser.add_argument('--models', nargs='+', default=['gpt4', 'claude', 'gemini'],
                                 choices=['gpt4', 'claude', 'gemini'],
                                 help='AI models to use')
        single_parser.add_argument('--validation-mode', choices=['strict', 'balanced', 'lenient'],
                                 default='balanced', help='Validation mode')
        single_parser.add_argument('--config', help='Path to YAML configuration file')
        single_parser.add_argument('--timeout', type=int, default=300,
                                 help='Timeout per phase in seconds')
        single_parser.add_argument('--test-name', help='Custom test name')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Run multiple phases in sequence')
        pipeline_parser.add_argument('--phases', nargs='+', type=int, default=[0, 1, 2],
                                   choices=[0, 1, 2], help='Phases to run in sequence')
        pipeline_parser.add_argument('--models', nargs='+', default=['gpt4', 'claude', 'gemini'],
                                   choices=['gpt4', 'claude', 'gemini'],
                                   help='AI models to use')
        pipeline_parser.add_argument('--validation-mode', choices=['strict', 'balanced', 'lenient'],
                                   default='balanced', help='Validation mode')
        pipeline_parser.add_argument('--continue-on-failure', action='store_true',
                                   help='Continue pipeline even if a phase fails')
        pipeline_parser.add_argument('--timeout', type=int, default=300,
                                   help='Timeout per phase in seconds')
        pipeline_parser.add_argument('--test-name', help='Custom test name')
        
        # Comparison command
        compare_parser = subparsers.add_parser('compare', help='Compare different configurations')
        compare_parser.add_argument('--phase', type=int, required=True,
                                  choices=[0, 1, 2], help='Phase number to compare')
        compare_parser.add_argument('--compare-type', choices=['models', 'validation'], 
                                  default='models', help='What to compare')
        compare_parser.add_argument('--models', nargs='+', default=['gpt4', 'claude', 'gemini'],
                                  choices=['gpt4', 'claude', 'gemini'],
                                  help='Models to include in comparison')
        compare_parser.add_argument('--validation-modes', nargs='+', 
                                  choices=['strict', 'balanced', 'lenient'],
                                  default=['strict', 'balanced', 'lenient'],
                                  help='Validation modes to compare')
        
        # Scenario command
        scenario_parser = subparsers.add_parser('scenario', help='Run predefined test scenarios')
        scenario_parser.add_argument('scenario_name', choices=['validation_comparison', 'full_pipeline', 'performance'],
                                   help='Predefined scenario to run')
        scenario_parser.add_argument('--phase', type=int, help='Phase number for performance scenario')
        
        # List configurations command
        list_parser = subparsers.add_parser('list-configs', help='List available configurations')
        
        # List results command
        results_parser = subparsers.add_parser('list-results', help='List recent test results')
        results_parser.add_argument('--limit', type=int, default=10, help='Number of results to show')
        
        # Global options
        parser.add_argument('--output-dir', default='test_outputs',
                          help='Output directory for test results')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='Verbose output')
        parser.add_argument('--quiet', '-q', action='store_true',
                          help='Quiet mode - minimal output')
        
        return parser
    
    async def run_single_phase(self, args) -> int:
        """Run a single phase test"""
        if not self.quiet_mode:
            print(f"Running Phase {args.phase} test...")
            print(f"Models: {args.models}")
            print(f"Validation mode: {args.validation_mode}")
        
        try:
            if args.config:
                # Load from config file
                config = PhaseTestBase().load_test_config(Path(args.config).stem)
                config.phase_numbers = [args.phase]
                config.models = args.models
            else:
                # Create config from arguments
                config = PhaseTestConfig(
                    phase_numbers=[args.phase],
                    models=args.models,
                    validation_mode=args.validation_mode,
                    timeout_seconds=args.timeout,
                    test_name=args.test_name or f"cli_phase{args.phase}",
                    description=f"CLI test of Phase {args.phase}"
                )
            
            result = await self.orchestrator.run_single_phase(args.phase, config)
            
            # Display results
            self.display_phase_result(result, args.phase)
            
            return 0 if result.success else 1
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"Error running phase test: {e}")
            return 1
    
    async def run_pipeline(self, args) -> int:
        """Run pipeline test"""
        if not self.quiet_mode:
            print(f"Running pipeline with phases: {args.phases}")
            print(f"Models: {args.models}")
            print(f"Continue on failure: {args.continue_on_failure}")
        
        try:
            config = PipelineTestConfig(
                phases=args.phases,
                models=args.models,
                validation_mode=args.validation_mode,
                continue_on_failure=args.continue_on_failure,
                timeout_per_phase=args.timeout,
                test_name=args.test_name or f"cli_pipeline_{'_'.join(map(str, args.phases))}",
                description=f"CLI pipeline test of phases {args.phases}"
            )
            
            session = await self.orchestrator.run_phase_pipeline(config)
            
            # Display results
            self.display_pipeline_results(session)
            
            return 0 if session.overall_success else 1
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"Error running pipeline: {e}")
            return 1
    
    async def run_comparison(self, args) -> int:
        """Run comparison test"""
        if not self.quiet_mode:
            print(f"Running comparison for Phase {args.phase}")
            print(f"Comparing: {args.compare_type}")
        
        try:
            configs = []
            
            if args.compare_type == 'models':
                # Compare different model combinations
                model_combinations = [
                    [args.models[0]] if args.models else ['gpt4'],  # Single model
                    args.models[:2] if len(args.models) >= 2 else ['gpt4', 'claude'],  # Two models
                    args.models if len(args.models) >= 3 else ['gpt4', 'claude', 'gemini']  # All models
                ]
                
                for i, models in enumerate(model_combinations):
                    configs.append(PhaseTestConfig(
                        phase_numbers=[args.phase],
                        models=models,
                        test_name=f"model_comparison_{i+1}",
                        description=f"Model comparison test {i+1}: {models}"
                    ))
            
            elif args.compare_type == 'validation':
                # Compare validation modes
                for mode in args.validation_modes:
                    configs.append(PhaseTestConfig(
                        phase_numbers=[args.phase],
                        models=args.models,
                        validation_mode=mode,
                        test_name=f"validation_{mode}",
                        description=f"Validation mode comparison: {mode}"
                    ))
            
            comparison_result = await self.orchestrator.run_comparison_test(args.phase, configs)
            
            # Display results
            self.display_comparison_results(comparison_result)
            
            return 0
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"Error running comparison: {e}")
            return 1
    
    async def run_scenario(self, args) -> int:
        """Run predefined scenario"""
        if not self.quiet_mode:
            print(f"Running scenario: {args.scenario_name}")
        
        try:
            if args.scenario_name == 'validation_comparison':
                scenario = PhaseTestPresets.validation_comparison()
            elif args.scenario_name == 'full_pipeline':
                pipeline_config = PhaseTestPresets.full_pipeline_test()
                session = await self.orchestrator.run_phase_pipeline(pipeline_config)
                self.display_pipeline_results(session)
                return 0 if session.overall_success else 1
            elif args.scenario_name == 'performance':
                if not args.phase:
                    print("--phase required for performance scenario")
                    return 1
                scenario = PhaseTestPresets.performance_comparison(args.phase)
            else:
                print(f"Unknown scenario: {args.scenario_name}")
                return 1
            
            scenario_result = await self.orchestrator.run_test_scenario(scenario)
            self.display_scenario_results(scenario_result)
            
            return 0 if scenario_result['overall_success'] else 1
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"Error running scenario: {e}")
            return 1
    
    def list_configurations(self, args) -> int:
        """List available configurations"""
        config_dir = Path(__file__).parent / "configs"
        
        print("Available test configurations:")
        print("=" * 50)
        
        if not config_dir.exists():
            print("No configuration directory found")
            return 0
        
        for config_file in sorted(config_dir.glob("*.yaml")):
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                print(f"\n{config_file.stem}:")
                print(f"  Description: {config_data.get('description', 'N/A')}")
                print(f"  Phases: {config_data.get('phase_numbers', 'N/A')}")
                print(f"  Models: {config_data.get('models', 'N/A')}")
                print(f"  Validation: {config_data.get('validation_mode', 'N/A')}")
                
            except Exception as e:
                print(f"\n{config_file.stem}: Error loading config - {e}")
        
        return 0
    
    def list_results(self, args) -> int:
        """List recent test results"""
        output_dir = Path(args.output_dir)
        
        if not output_dir.exists():
            print("No output directory found")
            return 0
        
        # Find all result files
        result_files = []
        for pattern in ["*.json", "sessions/*.json", "comparisons/*.json", "scenarios/*.json"]:
            result_files.extend(output_dir.glob(pattern))
        
        # Sort by modification time
        result_files = sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"Recent test results (showing last {args.limit}):")
        print("=" * 60)
        
        for i, result_file in enumerate(result_files[:args.limit]):
            try:
                stat = result_file.stat()
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                
                print(f"\n{i+1}. {result_file.name}")
                print(f"   Path: {result_file}")
                print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Size: {stat.st_size:,} bytes")
                
                # Try to extract basic info from the file
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    if 'test_metadata' in data:
                        metadata = data['test_metadata']
                        print(f"   Phase: {metadata.get('phase_number', 'N/A')}")
                        print(f"   Test: {metadata.get('test_config', {}).get('test_name', 'N/A')}")
                    elif 'session_id' in data:
                        print(f"   Type: Pipeline Session")
                        print(f"   Success: {data.get('overall_success', 'N/A')}")
                    elif 'scenario_name' in data:
                        print(f"   Type: Scenario - {data.get('scenario_name')}")
                        print(f"   Success: {data.get('overall_success', 'N/A')}")
                
                except:
                    pass  # Skip if can't parse JSON
                    
            except Exception as e:
                print(f"\n{i+1}. {result_file.name} - Error: {e}")
        
        return 0
    
    def display_phase_result(self, result, phase_number: int):
        """Display single phase test result"""
        if self.quiet_mode:
            print("SUCCESS" if result.success else "FAILURE")
            return
        
        print(f"\nPhase {phase_number} Test Results:")
        print("=" * 40)
        print(f"Success: {'OK' if result.success else 'FAIL'}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        if result.success:
            print(f"Confidence Score: {result.confidence_score:.2f}")
            print(f"Consensus Level: {result.consensus_level:.2f}")
            print(f"Models Responded: {result.models_responded}/{result.total_models}")
            
            if result.validation_results:
                print("\nValidation Results:")
                for key, value in result.validation_results.items():
                    if isinstance(value, dict) and 'valid' in value:
                        status = 'OK' if value['valid'] else 'FAIL'
                        print(f"  {key}: {status}")
        else:
            print(f"Error: {result.error_message}")
    
    def display_pipeline_results(self, session):
        """Display pipeline test results"""
        if self.quiet_mode:
            print("SUCCESS" if session.overall_success else "FAILURE")
            return
        
        print(f"\nPipeline Results:")
        print("=" * 50)
        print(f"Session ID: {session.session_id}")
        print(f"Overall Success: {'OK' if session.overall_success else 'FAIL'}")
        print(f"Total Time: {session.total_execution_time:.2f}s")
        print(f"Phases Completed: {len(session.phase_results)}")
        
        print("\nPhase Results:")
        for result in session.phase_results:
            status = 'OK' if result.success else 'FAIL'
            print(f"  Phase {result.phase_number}: {status} ({result.execution_time:.2f}s)")
            if not result.success and result.error_message:
                print(f"    Error: {result.error_message}")
    
    def display_comparison_results(self, comparison_result):
        """Display comparison test results"""
        if self.quiet_mode:
            best_index = comparison_result.get('best_config_index', 0)
            print(f"BEST_CONFIG: {best_index + 1}")
            return
        
        print(f"\nComparison Results for Phase {comparison_result['phase_number']}:")
        print("=" * 60)
        
        results = comparison_result['results']
        analysis = comparison_result['analysis']
        
        print("\nConfiguration Results:")
        for i, result_data in enumerate(results):
            result = result_data['result']
            config = result_data['config']
            
            status = 'OK' if result.success else 'FAIL'
            print(f"\n{i+1}. {config.test_name}: {status}")
            print(f"   Models: {config.models}")
            print(f"   Validation: {config.validation_mode}")
            print(f"   Time: {result.execution_time:.2f}s")
            if result.success:
                print(f"   Confidence: {result.confidence_score:.2f}")
                print(f"   Score: {analysis.get('scores', [0])[i]:.1f}")
        
        best_index = analysis.get('best_config_index', 0)
        print(f"\nBest Configuration: #{best_index + 1}")
        
        if 'recommendations' in analysis:
            print("\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
    
    def display_scenario_results(self, scenario_result):
        """Display scenario test results"""
        if self.quiet_mode:
            print("SUCCESS" if scenario_result['overall_success'] else "FAILURE")
            return
        
        print(f"\nScenario Results: {scenario_result['scenario_name']}")
        print("=" * 60)
        print(f"Description: {scenario_result['description']}")
        print(f"Overall Success: {'OK' if scenario_result['overall_success'] else 'FAIL'}")
        
        print("\nConfiguration Results:")
        for i, result in enumerate(scenario_result['results']):
            status = '✓' if result.get('success', False) else '✗'
            print(f"  {i+1}. {result.get('config_name', 'Unknown')}: {status}")
            
            if result.get('config_type') == 'single_phase':
                print(f"      Phase: {result.get('phase_number')}")
                print(f"      Time: {result.get('execution_time', 0):.2f}s")
            elif result.get('config_type') == 'pipeline':
                print(f"      Phases: {result.get('phases_completed', 0)}")
                print(f"      Total Time: {result.get('total_time', 0):.2f}s")
    
    async def run(self, args=None) -> int:
        """Run the CLI with given arguments"""
        if args is None:
            args = sys.argv[1:]
        
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Set global flags
        self.verbose_mode = parsed_args.verbose
        self.quiet_mode = parsed_args.quiet
        
        # Ensure output directory exists
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
        self.orchestrator.output_dir = Path(parsed_args.output_dir)
        
        # Route to appropriate command handler
        if parsed_args.command == 'single':
            return await self.run_single_phase(parsed_args)
        elif parsed_args.command == 'pipeline':
            return await self.run_pipeline(parsed_args)
        elif parsed_args.command == 'compare':
            return await self.run_comparison(parsed_args)
        elif parsed_args.command == 'scenario':
            return await self.run_scenario(parsed_args)
        elif parsed_args.command == 'list-configs':
            return self.list_configurations(parsed_args)
        elif parsed_args.command == 'list-results':
            return self.list_results(parsed_args)
        else:
            parser.print_help()
            return 1


async def main():
    """Main entry point"""
    cli = PhaseTestCLI()
    try:
        return await cli.run()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))