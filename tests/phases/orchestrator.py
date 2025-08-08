"""
Phase Test Orchestrator - Manages running phase combinations and test scenarios
"""
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass

from .base import PhaseTestBase, PhaseTestConfig, PhaseTestResult, TestSession


@dataclass
class PipelineTestConfig:
    """Configuration for running multiple phases in sequence"""
    phases: List[int]  # Phase numbers to run
    models: List[str]  # AI models to use
    validation_mode: str = "balanced"
    continue_on_failure: bool = False  # Whether to continue if a phase fails
    save_intermediate: bool = True  # Save outputs between phases
    timeout_per_phase: int = 300  # Timeout per phase in seconds
    max_parallel: int = 1  # Number of phases to run in parallel (future feature)
    test_name: str = ""
    description: str = ""


@dataclass
class PhaseTestScenario:
    """Defines a specific test scenario with multiple configurations"""
    name: str
    description: str
    configs: List[Union[PhaseTestConfig, PipelineTestConfig]]
    expected_outcomes: Dict[str, Any] = None  # Expected results for validation
    tags: List[str] = None  # Tags for categorization
    

class PhaseTestOrchestrator:
    """
    Orchestrates running multiple phase tests with different configurations
    Supports single phases, phase pipelines, and comparison testing
    """
    
    def __init__(self, output_directory: str = "test_outputs"):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        self.test_registry: Dict[int, PhaseTestBase] = {}
        self.sessions: Dict[str, TestSession] = {}
    
    def register_phase_test(self, phase_test: PhaseTestBase):
        """Register a phase test implementation"""
        self.test_registry[phase_test.phase_number] = phase_test
    
    def get_registered_phases(self) -> List[int]:
        """Get list of registered phase numbers"""
        return sorted(self.test_registry.keys())
    
    async def run_single_phase(self, phase_number: int, 
                             config: PhaseTestConfig) -> PhaseTestResult:
        """Run a single phase test"""
        if phase_number not in self.test_registry:
            raise ValueError(f"Phase {phase_number} not registered")
        
        phase_test = self.test_registry[phase_number]
        return await phase_test.run_test(config)
    
    async def run_phase_pipeline(self, pipeline_config: PipelineTestConfig) -> TestSession:
        """Run multiple phases in sequence"""
        session_id = str(uuid.uuid4())
        session = TestSession(
            session_id=session_id,
            test_config=PhaseTestConfig(
                phase_numbers=pipeline_config.phases,
                models=pipeline_config.models,
                validation_mode=pipeline_config.validation_mode,
                test_name=pipeline_config.test_name,
                description=pipeline_config.description
            ),
            phase_results=[],
            start_time=datetime.now()
        )
        
        self.sessions[session_id] = session
        
        # Track data flow between phases
        phase_data = None
        overall_success = True
        
        try:
            for phase_num in pipeline_config.phases:
                if phase_num not in self.test_registry:
                    raise ValueError(f"Phase {phase_num} not registered")
                
                # Create config for this phase
                phase_config = PhaseTestConfig(
                    phase_numbers=[phase_num],
                    models=pipeline_config.models,
                    validation_mode=pipeline_config.validation_mode,
                    timeout_seconds=pipeline_config.timeout_per_phase,
                    save_outputs=pipeline_config.save_intermediate,
                    test_name=f"{pipeline_config.test_name}_phase{phase_num}"
                )
                
                # If we have data from previous phase, use it
                if phase_data is not None:
                    # Inject previous phase data into test preparation
                    phase_test = self.test_registry[phase_num]
                    if hasattr(phase_test, 'set_input_data'):
                        phase_test.set_input_data(phase_data)
                
                # Run the phase
                result = await self.run_single_phase(phase_num, phase_config)
                session.phase_results.append(result)
                
                # Check if phase succeeded
                if not result.success:
                    overall_success = False
                    if not pipeline_config.continue_on_failure:
                        break
                
                # Prepare data for next phase
                if result.output_data:
                    phase_data = result.output_data
        
        except Exception as e:
            # Add error result if pipeline failed
            error_result = PhaseTestResult(
                phase_number=-1,
                success=False,
                execution_time=0,
                error_message=f"Pipeline error: {str(e)}"
            )
            session.phase_results.append(error_result)
            overall_success = False
        
        # Finalize session
        session.end_time = datetime.now()
        session.total_execution_time = sum(r.execution_time for r in session.phase_results)
        session.overall_success = overall_success
        
        # Save session results
        await self.save_session_results(session)
        
        return session
    
    async def run_comparison_test(self, phase_number: int, 
                                configs: List[PhaseTestConfig]) -> Dict[str, Any]:
        """Run the same phase with different configurations for comparison"""
        if phase_number not in self.test_registry:
            raise ValueError(f"Phase {phase_number} not registered")
        
        results = []
        
        # Run all configurations
        tasks = []
        for i, config in enumerate(configs):
            config.test_name = f"comparison_{i+1}"
            task = self.run_single_phase(phase_number, config)
            tasks.append((i, task))
        
        # Collect results
        for i, task in tasks:
            result = await task
            results.append({
                'config_index': i,
                'config': configs[i],
                'result': result
            })
        
        # Analyze comparisons
        comparison_analysis = self.analyze_comparison_results(results)
        
        # Save comparison report
        await self.save_comparison_report(phase_number, results, comparison_analysis)
        
        return {
            'phase_number': phase_number,
            'results': results,
            'analysis': comparison_analysis,
            'best_config_index': comparison_analysis.get('best_config_index', 0)
        }
    
    async def run_test_scenario(self, scenario: PhaseTestScenario) -> Dict[str, Any]:
        """Run a complete test scenario with multiple configurations"""
        scenario_results = {
            'scenario_name': scenario.name,
            'description': scenario.description,
            'tags': scenario.tags or [],
            'start_time': datetime.now().isoformat(),
            'results': [],
            'overall_success': True
        }
        
        for i, config in enumerate(scenario.configs):
            config_name = f"{scenario.name}_config_{i+1}"
            
            try:
                if isinstance(config, PipelineTestConfig):
                    # Pipeline configuration
                    config.test_name = config_name
                    session = await self.run_phase_pipeline(config)
                    result_summary = {
                        'config_type': 'pipeline',
                        'config_name': config_name,
                        'session_id': session.session_id,
                        'success': session.overall_success,
                        'total_time': session.total_execution_time,
                        'phases_completed': len(session.phase_results),
                        'phases_successful': sum(1 for r in session.phase_results if r.success)
                    }
                    
                elif isinstance(config, PhaseTestConfig):
                    # Single phase configuration
                    config.test_name = config_name
                    if len(config.phase_numbers) == 1:
                        result = await self.run_single_phase(config.phase_numbers[0], config)
                        result_summary = {
                            'config_type': 'single_phase',
                            'config_name': config_name,
                            'phase_number': config.phase_numbers[0],
                            'success': result.success,
                            'execution_time': result.execution_time,
                            'confidence_score': result.confidence_score
                        }
                    else:
                        raise ValueError("PhaseTestConfig with multiple phases not supported")
                else:
                    raise ValueError(f"Unknown config type: {type(config)}")
                
                scenario_results['results'].append(result_summary)
                
            except Exception as e:
                scenario_results['overall_success'] = False
                scenario_results['results'].append({
                    'config_name': config_name,
                    'success': False,
                    'error': str(e)
                })
        
        scenario_results['end_time'] = datetime.now().isoformat()
        
        # Save scenario results
        await self.save_scenario_results(scenario_results)
        
        return scenario_results
    
    def analyze_comparison_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison test results to find best configuration"""
        if not results:
            return {}
        
        # Score each configuration
        scores = []
        for result_data in results:
            result = result_data['result']
            config = result_data['config']
            
            # Calculate composite score (adjust weights as needed)
            score = 0.0
            
            # Success is most important
            if result.success:
                score += 40
            
            # Confidence score
            score += result.confidence_score * 25
            
            # Consensus level
            score += result.consensus_level * 20
            
            # Model success rate
            score += result.model_success_rate * 10
            
            # Execution time (inverse scoring - faster is better)
            if result.execution_time > 0:
                time_score = max(0, 5 - (result.execution_time / 60))  # 5 points if under 1 minute
                score += time_score
            
            scores.append(score)
        
        # Find best configuration
        best_index = scores.index(max(scores)) if scores else 0
        
        return {
            'scores': scores,
            'best_config_index': best_index,
            'best_score': scores[best_index] if scores else 0,
            'performance_comparison': {
                'execution_times': [r['result'].execution_time for r in results],
                'confidence_scores': [r['result'].confidence_score for r in results],
                'consensus_levels': [r['result'].consensus_level for r in results],
                'success_rates': [r['result'].model_success_rate for r in results]
            }
        }
    
    async def save_session_results(self, session: TestSession):
        """Save test session results to file"""
        timestamp = session.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session.session_id[:8]}_{timestamp}.json"
        
        output_file = self.output_dir / "sessions" / filename
        output_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
    
    async def save_comparison_report(self, phase_number: int, results: List[Dict[str, Any]], 
                                   analysis: Dict[str, Any]):
        """Save comparison test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_phase{phase_number}_{timestamp}.json"
        
        output_file = self.output_dir / "comparisons" / filename
        output_file.parent.mkdir(exist_ok=True)
        
        report = {
            'phase_number': phase_number,
            'timestamp': timestamp,
            'results': results,
            'analysis': analysis,
            'recommendations': self.generate_comparison_recommendations(analysis)
        }
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    async def save_scenario_results(self, scenario_results: Dict[str, Any]):
        """Save test scenario results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = scenario_results['scenario_name'].replace(' ', '_').lower()
        filename = f"scenario_{scenario_name}_{timestamp}.json"
        
        output_file = self.output_dir / "scenarios" / filename
        output_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(scenario_results, f, indent=2, ensure_ascii=False)
    
    def generate_comparison_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison analysis"""
        recommendations = []
        
        if 'performance_comparison' in analysis:
            perf = analysis['performance_comparison']
            
            # Execution time recommendations
            times = perf.get('execution_times', [])
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > 60:
                    recommendations.append("Consider optimizing for better performance - average execution time exceeds 1 minute")
                
                fastest_config = times.index(min(times))
                recommendations.append(f"Configuration {fastest_config + 1} shows best performance")
            
            # Confidence recommendations
            confidence_scores = perf.get('confidence_scores', [])
            if confidence_scores:
                best_confidence_config = confidence_scores.index(max(confidence_scores))
                if max(confidence_scores) > 0.8:
                    recommendations.append(f"Configuration {best_confidence_config + 1} shows highest confidence")
                else:
                    recommendations.append("All configurations show relatively low confidence - review input quality")
        
        best_index = analysis.get('best_config_index', 0)
        recommendations.append(f"Overall recommended configuration: {best_index + 1}")
        
        return recommendations
    
    def get_session(self, session_id: str) -> Optional[TestSession]:
        """Get test session by ID"""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())


class PhaseTestPresets:
    """Predefined test configurations for common scenarios"""
    
    @staticmethod
    def single_model_test(phase_numbers: List[int], model: str) -> PhaseTestConfig:
        """Create config for testing with a single model"""
        return PhaseTestConfig(
            phase_numbers=phase_numbers,
            models=[model],
            test_name=f"single_{model}",
            description=f"Test with {model} model only"
        )
    
    @staticmethod
    def multi_model_test(phase_numbers: List[int]) -> PhaseTestConfig:
        """Create config for testing with all models"""
        return PhaseTestConfig(
            phase_numbers=phase_numbers,
            models=["gpt4", "claude", "gemini"],
            test_name="multi_model",
            description="Test with all available models"
        )
    
    @staticmethod
    def validation_comparison() -> PhaseTestScenario:
        """Create scenario comparing different validation modes"""
        return PhaseTestScenario(
            name="validation_comparison",
            description="Compare different validation modes",
            configs=[
                PhaseTestConfig(
                    phase_numbers=[1],
                    models=["gpt4", "claude"],
                    validation_mode="strict",
                    test_name="strict_validation"
                ),
                PhaseTestConfig(
                    phase_numbers=[1],
                    models=["gpt4", "claude"],
                    validation_mode="balanced",
                    test_name="balanced_validation"
                ),
                PhaseTestConfig(
                    phase_numbers=[1],
                    models=["gpt4", "claude"],
                    validation_mode="lenient",
                    test_name="lenient_validation"
                )
            ],
            tags=["validation", "comparison"]
        )
    
    @staticmethod
    def full_pipeline_test() -> PipelineTestConfig:
        """Create config for testing complete pipeline"""
        return PipelineTestConfig(
            phases=[0, 1, 2],  # Only implemented phases
            models=["gpt4", "claude", "gemini"],
            test_name="full_pipeline",
            description="Complete pipeline test",
            continue_on_failure=False,
            save_intermediate=True
        )
    
    @staticmethod
    def performance_comparison(phase_number: int) -> PhaseTestScenario:
        """Create scenario for performance testing different model combinations"""
        return PhaseTestScenario(
            name=f"performance_phase{phase_number}",
            description=f"Performance comparison for Phase {phase_number}",
            configs=[
                PhaseTestConfig(
                    phase_numbers=[phase_number],
                    models=["gpt4"],
                    test_name="gpt4_only"
                ),
                PhaseTestConfig(
                    phase_numbers=[phase_number],
                    models=["claude"],
                    test_name="claude_only"
                ),
                PhaseTestConfig(
                    phase_numbers=[phase_number],
                    models=["gemini"],
                    test_name="gemini_only"
                ),
                PhaseTestConfig(
                    phase_numbers=[phase_number],
                    models=["gpt4", "claude"],
                    test_name="gpt4_claude"
                ),
                PhaseTestConfig(
                    phase_numbers=[phase_number],
                    models=["gpt4", "claude", "gemini"],
                    test_name="all_models"
                )
            ],
            tags=["performance", "model_comparison"]
        )