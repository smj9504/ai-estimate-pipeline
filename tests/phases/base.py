"""
Base classes and utilities for phase testing framework
"""
import asyncio
import json
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from ..utils.test_data_loader import TestDataLoader


@dataclass
class PhaseTestConfig:
    """Configuration for phase testing"""
    phase_numbers: List[int]
    models: List[str]
    validation_mode: str = "balanced"
    timeout_seconds: int = 300
    retry_attempts: int = 2
    save_outputs: bool = True
    output_directory: str = "test_outputs"
    test_name: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhaseTestConfig':
        return cls(**data)


@dataclass 
class PhaseTestResult:
    """Result from running a phase test"""
    phase_number: int
    success: bool
    execution_time: float
    confidence_score: float = 0.0
    consensus_level: float = 0.0
    models_responded: int = 0
    total_models: int = 0
    validation_results: Dict[str, Any] = None
    error_message: str = ""
    output_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def model_success_rate(self) -> float:
        """Calculate model success rate"""
        if self.total_models == 0:
            return 0.0
        return self.models_responded / self.total_models


@dataclass
class TestSession:
    """Test session containing multiple phase results"""
    session_id: str
    test_config: PhaseTestConfig
    phase_results: List[PhaseTestResult] 
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    overall_success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'test_config': self.test_config.to_dict(),
            'phase_results': [result.to_dict() for result in self.phase_results],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_execution_time': self.total_execution_time,
            'overall_success': self.overall_success
        }


class PhaseTestBase(ABC):
    """Base class for all phase tests"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.logger = get_logger(__name__)
        self.test_data_loader = TestDataLoader()  # Use new centralized data loader
        self.test_data_dir = Path(__file__).parent / "fixtures"  # Keep for backward compatibility
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
    
    @property
    @abstractmethod
    def phase_number(self) -> int:
        """Phase number this test is for"""
        pass
    
    @property
    @abstractmethod 
    def phase_name(self) -> str:
        """Human readable phase name"""
        pass
    
    @property
    def default_models(self) -> List[str]:
        """Default AI models to use for testing"""
        return ["gpt4", "claude", "gemini"]
    
    @abstractmethod
    async def prepare_test_data(self, test_config: PhaseTestConfig) -> Dict[str, Any]:
        """Prepare input data for the phase test"""
        pass
    
    @abstractmethod
    async def execute_phase(self, input_data: Dict[str, Any], 
                          test_config: PhaseTestConfig) -> PhaseTestResult:
        """Execute the specific phase with given configuration"""
        pass
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data meets phase requirements"""
        # Default implementation - override in subclasses for specific validation
        return input_data is not None and isinstance(input_data, dict)
    
    def analyze_results(self, result: PhaseTestResult) -> Dict[str, Any]:
        """Analyze phase test results and provide insights"""
        analysis = {
            'performance_metrics': {
                'execution_time': result.execution_time,
                'model_success_rate': result.model_success_rate,
                'confidence_score': result.confidence_score,
                'consensus_level': result.consensus_level
            },
            'quality_metrics': {
                'overall_success': result.success,
                'validation_passed': result.validation_results is not None
            },
            'recommendations': []
        }
        
        # Add performance recommendations
        if result.execution_time > 30:
            analysis['recommendations'].append("Consider optimizing for faster execution")
        
        if result.model_success_rate < 0.8:
            analysis['recommendations'].append("Check API connectivity and error handling")
        
        if result.confidence_score < 0.7:
            analysis['recommendations'].append("Review model agreement and input quality")
        
        return analysis
    
    async def run_test(self, test_config: PhaseTestConfig) -> PhaseTestResult:
        """Run the complete phase test"""
        start_time = datetime.now()
        
        try:
            # Prepare test data
            input_data = await self.prepare_test_data(test_config)
            
            # Validate input
            if not self.validate_input_data(input_data):
                raise ValueError(f"Invalid input data for Phase {self.phase_number}")
            
            # Execute phase
            result = await self.execute_phase(input_data, test_config)
            
            # Save outputs if requested
            if test_config.save_outputs:
                await self.save_test_output(result, test_config)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Phase {self.phase_number} test failed: {e}")
            
            return PhaseTestResult(
                phase_number=self.phase_number,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                total_models=len(test_config.models)
            )
    
    async def save_test_output(self, result: PhaseTestResult, 
                             test_config: PhaseTestConfig) -> Path:
        """Save test output to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filename
        models_str = '_'.join([m[:3].upper() for m in test_config.models])
        test_name = test_config.test_name or f"phase{self.phase_number}"
        filename = f"{test_name}_{models_str}_{test_config.validation_mode}_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        # Prepare output data
        output_data = {
            'test_metadata': {
                'phase_number': self.phase_number,
                'phase_name': self.phase_name,
                'test_config': test_config.to_dict(),
                'timestamp': timestamp
            },
            'test_result': result.to_dict(),
            'analysis': self.analyze_results(result)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Test output saved: {output_file}")
        return output_file
    
    def load_fixture(self, fixture_name: str) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """Load test fixture data - now uses centralized test data loader"""
        try:
            # Try loading from new centralized test data first
            if fixture_name == "sample_demo":
                return self.test_data_loader.load_demo_data()
            elif fixture_name == "sample_measurement":
                return self.test_data_loader.load_measurement_data() 
            elif fixture_name == "sample_intake_form":
                return self.test_data_loader.load_intake_form()
            elif fixture_name == "combined_project_data":
                return self.test_data_loader.create_combined_project_data()
            else:
                # Fall back to old fixture loading for backward compatibility
                fixture_file = self.test_data_dir / f"{fixture_name}.json"
                if not fixture_file.exists():
                    raise FileNotFoundError(f"Fixture not found: {fixture_file}")
                
                with open(fixture_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load fixture '{fixture_name}': {e}")
            # Try fallback to old method
            fixture_file = self.test_data_dir / f"{fixture_name}.json" 
            if fixture_file.exists():
                with open(fixture_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            raise
    
    def load_test_config(self, config_name: str) -> PhaseTestConfig:
        """Load test configuration"""
        config_file = Path(__file__).parent / "configs" / f"{config_name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Test config not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return PhaseTestConfig.from_dict(config_data)


class PhaseTestMixin:
    """Mixin providing common testing utilities"""
    
    def assert_phase_success(self, result: PhaseTestResult, min_confidence: float = 0.7):
        """Assert that phase executed successfully with minimum confidence"""
        assert result.success, f"Phase {result.phase_number} failed: {result.error_message}"
        assert result.confidence_score >= min_confidence, \
            f"Confidence score {result.confidence_score} below threshold {min_confidence}"
    
    def assert_model_consensus(self, result: PhaseTestResult, min_consensus: float = 0.6):
        """Assert minimum model consensus level"""
        assert result.consensus_level >= min_consensus, \
            f"Consensus level {result.consensus_level} below threshold {min_consensus}"
    
    def assert_validation_passed(self, result: PhaseTestResult):
        """Assert that validation checks passed"""
        assert result.validation_results is not None, "No validation results found"
        
        if isinstance(result.validation_results, dict):
            overall_valid = result.validation_results.get('overall_valid', False)
            assert overall_valid, f"Validation failed: {result.validation_results}"
    
    def assert_execution_time(self, result: PhaseTestResult, max_time: float = 60.0):
        """Assert execution time within acceptable limits"""
        assert result.execution_time <= max_time, \
            f"Execution time {result.execution_time}s exceeds limit {max_time}s"
    
    def compare_results(self, result1: PhaseTestResult, result2: PhaseTestResult) -> Dict[str, Any]:
        """Compare two phase test results"""
        return {
            'confidence_diff': result1.confidence_score - result2.confidence_score,
            'consensus_diff': result1.consensus_level - result2.consensus_level,
            'time_diff': result1.execution_time - result2.execution_time,
            'success_comparison': {
                'result1_success': result1.success,
                'result2_success': result2.success,
                'both_successful': result1.success and result2.success
            },
            'model_performance': {
                'result1_rate': result1.model_success_rate,
                'result2_rate': result2.model_success_rate
            }
        }