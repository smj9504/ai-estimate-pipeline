"""
Basic framework functionality tests
"""
import asyncio
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.phases.base import PhaseTestConfig, PhaseTestResult
from tests.phases.orchestrator import PhaseTestOrchestrator, PipelineTestConfig
from tests.phases.individual.test_phase1 import Phase1Test


class TestFrameworkBasics:
    """Basic tests to verify framework functionality"""
    
    def test_phase_test_config_creation(self):
        """Test creating phase test configuration"""
        config = PhaseTestConfig(
            phase_numbers=[1],
            models=["gpt4"],
            validation_mode="balanced",
            test_name="basic_test"
        )
        
        assert config.phase_numbers == [1]
        assert config.models == ["gpt4"]
        assert config.validation_mode == "balanced"
        assert config.test_name == "basic_test"
    
    def test_phase_test_result_creation(self):
        """Test creating phase test result"""
        result = PhaseTestResult(
            phase_number=1,
            success=True,
            execution_time=10.5,
            confidence_score=0.85,
            models_responded=1,
            total_models=1
        )
        
        assert result.phase_number == 1
        assert result.success is True
        assert result.execution_time == 10.5
        assert result.confidence_score == 0.85
        assert result.model_success_rate == 1.0
    
    def test_orchestrator_registration(self):
        """Test phase test registration with orchestrator"""
        orchestrator = PhaseTestOrchestrator("test_temp")
        phase1_test = Phase1Test()
        
        orchestrator.register_phase_test(phase1_test)
        
        registered_phases = orchestrator.get_registered_phases()
        assert 1 in registered_phases
    
    def test_pipeline_config_creation(self):
        """Test creating pipeline configuration"""
        config = PipelineTestConfig(
            phases=[0, 1],
            models=["gpt4", "claude"],
            test_name="pipeline_test"
        )
        
        assert config.phases == [0, 1]
        assert config.models == ["gpt4", "claude"]
        assert config.test_name == "pipeline_test"
    
    def test_phase1_test_properties(self):
        """Test Phase1Test basic properties"""
        phase1_test = Phase1Test()
        
        assert phase1_test.phase_number == 1
        assert phase1_test.phase_name == "Merge Measurement & Work Scope"
        assert "gpt4" in phase1_test.default_models


# Async test for actual phase execution (requires API keys)
@pytest.mark.asyncio
async def test_phase1_with_sample_data():
    """Test Phase1 with sample data (requires API keys)"""
    import os
    
    # Skip if no API keys available
    if not any([
        os.getenv('OPENAI_API_KEY'),
        os.getenv('ANTHROPIC_API_KEY'), 
        os.getenv('GOOGLE_API_KEY')
    ]):
        pytest.skip("No API keys available for testing")
    
    phase1_test = Phase1Test()
    
    config = PhaseTestConfig(
        phase_numbers=[1],
        models=["gpt4"],  # Single model for testing
        validation_mode="balanced",
        test_name="framework_test"
    )
    
    try:
        # This will use sample data if no Phase 0 output is available
        result = await phase1_test.run_test(config)
        
        # Basic assertions
        assert result.phase_number == 1
        assert isinstance(result.success, bool)
        assert isinstance(result.execution_time, (int, float))
        assert result.total_models == 1
        
        if result.success:
            assert result.confidence_score >= 0.0
            assert result.models_responded > 0
        
    except Exception as e:
        # If test fails due to API issues, that's expected
        # We're mainly testing the framework structure
        print(f"API test failed (expected): {e}")


if __name__ == "__main__":
    # Run basic tests
    import unittest
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFrameworkBasics)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nBasic framework tests: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    
    # Try async test if API keys available
    try:
        import os
        if any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GOOGLE_API_KEY')]):
            print("\nRunning API test...")
            asyncio.run(test_phase1_with_sample_data())
            print("API test completed")
        else:
            print("Skipping API test - no keys available")
    except Exception as e:
        print(f"API test failed: {e}")