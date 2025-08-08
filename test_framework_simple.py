"""
Simple framework functionality test (no pytest required)
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_framework_basics():
    """Test basic framework functionality"""
    print("Testing Phase Testing Framework...")
    
    try:
        # Test imports
        from tests.phases.base import PhaseTestConfig, PhaseTestResult
        from tests.phases.orchestrator import PhaseTestOrchestrator, PipelineTestConfig
        from tests.phases.individual.test_phase1 import Phase1Test
        print("OK Imports successful")
        
        # Test configuration creation
        config = PhaseTestConfig(
            phase_numbers=[1],
            models=["gpt4"],
            validation_mode="balanced",
            test_name="basic_test"
        )
        assert config.phase_numbers == [1]
        assert config.models == ["gpt4"]
        print("OK PhaseTestConfig creation successful")
        
        # Test result creation
        result = PhaseTestResult(
            phase_number=1,
            success=True,
            execution_time=10.5,
            confidence_score=0.85,
            models_responded=1,
            total_models=1
        )
        assert result.phase_number == 1
        assert result.model_success_rate == 1.0
        print("OK PhaseTestResult creation successful")
        
        # Test orchestrator
        orchestrator = PhaseTestOrchestrator("test_temp")
        phase1_test = Phase1Test()
        orchestrator.register_phase_test(phase1_test)
        
        registered_phases = orchestrator.get_registered_phases()
        assert 1 in registered_phases
        print("OK Orchestrator registration successful")
        
        # Test Phase1Test properties
        assert phase1_test.phase_number == 1
        assert phase1_test.phase_name == "Merge Measurement & Work Scope"
        print("OK Phase1Test properties correct")
        
        print("\nSUCCESS: All basic framework tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\nFAILED: Framework test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_help():
    """Test CLI help functionality"""
    print("\nTesting CLI help...")
    
    try:
        from tests.phases.cli import PhaseTestCLI
        cli = PhaseTestCLI()
        parser = cli.create_parser()
        
        # Test that help can be generated without errors
        help_text = parser.format_help()
        assert "Phase Testing CLI" in help_text
        assert "single" in help_text
        assert "pipeline" in help_text
        
        print("OK CLI help generation successful")
        return True
        
    except Exception as e:
        print(f"FAILED: CLI test FAILED: {e}")
        return False


def main():
    """Run all basic tests"""
    print("=" * 60)
    print("AI Estimate Pipeline - Phase Testing Framework")
    print("Basic Functionality Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    if test_framework_basics():
        tests_passed += 1
    
    if test_cli_help():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("SUCCESS: Framework is ready to use!")
        print("\nQuick start commands:")
        print("  python run_phase_tests.py --help")
        print("  python run_phase_tests.py single --phase 1")
        print("  python run_phase_tests.py list-configs")
    else:
        print("FAILED: Some tests failed - check the framework setup")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)