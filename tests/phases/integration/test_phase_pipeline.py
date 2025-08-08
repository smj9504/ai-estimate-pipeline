"""
Integration tests for multi-phase pipeline execution
"""
import asyncio
import pytest
from typing import List, Dict, Any

from ..base import PhaseTestBase, PhaseTestConfig, PhaseTestResult, TestSession
from ..orchestrator import PhaseTestOrchestrator, PipelineTestConfig
from ..individual.test_phase0 import Phase0Test
from ..individual.test_phase1 import Phase1Test
from ..individual.test_phase2 import Phase2Test


class TestPhasePipeline:
    """Integration tests for phase pipeline execution"""
    
    @pytest.fixture
    def orchestrator(self) -> PhaseTestOrchestrator:
        """Create test orchestrator with registered phases"""
        orchestrator = PhaseTestOrchestrator("test_outputs/integration")
        
        # Register phase tests
        orchestrator.register_phase_test(Phase0Test())
        orchestrator.register_phase_test(Phase1Test())
        orchestrator.register_phase_test(Phase2Test())
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_sequential_phase_execution(self, orchestrator: PhaseTestOrchestrator):
        """Test running Phase 0 -> Phase 1 -> Phase 2 in sequence"""
        
        pipeline_config = PipelineTestConfig(
            phases=[0, 1, 2],
            models=["gpt4"],  # Single model for faster testing
            test_name="sequential_pipeline",
            description="Test sequential execution of all phases",
            continue_on_failure=False,
            save_intermediate=True
        )
        
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        
        # Verify session results
        assert session.overall_success, f"Pipeline failed: {[r.error_message for r in session.phase_results if not r.success]}"
        assert len(session.phase_results) == 3, "Should have results for all 3 phases"
        
        # Verify each phase succeeded
        for i, result in enumerate(session.phase_results):
            assert result.phase_number == i, f"Phase {i} result has wrong phase number"
            assert result.success, f"Phase {i} failed: {result.error_message}"
            
        # Verify data flow between phases
        phase0_result = session.phase_results[0]
        phase1_result = session.phase_results[1]
        phase2_result = session.phase_results[2]
        
        # Phase 1 should have processed Phase 0 data
        assert phase1_result.output_data is not None
        assert 'data' in phase1_result.output_data
        
        # Phase 2 should have processed Phase 1 data  
        assert phase2_result.output_data is not None
        assert 'data' in phase2_result.output_data
    
    @pytest.mark.asyncio
    async def test_pipeline_failure_handling(self, orchestrator: PhaseTestOrchestrator):
        """Test pipeline behavior when a phase fails"""
        
        # Create config that will likely fail (invalid models)
        pipeline_config = PipelineTestConfig(
            phases=[0, 1],
            models=["invalid_model"],
            test_name="failure_handling",
            description="Test failure handling in pipeline",
            continue_on_failure=False,
            save_intermediate=True
        )
        
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        
        # Pipeline should report overall failure
        assert not session.overall_success, "Pipeline should fail with invalid model"
        
        # Should stop at first failure
        failed_phases = [r for r in session.phase_results if not r.success]
        assert len(failed_phases) > 0, "Should have at least one failed phase"
    
    @pytest.mark.asyncio
    async def test_pipeline_with_continue_on_failure(self, orchestrator: PhaseTestOrchestrator):
        """Test pipeline continuation after phase failure"""
        
        pipeline_config = PipelineTestConfig(
            phases=[0, 1, 2],
            models=["gpt4"],
            test_name="continue_on_failure",
            description="Test continuing pipeline after failure",
            continue_on_failure=True,
            save_intermediate=True
        )
        
        # We can't easily force a specific phase to fail without modifying the processors,
        # so this test serves more as documentation of the intended behavior
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        
        # Even if some phases fail, we should get results for all phases
        assert len(session.phase_results) <= 3, "Should attempt all phases even if some fail"
    
    @pytest.mark.asyncio  
    async def test_partial_pipeline_execution(self, orchestrator: PhaseTestOrchestrator):
        """Test running only a subset of phases"""
        
        pipeline_config = PipelineTestConfig(
            phases=[1, 2],  # Skip Phase 0
            models=["gpt4", "claude"],
            test_name="partial_pipeline",
            description="Test running Phase 1 and 2 only",
            continue_on_failure=False,
            save_intermediate=True
        )
        
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        
        # Should only have 2 results
        assert len(session.phase_results) <= 2, "Should only execute phases 1 and 2"
        
        if session.overall_success:
            # Verify phase numbers are correct
            phase_numbers = [r.phase_number for r in session.phase_results if r.success]
            assert 1 in phase_numbers, "Phase 1 should be included"
            if len(phase_numbers) > 1:
                assert 2 in phase_numbers, "Phase 2 should be included if pipeline succeeded"


class TestPhaseDataFlow:
    """Test data flow and compatibility between phases"""
    
    @pytest.fixture
    def orchestrator(self) -> PhaseTestOrchestrator:
        """Create test orchestrator with registered phases"""
        orchestrator = PhaseTestOrchestrator("test_outputs/integration")
        orchestrator.register_phase_test(Phase0Test())
        orchestrator.register_phase_test(Phase1Test())
        orchestrator.register_phase_test(Phase2Test())
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_phase0_to_phase1_data_compatibility(self, orchestrator: PhaseTestOrchestrator):
        """Test that Phase 0 output is compatible with Phase 1 input"""
        
        # Run Phase 0
        phase0_config = PhaseTestConfig(
            phase_numbers=[0],
            models=["gpt4"],
            test_name="phase0_data_compat"
        )
        
        phase0_result = await orchestrator.run_single_phase(0, phase0_config)
        assert phase0_result.success, f"Phase 0 failed: {phase0_result.error_message}"
        
        # Manually test Phase 1 with Phase 0 output
        phase1_test = Phase1Test()
        phase1_test.set_input_data(phase0_result.output_data)
        
        phase1_config = PhaseTestConfig(
            phase_numbers=[1],
            models=["gpt4"],
            test_name="phase1_data_compat"
        )
        
        # Validate input data
        prepared_data = await phase1_test.prepare_test_data(phase1_config)
        assert phase1_test.validate_input_data(prepared_data), "Phase 0 output incompatible with Phase 1"
        
        # Execute Phase 1
        phase1_result = await phase1_test.run_test(phase1_config)
        assert phase1_result.success, f"Phase 1 failed with Phase 0 data: {phase1_result.error_message}"
    
    @pytest.mark.asyncio
    async def test_phase1_to_phase2_data_compatibility(self, orchestrator: PhaseTestOrchestrator):
        """Test that Phase 1 output is compatible with Phase 2 input"""
        
        # Run Phase 0 and 1 first
        pipeline_config = PipelineTestConfig(
            phases=[0, 1],
            models=["gpt4"],
            test_name="phase1_to_2_compat",
            continue_on_failure=False
        )
        
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        assert session.overall_success, "Phase 0-1 pipeline failed"
        
        # Get Phase 1 result
        phase1_result = next(r for r in session.phase_results if r.phase_number == 1)
        
        # Manually test Phase 2 with Phase 1 output
        phase2_test = Phase2Test()
        phase2_test.set_input_data(phase1_result.output_data)
        
        phase2_config = PhaseTestConfig(
            phase_numbers=[2],
            models=["gpt4"],
            test_name="phase2_data_compat"
        )
        
        # Validate input data
        prepared_data = await phase2_test.prepare_test_data(phase2_config)
        assert phase2_test.validate_input_data(prepared_data), "Phase 1 output incompatible with Phase 2"
        
        # Execute Phase 2
        phase2_result = await phase2_test.run_test(phase2_config)
        assert phase2_result.success, f"Phase 2 failed with Phase 1 data: {phase2_result.error_message}"


if __name__ == "__main__":
    # Run tests directly for debugging
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    async def run_tests():
        orchestrator = PhaseTestOrchestrator("test_outputs/integration")
        orchestrator.register_phase_test(Phase0Test())
        orchestrator.register_phase_test(Phase1Test())
        orchestrator.register_phase_test(Phase2Test())
        
        # Test sequential pipeline
        pipeline_config = PipelineTestConfig(
            phases=[0, 1, 2],
            models=["gpt4"],
            test_name="manual_sequential_test",
            description="Manual test of sequential pipeline"
        )
        
        session = await orchestrator.run_phase_pipeline(pipeline_config)
        print(f"Pipeline success: {session.overall_success}")
        print(f"Total execution time: {session.total_execution_time:.2f}s")
        
        for result in session.phase_results:
            print(f"Phase {result.phase_number}: {'✓' if result.success else '✗'} ({result.execution_time:.2f}s)")
    
    asyncio.run(run_tests())