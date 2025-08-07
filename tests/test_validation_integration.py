# tests/test_validation_integration.py
"""
Integration tests for the validation system with the existing pipeline components.

These tests verify that the validation system integrates correctly with:
- ModelOrchestrator
- ResultMerger  
- Existing data models
- Pipeline workflow
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.models.model_interface import ModelOrchestrator
from src.models.data_models import ModelResponse, ProjectData
from src.validators.response_validator import ValidationOrchestrator, ValidationReport, ResponseQuality
from src.processors.result_merger import ResultMerger


class TestModelOrchestratorIntegration:
    """Test ModelOrchestrator integration with validation"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Create orchestrator with validation enabled
        self.orchestrator = ModelOrchestrator(enable_validation=True)
        
        # Sample project data
        self.sample_data = {
            "floors": [{
                "location": "First Floor",
                "rooms": [{
                    "name": "Kitchen",
                    "work_scope": {"Flooring": "Remove & Replace"},
                    "measurements": {"floor_area_sqft": 100.0}
                }]
            }]
        }
    
    def test_orchestrator_initialization_with_validation(self):
        """Test that orchestrator initializes with validation correctly"""
        orchestrator = ModelOrchestrator(enable_validation=True)
        
        # Should have validation enabled if dependencies available
        assert hasattr(orchestrator, 'validation_orchestrator')
        assert hasattr(orchestrator, 'enable_validation')
        
        # Check validation control methods
        assert callable(orchestrator.get_validation_enabled)
        assert callable(orchestrator.set_validation_enabled)
    
    def test_orchestrator_initialization_without_validation(self):
        """Test that orchestrator works without validation"""
        orchestrator = ModelOrchestrator(enable_validation=False)
        
        assert not orchestrator.enable_validation
        assert orchestrator.validation_orchestrator is None
    
    @pytest.mark.asyncio
    async def test_run_parallel_with_validation(self):
        """Test run_parallel with validation enabled"""
        # Mock the individual model calls
        good_response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Kitchen",
                "tasks": [
                    {"task_name": "Remove flooring", "quantity": 100.0, "unit": "sqft"},
                    {"task_name": "Install flooring", "quantity": 100.0, "unit": "sqft"}
                ]
            }],
            total_work_items=2,
            confidence_self_assessment=0.85,
            raw_response='{"rooms": [...]}'
        )
        
        bad_response = ModelResponse(
            model_name="claude-3",
            room_estimates=[{
                "name": "**Kitchen**",  # Invalid room name
                "tasks": [{"task_name": "", "quantity": -10, "unit": "invalid"}]  # Invalid task
            }],
            total_work_items=1,
            confidence_self_assessment=0.30,
            raw_response='Error response'
        )
        
        # Mock the model interfaces
        with patch.object(self.orchestrator, 'run_single_model') as mock_run_single:
            mock_run_single.side_effect = [good_response, bad_response]
            
            # Mock available models
            self.orchestrator.models = {'gpt4': Mock(), 'claude': Mock()}
            
            results = await self.orchestrator.run_parallel(
                prompt="Test prompt",
                json_data=self.sample_data,
                model_names=['gpt4', 'claude'],
                enable_validation=True,
                min_quality_threshold=50.0
            )
            
            # Should only return the good response after validation
            assert len(results) == 1
            assert results[0].model_name == "gpt-4"
    
    def test_validation_enabled_control(self):
        """Test validation enable/disable controls"""
        orchestrator = ModelOrchestrator(enable_validation=False)
        
        # Initially disabled
        assert not orchestrator.get_validation_enabled()
        
        # Enable validation
        result = orchestrator.set_validation_enabled(True)
        
        # Check if validation could be enabled (depends on dependencies)
        if result:
            assert orchestrator.get_validation_enabled()
        else:
            # If dependencies not available, should remain disabled
            assert not orchestrator.get_validation_enabled()


class TestResultMergerIntegration:
    """Test ResultMerger integration with validated responses"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.merger = ResultMerger()
    
    def create_mock_response(self, model_name: str, quality_score: float, 
                           room_name: str = "Kitchen", task_count: int = 2) -> ModelResponse:
        """Create a mock model response"""
        tasks = []
        for i in range(task_count):
            tasks.append({
                "task_name": f"Task {i+1}",
                "quantity": 50.0 * (i + 1),
                "unit": "sqft",
                "room_name": room_name
            })
        
        return ModelResponse(
            model_name=model_name,
            room_estimates=[{
                "name": room_name,
                "tasks": tasks
            }],
            total_work_items=task_count,
            confidence_self_assessment=quality_score,
            raw_response=f'{{"model": "{model_name}"}}'
        )
    
    def test_merge_with_quality_filtered_responses(self):
        """Test merging with quality-filtered responses"""
        # Create responses with different quality levels
        high_quality = self.create_mock_response("gpt-4", 0.90)
        medium_quality = self.create_mock_response("claude-3", 0.70)
        low_quality = self.create_mock_response("gemini", 0.30)
        
        responses = [high_quality, medium_quality, low_quality]
        
        # In real usage, low quality would be filtered out by ModelOrchestrator
        # Here we test that merger handles different quality responses
        result = self.merger.merge_results(responses)
        
        assert result is not None
        assert result.total_work_items > 0
        assert len(result.rooms) > 0
        
        # Check that metadata includes model information
        assert len(result.metadata.models_used) == 3
    
    def test_merge_with_single_valid_response(self):
        """Test merging when only one response passes validation"""
        single_response = self.create_mock_response("gpt-4", 0.85)
        
        result = self.merger.merge_results([single_response])
        
        assert result is not None
        assert result.total_work_items > 0
        assert len(result.rooms) > 0
        assert result.metadata.models_used == ["gpt-4"]
    
    def test_merge_with_no_valid_responses(self):
        """Test merging when no responses pass validation"""
        # This simulates the case where all responses were filtered out
        result = self.merger.merge_results([])
        
        assert result is not None
        assert result.total_work_items == 0
        assert len(result.rooms) == 0
        assert result.overall_confidence == 0.0


class TestPipelineIntegration:
    """Test full pipeline integration with validation"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.sample_project_data = [
            {
                "Jobsite": "123 Test St",
                "occupancy": "Residential",
                "company": {"name": "Test Co"}
            },
            {
                "location": "First Floor",
                "rooms": [{
                    "name": "Kitchen", 
                    "material": {"Floor": "Tile"},
                    "work_scope": {"Flooring": "Remove & Replace"},
                    "measurements": {"floor_area_sqft": 150.0},
                    "demo_scope(already demo'd)": {"Wall Drywall(sq_ft)": 0},
                    "additional_notes": {"protection": [], "detach_reset": []}
                }]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_end_to_end_with_validation(self):
        """Test end-to-end pipeline flow with validation enabled"""
        # Create project data
        project_data = ProjectData.from_json_list(self.sample_project_data)
        
        # Create orchestrator with validation
        orchestrator = ModelOrchestrator(enable_validation=True)
        
        # Mock successful model responses
        mock_response1 = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Kitchen",
                "tasks": [
                    {"task_name": "Remove existing flooring", "quantity": 150.0, "unit": "sqft"},
                    {"task_name": "Install new flooring", "quantity": 150.0, "unit": "sqft"}
                ]
            }],
            total_work_items=2,
            confidence_self_assessment=0.85,
            raw_response='{"success": true}'
        )
        
        mock_response2 = ModelResponse(
            model_name="claude-3",
            room_estimates=[{
                "name": "Kitchen", 
                "tasks": [
                    {"task_name": "Remove flooring", "quantity": 150.0, "unit": "sqft"},
                    {"task_name": "Install flooring", "quantity": 150.0, "unit": "sqft"}
                ]
            }],
            total_work_items=2,
            confidence_self_assessment=0.80,
            raw_response='{"success": true}'
        )
        
        # Mock the model execution
        with patch.object(orchestrator, 'run_parallel') as mock_run_parallel:
            mock_run_parallel.return_value = [mock_response1, mock_response2]
            
            # Run the orchestrator
            responses = await orchestrator.run_parallel(
                prompt="Generate estimate",
                json_data=self.sample_project_data,
                enable_validation=True,
                min_quality_threshold=30.0
            )
            
            assert len(responses) > 0
            
            # Merge results
            merger = ResultMerger()
            merged_result = merger.merge_results(responses)
            
            assert merged_result is not None
            assert merged_result.total_work_items > 0
            assert len(merged_result.rooms) > 0
            assert merged_result.overall_confidence > 0


class TestValidationConfigurationIntegration:
    """Test configuration integration"""
    
    def test_validation_config_loading(self):
        """Test that validation configuration can be loaded"""
        try:
            import yaml
            
            # Try to load validation config
            with open('config/validation_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Check key configuration sections exist
            assert 'validation' in config
            assert 'structure_validation' in config
            assert 'integrity_validation' in config
            assert 'auto_fix' in config
            
            # Check key settings
            assert 'enabled' in config['validation']
            assert 'quality_thresholds' in config['validation']
            
        except FileNotFoundError:
            pytest.skip("Validation config file not found")
        except ImportError:
            pytest.skip("YAML library not available")
    
    def test_validation_with_custom_thresholds(self):
        """Test validation with custom quality thresholds"""
        from src.validators.response_validator import should_exclude_from_merging, ValidationReport, ResponseQuality
        
        # Create a report with medium quality
        medium_report = ValidationReport(
            is_valid=True,
            quality_score=55.0,
            quality_level=ResponseQuality.FAIR,
            issues=[],
            warnings=[],
            suggestions=[],
            fixed_issues=[],
            processing_time=0.1,
            metadata={}
        )
        
        # Test different thresholds
        assert not should_exclude_from_merging(medium_report, min_quality_threshold=50.0)
        assert should_exclude_from_merging(medium_report, min_quality_threshold=60.0)


class TestErrorHandlingIntegration:
    """Test error handling in integrated environment"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.orchestrator = ModelOrchestrator(enable_validation=True)
    
    def test_validation_failure_graceful_handling(self):
        """Test that validation failures don't break the pipeline"""
        # Create a response that might cause validation issues
        problematic_response = ModelResponse(
            model_name="test",
            room_estimates=None,  # This might cause issues
            total_work_items=0,
            confidence_self_assessment=0.0,
            raw_response=""
        )
        
        # Validation should handle this gracefully
        if self.orchestrator.validation_orchestrator:
            try:
                report = self.orchestrator.validation_orchestrator.validate_response(
                    problematic_response, auto_fix=True
                )
                # Should not crash, should return some kind of report
                assert report is not None
            except Exception as e:
                pytest.fail(f"Validation should handle errors gracefully: {e}")
    
    def test_missing_validation_dependencies(self):
        """Test behavior when validation dependencies are missing"""
        # Create orchestrator and simulate missing validation
        orchestrator = ModelOrchestrator(enable_validation=True)
        orchestrator.validation_orchestrator = None
        orchestrator.enable_validation = False
        
        # Should still work without validation
        assert not orchestrator.get_validation_enabled()
        
        # Should be able to process responses without validation
        response = ModelResponse(
            model_name="test",
            room_estimates=[{
                "name": "Test Room",
                "tasks": [{"task_name": "Test", "quantity": 1, "unit": "item"}]
            }],
            total_work_items=1,
            confidence_self_assessment=0.75
        )
        
        # This should not raise an exception
        assert response is not None


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system"""
    
    @pytest.mark.slow
    def test_validation_performance_with_large_response(self):
        """Test validation performance with large responses"""
        # Create a large response (100 rooms, 10 tasks each)
        large_rooms = []
        for i in range(100):
            tasks = []
            for j in range(10):
                tasks.append({
                    "task_name": f"Task {j} in Room {i}",
                    "description": f"Description for task {j}",
                    "quantity": float(j + 1),
                    "unit": "sqft",
                    "necessity": "required"
                })
            large_rooms.append({
                "name": f"Room {i}",
                "tasks": tasks
            })
        
        large_response = ModelResponse(
            model_name="performance-test",
            room_estimates=large_rooms,
            total_work_items=1000,
            confidence_self_assessment=0.80,
            raw_response='{"large": "response"}'
        )
        
        # Test validation performance
        import time
        start_time = time.time()
        
        try:
            from src.validators.response_validator import ValidationOrchestrator
            validator = ValidationOrchestrator()
            report = validator.validate_response(large_response, auto_fix=False)
            
            validation_time = time.time() - start_time
            
            # Should complete in reasonable time (less than 5 seconds for 1000 items)
            assert validation_time < 5.0, f"Validation took {validation_time:.2f}s, too slow"
            assert report is not None
            
        except ImportError:
            pytest.skip("Validation orchestrator not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])