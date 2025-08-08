"""
Comprehensive Model Combination Tests
Tests all AI model combinations using actual test data and the new testing framework.
"""
import pytest
import asyncio
import logging
from unittest.mock import patch, Mock, AsyncMock

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
from tests.utils.test_data_loader import TestDataLoader, get_test_data_loader
from src.models.data_models import ModelResponse


logger = logging.getLogger(__name__)


@pytest.fixture
def test_data_loader():
    """Get test data loader instance"""
    return get_test_data_loader()


@pytest.fixture
def test_dataset(test_data_loader):
    """Get complete test dataset"""
    return test_data_loader.get_test_dataset()


@pytest.fixture
def model_tester(test_data_loader):
    """Get model combination tester"""
    return ModelCombinationTester(
        test_data_loader=test_data_loader,
        output_directory="test_outputs/comprehensive_tests"
    )


@pytest.fixture
def test_matrix():
    """Create test matrix for testing"""
    return ModelTestMatrix(
        available_models=[ModelType.GPT4, ModelType.CLAUDE, ModelType.GEMINI],
        validation_modes=[ValidationMode.BALANCED],
        processing_modes=[ProcessingMode.PARALLEL],
        include_single_model=True,
        include_multi_model=True
    )


@pytest.fixture
def mock_model_responses(test_data_loader):
    """Create realistic mock responses based on actual test data"""
    combined_data = test_data_loader.create_combined_project_data()
    
    # Extract room information for realistic responses
    rooms = []
    for floor_data in combined_data[1:]:  # Skip jobsite info
        rooms.extend(floor_data.get('rooms', []))
    
    # Create mock responses for each model
    responses = []
    
    # GPT-4 Response
    gpt4_tasks = []
    for room in rooms[:2]:  # First 2 rooms
        room_name = room['name']
        measurements = room.get('measurements', {})
        floor_area = measurements.get('floor_area_sqft', 100)
        
        gpt4_tasks.extend([
            {
                "task_name": f"Remove existing flooring - {room_name}",
                "description": f"Remove old flooring from {room_name}",
                "necessity": "required",
                "quantity": floor_area,
                "unit": "sq_ft"
            },
            {
                "task_name": f"Install new flooring - {room_name}",
                "description": f"Install new flooring in {room_name}",
                "necessity": "required", 
                "quantity": floor_area,
                "unit": "sq_ft"
            }
        ])
    
    responses.append(ModelResponse(
        model_name="gpt-4",
        room_estimates=[{"name": "Multi-room estimate", "tasks": gpt4_tasks}],
        processing_time=2.3,
        total_work_items=len(gpt4_tasks),
        confidence_self_assessment=0.87
    ))
    
    # Claude Response - slightly different approach
    claude_tasks = []
    for room in rooms[:2]:
        room_name = room['name']
        measurements = room.get('measurements', {})
        floor_area = measurements.get('floor_area_sqft', 100)
        wall_area = measurements.get('wall_area_sqft', 200)
        
        claude_tasks.extend([
            {
                "task_name": f"Flooring removal and installation - {room_name}",
                "description": f"Complete flooring replacement for {room_name}",
                "necessity": "required",
                "quantity": floor_area,
                "unit": "sq_ft"
            },
            {
                "task_name": f"Wall preparation - {room_name}",
                "description": f"Patch and prepare walls in {room_name}",
                "necessity": "required",
                "quantity": wall_area,
                "unit": "sq_ft"
            }
        ])
    
    responses.append(ModelResponse(
        model_name="claude-3-sonnet",
        room_estimates=[{"name": "Multi-room estimate", "tasks": claude_tasks}],
        processing_time=2.1,
        total_work_items=len(claude_tasks),
        confidence_self_assessment=0.84
    ))
    
    # Gemini Response
    gemini_tasks = []
    for room in rooms[:2]:
        room_name = room['name']
        measurements = room.get('measurements', {})
        floor_area = measurements.get('floor_area_sqft', 100)
        
        gemini_tasks.extend([
            {
                "task_name": f"Remove old {room_name} flooring",
                "description": f"Demolition of existing flooring materials in {room_name}",
                "necessity": "required",
                "quantity": floor_area * 1.05,  # Slight variation
                "unit": "sq_ft"
            },
            {
                "task_name": f"Install replacement flooring {room_name}",
                "description": f"Installation of new flooring materials in {room_name}",
                "necessity": "required",
                "quantity": floor_area,
                "unit": "sq_ft"
            }
        ])
    
    responses.append(ModelResponse(
        model_name="gemini-pro",
        room_estimates=[{"name": "Multi-room estimate", "tasks": gemini_tasks}],
        processing_time=1.8,
        total_work_items=len(gemini_tasks),
        confidence_self_assessment=0.82
    ))
    
    return responses


class TestModelCombinationFramework:
    """Test the model combination testing framework itself"""
    
    def test_test_data_loader_initialization(self, test_data_loader):
        """Test that test data loader works properly"""
        assert test_data_loader is not None
        
        # Test loading each data type
        demo_data = test_data_loader.load_demo_data()
        assert isinstance(demo_data, list)
        assert len(demo_data) > 0
        
        measurement_data = test_data_loader.load_measurement_data()
        assert isinstance(measurement_data, list)
        assert len(measurement_data) > 0
        
        intake_form = test_data_loader.load_intake_form()
        assert isinstance(intake_form, str)
        assert len(intake_form) > 0
        
        combined_data = test_data_loader.create_combined_project_data()
        assert isinstance(combined_data, list)
        assert len(combined_data) > 1  # Should have jobsite info + floors
    
    def test_test_matrix_generation(self, test_matrix):
        """Test test matrix generation"""
        # Test single model configurations
        single_configs = test_matrix.generate_single_model_configurations()
        assert len(single_configs) > 0
        assert all(len(config.models) == 1 for config in single_configs)
        
        # Test multi-model configurations
        multi_configs = test_matrix.generate_multi_model_configurations()
        assert len(multi_configs) > 0
        assert all(len(config.models) > 1 for config in multi_configs)
        
        # Test essential configurations
        essential_configs = test_matrix.generate_essential_configurations()
        assert len(essential_configs) > 0
        
        # Test performance comparison matrix
        performance_configs = test_matrix.generate_performance_comparison_matrix()
        assert len(performance_configs) > 0
    
    def test_model_tester_initialization(self, model_tester):
        """Test model tester initialization"""
        assert model_tester is not None
        assert model_tester.orchestrator is not None
        assert model_tester.result_merger is not None
        assert model_tester.validator is not None
    
    @pytest.mark.asyncio
    async def test_single_test_execution_mock(self, model_tester, test_dataset, mock_model_responses):
        """Test single model combination test execution with mocking"""
        # Create a simple test configuration
        config = TestConfiguration(
            models=[ModelType.GPT4],
            validation_mode=ValidationMode.BALANCED,
            processing_mode=ProcessingMode.SINGLE_MODEL,
            test_name="test_single_gpt4"
        )
        
        # Mock the orchestrator run_parallel method
        with patch.object(model_tester.orchestrator, 'run_parallel', 
                         return_value=[mock_model_responses[0]]):
            
            result = await model_tester.run_single_test(
                configuration=config,
                test_dataset=test_dataset,
                prompt="Test prompt for estimation"
            )
            
            assert result is not None
            assert result.success
            assert result.configuration.test_name == "test_single_gpt4"
            assert result.models_responded == 1
            assert result.total_models == 1
            assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_multi_model_test_execution_mock(self, model_tester, test_dataset, mock_model_responses):
        """Test multi-model combination test execution with mocking"""
        config = TestConfiguration(
            models=[ModelType.GPT4, ModelType.CLAUDE],
            validation_mode=ValidationMode.BALANCED,
            processing_mode=ProcessingMode.PARALLEL,
            test_name="test_gpt4_claude"
        )
        
        # Mock the orchestrator run_parallel method
        with patch.object(model_tester.orchestrator, 'run_parallel', 
                         return_value=mock_model_responses[:2]):
            
            result = await model_tester.run_single_test(
                configuration=config,
                test_dataset=test_dataset,
                prompt="Test prompt for estimation"
            )
            
            assert result is not None
            assert result.success
            assert result.models_responded == 2
            assert result.total_models == 2
            assert result.merged_estimate is not None
    
    def test_performance_analyzer(self, mock_model_responses):
        """Test performance analyzer functionality"""
        analyzer = PerformanceAnalyzer()
        
        # Create mock test results
        from tests.model_combinations.combination_tester import CombinationTestResult
        from datetime import datetime
        
        config = TestConfiguration(
            models=[ModelType.GPT4],
            validation_mode=ValidationMode.BALANCED,
            processing_mode=ProcessingMode.SINGLE_MODEL
        )
        
        mock_results = [
            CombinationTestResult(
                configuration=config,
                success=True,
                execution_time=2.5,
                model_responses=[mock_model_responses[0]],
                models_responded=1,
                timestamp=datetime.now()
            )
        ]
        
        metrics = analyzer.analyze_single_configuration(mock_results)
        
        assert metrics is not None
        assert metrics.total_tests == 1
        assert metrics.successful_tests == 1
        assert metrics.success_rate == 1.0
        assert metrics.avg_execution_time == 2.5
    
    def test_comparison_reporter(self):
        """Test comparison reporter functionality"""
        reporter = ComparisonReporter(output_directory="test_outputs/reports")
        
        # Create mock test results
        from tests.model_combinations.combination_tester import CombinationTestResult
        from datetime import datetime
        
        config1 = TestConfiguration(
            models=[ModelType.GPT4],
            validation_mode=ValidationMode.BALANCED,
            processing_mode=ProcessingMode.SINGLE_MODEL,
            test_name="test_gpt4"
        )
        
        mock_results = [
            CombinationTestResult(
                configuration=config1,
                success=True,
                execution_time=2.5,
                overall_confidence=0.85,
                consensus_level=1.0,
                validation_score=0.8,
                total_work_items=5,
                unique_rooms_processed=2,
                models_responded=1,
                timestamp=datetime.now()
            )
        ]
        
        report = reporter.generate_comprehensive_report(
            test_results=mock_results,
            report_title="Test Report"
        )
        
        assert report is not None
        assert report.title == "Test Report"
        assert len(report.detailed_results) == 1
        assert len(report.recommendations) > 0


class TestModelCombinations:
    """Test specific model combinations"""
    
    @pytest.mark.asyncio
    async def test_single_model_combinations(self, model_tester, test_dataset, mock_model_responses):
        """Test all single model configurations"""
        configurations = [
            TestConfiguration(
                models=[ModelType.GPT4],
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.SINGLE_MODEL,
                test_name="single_gpt4"
            ),
            TestConfiguration(
                models=[ModelType.CLAUDE],
                validation_mode=ValidationMode.BALANCED,
                processing_mode=ProcessingMode.SINGLE_MODEL,
                test_name="single_claude"
            )
        ]
        
        # Mock responses for each model
        response_map = {
            "gpt4": [mock_model_responses[0]],
            "claude": [mock_model_responses[1]]
        }
        
        def mock_run_parallel(prompt, json_data, model_names):
            model_name = model_names[0] if model_names else "gpt4"
            return response_map.get(model_name, [mock_model_responses[0]])
        
        with patch.object(model_tester.orchestrator, 'run_parallel', side_effect=mock_run_parallel):
            
            results = await model_tester.run_test_batch(
                configurations=configurations,
                test_dataset=test_dataset,
                max_concurrent=2
            )
            
            assert len(results) == 2
            assert all(result.success for result in results)
            assert results[0].configuration.test_name == "single_gpt4"
            assert results[1].configuration.test_name == "single_claude"
    
    @pytest.mark.asyncio 
    async def test_essential_configurations(self, test_matrix, model_tester, test_dataset, mock_model_responses):
        """Test essential model configurations"""
        essential_configs = test_matrix.generate_essential_configurations()
        
        # Limit to first 3 for faster testing
        test_configs = essential_configs[:3]
        
        # Mock all model combinations
        def mock_run_parallel(prompt, json_data, model_names):
            if len(model_names) == 1:
                model_name = model_names[0]
                if "gpt4" in model_name:
                    return [mock_model_responses[0]]
                elif "claude" in model_name:
                    return [mock_model_responses[1]]
                else:
                    return [mock_model_responses[2]]
            else:
                # Multi-model response
                return mock_model_responses[:len(model_names)]
        
        with patch.object(model_tester.orchestrator, 'run_parallel', side_effect=mock_run_parallel):
            
            results = await model_tester.run_test_batch(
                configurations=test_configs,
                test_dataset=test_dataset,
                max_concurrent=2
            )
            
            assert len(results) == len(test_configs)
            success_count = sum(1 for result in results if result.success)
            assert success_count >= len(test_configs) * 0.8  # At least 80% success rate


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete testing framework"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_mock(self, test_data_loader, mock_model_responses):
        """Test complete end-to-end workflow with mocking"""
        # Initialize components
        tester = ModelCombinationTester(
            test_data_loader=test_data_loader,
            output_directory="test_outputs/integration_test"
        )
        
        # Create simple test matrix
        matrix = ModelTestMatrix(
            available_models=[ModelType.GPT4, ModelType.CLAUDE],
            validation_modes=[ValidationMode.BALANCED],
            processing_modes=[ProcessingMode.PARALLEL],
            include_single_model=True,
            include_multi_model=True
        )
        
        # Generate configurations (limit for testing)
        configurations = matrix.generate_essential_configurations()[:2]
        
        # Mock the orchestrator
        def mock_run_parallel(prompt, json_data, model_names):
            if len(model_names) == 1:
                return [mock_model_responses[0]]
            else:
                return mock_model_responses[:2]
        
        with patch.object(tester.orchestrator, 'run_parallel', side_effect=mock_run_parallel):
            
            # Run tests
            test_dataset = test_data_loader.get_test_dataset()
            results = await tester.run_test_batch(configurations, test_dataset)
            
            assert len(results) == len(configurations)
            
            # Analyze results
            analyzer = PerformanceAnalyzer()
            
            # Group results by configuration for analysis
            results_by_config = {}
            for result in results:
                config_name = result.configuration.test_name
                if config_name not in results_by_config:
                    results_by_config[config_name] = []
                results_by_config[config_name].append(result)
            
            metrics_list = analyzer.analyze_multiple_configurations(results_by_config)
            assert len(metrics_list) > 0
            
            # Generate report
            reporter = ComparisonReporter("test_outputs/integration_reports")
            report = reporter.generate_comprehensive_report(results)
            
            assert report is not None
            assert len(report.detailed_results) == len(results)
    
    def test_data_consistency(self, test_data_loader):
        """Test that test data is consistent and valid"""
        # Load all test data
        demo_data = test_data_loader.load_demo_data()
        measurement_data = test_data_loader.load_measurement_data()
        intake_form = test_data_loader.load_intake_form()
        combined_data = test_data_loader.create_combined_project_data()
        
        # Validate combined data can be converted to ProjectData
        assert test_data_loader.validate_project_data(combined_data)
        
        # Check data consistency
        demo_rooms = set()
        for floor in demo_data:
            for room in floor.get('rooms', []):
                demo_rooms.add((floor['location'], room['name']))
        
        measurement_rooms = set()
        for floor in measurement_data:
            for room in floor.get('rooms', []):
                measurement_rooms.add((floor['location'], room['name']))
        
        # Should have overlapping rooms between demo and measurement data
        common_rooms = demo_rooms & measurement_rooms
        assert len(common_rooms) > 0, "No common rooms between demo and measurement data"