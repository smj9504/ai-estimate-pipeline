# tests/test_response_validator.py
"""
Comprehensive tests for the Response Validation System

Tests cover:
- ResponseStructureValidator functionality
- DataIntegrityValidator functionality  
- ValidationOrchestrator coordination
- Error recovery mechanisms
- Integration with existing pipeline components
"""

import pytest
import json
from typing import Dict, Any, List

from src.validators.response_validator import (
    ResponseStructureValidator,
    DataIntegrityValidator, 
    ValidationOrchestrator,
    ValidationSeverity,
    ResponseQuality,
    ValidationIssue,
    ValidationReport,
    validate_model_response,
    should_exclude_from_merging
)
from src.models.data_models import ModelResponse


class TestResponseStructureValidator:
    """Test the ResponseStructureValidator class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = ResponseStructureValidator()
    
    def test_valid_response_structure(self):
        """Test validation of a properly structured response"""
        response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Living Room",
                "tasks": [
                    {
                        "task_name": "Install flooring",
                        "description": "Install hardwood flooring",
                        "quantity": 150.0,
                        "unit": "sqft",
                        "necessity": "required"
                    }
                ]
            }],
            total_work_items=1,
            confidence_self_assessment=0.85,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.validator.validate_response_structure(response)
        
        assert report.is_valid
        assert report.quality_score >= 70  # Should be good quality
        assert report.quality_level in [ResponseQuality.GOOD, ResponseQuality.EXCELLENT]
        assert len(report.issues) == 0
    
    def test_invalid_room_names(self):
        """Test detection of invalid room names"""
        response = ModelResponse(
            model_name="claude-3",
            room_estimates=[
                {
                    "name": "**Kitchen**",  # Invalid asterisks
                    "tasks": [{"task_name": "Test task", "quantity": 1, "unit": "item"}]
                },
                {
                    "name": "",  # Empty name
                    "tasks": [{"task_name": "Test task", "quantity": 1, "unit": "item"}]
                },
                {
                    "name": "Room 1",  # Generic name
                    "tasks": [{"task_name": "Test task", "quantity": 1, "unit": "item"}]
                }
            ],
            total_work_items=3,
            confidence_self_assessment=0.60,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.validator.validate_response_structure(response)
        
        assert not report.is_valid  # Should fail due to critical issues
        assert report.quality_score < 70  # Should be lower quality
        
        # Check for specific issues
        room_name_issues = [i for i in report.issues if i.category == "invalid_room_name"]
        assert len(room_name_issues) >= 2  # Should find asterisk and empty name issues
        
        # Check severity levels
        critical_issues = [i for i in report.issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical_issues) > 0
    
    def test_missing_required_fields(self):
        """Test detection of missing required fields"""
        response = ModelResponse(
            model_name="gemini",
            room_estimates=[
                {
                    # Missing 'name' field
                    "tasks": [{"task_name": "Test task", "quantity": 1, "unit": "item"}]
                },
                {
                    "name": "Valid Room",
                    "tasks": [
                        {
                            # Missing 'task_name' field
                            "description": "Some task",
                            "quantity": 1,
                            "unit": "item"
                        }
                    ]
                }
            ],
            total_work_items=2,
            confidence_self_assessment=0.50,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.validator.validate_response_structure(response)
        
        assert not report.is_valid
        
        missing_field_issues = [i for i in report.issues if i.category == "missing_field"]
        assert len(missing_field_issues) >= 2  # Room name and task name
    
    def test_invalid_quantities_and_units(self):
        """Test validation of quantity and unit fields"""
        response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Test Room", 
                "tasks": [
                    {
                        "task_name": "Task with negative quantity",
                        "quantity": -50.0,  # Invalid negative
                        "unit": "sqft"
                    },
                    {
                        "task_name": "Task with invalid unit",
                        "quantity": 100.0,
                        "unit": "invalid_unit"  # Invalid unit
                    },
                    {
                        "task_name": "Task with string quantity",
                        "quantity": "not_a_number",  # Invalid type
                        "unit": "lf"
                    }
                ]
            }],
            total_work_items=3,
            confidence_self_assessment=0.70,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.validator.validate_response_structure(response)
        
        quantity_issues = [i for i in report.issues if i.category == "invalid_quantity"]
        unit_issues = [i for i in report.issues if i.category == "invalid_unit"]
        
        assert len(quantity_issues) >= 2  # Negative and invalid type
        assert len(unit_issues) >= 1  # Invalid unit
    
    def test_work_items_count_mismatch(self):
        """Test detection of work items count mismatch"""
        response = ModelResponse(
            model_name="claude-3",
            room_estimates=[{
                "name": "Test Room",
                "tasks": [
                    {"task_name": "Task 1", "quantity": 1, "unit": "item"},
                    {"task_name": "Task 2", "quantity": 1, "unit": "item"}
                ]
            }],
            total_work_items=5,  # Mismatch: actual is 2
            confidence_self_assessment=0.80,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.validator.validate_response_structure(response)
        
        count_issues = [i for i in report.issues if i.category == "count_mismatch"]
        assert len(count_issues) == 1
        assert count_issues[0].auto_fixable  # Should be auto-fixable
    
    def test_empty_response(self):
        """Test handling of empty responses"""
        response = ModelResponse(
            model_name="test",
            room_estimates=[],
            total_work_items=0,
            confidence_self_assessment=0.0,
            raw_response=""
        )
        
        report = self.validator.validate_response_structure(response)
        
        assert not report.is_valid
        assert report.quality_level == ResponseQuality.UNUSABLE


class TestDataIntegrityValidator:
    """Test the DataIntegrityValidator class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.validator = DataIntegrityValidator()
        
        # Sample original data for business logic validation
        self.original_data = {
            "floors": [{
                "rooms": [{
                    "name": "Kitchen",
                    "work_scope": {
                        "Flooring": "Remove & Replace",
                        "Wall": "Paint"
                    },
                    "measurements": {
                        "floor_area_sqft": 100.0,
                        "wall_area_sqft": 200.0
                    }
                }]
            }]
        }
    
    def test_quantity_range_validation(self):
        """Test validation of quantity ranges"""
        response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Test Room",
                "tasks": [
                    {
                        "task_name": "Tiny flooring job",
                        "quantity": 0.01,  # Too small
                        "unit": "sqft"
                    },
                    {
                        "task_name": "Massive flooring job", 
                        "quantity": 50000.0,  # Too large
                        "unit": "sqft"
                    },
                    {
                        "task_name": "Reasonable flooring job",
                        "quantity": 150.0,  # Normal range
                        "unit": "sqft"
                    }
                ]
            }],
            total_work_items=3,
            confidence_self_assessment=0.75
        )
        
        report = self.validator.validate_data_integrity(response)
        
        range_issues = [i for i in report.issues if i.category == "quantity_range"]
        assert len(range_issues) == 2  # Too small and too large
    
    def test_remove_replace_logic_validation(self):
        """Test Remove & Replace business logic validation"""
        response = ModelResponse(
            model_name="claude-3",
            room_estimates=[{
                "name": "Kitchen",
                "tasks": [
                    # Missing removal task for Remove & Replace flooring
                    {
                        "task_name": "Install new flooring",
                        "quantity": 100.0,
                        "unit": "sqft"
                    },
                    # Has both removal and installation for walls (correct)
                    {
                        "task_name": "Remove existing wall covering",
                        "quantity": 200.0,
                        "unit": "sqft"
                    }
                ]
            }],
            total_work_items=2,
            confidence_self_assessment=0.70
        )
        
        report = self.validator.validate_data_integrity(response, self.original_data)
        
        # Should find missing removal task for flooring
        missing_removal = [i for i in report.issues if i.category == "missing_removal"]
        assert len(missing_removal) >= 1
    
    def test_measurement_consistency(self):
        """Test measurement consistency validation"""
        response = ModelResponse(
            model_name="gemini",
            room_estimates=[{
                "name": "Kitchen",
                "tasks": [
                    {
                        "task_name": "Install flooring",
                        "quantity": 500.0,  # Much larger than measured area (100.0)
                        "unit": "sqft"
                    }
                ]
            }],
            total_work_items=1,
            confidence_self_assessment=0.65
        )
        
        report = self.validator.validate_data_integrity(response, self.original_data)
        
        measurement_issues = [i for i in report.issues if i.category == "measurement_mismatch"]
        assert len(measurement_issues) >= 1
    
    def test_task_relationship_validation(self):
        """Test logical task relationship validation"""
        response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Test Room",
                "tasks": [
                    # Installation without corresponding removal (orphaned)
                    {
                        "task_name": "Install new baseboard", 
                        "quantity": 50.0,
                        "unit": "lf"
                    },
                    # Proper pair: removal and installation
                    {
                        "task_name": "Remove existing flooring",
                        "quantity": 100.0,
                        "unit": "sqft"
                    },
                    {
                        "task_name": "Install new flooring",
                        "quantity": 100.0,
                        "unit": "sqft"
                    }
                ]
            }],
            total_work_items=3,
            confidence_self_assessment=0.80
        )
        
        report = self.validator.validate_data_integrity(response)
        
        orphaned_issues = [i for i in report.issues if i.category == "orphaned_installation"]
        assert len(orphaned_issues) >= 1  # Should find baseboard installation without removal


class TestValidationOrchestrator:
    """Test the ValidationOrchestrator class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.orchestrator = ValidationOrchestrator()
        
        self.sample_original_data = {
            "floors": [{
                "rooms": [{
                    "name": "Living Room",
                    "work_scope": {"Flooring": "Remove & Replace"},
                    "measurements": {"floor_area_sqft": 150.0}
                }]
            }]
        }
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation combining structure and integrity"""
        response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Living Room",
                "tasks": [
                    {
                        "task_name": "Remove existing flooring",
                        "quantity": 150.0,
                        "unit": "sqft"
                    },
                    {
                        "task_name": "Install new flooring",
                        "quantity": 150.0,
                        "unit": "sqft"
                    }
                ]
            }],
            total_work_items=2,
            confidence_self_assessment=0.85,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.orchestrator.validate_response(
            response, self.sample_original_data, auto_fix=False
        )
        
        assert report.is_valid
        assert report.quality_score >= 70
        assert report.quality_level in [ResponseQuality.GOOD, ResponseQuality.EXCELLENT]
        
        # Should combine both structure and integrity metadata
        assert 'structure_score' in report.metadata
        assert 'integrity_score' in report.metadata
    
    def test_auto_fix_functionality(self):
        """Test automatic fixing of issues"""
        response = ModelResponse(
            model_name="claude-3",
            room_estimates=[{
                "name": "**Room Name**",  # Auto-fixable: remove asterisks
                "tasks": [
                    {
                        "task_name": "Valid task",
                        "quantity": -10.0,  # Auto-fixable: make positive
                        "unit": "sqft"
                    }
                ]
            }],
            total_work_items=5,  # Auto-fixable: count mismatch
            confidence_self_assessment=0.60,
            raw_response='{"rooms": [...]}'
        )
        
        report = self.orchestrator.validate_response(
            response, self.sample_original_data, auto_fix=True
        )
        
        assert len(report.fixed_issues) > 0  # Should have applied some fixes
        
        # Specific fixes that should be applied
        fixed_categories = [fix.split(':')[0] for fix in report.fixed_issues]
        possible_fixes = ['invalid_room_name', 'invalid_quantity', 'count_mismatch']
        
        # Should have fixed at least one category
        assert any(cat in str(report.fixed_issues) for cat in possible_fixes)
    
    def test_validation_summary_generation(self):
        """Test human-readable validation summary generation"""
        response = ModelResponse(
            model_name="test",
            room_estimates=[],
            total_work_items=0,
            confidence_self_assessment=0.0,
            raw_response=""
        )
        
        report = self.orchestrator.validate_response(response, auto_fix=False)
        summary = self.orchestrator.create_validation_summary(report)
        
        assert isinstance(summary, str)
        assert "Quality:" in summary
        assert "Status:" in summary
        assert "Processing Time:" in summary
        
        # Should indicate poor quality for empty response
        assert report.quality_level == ResponseQuality.UNUSABLE
    
    def test_quality_level_determination(self):
        """Test quality level determination logic"""
        # Test excellent quality
        excellent_response = ModelResponse(
            model_name="gpt-4",
            room_estimates=[{
                "name": "Perfect Room",
                "tasks": [
                    {"task_name": "Perfect task", "quantity": 100.0, "unit": "sqft"},
                    {"task_name": "Another perfect task", "quantity": 50.0, "unit": "lf"}
                ]
            }],
            total_work_items=2,
            confidence_self_assessment=0.95,
            raw_response='{"perfect": "json"}'
        )
        
        excellent_report = self.orchestrator.validate_response(
            excellent_response, auto_fix=False
        )
        
        assert excellent_report.quality_level in [ResponseQuality.EXCELLENT, ResponseQuality.GOOD]
        assert excellent_report.quality_score >= 65


class TestIntegrationFunctions:
    """Test integration helper functions"""
    
    def test_validate_model_response_function(self):
        """Test the validate_model_response convenience function"""
        response = ModelResponse(
            model_name="test",
            room_estimates=[{
                "name": "Test Room",
                "tasks": [{"task_name": "Test task", "quantity": 1, "unit": "item"}]
            }],
            total_work_items=1,
            confidence_self_assessment=0.80,
            raw_response='{"test": true}'
        )
        
        validated_response, report = validate_model_response(response, auto_fix=True)
        
        assert isinstance(validated_response, ModelResponse)
        assert isinstance(report, ValidationReport)
        assert report.is_valid
    
    def test_should_exclude_from_merging_function(self):
        """Test the should_exclude_from_merging function"""
        # Create a report that should be excluded (low quality)
        low_quality_report = ValidationReport(
            is_valid=False,
            quality_score=20.0,  # Below threshold
            quality_level=ResponseQuality.POOR,
            issues=[ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="test",
                message="Critical issue"
            )],
            warnings=[],
            suggestions=[],
            fixed_issues=[],
            processing_time=0.1,
            metadata={}
        )
        
        assert should_exclude_from_merging(low_quality_report, min_quality_threshold=30.0)
        
        # Create a report that should be included (good quality)
        good_quality_report = ValidationReport(
            is_valid=True,
            quality_score=80.0,
            quality_level=ResponseQuality.GOOD,
            issues=[],
            warnings=[],
            suggestions=[],
            fixed_issues=[],
            processing_time=0.1,
            metadata={}
        )
        
        assert not should_exclude_from_merging(good_quality_report, min_quality_threshold=30.0)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.orchestrator = ValidationOrchestrator()
    
    def test_none_response_handling(self):
        """Test handling of None or invalid response objects"""
        # This would need to be handled by the calling code
        # as our validators expect ModelResponse objects
        pass
    
    def test_malformed_original_data(self):
        """Test handling of malformed original data"""
        response = ModelResponse(
            model_name="test",
            room_estimates=[{
                "name": "Test Room",
                "tasks": [{"task_name": "Test task", "quantity": 1, "unit": "item"}]
            }],
            total_work_items=1,
            confidence_self_assessment=0.80
        )
        
        malformed_data = {"invalid": "structure"}
        
        # Should handle gracefully without crashing
        report = self.orchestrator.validate_response(response, malformed_data)
        
        assert isinstance(report, ValidationReport)
        # May have warnings about data structure, but shouldn't crash
    
    def test_very_large_response(self):
        """Test handling of very large responses"""
        # Create a response with many rooms and tasks
        large_rooms = []
        for i in range(100):  # 100 rooms
            tasks = []
            for j in range(10):  # 10 tasks per room
                tasks.append({
                    "task_name": f"Task {j} in Room {i}",
                    "quantity": float(j + 1),
                    "unit": "sqft"
                })
            large_rooms.append({
                "name": f"Room {i}",
                "tasks": tasks
            })
        
        large_response = ModelResponse(
            model_name="test",
            room_estimates=large_rooms,
            total_work_items=1000,
            confidence_self_assessment=0.75,
            raw_response='{"large": "response"}'
        )
        
        report = self.orchestrator.validate_response(large_response, auto_fix=False)
        
        assert isinstance(report, ValidationReport)
        # Should complete without timeout or memory issues
        assert report.processing_time < 10.0  # Should be reasonably fast


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])