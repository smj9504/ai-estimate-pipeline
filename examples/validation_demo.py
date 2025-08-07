# examples/validation_demo.py
"""
Demonstration of the Response Validation System

This script shows how to use the new response validation system
to validate AI model responses before they enter the merging pipeline.
"""

import asyncio
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_interface import ModelOrchestrator
from src.validators.response_validator import (
    ValidationOrchestrator, 
    validate_model_response,
    should_exclude_from_merging
)
from src.models.data_models import ModelResponse
from src.utils.logger import get_logger


async def demonstrate_validation():
    """Demonstrate the validation system with example data"""
    logger = get_logger('validation_demo')
    
    # Sample project data (simplified)
    sample_project_data = {
        "floors": [{
            "location": "First Floor",
            "rooms": [{
                "name": "Living Room",
                "work_scope": {
                    "Flooring": "Remove & Replace",
                    "Wall": "Paint",
                    "Ceiling": ""
                },
                "measurements": {
                    "floor_area_sqft": 250.0,
                    "wall_area_sqft": 400.0,
                    "ceiling_area_sqft": 250.0,
                    "height": 9.5
                }
            }]
        }]
    }
    
    # Example 1: Good response
    logger.info("=" * 60)
    logger.info("Example 1: Validating a GOOD response")
    logger.info("=" * 60)
    
    good_response = ModelResponse(
        model_name="gpt-4",
        room_estimates=[{
            "name": "Living Room",
            "tasks": [
                {
                    "task_name": "Remove existing flooring",
                    "description": "Remove existing hardwood flooring",
                    "quantity": 250.0,
                    "unit": "sqft",
                    "necessity": "required"
                },
                {
                    "task_name": "Install new flooring",
                    "description": "Install new hardwood flooring",
                    "quantity": 250.0,
                    "unit": "sqft",
                    "necessity": "required"
                },
                {
                    "task_name": "Paint walls",
                    "description": "Prime and paint all walls",
                    "quantity": 400.0,
                    "unit": "sqft",
                    "necessity": "required"
                }
            ]
        }],
        total_work_items=3,
        processing_time=2.5,
        confidence_self_assessment=0.85,
        raw_response='{"rooms": [{"name": "Living Room", "tasks": [...]}]}'
    )
    
    validated_response, validation_report = validate_model_response(
        good_response, sample_project_data
    )
    
    logger.info(f"Validation Result: {validation_report.quality_level.value.upper()}")
    logger.info(f"Quality Score: {validation_report.quality_score:.1f}/100")
    logger.info(f"Is Valid: {validation_report.is_valid}")
    logger.info(f"Issues: {len(validation_report.issues)}")
    logger.info(f"Auto-fixes: {len(validation_report.fixed_issues)}")
    
    # Example 2: Bad response with issues
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Validating a BAD response with issues")
    logger.info("=" * 60)
    
    bad_response = ModelResponse(
        model_name="claude-3",
        room_estimates=[{
            "name": "**Room Name**",  # Invalid room name with asterisks
            "tasks": [
                {
                    "task_name": "",  # Empty task name
                    "description": "Remove something",
                    "quantity": -50.0,  # Negative quantity
                    "unit": "invalid_unit",  # Invalid unit
                    "necessity": "required"
                },
                {
                    "task_name": "TODO: Add task name",  # Placeholder task name
                    "description": "Install something",
                    "quantity": "not_a_number",  # Invalid quantity type
                    "unit": "sqft",
                    "necessity": "required"
                }
            ]
        }],
        total_work_items=5,  # Mismatch with actual count (2)
        processing_time=1.8,
        confidence_self_assessment=0.60,
        raw_response='Error: Partial parsing failure'
    )
    
    validated_response, validation_report = validate_model_response(
        bad_response, sample_project_data, auto_fix=True
    )
    
    logger.info(f"Validation Result: {validation_report.quality_level.value.upper()}")
    logger.info(f"Quality Score: {validation_report.quality_score:.1f}/100")
    logger.info(f"Is Valid: {validation_report.is_valid}")
    logger.info(f"Issues: {len(validation_report.issues)}")
    logger.info(f"Auto-fixes: {len(validation_report.fixed_issues)}")
    
    # Show detailed issues
    if validation_report.issues:
        logger.info("\nDetailed Issues:")
        for issue in validation_report.issues[:3]:  # Show first 3
            logger.info(f"  - {issue.severity.value.upper()}: {issue.message}")
    
    # Show auto-fixes applied
    if validation_report.fixed_issues:
        logger.info("\nAuto-fixes Applied:")
        for fix in validation_report.fixed_issues[:3]:  # Show first 3
            logger.info(f"  - {fix}")
    
    # Check if response should be excluded from merging
    should_exclude = should_exclude_from_merging(validation_report)
    logger.info(f"\nShould exclude from merging: {should_exclude}")
    
    # Example 3: Using ValidationOrchestrator directly
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Using ValidationOrchestrator directly")
    logger.info("=" * 60)
    
    orchestrator = ValidationOrchestrator()
    detailed_report = orchestrator.validate_response(
        good_response, sample_project_data, auto_fix=False
    )
    
    # Create human-readable summary
    summary = orchestrator.create_validation_summary(detailed_report)
    logger.info("Validation Summary:")
    logger.info(summary)


async def demonstrate_model_orchestrator_integration():
    """Demonstrate validation integration with ModelOrchestrator"""
    logger = get_logger('orchestrator_demo')
    
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: ModelOrchestrator with Validation")
    logger.info("=" * 60)
    
    # Create orchestrator with validation enabled
    orchestrator = ModelOrchestrator(enable_validation=True)
    
    logger.info(f"Validation enabled: {orchestrator.get_validation_enabled()}")
    logger.info(f"Available models: {orchestrator.get_available_models()}")
    
    # Note: Actual model calls would require API keys
    logger.info("\nTo test with real models:")
    logger.info("1. Set API keys in .env file")
    logger.info("2. Use orchestrator.run_parallel() with validation")
    logger.info("3. Quality threshold filtering will be applied automatically")
    
    # Demonstrate validation settings
    logger.info(f"\nValidation can be controlled:")
    logger.info(f"- orchestrator.set_validation_enabled(False)")
    logger.info(f"- run_parallel(..., enable_validation=False)")
    logger.info(f"- run_parallel(..., min_quality_threshold=50.0)")


def create_test_data():
    """Create test data files for validation testing"""
    logger = get_logger('test_data_creator')
    
    # Create test directory if it doesn't exist
    test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(test_dir, exist_ok=True)
    
    # Sample project data
    project_data = [
        {
            "Jobsite": "123 Test Street",
            "occupancy": "Single Family Residence",
            "company": {"name": "Test Construction"}
        },
        {
            "location": "First Floor",
            "rooms": [
                {
                    "name": "Kitchen",
                    "material": {
                        "Floor": "Tile",
                        "wall": "Drywall", 
                        "ceiling": "Drywall",
                        "Baseboard": "Wood",
                        "Quarter Round": "Wood"
                    },
                    "work_scope": {
                        "Flooring": "Remove & Replace",
                        "Wall": "Remove & Replace",
                        "Ceiling": "Paint",
                        "Baseboard": "Remove & Replace",
                        "Quarter Round": "Remove & Replace"
                    },
                    "measurements": {
                        "height": 10.0,
                        "wall_area_sqft": 320.0,
                        "ceiling_area_sqft": 150.0,
                        "floor_area_sqft": 150.0,
                        "floor_perimeter_lf": 50.0
                    },
                    "demo_scope(already demo'd)": {
                        "Ceiling Drywall(sq_ft)": 0,
                        "Wall Drywall(sq_ft)": 100.0
                    },
                    "additional_notes": {
                        "protection": ["Appliances", "Countertops"],
                        "detach_reset": ["Light fixtures", "Cabinet doors"]
                    }
                },
                {
                    "name": "Bathroom",
                    "material": {
                        "Floor": "Vinyl",
                        "wall": "Tile",
                        "ceiling": "Drywall",
                        "Baseboard": "Tile",
                        "Quarter Round": ""
                    },
                    "work_scope": {
                        "Flooring": "Remove & Replace",
                        "Wall": "Remove & Replace", 
                        "Ceiling": "Paint",
                        "Baseboard": "Remove & Replace",
                        "Quarter Round": ""
                    },
                    "measurements": {
                        "height": 9.0,
                        "wall_area_sqft": 200.0,
                        "ceiling_area_sqft": 50.0,
                        "floor_area_sqft": 50.0,
                        "floor_perimeter_lf": 30.0
                    },
                    "demo_scope(already demo'd)": {
                        "Ceiling Drywall(sq_ft)": 0,
                        "Wall Drywall(sq_ft)": 0
                    },
                    "additional_notes": {
                        "protection": ["Vanity", "Toilet"],
                        "detach_reset": ["Mirror", "Towel bars"]
                    }
                }
            ]
        }
    ]
    
    # Save test data
    test_file = os.path.join(test_dir, 'sample_project.json')
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created test data file: {test_file}")
    
    return project_data


async def main():
    """Main demonstration function"""
    logger = get_logger('main')
    
    logger.info("🔍 AI Response Validation System Demonstration")
    logger.info("=" * 60)
    
    # Create test data
    test_data = create_test_data()
    
    # Run validation demonstrations
    await demonstrate_validation()
    await demonstrate_model_orchestrator_integration()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Validation demonstration complete!")
    logger.info("=" * 60)
    
    logger.info("\nKey Features Demonstrated:")
    logger.info("1. ✅ Room name validation (preventing '**' placeholders)")
    logger.info("2. ✅ JSON structure validation")
    logger.info("3. ✅ Response quality scoring (0-100)")
    logger.info("4. ✅ Error recovery mechanisms (auto-fixing)")
    logger.info("5. ✅ Integration with ModelOrchestrator")
    logger.info("6. ✅ Business logic validation (Remove & Replace)")
    logger.info("7. ✅ Measurement consistency checks")
    logger.info("8. ✅ Quality threshold filtering")
    
    logger.info("\nNext Steps:")
    logger.info("- Add API keys to test with real models")
    logger.info("- Integrate with existing pipeline")
    logger.info("- Customize validation rules as needed")


if __name__ == "__main__":
    asyncio.run(main())