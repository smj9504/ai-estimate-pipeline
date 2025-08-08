"""
Phase 0 Test Implementation - Generate Scope of Work
"""
import asyncio
from datetime import datetime
from typing import Dict, Any

from ..base import PhaseTestBase, PhaseTestConfig, PhaseTestResult
from src.phases.phase0_processor import Phase0Processor


class Phase0Test(PhaseTestBase):
    """Test implementation for Phase 0 - Generate Scope of Work"""
    
    @property
    def phase_number(self) -> int:
        return 0
    
    @property
    def phase_name(self) -> str:
        return "Generate Scope of Work"
    
    async def prepare_test_data(self, test_config: PhaseTestConfig) -> Dict[str, Any]:
        """Prepare test data for Phase 0"""
        try:
            # Try to load from existing test data
            measurement_data = self.load_fixture("sample_measurement")
            demolition_data = self.load_fixture("sample_demo") 
            intake_form = self.load_fixture("sample_intake_form")
            
            return {
                'measurement_data': measurement_data,
                'demolition_scope_data': demolition_data,
                'intake_form': intake_form,
                'project_id': f"TEST_PHASE0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        except FileNotFoundError:
            # Fall back to sample data
            return self._create_sample_data()
    
    def _create_sample_data(self) -> Dict[str, Any]:
        """Create sample test data when fixtures are not available"""
        return {
            'measurement_data': [
                {
                    "floor_name": "First Floor",
                    "rooms": [
                        {
                            "name": "Living Room",
                            "dimensions": {
                                "width": 16,
                                "length": 20,
                                "height": 9
                            },
                            "features": {
                                "windows": 3,
                                "doors": 2,
                                "fireplace": 1
                            }
                        },
                        {
                            "name": "Kitchen",
                            "dimensions": {
                                "width": 12,
                                "length": 14,
                                "height": 9
                            },
                            "features": {
                                "windows": 2,
                                "doors": 1,
                                "cabinets": "standard"
                            }
                        }
                    ]
                }
            ],
            'demolition_scope_data': {
                "completed_demolition": [
                    {
                        "room": "Living Room",
                        "items": ["Old carpet - 320 sq ft", "Damaged drywall - 150 sq ft"]
                    },
                    {
                        "room": "Kitchen", 
                        "items": ["Old cabinets - 6 units", "Countertop - 25 sq ft"]
                    }
                ]
            },
            'intake_form': "Residential reconstruction project following water damage. Client requests modern finishes with focus on durability. Budget range: $50,000-75,000. Timeline: 8-10 weeks.",
            'project_id': f"TEST_PHASE0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for Phase 0"""
        required_keys = ['measurement_data', 'demolition_scope_data', 'intake_form']
        
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        # Validate measurement data structure
        measurement_data = input_data['measurement_data']
        if not isinstance(measurement_data, list) or not measurement_data:
            self.logger.error("measurement_data must be a non-empty list")
            return False
        
        for floor in measurement_data:
            if not isinstance(floor, dict) or 'rooms' not in floor:
                self.logger.error("Each floor must have 'rooms' key")
                return False
        
        return True
    
    async def execute_phase(self, input_data: Dict[str, Any], 
                          test_config: PhaseTestConfig) -> PhaseTestResult:
        """Execute Phase 0 test"""
        start_time = datetime.now()
        
        try:
            # Initialize processor
            processor = Phase0Processor()
            
            # Phase 0 uses single model only
            model_to_use = test_config.models[0] if test_config.models else "gpt4"
            
            # Execute phase
            result = await processor.process(
                measurement_data=input_data['measurement_data'],
                demolition_scope_data=input_data['demolition_scope_data'],
                intake_form=input_data['intake_form'],
                model_to_use=model_to_use,
                project_id=input_data.get('project_id')
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Analyze results
            success = result.get('success', False)
            confidence_score = 0.8 if success else 0.0  # Phase 0 doesn't have multi-model confidence
            
            # Count generated rooms/scopes
            data_quality_score = 0.0
            if success and 'data' in result:
                total_rooms = 0
                for floor_data in result['data'][1:]:  # Skip jobsite_info
                    if 'rooms' in floor_data:
                        total_rooms += len(floor_data['rooms'])
                
                data_quality_score = min(1.0, total_rooms / 5)  # Normalize based on expected rooms
            
            return PhaseTestResult(
                phase_number=self.phase_number,
                success=success,
                execution_time=execution_time,
                confidence_score=confidence_score,
                consensus_level=1.0,  # Single model, so perfect "consensus"
                models_responded=1 if success else 0,
                total_models=1,
                output_data=result,
                metadata={
                    'model_used': model_to_use,
                    'data_quality_score': data_quality_score,
                    'total_rooms_generated': total_rooms if success else 0
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Phase 0 execution failed: {e}")
            
            return PhaseTestResult(
                phase_number=self.phase_number,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                total_models=1
            )