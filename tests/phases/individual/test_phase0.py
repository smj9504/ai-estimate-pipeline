"""
Phase 0 Test Implementation - Generate Scope of Work
"""
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
        # Always use test_data fixtures directly
        measurement_data = self.load_fixture("sample_measurement")
        demolition_data = self.load_fixture("sample_demo") 
        intake_form = self.load_fixture("sample_intake_form")
        
        return {
            'measurement_data': measurement_data,
            'demolition_scope_data': demolition_data,
            'intake_form': intake_form,
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