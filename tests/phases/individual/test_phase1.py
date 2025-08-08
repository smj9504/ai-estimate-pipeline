"""
Phase 1 Test Implementation - Merge Measurement & Work Scope
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from ..base import PhaseTestBase, PhaseTestConfig, PhaseTestResult
from src.phases.phase1_processor import Phase1Processor


class Phase1Test(PhaseTestBase):
    """Test implementation for Phase 1 - Merge Measurement & Work Scope"""
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.cached_input_data = None  # For pipeline testing
    
    @property
    def phase_number(self) -> int:
        return 1
    
    @property
    def phase_name(self) -> str:
        return "Merge Measurement & Work Scope"
    
    def set_input_data(self, input_data: Dict[str, Any]):
        """Set input data from previous phase (for pipeline testing)"""
        self.cached_input_data = input_data
    
    async def prepare_test_data(self, test_config: PhaseTestConfig) -> Dict[str, Any]:
        """Prepare test data for Phase 1"""
        # If we have cached data from pipeline, use it
        if self.cached_input_data:
            return self.cached_input_data
        
        # Load most recent Phase 0 output, create sample if none exists
        try:
            return await self._load_phase0_output()
        except FileNotFoundError:
            self.logger.warning("No Phase 0 output found, using sample data")
            return self._create_sample_phase0_output()
    
    async def _load_phase0_output(self) -> Dict[str, Any]:
        """Load most recent Phase 0 output"""
        output_dir = Path("output")
        if not output_dir.exists():
            raise FileNotFoundError("No output directory found")
        
        # Find most recent Phase 0 result
        phase0_files = sorted(output_dir.glob("phase0_result*.json"), reverse=True)
        if not phase0_files:
            # Try test outputs
            test_output_dir = Path("test_outputs")
            if test_output_dir.exists():
                phase0_files = sorted(test_output_dir.glob("*phase0*.json"), reverse=True)
        
        if not phase0_files:
            raise FileNotFoundError("No Phase 0 output files found")
        
        with open(phase0_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the actual output data if it's wrapped in test metadata
        if 'test_result' in data and 'output_data' in data['test_result']:
            return data['test_result']['output_data']
        elif 'output_data' in data:
            return data['output_data']
        else:
            return data
    
    def _create_sample_phase0_output(self) -> Dict[str, Any]:
        """Create sample Phase 0 output for testing"""
        return {
            "phase": 0,
            "phase_name": "Generate Scope of Work",
            "timestamp": datetime.now().isoformat(),
            "project_id": f"TEST_PHASE1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "success": True,
            "data": [
                {
                    "jobsite_info": {
                        "property_type": "Single Family Home",
                        "damage_cause": "Water Damage",
                        "floors": 1
                    }
                },
                {
                    "floor_name": "First Floor",
                    "rooms": [
                        {
                            "name": "Living Room",
                            "materials": {
                                "Paint - Walls": "Existing",
                                "Paint - Ceiling": "Existing", 
                                "Carpet": "Existing",
                                "Baseboards": "Wood",
                                "Windows": "Double Pane"
                            },
                            "work_scope": {
                                "Paint - Walls": "Remove & Replace",
                                "Paint - Ceiling": "Remove & Replace",
                                "Carpet": "Remove & Replace",
                                "Baseboards": "Remove & Replace",
                                "Windows": "Clean"
                            },
                            "measurements": {
                                "width": 16,
                                "length": 20,
                                "height": 9,
                                "windows": 3,
                                "doors": 2
                            },
                            "demo_scope(already demo'd)": {
                                "Carpet": 320,
                                "Damaged Drywall": 150
                            },
                            "additional_notes": "Water damage on walls requires primer before painting"
                        },
                        {
                            "name": "Kitchen",
                            "materials": {
                                "Cabinets - Upper": "Wood",
                                "Cabinets - Lower": "Wood",
                                "Countertop": "Granite",
                                "Flooring": "Tile",
                                "Paint - Walls": "Existing",
                                "Paint - Ceiling": "Existing"
                            },
                            "work_scope": {
                                "Cabinets - Upper": "Remove & Replace",
                                "Cabinets - Lower": "Remove & Replace", 
                                "Countertop": "Remove & Replace",
                                "Flooring": "Clean & Seal",
                                "Paint - Walls": "Remove & Replace",
                                "Paint - Ceiling": "Remove & Replace"
                            },
                            "measurements": {
                                "width": 12,
                                "length": 14,
                                "height": 9,
                                "linear_feet_upper": 12,
                                "linear_feet_lower": 14,
                                "countertop_sf": 25
                            },
                            "demo_scope(already demo'd)": {
                                "Cabinets - Upper": 6,
                                "Countertop": 10
                            },
                            "additional_notes": "High-end finishes requested by client"
                        }
                    ]
                }
            ]
        }
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for Phase 1"""
        # Check basic structure
        if not isinstance(input_data, dict):
            self.logger.error("Input data must be a dictionary")
            return False
        
        required_keys = ['data', 'success']
        for key in required_keys:
            if key not in input_data:
                self.logger.error(f"Missing required key: {key}")
                return False
        
        # Validate success status
        if not input_data.get('success', False):
            self.logger.error("Phase 0 input shows unsuccessful status")
            return False
        
        # Validate data structure
        data = input_data['data']
        if not isinstance(data, list) or len(data) < 2:
            self.logger.error("Data must be a list with at least jobsite_info and one floor")
            return False
        
        # Check floor data
        for floor_data in data[1:]:  # Skip jobsite_info
            if not isinstance(floor_data, dict) or 'rooms' not in floor_data:
                self.logger.error("Floor data must contain 'rooms'")
                return False
            
            for room in floor_data['rooms']:
                required_room_keys = ['name', 'work_scope', 'measurements']
                # Check for materials or material key (Phase 0 outputs use 'material')
                has_materials = 'materials' in room or 'material' in room
                for key in required_room_keys:
                    if key not in room:
                        self.logger.error(f"Room missing required key: {key}")
                        return False
                
                if not has_materials:
                    self.logger.error("Room missing required key: materials/material")
                    return False
        
        return True
    
    async def execute_phase(self, input_data: Dict[str, Any], 
                          test_config: PhaseTestConfig) -> PhaseTestResult:
        """Execute Phase 1 test"""
        start_time = datetime.now()
        
        try:
            # Initialize processor
            processor = Phase1Processor()
            
            # Execute phase
            result = await processor.process(
                phase0_output=input_data,
                models_to_use=test_config.models,
                project_id=input_data.get('project_id', f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Analyze results
            success = result.get('success', False)
            confidence_score = result.get('confidence_score', 0.0)
            consensus_level = result.get('consensus_level', 0.0)
            models_responded = result.get('models_responded', 0)
            total_models = len(test_config.models)
            
            # Extract validation results
            validation_results = result.get('validation', {})
            
            return PhaseTestResult(
                phase_number=self.phase_number,
                success=success,
                execution_time=execution_time,
                confidence_score=confidence_score,
                consensus_level=consensus_level,
                models_responded=models_responded,
                total_models=total_models,
                validation_results=validation_results,
                output_data=result,
                metadata={
                    'models_used': test_config.models,
                    'validation_mode': test_config.validation_mode,
                    'rooms_processed': self._count_rooms_processed(result)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Phase 1 execution failed: {e}")
            
            return PhaseTestResult(
                phase_number=self.phase_number,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                total_models=len(test_config.models)
            )
    
    def _count_rooms_processed(self, result: Dict[str, Any]) -> int:
        """Count number of rooms processed in the result"""
        if not result.get('success') or 'data' not in result:
            return 0
        
        data = result['data']
        
        # Handle different data structures
        if isinstance(data, dict):
            # Phase 1 merged result format
            if 'rooms' in data:
                return len(data['rooms'])
            # Phase 0 format with floors
            elif isinstance(data, list) and len(data) > 1:
                total_rooms = 0
                try:
                    for floor_data in data[1:]:  # Skip jobsite_info
                        if isinstance(floor_data, dict) and 'rooms' in floor_data:
                            total_rooms += len(floor_data['rooms'])
                except (TypeError, IndexError):
                    return 0
                return total_rooms
        
        # Fallback - try to find rooms anywhere in the data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'rooms' in item:
                    return len(item['rooms'])
        
        return 0
    
    def analyze_results(self, result: PhaseTestResult) -> Dict[str, Any]:
        """Enhanced analysis for Phase 1 results"""
        analysis = super().analyze_results(result)
        
        # Add Phase 1 specific analysis
        if result.validation_results:
            validation_summary = {
                'remove_replace_valid': result.validation_results.get('remove_replace_logic', {}).get('valid', False),
                'measurements_valid': result.validation_results.get('measurements_accuracy', {}).get('valid', False),
                'special_tasks_valid': result.validation_results.get('special_tasks', {}).get('valid', False),
                'overall_valid': result.validation_results.get('overall_valid', False)
            }
            analysis['validation_summary'] = validation_summary
            
            # Count validation issues
            total_issues = 0
            for category, info in result.validation_results.items():
                if isinstance(info, dict) and 'issues' in info:
                    total_issues += len(info['issues'])
            
            analysis['validation_issues_count'] = total_issues
        
        # Model performance analysis
        if result.total_models > 1:
            if result.consensus_level < 0.6:
                analysis['recommendations'].append("Low consensus between models - review input quality")
            if result.model_success_rate < 0.8:
                analysis['recommendations'].append("Some models failed - check API connectivity")
        
        return analysis