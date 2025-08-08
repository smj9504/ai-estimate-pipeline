"""
Phase 2 Test Implementation - Quantity Survey
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from ..base import PhaseTestBase, PhaseTestConfig, PhaseTestResult
from src.phases.phase2_processor import Phase2Processor


class Phase2Test(PhaseTestBase):
    """Test implementation for Phase 2 - Quantity Survey"""
    
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        self.cached_input_data = None  # For pipeline testing
    
    @property
    def phase_number(self) -> int:
        return 2
    
    @property
    def phase_name(self) -> str:
        return "Quantity Survey"
    
    def set_input_data(self, input_data: Dict[str, Any]):
        """Set input data from previous phase (for pipeline testing)"""
        self.cached_input_data = input_data
    
    async def prepare_test_data(self, test_config: PhaseTestConfig) -> Dict[str, Any]:
        """Prepare test data for Phase 2"""
        # If we have cached data from pipeline, use it
        if self.cached_input_data:
            return self.cached_input_data
        
        try:
            # Try to load from existing Phase 1 output
            return await self._load_phase1_output()
        except FileNotFoundError:
            # Fall back to sample data
            return self._create_sample_phase1_output()
    
    async def _load_phase1_output(self) -> Dict[str, Any]:
        """Load most recent Phase 1 output"""
        # Look in multiple locations for Phase 1 output
        search_patterns = [
            "output/phase1_*.json",
            "test_outputs/*phase1*.json",
            "test_outputs/sessions/session_*.json"
        ]
        
        phase1_files = []
        for pattern in search_patterns:
            phase1_files.extend(Path().glob(pattern))
        
        # Sort by modification time
        phase1_files = sorted(phase1_files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not phase1_files:
            raise FileNotFoundError("No Phase 1 output files found")
        
        with open(phase1_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the actual output data from different possible formats
        if 'test_result' in data and 'output_data' in data['test_result']:
            return data['test_result']['output_data']
        elif 'phase_results' in data:
            # Session format - find Phase 1 result
            for phase_result in data['phase_results']:
                if phase_result.get('phase_number') == 1:
                    return phase_result.get('output_data', {})
        elif 'output_data' in data:
            return data['output_data']
        else:
            return data
    
    def _create_sample_phase1_output(self) -> Dict[str, Any]:
        """Create sample Phase 1 output for testing"""
        return {
            "phase": 1,
            "phase_name": "Merge Measurement & Work Scope",
            "timestamp": datetime.now().isoformat(),
            "project_id": f"TEST_PHASE2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "success": True,
            "confidence_score": 0.85,
            "consensus_level": 0.78,
            "models_responded": 3,
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
                                "Baseboards": "Wood"
                            },
                            "work_scope": {
                                "Paint - Walls": "Remove & Replace",
                                "Paint - Ceiling": "Remove & Replace",
                                "Carpet": "Remove & Replace",
                                "Baseboards": "Remove & Replace"
                            },
                            "measurements": {
                                "width": 16,
                                "length": 20,
                                "height": 9,
                                "windows": 3,
                                "doors": 2
                            },
                            "demo_scope(already demo'd)": {
                                "Carpet": 320
                            },
                            "additional_notes": "High ceiling requires scaffolding"
                        },
                        {
                            "name": "Kitchen",
                            "materials": {
                                "Cabinets - Upper": "Wood",
                                "Cabinets - Lower": "Wood",
                                "Countertop": "Granite",
                                "Paint - Walls": "Existing",
                                "Paint - Ceiling": "Existing"
                            },
                            "work_scope": {
                                "Cabinets - Upper": "Remove & Replace",
                                "Cabinets - Lower": "Remove & Replace", 
                                "Countertop": "Remove & Replace",
                                "Paint - Walls": "Remove & Replace",
                                "Paint - Ceiling": "Remove & Replace"
                            },
                            "measurements": {
                                "width": 12,
                                "length": 14,
                                "height": 10,
                                "linear_feet_upper": 12,
                                "linear_feet_lower": 14,
                                "countertop_sf": 25
                            },
                            "demo_scope(already demo'd)": {
                                "Cabinets - Upper": 6,
                                "Countertop": 10
                            },
                            "additional_notes": "Premium finishes specified"
                        }
                    ]
                }
            ],
            "validation": {
                "remove_replace_logic": {"valid": True, "issues": []},
                "measurements_accuracy": {"valid": True, "issues": []},
                "special_tasks": {"valid": True, "issues": []},
                "overall_valid": True
            }
        }
    
    def validate_input_data(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data for Phase 2"""
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
            self.logger.error("Phase 1 input shows unsuccessful status")
            return False
        
        # Validate data structure
        data = input_data['data']
        if not isinstance(data, list) or len(data) < 2:
            self.logger.error("Data must be a list with at least jobsite_info and one floor")
            return False
        
        # Check that rooms have required data for quantity calculation
        for floor_data in data[1:]:  # Skip jobsite_info
            if not isinstance(floor_data, dict) or 'rooms' not in floor_data:
                self.logger.error("Floor data must contain 'rooms'")
                return False
            
            for room in floor_data['rooms']:
                required_room_keys = ['name', 'materials', 'work_scope', 'measurements']
                for key in required_room_keys:
                    if key not in room:
                        self.logger.error(f"Room missing required key for quantities: {key}")
                        return False
                
                # Validate measurements have numeric values
                measurements = room.get('measurements', {})
                if not any(isinstance(v, (int, float)) for v in measurements.values()):
                    self.logger.error(f"Room {room['name']} has no valid numeric measurements")
                    return False
        
        return True
    
    async def execute_phase(self, input_data: Dict[str, Any], 
                          test_config: PhaseTestConfig) -> PhaseTestResult:
        """Execute Phase 2 test"""
        start_time = datetime.now()
        
        try:
            # Initialize processor
            processor = Phase2Processor()
            
            # Execute phase
            result = await processor.process(
                phase1_output=input_data,
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
                    'total_tasks_calculated': self._count_tasks_with_quantities(result),
                    'rooms_processed': self._count_rooms_processed(result)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Phase 2 execution failed: {e}")
            
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
        
        total_rooms = 0
        for floor_data in result['data'][1:]:  # Skip jobsite_info
            if 'rooms' in floor_data:
                total_rooms += len(floor_data['rooms'])
        
        return total_rooms
    
    def _count_tasks_with_quantities(self, result: Dict[str, Any]) -> int:
        """Count total tasks that have quantities calculated"""
        if not result.get('success') or 'data' not in result:
            return 0
        
        total_tasks = 0
        for floor_data in result['data'][1:]:  # Skip jobsite_info
            if 'rooms' in floor_data:
                for room in floor_data['rooms']:
                    if 'tasks' in room:
                        for task in room['tasks']:
                            if 'quantity' in task and task['quantity'] > 0:
                                total_tasks += 1
        
        return total_tasks
    
    def analyze_results(self, result: PhaseTestResult) -> Dict[str, Any]:
        """Enhanced analysis for Phase 2 results"""
        analysis = super().analyze_results(result)
        
        # Add Phase 2 specific analysis
        if result.output_data and result.success:
            # Analyze quantity calculations
            quantity_analysis = self._analyze_quantities(result.output_data)
            analysis['quantity_analysis'] = quantity_analysis
        
        # Validation analysis
        if result.validation_results:
            validation_summary = {
                'quantity_logic_valid': result.validation_results.get('quantity_logic', {}).get('valid', False),
                'remove_replace_valid': result.validation_results.get('remove_replace_logic', {}).get('valid', False),
                'measurements_valid': result.validation_results.get('measurements_accuracy', {}).get('valid', False),
                'overall_valid': result.validation_results.get('overall_valid', False)
            }
            analysis['validation_summary'] = validation_summary
        
        # Model consensus analysis for quantities
        if result.total_models > 1:
            if result.consensus_level < 0.7:
                analysis['recommendations'].append("Low consensus on quantities - review calculation accuracy")
            if result.confidence_score > 0.9:
                analysis['recommendations'].append("High confidence in quantity calculations")
        
        return analysis
    
    def _analyze_quantities(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quantity calculations in the output"""
        analysis = {
            'total_tasks': 0,
            'tasks_with_quantities': 0,
            'quantity_ranges': {
                'very_small': 0,  # < 10 units
                'small': 0,       # 10-100 units
                'medium': 0,      # 100-1000 units
                'large': 0        # > 1000 units
            },
            'average_quantity': 0.0,
            'rooms_with_calculations': 0
        }
        
        total_quantity = 0.0
        quantity_count = 0
        
        if 'data' in output_data:
            for floor_data in output_data['data'][1:]:  # Skip jobsite_info
                if 'rooms' in floor_data:
                    for room in floor_data['rooms']:
                        room_has_calculations = False
                        
                        if 'tasks' in room:
                            for task in room['tasks']:
                                analysis['total_tasks'] += 1
                                
                                if 'quantity' in task and isinstance(task['quantity'], (int, float)):
                                    analysis['tasks_with_quantities'] += 1
                                    room_has_calculations = True
                                    
                                    quantity = task['quantity']
                                    total_quantity += quantity
                                    quantity_count += 1
                                    
                                    # Categorize quantity size
                                    if quantity < 10:
                                        analysis['quantity_ranges']['very_small'] += 1
                                    elif quantity < 100:
                                        analysis['quantity_ranges']['small'] += 1
                                    elif quantity < 1000:
                                        analysis['quantity_ranges']['medium'] += 1
                                    else:
                                        analysis['quantity_ranges']['large'] += 1
                        
                        if room_has_calculations:
                            analysis['rooms_with_calculations'] += 1
        
        if quantity_count > 0:
            analysis['average_quantity'] = total_quantity / quantity_count
        
        # Calculate completion percentage
        if analysis['total_tasks'] > 0:
            analysis['quantity_completion_rate'] = analysis['tasks_with_quantities'] / analysis['total_tasks']
        else:
            analysis['quantity_completion_rate'] = 0.0
        
        return analysis