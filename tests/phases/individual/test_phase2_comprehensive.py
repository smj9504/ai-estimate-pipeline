"""
Comprehensive Test Suite for Phase 2 (Market Research & Pricing)
Implements the testing strategy with golden datasets and error recovery
"""
import asyncio
import json
import pytest
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.phases.phase2_processor import Phase2Processor


class Phase2TestDatasets:
    """Golden datasets for Phase 2 testing"""
    
    @staticmethod
    def get_perfect_dataset() -> Dict[str, Any]:
        """Perfect Phase 1 output - happy path"""
        return {
            'phase': 1,
            'phase_name': 'Work Scope & Quantity Calculation (Integrated)',
            'success': True,
            'confidence_score': 0.92,
            'project_id': 'TEST_PERFECT_001',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'rooms': [
                    {
                        'name': 'Living Room',
                        'measurements': {
                            'sqft': 200,
                            'height': 9,
                            'perimeter': 56
                        },
                        'tasks': [
                            {
                                'description': 'Remove damaged drywall',
                                'quantity': 400,
                                'unit': 'sqft',
                                'category': 'Demolition'
                            },
                            {
                                'description': 'Install new drywall',
                                'quantity': 400,
                                'unit': 'sqft',
                                'quantity_with_waste': 440,
                                'waste_factor': 10,
                                'category': 'Installation'
                            },
                            {
                                'description': 'Paint walls and ceiling',
                                'quantity': 800,
                                'unit': 'sqft',
                                'quantity_with_waste': 840,
                                'waste_factor': 5,
                                'category': 'Finishing'
                            },
                            {
                                'description': 'Install hardwood flooring',
                                'quantity': 200,
                                'unit': 'sqft',
                                'quantity_with_waste': 224,
                                'waste_factor': 12,
                                'category': 'Installation'
                            }
                        ]
                    },
                    {
                        'name': 'Kitchen',
                        'measurements': {
                            'sqft': 150,
                            'height': 9,
                            'perimeter': 50
                        },
                        'tasks': [
                            {
                                'description': 'Remove cabinets',
                                'quantity': 20,
                                'unit': 'LF',
                                'category': 'Demolition'
                            },
                            {
                                'description': 'Install upper cabinets',
                                'quantity': 12,
                                'unit': 'LF',
                                'quantity_with_waste': 13.2,
                                'waste_factor': 10,
                                'category': 'Installation'
                            },
                            {
                                'description': 'Install base cabinets',
                                'quantity': 15,
                                'unit': 'LF',
                                'quantity_with_waste': 16.5,
                                'waste_factor': 10,
                                'category': 'Installation'
                            },
                            {
                                'description': 'Install granite countertops',
                                'quantity': 40,
                                'unit': 'sqft',
                                'quantity_with_waste': 44,
                                'waste_factor': 10,
                                'category': 'Installation'
                            }
                        ]
                    }
                ],
                'waste_summary': {
                    'drywall': {
                        'base_quantity': 400,
                        'waste_amount': 40,
                        'waste_factor': 0.10
                    },
                    'paint': {
                        'base_quantity': 800,
                        'waste_amount': 40,
                        'waste_factor': 0.05
                    },
                    'hardwood': {
                        'base_quantity': 200,
                        'waste_amount': 24,
                        'waste_factor': 0.12
                    },
                    'cabinets': {
                        'base_quantity': 27,
                        'waste_amount': 2.7,
                        'waste_factor': 0.10
                    },
                    'countertops': {
                        'base_quantity': 40,
                        'waste_amount': 4,
                        'waste_factor': 0.10
                    }
                },
                'waste_factors_applied': True
            },
            'validation': {
                'remove_replace_logic': {'valid': True, 'issues': []},
                'quantity_accuracy': {'valid': True, 'issues': []},
                'waste_factors': {'valid': True, 'issues': []},
                'overall_valid': True
            }
        }
    
    @staticmethod
    def get_realistic_dataset() -> Dict[str, Any]:
        """Realistic Phase 1 output with minor issues"""
        return {
            'phase': 1,
            'phase_name': 'Work Scope & Quantity Calculation (Integrated)',
            'success': True,
            'confidence_score': 0.78,
            'project_id': 'TEST_REALISTIC_002',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'rooms': [
                    {
                        'name': 'Master Bedroom',
                        'measurements': {
                            'sqft': 180
                            # Missing height - common issue
                        },
                        'tasks': [
                            {
                                'description': 'Remove damaged carpet',
                                'quantity': 180,
                                'unit': 'sqft',
                                'category': 'Demolition'
                                # Missing waste factor fields
                            },
                            {
                                'description': 'Install luxury vinyl plank',
                                'quantity': 195,  # Has quantity but missing waste breakdown
                                'unit': 'sqft',
                                'category': 'Installation'
                            },
                            {
                                'description': 'Install baseboards',
                                'quantity': 65,
                                'unit': 'LF',
                                'quantity_with_waste': 71.5,
                                'waste_factor': 10,
                                'category': 'Installation'
                            }
                        ]
                    },
                    {
                        'name': 'Bathroom',
                        'measurements': {
                            'sqft': 50,
                            'height': 8,
                            'perimeter': 30
                        },
                        'tasks': [
                            {
                                'description': 'Remove tile flooring',
                                'quantity': 50,
                                'unit': 'sqft',
                                'category': 'Demolition'
                            },
                            {
                                'description': 'Install ceramic tile',
                                'quantity': 50,
                                'unit': 'sqft',
                                'quantity_with_waste': 56,
                                'waste_factor': 12,
                                'category': 'Installation'
                            },
                            {
                                'description': 'Install vanity',
                                'quantity': 1,
                                'unit': 'EA',
                                'category': 'Installation'
                            }
                        ]
                    }
                ],
                'waste_summary': {
                    # Partial waste summary
                    'baseboards': {
                        'base_quantity': 65,
                        'waste_amount': 6.5
                        # Missing waste_factor
                    },
                    'tile': {
                        'base_quantity': 50
                        # Missing waste_amount
                    }
                },
                'waste_factors_applied': True  # Claimed but incomplete
            },
            'validation': {
                'remove_replace_logic': {'valid': True, 'issues': []},
                'quantity_accuracy': {
                    'valid': False, 
                    'issues': ['Missing height measurements for Master Bedroom']
                },
                'waste_factors': {
                    'valid': False, 
                    'issues': ['Incomplete waste factor data for some materials']
                },
                'overall_valid': False
            }
        }
    
    @staticmethod
    def get_edge_case_dataset() -> Dict[str, Any]:
        """Edge case Phase 1 output with severe issues"""
        return {
            'phase': 1,
            'phase_name': 'Work Scope & Quantity Calculation (Integrated)',
            'success': True,  # Marked as success despite issues
            'confidence_score': 0.45,
            'project_id': 'TEST_EDGE_003',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'rooms': [
                    {
                        'name': 'Unknown Room',
                        'measurements': {},  # Empty measurements
                        'tasks': [
                            {
                                'description': 'Tile installation',
                                'quantity': 0,  # Zero quantity
                                'unit': 'sqft',
                                'category': 'Installation'
                            },
                            {
                                'description': 'Plumbing fixture replacement',
                                'quantity': 99999,  # Extreme value
                                'unit': 'EA',
                                'category': 'Installation'
                            },
                            {
                                'description': 'Unknown work item',
                                # Missing all quantity fields
                                'category': 'Unknown'
                            }
                        ]
                    },
                    {
                        'name': '',  # Empty room name
                        'tasks': []  # Empty tasks
                    },
                    {
                        # Room with null values
                        'name': None,
                        'measurements': None,
                        'tasks': None
                    }
                ]
                # Missing waste_summary entirely
            },
            'validation': {
                'overall_valid': False,
                'errors': ['Multiple critical data issues detected']
            }
        }


class Phase2InputValidator:
    """Validates and sanitizes Phase 1 outputs for Phase 2"""
    
    WASTE_FACTORS = {
        'drywall': 0.10,
        'paint': 0.05,
        'carpet': 0.08,
        'hardwood': 0.12,
        'tile': 0.12,
        'vinyl': 0.08,
        'lvp': 0.08,
        'cabinets': 0.10,
        'countertops': 0.10,
        'default': 0.10
    }
    
    def __init__(self):
        self.corrections_log = []
    
    def validate_and_sanitize(self, phase1_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize Phase 1 output"""
        
        # Deep copy to avoid modifying original
        import copy
        sanitized = copy.deepcopy(phase1_output)
        
        # Apply corrections
        sanitized = self._ensure_required_structure(sanitized)
        sanitized = self._fix_missing_waste_factors(sanitized)
        sanitized = self._fix_extreme_values(sanitized)
        sanitized = self._fix_missing_measurements(sanitized)
        sanitized = self._ensure_valid_room_names(sanitized)
        sanitized = self._remove_invalid_rooms(sanitized)
        
        # Validate after sanitization
        validation_result = self._validate_sanitized(sanitized)
        
        return {
            'data': sanitized,
            'validation': validation_result,
            'corrections_applied': self.corrections_log,
            'input_quality_score': self._calculate_quality_score(sanitized)
        }
    
    def _ensure_required_structure(self, data: Dict) -> Dict:
        """Ensure required data structure exists"""
        if 'data' not in data:
            data['data'] = {}
            self.corrections_log.append("Added missing 'data' key")
        
        if 'rooms' not in data['data']:
            data['data']['rooms'] = []
            self.corrections_log.append("Added missing 'rooms' array")
        
        if 'waste_summary' not in data['data']:
            data['data']['waste_summary'] = {}
            self.corrections_log.append("Added missing 'waste_summary'")
        
        return data
    
    def _fix_missing_waste_factors(self, data: Dict) -> Dict:
        """Apply default waste factors where missing"""
        for room in data.get('data', {}).get('rooms', []):
            if not room or not isinstance(room, dict):
                continue
                
            for task in room.get('tasks', []):
                if not task or not isinstance(task, dict):
                    continue
                    
                if 'quantity' in task and task['quantity'] > 0:
                    if 'quantity_with_waste' not in task:
                        material_type = self._identify_material_type(
                            task.get('description', ''),
                            task.get('category', '')
                        )
                        waste_factor = self.WASTE_FACTORS.get(
                            material_type, 
                            self.WASTE_FACTORS['default']
                        )
                        task['waste_factor'] = waste_factor * 100
                        task['quantity_with_waste'] = task['quantity'] * (1 + waste_factor)
                        self.corrections_log.append(
                            f"Applied {waste_factor*100:.0f}% waste factor to {task.get('description', 'unknown task')}"
                        )
        
        return data
    
    def _fix_extreme_values(self, data: Dict) -> Dict:
        """Fix extreme or invalid quantity values"""
        for room in data.get('data', {}).get('rooms', []):
            if not room or not isinstance(room, dict):
                continue
                
            room_sqft = room.get('measurements', {}).get('sqft', 100)  # Default
            
            for task in room.get('tasks', []):
                if not task or not isinstance(task, dict):
                    continue
                
                # Fix zero quantities
                if task.get('quantity') == 0:
                    if 'sqft' in task.get('unit', '').lower():
                        task['quantity'] = room_sqft
                    else:
                        task['quantity'] = 1  # Default for EA units
                    task['needs_review'] = True
                    self.corrections_log.append(
                        f"Fixed zero quantity for {task.get('description', 'unknown')}"
                    )
                
                # Fix extreme quantities
                if task.get('quantity', 0) > 10000:
                    if 'sqft' in task.get('unit', '').lower():
                        task['quantity'] = min(task['quantity'], room_sqft * 4)
                    else:
                        task['quantity'] = min(task['quantity'], 100)
                    task['needs_review'] = True
                    self.corrections_log.append(
                        f"Capped extreme quantity for {task.get('description', 'unknown')}"
                    )
        
        return data
    
    def _fix_missing_measurements(self, data: Dict) -> Dict:
        """Add default measurements where missing"""
        for room in data.get('data', {}).get('rooms', []):
            if not room or not isinstance(room, dict):
                continue
                
            if 'measurements' not in room or not room['measurements']:
                room['measurements'] = {}
                
            measurements = room['measurements']
            
            # Add default height if missing
            if 'height' not in measurements:
                measurements['height'] = 9  # Standard ceiling height
                self.corrections_log.append(
                    f"Added default height (9ft) for {room.get('name', 'unknown room')}"
                )
            
            # Estimate perimeter if missing
            if 'perimeter' not in measurements and 'sqft' in measurements:
                # Rough estimate assuming square room
                import math
                side_length = math.sqrt(measurements['sqft'])
                measurements['perimeter'] = side_length * 4
                self.corrections_log.append(
                    f"Estimated perimeter for {room.get('name', 'unknown room')}"
                )
        
        return data
    
    def _ensure_valid_room_names(self, data: Dict) -> Dict:
        """Ensure all rooms have valid names"""
        room_counter = 1
        for room in data.get('data', {}).get('rooms', []):
            if not room or not isinstance(room, dict):
                continue
                
            if not room.get('name') or room['name'] == '':
                room['name'] = f"Room {room_counter}"
                self.corrections_log.append(f"Added default name: Room {room_counter}")
                room_counter += 1
        
        return data
    
    def _remove_invalid_rooms(self, data: Dict) -> Dict:
        """Remove completely invalid rooms"""
        valid_rooms = []
        
        for room in data.get('data', {}).get('rooms', []):
            if not room or not isinstance(room, dict):
                self.corrections_log.append("Removed invalid room (null or non-dict)")
                continue
                
            if not room.get('tasks') or len(room['tasks']) == 0:
                self.corrections_log.append(
                    f"Removed room '{room.get('name', 'unknown')}' with no tasks"
                )
                continue
                
            valid_rooms.append(room)
        
        data['data']['rooms'] = valid_rooms
        return data
    
    def _identify_material_type(self, description: str, category: str) -> str:
        """Identify material type from description"""
        description_lower = description.lower()
        
        material_mapping = {
            'drywall': ['drywall', 'sheetrock', 'gypsum'],
            'paint': ['paint', 'primer'],
            'carpet': ['carpet'],
            'hardwood': ['hardwood', 'wood floor'],
            'tile': ['tile', 'ceramic', 'porcelain'],
            'vinyl': ['vinyl', 'lvt'],
            'lvp': ['lvp', 'luxury vinyl'],
            'cabinets': ['cabinet'],
            'countertops': ['countertop', 'granite', 'quartz']
        }
        
        for material_type, keywords in material_mapping.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return material_type
        
        return 'default'
    
    def _validate_sanitized(self, data: Dict) -> Dict:
        """Validate sanitized data"""
        issues = []
        warnings = []
        
        # Check for required fields
        if not data.get('data', {}).get('rooms'):
            issues.append("No valid rooms found after sanitization")
        
        # Check data quality
        total_tasks = sum(
            len(r.get('tasks', [])) 
            for r in data.get('data', {}).get('rooms', [])
        )
        
        if total_tasks < 3:
            warnings.append("Very few tasks found - results may be unreliable")
        
        # Check for review flags
        tasks_needing_review = sum(
            1 for r in data.get('data', {}).get('rooms', [])
            for t in r.get('tasks', [])
            if t.get('needs_review')
        )
        
        if tasks_needing_review > 0:
            warnings.append(f"{tasks_needing_review} tasks need manual review")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def _calculate_quality_score(self, data: Dict) -> float:
        """Calculate quality score of sanitized data (0-1)"""
        score = 1.0
        
        # Penalize for corrections
        score -= len(self.corrections_log) * 0.02
        
        # Check completeness
        rooms = data.get('data', {}).get('rooms', [])
        if not rooms:
            return 0.0
        
        # Check task completeness
        total_tasks = sum(len(r.get('tasks', [])) for r in rooms)
        complete_tasks = sum(
            1 for r in rooms
            for t in r.get('tasks', [])
            if all(k in t for k in ['description', 'quantity', 'unit', 'category'])
        )
        
        if total_tasks > 0:
            completeness_ratio = complete_tasks / total_tasks
            score *= completeness_ratio
        
        # Check for waste factors
        tasks_with_waste = sum(
            1 for r in rooms
            for t in r.get('tasks', [])
            if 'quantity_with_waste' in t
        )
        
        if total_tasks > 0:
            waste_ratio = tasks_with_waste / total_tasks
            score *= (0.5 + 0.5 * waste_ratio)  # Partial penalty
        
        return max(0.0, min(1.0, score))


class TestPhase2Comprehensive:
    """Comprehensive test suite for Phase 2"""
    
    @pytest.fixture
    def processor(self):
        """Create Phase 2 processor instance"""
        return Phase2Processor()
    
    @pytest.fixture
    def validator(self):
        """Create input validator instance"""
        return Phase2InputValidator()
    
    @pytest.fixture
    def datasets(self):
        """Get all test datasets"""
        return {
            'perfect': Phase2TestDatasets.get_perfect_dataset(),
            'realistic': Phase2TestDatasets.get_realistic_dataset(),
            'edge_case': Phase2TestDatasets.get_edge_case_dataset()
        }
    
    # Unit Tests
    
    def test_validator_perfect_input(self, validator):
        """Test validator with perfect input"""
        perfect_data = Phase2TestDatasets.get_perfect_dataset()
        result = validator.validate_and_sanitize(perfect_data)
        
        assert result['validation']['valid'] == True
        assert len(result['corrections_applied']) == 0
        assert result['input_quality_score'] > 0.9
    
    def test_validator_realistic_input(self, validator):
        """Test validator with realistic input"""
        realistic_data = Phase2TestDatasets.get_realistic_dataset()
        result = validator.validate_and_sanitize(realistic_data)
        
        assert result['validation']['valid'] == True
        assert len(result['corrections_applied']) > 0
        assert 0.6 < result['input_quality_score'] < 0.9
    
    def test_validator_edge_case_input(self, validator):
        """Test validator with edge case input"""
        edge_data = Phase2TestDatasets.get_edge_case_dataset()
        result = validator.validate_and_sanitize(edge_data)
        
        # Should handle edge cases gracefully
        assert 'data' in result
        assert len(result['corrections_applied']) > 5
        assert result['input_quality_score'] < 0.5
    
    # Integration Tests (with mocked AI models)
    
    @pytest.mark.asyncio
    async def test_phase2_perfect_input(self, processor):
        """Test Phase 2 with perfect input"""
        perfect_data = Phase2TestDatasets.get_perfect_dataset()
        
        # Mock model responses
        mock_responses = self._create_mock_model_responses('perfect')
        
        with patch.object(processor.orchestrator, 'run_parallel', 
                         return_value=mock_responses):
            result = await processor.process(
                perfect_data,
                models_to_use=['gpt4', 'claude', 'gemini']
            )
        
        assert result['success'] == True
        assert result['confidence_score'] > 0.85
        assert 'data' in result
        assert result['phase3_ready'] == True
    
    @pytest.mark.asyncio
    async def test_phase2_realistic_input(self, processor, validator):
        """Test Phase 2 with realistic input after validation"""
        realistic_data = Phase2TestDatasets.get_realistic_dataset()
        
        # Validate and sanitize first
        validated = validator.validate_and_sanitize(realistic_data)
        
        # Mock model responses
        mock_responses = self._create_mock_model_responses('realistic')
        
        with patch.object(processor.orchestrator, 'run_parallel',
                         return_value=mock_responses):
            result = await processor.process(
                validated['data'],
                models_to_use=['gpt4', 'claude', 'gemini']
            )
        
        assert result['success'] == True
        assert 0.6 < result['confidence_score'] < 0.85
        assert len(result.get('validation', {}).get('warnings', [])) > 0
    
    @pytest.mark.asyncio
    async def test_phase2_edge_case_recovery(self, processor, validator):
        """Test Phase 2 recovery with edge case input"""
        edge_data = Phase2TestDatasets.get_edge_case_dataset()
        
        # Validate and sanitize
        validated = validator.validate_and_sanitize(edge_data)
        
        if validated['input_quality_score'] < 0.3:
            # Too poor quality - use fallback
            result = self._apply_fallback_pricing(validated['data'])
        else:
            # Try processing with low expectations
            mock_responses = self._create_mock_model_responses('edge')
            
            with patch.object(processor.orchestrator, 'run_parallel',
                            return_value=mock_responses):
                result = await processor.process(
                    validated['data'],
                    models_to_use=['gpt4']  # Use single model for edge cases
                )
        
        assert 'data' in result or 'fallback_result' in result
        assert result.get('confidence_score', 0) < 0.6
    
    # Helper Methods
    
    def _create_mock_model_responses(self, scenario: str) -> List:
        """Create mock model responses for testing"""
        from src.models.data_models import ModelResponse
        
        if scenario == 'perfect':
            return [
                ModelResponse(
                    model_name='gpt4',
                    success=True,
                    data={'market_prices': self._get_market_prices()},
                    processing_time=1.5,
                    confidence_score=0.9
                ),
                ModelResponse(
                    model_name='claude',
                    success=True,
                    data={'market_prices': self._get_market_prices(variation=0.05)},
                    processing_time=1.2,
                    confidence_score=0.88
                ),
                ModelResponse(
                    model_name='gemini',
                    success=True,
                    data={'market_prices': self._get_market_prices(variation=0.08)},
                    processing_time=1.8,
                    confidence_score=0.85
                )
            ]
        
        elif scenario == 'realistic':
            return [
                ModelResponse(
                    model_name='gpt4',
                    success=True,
                    data={'market_prices': self._get_market_prices()},
                    processing_time=2.0,
                    confidence_score=0.75
                ),
                ModelResponse(
                    model_name='claude',
                    success=True,
                    data={'market_prices': self._get_market_prices(variation=0.15)},
                    processing_time=1.8,
                    confidence_score=0.72
                ),
                ModelResponse(
                    model_name='gemini',
                    success=False,
                    error='API timeout',
                    processing_time=5.0,
                    confidence_score=0.0
                )
            ]
        
        else:  # edge case
            return [
                ModelResponse(
                    model_name='gpt4',
                    success=True,
                    data={'market_prices': self._get_market_prices(variation=0.3)},
                    processing_time=3.5,
                    confidence_score=0.4
                )
            ]
    
    def _get_market_prices(self, variation: float = 0.0) -> Dict:
        """Get mock market prices with optional variation"""
        import random
        
        base_prices = {
            'drywall': {'material': 12.50, 'labor': 35.00},
            'paint': {'material': 3.50, 'labor': 2.50},
            'hardwood': {'material': 8.50, 'labor': 4.50},
            'tile': {'material': 5.50, 'labor': 8.00},
            'cabinets': {'material': 150.00, 'labor': 75.00},
            'countertops': {'material': 65.00, 'labor': 45.00}
        }
        
        if variation > 0:
            # Apply random variation
            for material in base_prices:
                for cost_type in base_prices[material]:
                    original = base_prices[material][cost_type]
                    variance = original * variation
                    base_prices[material][cost_type] = original + random.uniform(-variance, variance)
        
        return base_prices
    
    def _apply_fallback_pricing(self, data: Dict) -> Dict:
        """Apply fallback pricing strategy"""
        return {
            'success': True,
            'confidence_score': 0.3,
            'fallback_used': True,
            'data': {
                'message': 'Used historical average pricing due to poor input quality',
                'estimated_total': 15000.00,
                'breakdown': {
                    'materials': 8000.00,
                    'labor': 7000.00
                }
            }
        }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])