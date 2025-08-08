# Phase 2 (Market Research & Pricing) Testing Strategy

## Executive Summary

This document outlines a comprehensive testing strategy for Phase 2 (Market Research & Pricing) of the AI construction estimation pipeline. The strategy ensures robust handling of Phase 1 outputs, maintains data integrity across the pipeline, and provides mechanisms for error recovery and edge case management.

## Architecture Overview

### Phase 2 Responsibilities
- Accept Phase 1 outputs (work scope, quantities, waste factors)
- Query market pricing for materials and labor
- Apply regional pricing adjustments (DMV area)
- Generate cost estimates with confidence scoring
- Prepare data for Phase 3 (Timeline & Disposal)

### Critical Integration Points
1. **Phase 1 → Phase 2 Interface**: Quantity data with waste factors
2. **Multi-Model Orchestration**: GPT-4, Claude, Gemini consensus
3. **Market Data Sources**: Material prices, labor rates, regional factors
4. **Output Contract**: Pricing data structure for Phase 3

## 1. Golden Dataset Design

### 1.1 Perfect Dataset (Happy Path)
```python
perfect_phase1_output = {
    'phase': 1,
    'success': True,
    'confidence_score': 0.92,
    'data': {
        'rooms': [
            {
                'name': 'Living Room',
                'measurements': {'sqft': 200, 'height': 9},
                'tasks': [
                    {
                        'description': 'Install drywall',
                        'quantity': 800,
                        'unit': 'sqft',
                        'quantity_with_waste': 880,  # 10% waste applied
                        'waste_factor': 10,
                        'category': 'Installation'
                    },
                    {
                        'description': 'Paint walls and ceiling',
                        'quantity': 800,
                        'unit': 'sqft',
                        'quantity_with_waste': 840,  # 5% waste applied
                        'waste_factor': 5,
                        'category': 'Finishing'
                    }
                ]
            }
        ],
        'waste_summary': {
            'drywall': {'base_quantity': 800, 'waste_amount': 80},
            'paint': {'base_quantity': 800, 'waste_amount': 40}
        }
    }
}
```

### 1.2 Realistic Dataset (Minor Issues)
```python
realistic_phase1_output = {
    'phase': 1,
    'success': True,
    'confidence_score': 0.78,  # Lower confidence
    'data': {
        'rooms': [
            {
                'name': 'Master Bedroom',
                'measurements': {'sqft': 150},  # Missing height
                'tasks': [
                    {
                        'description': 'Remove damaged carpet',
                        'quantity': 150,
                        'unit': 'sqft',
                        # Missing waste factor fields
                        'category': 'Demolition'
                    },
                    {
                        'description': 'Install luxury vinyl plank',
                        'quantity': 165,  # Has quantity but missing waste breakdown
                        'unit': 'sqft',
                        'category': 'Installation'
                    }
                ]
            }
        ],
        # Partial waste summary
        'waste_summary': {
            'lvp': {'base_quantity': 150}  # Missing waste_amount
        }
    }
}
```

### 1.3 Edge Case Dataset
```python
edge_case_phase1_output = {
    'phase': 1,
    'success': True,
    'confidence_score': 0.45,  # Very low confidence
    'data': {
        'rooms': [
            {
                'name': 'Bathroom',
                'measurements': {},  # Empty measurements
                'tasks': [
                    {
                        'description': 'Tile installation',
                        'quantity': 0,  # Zero quantity
                        'unit': 'sqft',
                        'category': 'Installation'
                    },
                    {
                        'description': 'Plumbing fixture',
                        'quantity': 99999,  # Extreme value
                        'unit': 'EA',
                        'category': 'Installation'
                    },
                    {
                        'description': 'Unknown work',
                        # Missing all quantity fields
                        'category': 'Unknown'
                    }
                ]
            },
            {
                'name': '',  # Empty room name
                'tasks': []  # Empty tasks
            }
        ],
        # Missing waste_summary entirely
    }
}
```

## 2. Test Architecture Implementation

### 2.1 Unit Tests

```python
# tests/phases/unit/test_phase2_components.py

import pytest
from unittest.mock import Mock, patch
from src.phases.phase2_processor import Phase2Processor

class TestPhase2Components:
    """Unit tests for Phase 2 individual components"""
    
    @pytest.fixture
    def processor(self):
        return Phase2Processor()
    
    def test_material_pricing_extraction(self, processor):
        """Test extraction of materials for pricing"""
        input_data = {...}  # Use golden dataset
        materials = processor._extract_materials_for_pricing(input_data)
        
        assert len(materials) > 0
        assert all('description' in m for m in materials)
        assert all('quantity' in m for m in materials)
    
    def test_labor_calculation(self, processor):
        """Test labor hour calculations"""
        tasks = [
            {'description': 'Install drywall', 'quantity': 100, 'unit': 'sqft'}
        ]
        labor_hours = processor._calculate_labor_hours(tasks)
        
        assert labor_hours > 0
        assert labor_hours == pytest.approx(2.5, rel=0.1)  # ~2.5 hrs per 100 sqft
    
    def test_regional_adjustment(self, processor):
        """Test DMV area pricing adjustments"""
        base_price = 100.0
        adjusted = processor._apply_regional_adjustment(base_price, 'DMV')
        
        assert adjusted > base_price  # DMV typically higher
        assert adjusted == pytest.approx(base_price * 1.15, rel=0.05)
    
    def test_waste_factor_pricing(self, processor):
        """Test pricing with waste factors included"""
        task = {
            'quantity': 100,
            'quantity_with_waste': 110,
            'unit_price': 5.0
        }
        total_cost = processor._calculate_material_cost(task)
        
        assert total_cost == 550.0  # 110 * 5.0
```

### 2.2 Integration Tests

```python
# tests/phases/integration/test_phase2_pipeline.py

import asyncio
import pytest
from src.phases.phase2_processor import Phase2Processor

class TestPhase2Integration:
    """Integration tests for Phase 2 with mocked AI models"""
    
    @pytest.fixture
    def mock_model_responses(self):
        return {
            'gpt4': self._create_gpt4_response(),
            'claude': self._create_claude_response(),
            'gemini': self._create_gemini_response()
        }
    
    @pytest.mark.asyncio
    async def test_perfect_input_processing(self, mock_model_responses):
        """Test Phase 2 with perfect Phase 1 output"""
        processor = Phase2Processor()
        
        with patch('src.models.model_interface.ModelOrchestrator.run_parallel',
                  return_value=mock_model_responses):
            result = await processor.process(perfect_phase1_output)
        
        assert result['success'] == True
        assert result['confidence_score'] > 0.8
        assert 'pricing_data' in result['data']
        assert all(room['total_cost'] > 0 for room in result['data']['rooms'])
    
    @pytest.mark.asyncio
    async def test_realistic_input_recovery(self):
        """Test Phase 2 with realistic (imperfect) input"""
        processor = Phase2Processor()
        result = await processor.process(realistic_phase1_output)
        
        # Should handle missing waste factors gracefully
        assert result['success'] == True
        assert 'warnings' in result['validation']
        assert 'Missing waste factors applied defaults' in result['validation']['warnings']
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self):
        """Test Phase 2 with edge cases"""
        processor = Phase2Processor()
        result = await processor.process(edge_case_phase1_output)
        
        assert result['success'] == True  # Graceful degradation
        assert result['confidence_score'] < 0.5  # Low confidence
        assert 'errors' in result['validation']
        assert len(result['validation']['errors']) > 0
```

### 2.3 Contract Tests

```python
# tests/phases/contract/test_phase1_phase2_contract.py

import pytest
from jsonschema import validate, ValidationError

class TestPhase1Phase2Contract:
    """Contract tests for Phase 1 → Phase 2 interface"""
    
    @property
    def phase2_input_schema(self):
        return {
            "type": "object",
            "required": ["phase", "success", "data"],
            "properties": {
                "phase": {"type": "number", "const": 1},
                "success": {"type": "boolean"},
                "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
                "data": {
                    "type": "object",
                    "required": ["rooms"],
                    "properties": {
                        "rooms": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "tasks"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "measurements": {"type": "object"},
                                    "tasks": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["description", "category"],
                                            "properties": {
                                                "description": {"type": "string"},
                                                "quantity": {"type": "number"},
                                                "unit": {"type": "string"},
                                                "quantity_with_waste": {"type": "number"},
                                                "waste_factor": {"type": "number"},
                                                "category": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "waste_summary": {"type": "object"}
                    }
                }
            }
        }
    
    def test_perfect_dataset_contract(self):
        """Validate perfect dataset against schema"""
        validate(instance=perfect_phase1_output, schema=self.phase2_input_schema)
    
    def test_realistic_dataset_contract(self):
        """Validate realistic dataset against schema"""
        # Should still pass as required fields are present
        validate(instance=realistic_phase1_output, schema=self.phase2_input_schema)
    
    def test_phase2_output_contract(self):
        """Validate Phase 2 output for Phase 3 compatibility"""
        phase2_output = {
            "phase": 2,
            "success": True,
            "data": {
                "rooms": [...],
                "total_material_cost": 5000.0,
                "total_labor_cost": 3000.0,
                "regional_adjustment": 1.15,
                "market_data_timestamp": "2024-01-15T10:00:00"
            }
        }
        validate(instance=phase2_output, schema=self.phase3_input_schema)
```

## 3. Error Recovery Mechanisms

### 3.1 Input Validation & Sanitization

```python
class Phase2InputValidator:
    """Validates and sanitizes Phase 1 outputs before Phase 2 processing"""
    
    def validate_and_sanitize(self, phase1_output: Dict) -> Dict:
        """Main validation and sanitization entry point"""
        
        # 1. Check required structure
        if not self._has_required_structure(phase1_output):
            raise ValueError("Invalid Phase 1 output structure")
        
        # 2. Sanitize data
        sanitized = self._deep_copy(phase1_output)
        
        # 3. Apply corrections
        sanitized = self._fix_missing_waste_factors(sanitized)
        sanitized = self._fix_extreme_values(sanitized)
        sanitized = self._fix_missing_units(sanitized)
        sanitized = self._ensure_room_names(sanitized)
        
        # 4. Validate after sanitization
        validation_result = self._validate_sanitized(sanitized)
        
        return {
            'data': sanitized,
            'validation': validation_result,
            'corrections_applied': self._get_corrections_log()
        }
    
    def _fix_missing_waste_factors(self, data: Dict) -> Dict:
        """Apply default waste factors where missing"""
        for room in data.get('data', {}).get('rooms', []):
            for task in room.get('tasks', []):
                if 'quantity' in task and 'quantity_with_waste' not in task:
                    # Apply default waste factor based on material type
                    material_type = self._identify_material_type(task['description'])
                    default_waste = WASTE_FACTORS.get(material_type, 0.10)
                    task['waste_factor'] = default_waste * 100
                    task['quantity_with_waste'] = task['quantity'] * (1 + default_waste)
                    self._log_correction(f"Applied default waste factor for {task['description']}")
        return data
    
    def _fix_extreme_values(self, data: Dict) -> Dict:
        """Correct extreme or invalid values"""
        for room in data.get('data', {}).get('rooms', []):
            for task in room.get('tasks', []):
                # Fix zero quantities
                if task.get('quantity') == 0:
                    task['quantity'] = self._estimate_quantity(task, room)
                    self._log_correction(f"Estimated zero quantity for {task['description']}")
                
                # Fix extreme quantities (>10000 for most units)
                if task.get('quantity', 0) > 10000:
                    task['quantity'] = self._cap_quantity(task)
                    task['needs_review'] = True
                    self._log_correction(f"Capped extreme quantity for {task['description']}")
        return data
```

### 3.2 Fallback Mechanisms

```python
class Phase2FallbackStrategy:
    """Fallback strategies for Phase 2 processing failures"""
    
    def __init__(self):
        self.historical_prices = self._load_historical_prices()
        self.default_multipliers = {
            'materials': 1.0,
            'labor': 1.0,
            'regional': 1.15  # DMV default
        }
    
    async def apply_fallback(self, phase1_data: Dict, failure_type: str) -> Dict:
        """Apply appropriate fallback based on failure type"""
        
        if failure_type == 'all_models_failed':
            return self._use_historical_pricing(phase1_data)
        
        elif failure_type == 'partial_model_failure':
            return self._use_majority_consensus(phase1_data)
        
        elif failure_type == 'market_data_unavailable':
            return self._use_cached_pricing(phase1_data)
        
        elif failure_type == 'invalid_input_structure':
            return self._reconstruct_from_partial(phase1_data)
        
        else:
            return self._apply_conservative_defaults(phase1_data)
    
    def _use_historical_pricing(self, data: Dict) -> Dict:
        """Use historical average prices when live pricing fails"""
        result = {
            'rooms': [],
            'pricing_method': 'historical_fallback',
            'confidence_score': 0.6
        }
        
        for room in data.get('rooms', []):
            room_pricing = {
                'name': room['name'],
                'line_items': []
            }
            
            for task in room.get('tasks', []):
                historical_price = self._lookup_historical_price(
                    task['description'],
                    task.get('unit', 'EA')
                )
                
                room_pricing['line_items'].append({
                    'description': task['description'],
                    'quantity': task.get('quantity_with_waste', task.get('quantity', 0)),
                    'unit_price': historical_price,
                    'total_price': historical_price * task.get('quantity_with_waste', 0),
                    'price_source': 'historical_average'
                })
            
            result['rooms'].append(room_pricing)
        
        return result
```

### 3.3 Graceful Degradation

```python
class Phase2GracefulDegradation:
    """Implements graceful degradation for Phase 2"""
    
    def process_with_degradation(self, phase1_output: Dict) -> Dict:
        """Process with progressive degradation based on data quality"""
        
        quality_score = self._assess_input_quality(phase1_output)
        
        if quality_score > 0.8:
            # High quality - full processing
            return self._full_processing(phase1_output)
        
        elif quality_score > 0.6:
            # Medium quality - simplified processing
            return self._simplified_processing(phase1_output)
        
        elif quality_score > 0.4:
            # Low quality - basic processing with warnings
            return self._basic_processing(phase1_output)
        
        else:
            # Very low quality - minimal processing with defaults
            return self._minimal_processing(phase1_output)
    
    def _assess_input_quality(self, data: Dict) -> float:
        """Assess quality of Phase 1 output (0-1 scale)"""
        score = 1.0
        
        # Check for missing required fields
        if 'data' not in data:
            score -= 0.3
        if 'rooms' not in data.get('data', {}):
            score -= 0.3
        
        # Check for data completeness
        rooms = data.get('data', {}).get('rooms', [])
        if not rooms:
            score -= 0.2
        
        # Check task completeness
        total_tasks = sum(len(r.get('tasks', [])) for r in rooms)
        tasks_with_quantity = sum(
            1 for r in rooms 
            for t in r.get('tasks', []) 
            if 'quantity' in t
        )
        
        if total_tasks > 0:
            completeness_ratio = tasks_with_quantity / total_tasks
            score *= completeness_ratio
        
        return max(0, min(1, score))
```

## 4. Test Execution Framework

### 4.1 Test Runner Configuration

```python
# tests/phases/phase2_test_runner.py

import asyncio
from pathlib import Path
import json
from datetime import datetime

class Phase2TestRunner:
    """Orchestrates Phase 2 testing with various datasets"""
    
    def __init__(self):
        self.test_results = []
        self.test_datasets = {
            'perfect': perfect_phase1_output,
            'realistic': realistic_phase1_output,
            'edge_case': edge_case_phase1_output
        }
    
    async def run_all_tests(self):
        """Execute all Phase 2 tests"""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {},
            'detailed_results': []
        }
        
        # 1. Run unit tests
        unit_results = await self._run_unit_tests()
        results['detailed_results'].append(unit_results)
        
        # 2. Run integration tests with each dataset
        for dataset_name, dataset in self.test_datasets.items():
            integration_result = await self._run_integration_test(
                dataset_name, 
                dataset
            )
            results['detailed_results'].append(integration_result)
        
        # 3. Run contract tests
        contract_results = await self._run_contract_tests()
        results['detailed_results'].append(contract_results)
        
        # 4. Run error recovery tests
        recovery_results = await self._run_recovery_tests()
        results['detailed_results'].append(recovery_results)
        
        # 5. Generate summary
        results['test_summary'] = self._generate_summary(results['detailed_results'])
        
        # 6. Save results
        self._save_results(results)
        
        return results
    
    async def _run_integration_test(self, name: str, dataset: Dict) -> Dict:
        """Run integration test with specific dataset"""
        
        processor = Phase2Processor()
        validator = Phase2InputValidator()
        fallback = Phase2FallbackStrategy()
        
        test_result = {
            'test_name': f'integration_{name}',
            'dataset': name,
            'phases': []
        }
        
        try:
            # Validate and sanitize input
            sanitized = validator.validate_and_sanitize(dataset)
            test_result['phases'].append({
                'phase': 'validation',
                'success': True,
                'corrections': sanitized['corrections_applied']
            })
            
            # Process with Phase 2
            result = await processor.process(sanitized['data'])
            test_result['phases'].append({
                'phase': 'processing',
                'success': result['success'],
                'confidence': result.get('confidence_score', 0)
            })
            
        except Exception as e:
            # Apply fallback
            fallback_result = await fallback.apply_fallback(
                dataset, 
                str(type(e).__name__)
            )
            test_result['phases'].append({
                'phase': 'fallback',
                'success': True,
                'method': fallback_result.get('pricing_method', 'unknown')
            })
        
        return test_result
```

### 4.2 Continuous Validation

```python
class Phase2ContinuousValidator:
    """Continuous validation during Phase 2 execution"""
    
    def __init__(self):
        self.validation_checkpoints = []
        self.thresholds = {
            'min_confidence': 0.6,
            'max_price_variance': 0.3,
            'min_consensus': 0.67
        }
    
    def validate_checkpoint(self, stage: str, data: Any) -> bool:
        """Validate at specific checkpoint"""
        
        checkpoint = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'valid': True,
            'issues': []
        }
        
        if stage == 'post_model_responses':
            valid = self._validate_model_consensus(data)
            checkpoint['valid'] = valid
            
        elif stage == 'post_pricing_calculation':
            valid = self._validate_pricing_sanity(data)
            checkpoint['valid'] = valid
            
        elif stage == 'pre_output':
            valid = self._validate_output_completeness(data)
            checkpoint['valid'] = valid
        
        self.validation_checkpoints.append(checkpoint)
        return checkpoint['valid']
    
    def get_validation_report(self) -> Dict:
        """Generate validation report"""
        return {
            'checkpoints': self.validation_checkpoints,
            'overall_valid': all(c['valid'] for c in self.validation_checkpoints),
            'issues_count': sum(len(c['issues']) for c in self.validation_checkpoints)
        }
```

## 5. Performance Monitoring

### 5.1 Metrics Collection

```python
class Phase2Metrics:
    """Collect and analyze Phase 2 performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'confidence_scores': [],
            'error_rates': {},
            'fallback_usage': 0,
            'correction_counts': []
        }
    
    def record_execution(self, result: Dict):
        """Record metrics from Phase 2 execution"""
        
        self.metrics['processing_times'].append(
            result.get('processing_time', 0)
        )
        
        self.metrics['confidence_scores'].append(
            result.get('confidence_score', 0)
        )
        
        if not result.get('success'):
            error_type = result.get('error_type', 'unknown')
            self.metrics['error_rates'][error_type] = \
                self.metrics['error_rates'].get(error_type, 0) + 1
        
        if result.get('used_fallback'):
            self.metrics['fallback_usage'] += 1
    
    def get_performance_summary(self) -> Dict:
        """Generate performance summary"""
        import numpy as np
        
        return {
            'avg_processing_time': np.mean(self.metrics['processing_times']),
            'median_confidence': np.median(self.metrics['confidence_scores']),
            'error_rate': sum(self.metrics['error_rates'].values()) / len(self.metrics['processing_times']),
            'fallback_rate': self.metrics['fallback_usage'] / len(self.metrics['processing_times'])
        }
```

## 6. Test Data Management

### 6.1 Test Data Generator

```python
class Phase2TestDataGenerator:
    """Generate diverse test datasets for Phase 2"""
    
    def generate_dataset(self, scenario: str) -> Dict:
        """Generate dataset for specific scenario"""
        
        generators = {
            'small_residential': self._generate_small_residential,
            'large_commercial': self._generate_large_commercial,
            'disaster_recovery': self._generate_disaster_recovery,
            'partial_renovation': self._generate_partial_renovation,
            'high_end_custom': self._generate_high_end_custom
        }
        
        if scenario in generators:
            return generators[scenario]()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _generate_small_residential(self) -> Dict:
        """Generate small residential project data"""
        return {
            'phase': 1,
            'success': True,
            'confidence_score': 0.85,
            'data': {
                'rooms': [
                    self._generate_room('Living Room', 200, ['drywall', 'paint', 'flooring']),
                    self._generate_room('Bedroom', 150, ['paint', 'carpet']),
                    self._generate_room('Kitchen', 120, ['cabinets', 'countertops', 'flooring'])
                ],
                'waste_summary': self._calculate_waste_summary()
            }
        }
    
    def _generate_room(self, name: str, sqft: int, materials: List[str]) -> Dict:
        """Generate room data with specified materials"""
        tasks = []
        
        for material in materials:
            tasks.extend(self._generate_tasks_for_material(material, sqft))
        
        return {
            'name': name,
            'measurements': {'sqft': sqft, 'height': 9},
            'tasks': tasks
        }
```

## Implementation Checklist

- [ ] Create golden dataset files in `tests/fixtures/phase2/`
- [ ] Implement unit tests for Phase 2 components
- [ ] Implement integration tests with mocked models
- [ ] Create contract test schemas
- [ ] Implement input validation and sanitization
- [ ] Create fallback strategies for different failure modes
- [ ] Implement graceful degradation logic
- [ ] Set up continuous validation checkpoints
- [ ] Create performance monitoring
- [ ] Generate comprehensive test datasets
- [ ] Create test execution framework
- [ ] Document test procedures
- [ ] Set up CI/CD integration
- [ ] Create test result reporting

## Success Criteria

1. **Coverage**: >90% code coverage for Phase 2 components
2. **Reliability**: <5% failure rate with realistic datasets
3. **Recovery**: 100% graceful handling of edge cases
4. **Performance**: <3 seconds processing time for standard datasets
5. **Accuracy**: >85% pricing accuracy vs. manual estimates
6. **Integration**: Zero breaking changes to Phase 1→2→3 pipeline

## Next Steps

1. Review and approve testing strategy
2. Prioritize implementation tasks
3. Create test data from real projects
4. Implement core validation logic
5. Set up automated test execution
6. Monitor and iterate based on results