# src/testing/benchmark_generator.py
"""
Benchmark Dataset Generator for Construction Estimation Testing
Creates diverse, realistic test scenarios with ground truth for validation
"""
import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools

from src.utils.logger import get_logger


@dataclass
class RoomSpecification:
    """Specification for generating room test data"""
    room_type: str
    size_range: Tuple[int, int]  # (min_sqft, max_sqft)
    height_range: Tuple[int, int]  # (min_feet, max_feet)
    typical_materials: List[str]
    work_scope_options: List[str]
    complexity_factors: List[str]
    damage_scenarios: List[str]


@dataclass
class GroundTruthEstimate:
    """Ground truth estimate for validation"""
    room_name: str
    expected_tasks: List[Dict[str, Any]]
    expected_cost_range: Tuple[float, float]
    expected_timeline_days: int
    critical_requirements: List[str]
    quality_benchmarks: Dict[str, float]


@dataclass
class BenchmarkTestCase:
    """Complete benchmark test case"""
    test_id: str
    category: str
    difficulty_level: str  # easy, medium, hard, expert
    description: str
    input_data: Dict[str, Any]
    ground_truth: GroundTruthEstimate
    validation_criteria: Dict[str, Any]
    tags: List[str]
    created_timestamp: str


class BenchmarkDatasetGenerator:
    """
    Generates comprehensive benchmark datasets for construction estimation testing
    """
    
    def __init__(self):
        self.logger = get_logger('benchmark_generator')
        
        # Room specifications database
        self.room_specs = {
            'bedroom': RoomSpecification(
                room_type='bedroom',
                size_range=(100, 400),
                height_range=(8, 12),
                typical_materials=['drywall', 'carpet', 'paint', 'baseboards', 'door', 'window'],
                work_scope_options=['remove_replace', 'paint_only', 'patch_repair', 'full_renovation'],
                complexity_factors=['water_damage', 'high_ceiling', 'custom_features', 'multiple_windows'],
                damage_scenarios=['water_damage', 'fire_damage', 'mold', 'structural_issues']
            ),
            'kitchen': RoomSpecification(
                room_type='kitchen',
                size_range=(150, 600),
                height_range=(8, 14),
                typical_materials=['cabinets', 'countertop', 'flooring', 'appliances', 'paint', 'plumbing', 'electrical'],
                work_scope_options=['full_renovation', 'cabinet_replacement', 'countertop_only', 'cosmetic_update'],
                complexity_factors=['island', 'high_end_finishes', 'electrical_upgrade', 'plumbing_relocation'],
                damage_scenarios=['water_damage', 'fire_damage', 'outdated_systems', 'structural_modifications']
            ),
            'bathroom': RoomSpecification(
                room_type='bathroom',
                size_range=(40, 200),
                height_range=(8, 10),
                typical_materials=['vanity', 'toilet', 'tub_shower', 'flooring', 'paint', 'mirror', 'plumbing'],
                work_scope_options=['full_renovation', 'vanity_replacement', 'tub_replacement', 'cosmetic_update'],
                complexity_factors=['accessibility_features', 'luxury_finishes', 'steam_shower', 'heated_floors'],
                damage_scenarios=['water_damage', 'mold', 'outdated_plumbing', 'accessibility_needs']
            ),
            'living_room': RoomSpecification(
                room_type='living_room',
                size_range=(200, 800),
                height_range=(8, 20),
                typical_materials=['flooring', 'paint', 'baseboards', 'windows', 'doors', 'lighting'],
                work_scope_options=['flooring_replacement', 'paint_only', 'full_renovation', 'entertainment_center'],
                complexity_factors=['vaulted_ceiling', 'fireplace', 'built_ins', 'large_windows'],
                damage_scenarios=['water_damage', 'fire_damage', 'outdated_systems', 'layout_changes']
            )
        }
        
        # DMV area cost database (simplified)
        self.dmv_cost_ranges = {
            'flooring_sqft': (8, 25),
            'paint_sqft': (2, 6),
            'drywall_sqft': (3, 8),
            'cabinet_lf': (200, 800),
            'countertop_sqft': (50, 150),
            'bathroom_renovation': (15000, 45000),
            'kitchen_renovation': (25000, 85000),
            'labor_hour': (45, 120)
        }
        
        # Complexity multipliers
        self.complexity_multipliers = {
            'easy': 1.0,
            'medium': 1.3,
            'hard': 1.8,
            'expert': 2.5
        }
    
    def generate_comprehensive_dataset(self, 
                                     num_cases_per_category: int = 50,
                                     output_path: Optional[str] = None) -> List[BenchmarkTestCase]:
        """
        Generate a comprehensive benchmark dataset covering various scenarios
        
        Args:
            num_cases_per_category: Number of test cases per category
            output_path: Optional path to save the dataset
        
        Returns:
            List of benchmark test cases
        """
        self.logger.info(f"Generating comprehensive benchmark dataset with {num_cases_per_category} cases per category")
        
        dataset = []
        categories = [
            'single_room_basic',
            'single_room_complex', 
            'multi_room_residential',
            'water_damage_scenarios',
            'high_end_renovations',
            'budget_constraints',
            'accessibility_requirements',
            'historic_property',
            'edge_cases'
        ]
        
        for category in categories:
            self.logger.info(f"Generating {category} test cases")
            category_cases = self._generate_category_cases(category, num_cases_per_category)
            dataset.extend(category_cases)
        
        # Add validation cases with known ground truth
        validation_cases = self._generate_validation_cases()
        dataset.extend(validation_cases)
        
        # Save dataset if path provided
        if output_path:
            self._save_dataset(dataset, output_path)
        
        self.logger.info(f"Generated {len(dataset)} total test cases")
        return dataset
    
    def _generate_category_cases(self, category: str, num_cases: int) -> List[BenchmarkTestCase]:
        """Generate test cases for a specific category"""
        
        cases = []
        
        for i in range(num_cases):
            if category == 'single_room_basic':
                case = self._generate_single_room_basic(i)
            elif category == 'single_room_complex':
                case = self._generate_single_room_complex(i)
            elif category == 'multi_room_residential':
                case = self._generate_multi_room_residential(i)
            elif category == 'water_damage_scenarios':
                case = self._generate_water_damage_scenario(i)
            elif category == 'high_end_renovations':
                case = self._generate_high_end_renovation(i)
            elif category == 'budget_constraints':
                case = self._generate_budget_constraint_case(i)
            elif category == 'accessibility_requirements':
                case = self._generate_accessibility_case(i)
            elif category == 'historic_property':
                case = self._generate_historic_property_case(i)
            elif category == 'edge_cases':
                case = self._generate_edge_case(i)
            else:
                continue
                
            cases.append(case)
        
        return cases
    
    def _generate_single_room_basic(self, case_id: int) -> BenchmarkTestCase:
        """Generate basic single room test case"""
        
        room_type = random.choice(list(self.room_specs.keys()))
        spec = self.room_specs[room_type]
        
        # Generate room dimensions
        area = random.randint(spec.size_range[0], spec.size_range[1])
        width = random.randint(8, 20)
        length = area // width
        height = random.randint(spec.height_range[0], spec.height_range[1])
        
        # Select materials and work scope
        selected_materials = random.sample(spec.typical_materials, k=min(4, len(spec.typical_materials)))
        work_scope = random.choice(spec.work_scope_options)
        
        # Generate input data
        input_data = {
            'prompt': f"Generate detailed work scope for {room_type} renovation",
            'data': [{
                'name': f"{room_type.title()} 1",
                'materials': {material: 'Existing' for material in selected_materials},
                'work_scope': {material: 'Remove & Replace' for material in selected_materials},
                'measurements': {
                    'width': width,
                    'length': length,
                    'height': height,
                    'area_sqft': area
                },
                'demo_scope(already demo\'d)': {},
                'additional_notes': f"Standard {room_type} renovation in DMV area"
            }],
            'expected_task_count': len(selected_materials) * 3  # Remove, install, finish for each material
        }
        
        # Generate ground truth
        expected_tasks = self._generate_expected_tasks(room_type, selected_materials, area, 'easy')
        cost_range = self._calculate_cost_range(room_type, area, 'easy')
        
        ground_truth = GroundTruthEstimate(
            room_name=f"{room_type.title()} 1",
            expected_tasks=expected_tasks,
            expected_cost_range=cost_range,
            expected_timeline_days=random.randint(3, 14),
            critical_requirements=['remove_replace_logic', 'proper_measurements', 'demo_scope_handling'],
            quality_benchmarks={'accuracy': 0.85, 'completeness': 0.90, 'consensus': 0.75}
        )
        
        return BenchmarkTestCase(
            test_id=f"single_basic_{room_type}_{case_id:03d}",
            category='single_room_basic',
            difficulty_level='easy',
            description=f"Basic {room_type} renovation with standard materials and Remove & Replace scope",
            input_data=input_data,
            ground_truth=ground_truth,
            validation_criteria={
                'min_tasks': len(selected_materials) * 2,
                'max_cost_variance': 0.2,
                'required_task_types': ['removal', 'installation'],
                'consensus_threshold': 0.66
            },
            tags=['single_room', 'basic', room_type, 'remove_replace'],
            created_timestamp=datetime.now().isoformat()
        )
    
    def _generate_single_room_complex(self, case_id: int) -> BenchmarkTestCase:
        """Generate complex single room test case"""
        
        room_type = random.choice(list(self.room_specs.keys()))
        spec = self.room_specs[room_type]
        
        # Generate larger, more complex room
        area = random.randint(spec.size_range[1]//2, spec.size_range[1])
        width = random.randint(12, 30)
        length = area // width
        height = random.randint(spec.height_range[1]-2, spec.height_range[1])
        
        # Add complexity factors
        complexity_factors = random.sample(spec.complexity_factors, k=random.randint(1, 3))
        damage_scenario = random.choice(spec.damage_scenarios)
        
        # More materials and mixed work scopes
        selected_materials = spec.typical_materials
        work_scopes = ['Remove & Replace', 'Patch & Paint', 'Clean & Seal', 'Repair']
        
        input_data = {
            'prompt': f"Generate detailed work scope for complex {room_type} with {damage_scenario}",
            'data': [{
                'name': f"{room_type.title()} 1",
                'materials': {material: 'Existing' for material in selected_materials},
                'work_scope': {material: random.choice(work_scopes) for material in selected_materials},
                'measurements': {
                    'width': width,
                    'length': length,
                    'height': height,
                    'area_sqft': area,
                    'perimeter_lf': 2 * (width + length)
                },
                'demo_scope(already demo\'d)': {
                    random.choice(selected_materials): random.randint(10, area//4)
                },
                'additional_notes': f"Complex {room_type} with {damage_scenario}. Features: {', '.join(complexity_factors)}"
            }],
            'expected_task_count': len(selected_materials) * 4  # More tasks due to complexity
        }
        
        # Generate ground truth for complex case
        expected_tasks = self._generate_expected_tasks(room_type, selected_materials, area, 'hard')
        cost_range = self._calculate_cost_range(room_type, area, 'hard')
        
        ground_truth = GroundTruthEstimate(
            room_name=f"{room_type.title()} 1",
            expected_tasks=expected_tasks,
            expected_cost_range=cost_range,
            expected_timeline_days=random.randint(7, 21),
            critical_requirements=[
                'remove_replace_logic', 'demo_scope_handling', 'complexity_factor_inclusion',
                'damage_scenario_tasks', 'safety_requirements'
            ],
            quality_benchmarks={'accuracy': 0.75, 'completeness': 0.85, 'consensus': 0.70}
        )
        
        return BenchmarkTestCase(
            test_id=f"single_complex_{room_type}_{case_id:03d}",
            category='single_room_complex',
            difficulty_level='hard',
            description=f"Complex {room_type} renovation with {damage_scenario} and multiple complexity factors",
            input_data=input_data,
            ground_truth=ground_truth,
            validation_criteria={
                'min_tasks': len(selected_materials) * 3,
                'max_cost_variance': 0.3,
                'required_task_types': ['removal', 'installation', 'preparation', 'safety'],
                'consensus_threshold': 0.5
            },
            tags=['single_room', 'complex', room_type, damage_scenario] + complexity_factors,
            created_timestamp=datetime.now().isoformat()
        )
    
    def _generate_multi_room_residential(self, case_id: int) -> BenchmarkTestCase:
        """Generate multi-room residential project"""
        
        # Select 3-5 rooms
        room_types = random.sample(list(self.room_specs.keys()), k=random.randint(3, 5))
        total_area = 0
        rooms_data = []
        
        for i, room_type in enumerate(room_types):
            spec = self.room_specs[room_type]
            
            # Generate room
            area = random.randint(spec.size_range[0], spec.size_range[1])
            total_area += area
            
            width = random.randint(8, 25)
            length = area // width
            height = random.randint(spec.height_range[0], spec.height_range[1])
            
            selected_materials = random.sample(spec.typical_materials, k=random.randint(3, len(spec.typical_materials)))
            
            room_data = {
                'name': f"{room_type.title()} {i+1}",
                'materials': {material: 'Existing' for material in selected_materials},
                'work_scope': {material: 'Remove & Replace' for material in selected_materials},
                'measurements': {
                    'width': width,
                    'length': length,
                    'height': height,
                    'area_sqft': area
                },
                'demo_scope(already demo\'d)': {},
                'additional_notes': f"Part of multi-room residential renovation"
            }
            rooms_data.append(room_data)
        
        input_data = {
            'prompt': "Generate comprehensive work scope for multi-room residential renovation",
            'data': [{'location': 'Main Floor', 'rooms': rooms_data}],
            'expected_task_count': len(room_types) * 12  # Average tasks per room
        }
        
        # Calculate combined ground truth
        all_expected_tasks = []
        total_cost_min = 0
        total_cost_max = 0
        
        for i, room_type in enumerate(room_types):
            room_area = rooms_data[i]['measurements']['area_sqft']
            tasks = self._generate_expected_tasks(room_type, list(rooms_data[i]['materials'].keys()), room_area, 'medium')
            all_expected_tasks.extend(tasks)
            
            cost_range = self._calculate_cost_range(room_type, room_area, 'medium')
            total_cost_min += cost_range[0]
            total_cost_max += cost_range[1]
        
        ground_truth = GroundTruthEstimate(
            room_name="Multi-Room Project",
            expected_tasks=all_expected_tasks,
            expected_cost_range=(total_cost_min, total_cost_max),
            expected_timeline_days=random.randint(14, 45),
            critical_requirements=[
                'room_coordination', 'material_consistency', 'work_sequence',
                'multi_room_logistics', 'project_management'
            ],
            quality_benchmarks={'accuracy': 0.80, 'completeness': 0.85, 'consensus': 0.70}
        )
        
        return BenchmarkTestCase(
            test_id=f"multi_room_{len(room_types)}rooms_{case_id:03d}",
            category='multi_room_residential',
            difficulty_level='medium',
            description=f"Multi-room residential renovation with {len(room_types)} rooms",
            input_data=input_data,
            ground_truth=ground_truth,
            validation_criteria={
                'min_tasks': len(room_types) * 8,
                'max_cost_variance': 0.25,
                'required_coordination': True,
                'consensus_threshold': 0.6
            },
            tags=['multi_room', 'residential'] + room_types,
            created_timestamp=datetime.now().isoformat()
        )
    
    def _generate_water_damage_scenario(self, case_id: int) -> BenchmarkTestCase:
        """Generate water damage specific test case"""
        
        room_type = random.choice(['bathroom', 'kitchen', 'bedroom'])
        spec = self.room_specs[room_type]
        
        # Water damage specifics
        damage_levels = ['minor', 'moderate', 'severe']
        damage_level = random.choice(damage_levels)
        
        area = random.randint(spec.size_range[0], spec.size_range[1])
        width = random.randint(10, 20)
        length = area // width
        height = random.randint(spec.height_range[0], spec.height_range[1])
        
        # Water damage affects specific materials
        water_affected_materials = ['drywall', 'flooring', 'baseboards', 'insulation']
        selected_materials = spec.typical_materials + water_affected_materials
        
        input_data = {
            'prompt': f"Generate work scope for {damage_level} water damage restoration in {room_type}",
            'data': [{
                'name': f"Water Damaged {room_type.title()}",
                'materials': {material: 'Water Damaged' for material in selected_materials},
                'work_scope': {material: 'Remove & Replace' for material in selected_materials},
                'measurements': {
                    'width': width,
                    'length': length,
                    'height': height,
                    'area_sqft': area,
                    'affected_area_sqft': area * random.uniform(0.3, 1.0)
                },
                'demo_scope(already demo\'d)': {},
                'additional_notes': f"{damage_level.title()} water damage. Requires mold remediation, moisture testing, and specialized drying."
            }],
            'expected_task_count': len(selected_materials) * 4  # More tasks for water damage
        }
        
        # Water damage specific tasks
        expected_tasks = self._generate_water_damage_tasks(selected_materials, area, damage_level)
        cost_multiplier = {'minor': 1.2, 'moderate': 1.6, 'severe': 2.2}[damage_level]
        base_cost = self._calculate_cost_range(room_type, area, 'medium')
        cost_range = (base_cost[0] * cost_multiplier, base_cost[1] * cost_multiplier)
        
        ground_truth = GroundTruthEstimate(
            room_name=f"Water Damaged {room_type.title()}",
            expected_tasks=expected_tasks,
            expected_cost_range=cost_range,
            expected_timeline_days=random.randint(7, 21),
            critical_requirements=[
                'water_damage_protocol', 'mold_remediation', 'moisture_testing',
                'specialized_equipment', 'safety_protocols', 'insurance_documentation'
            ],
            quality_benchmarks={'accuracy': 0.75, 'completeness': 0.90, 'consensus': 0.65}
        )
        
        return BenchmarkTestCase(
            test_id=f"water_damage_{damage_level}_{room_type}_{case_id:03d}",
            category='water_damage_scenarios',
            difficulty_level='hard',
            description=f"{damage_level.title()} water damage restoration in {room_type}",
            input_data=input_data,
            ground_truth=ground_truth,
            validation_criteria={
                'min_tasks': len(selected_materials) * 3,
                'max_cost_variance': 0.4,
                'required_task_types': ['removal', 'remediation', 'drying', 'testing', 'installation'],
                'consensus_threshold': 0.5
            },
            tags=['water_damage', damage_level, room_type, 'remediation', 'specialty'],
            created_timestamp=datetime.now().isoformat()
        )
    
    def _generate_validation_cases(self, num_cases: int = 20) -> List[BenchmarkTestCase]:
        """Generate validation cases with precisely known ground truth"""
        
        validation_cases = []
        
        # Create carefully controlled test cases
        for i in range(num_cases):
            # Simple bedroom with known specifications
            input_data = {
                'prompt': "Generate work scope for standard bedroom renovation",
                'data': [{
                    'name': f'Validation Bedroom {i+1}',
                    'materials': {
                        'Paint - Walls': 'Existing',
                        'Paint - Ceiling': 'Existing',
                        'Carpet': 'Existing',
                        'Baseboards': 'Existing'
                    },
                    'work_scope': {
                        'Paint - Walls': 'Remove & Replace',
                        'Paint - Ceiling': 'Remove & Replace',
                        'Carpet': 'Remove & Replace',
                        'Baseboards': 'Remove & Replace'
                    },
                    'measurements': {
                        'width': 12,
                        'length': 14,
                        'height': 9,
                        'area_sqft': 168,
                        'wall_area_sqft': 468,
                        'ceiling_area_sqft': 168,
                        'perimeter_lf': 52
                    },
                    'demo_scope(already demo\'d)': {
                        'Carpet': 50  # Partial demo completed
                    },
                    'additional_notes': 'Standard renovation, no special requirements'
                }],
                'expected_task_count': 12
            }
            
            # Precisely defined expected tasks
            expected_tasks = [
                {'task_name': 'Remove existing wall paint', 'quantity': 468, 'unit': 'sqft'},
                {'task_name': 'Remove existing ceiling paint', 'quantity': 168, 'unit': 'sqft'},
                {'task_name': 'Remove remaining carpet', 'quantity': 118, 'unit': 'sqft'},  # 168 - 50 demo'd
                {'task_name': 'Remove existing baseboards', 'quantity': 52, 'unit': 'lf'},
                {'task_name': 'Install new wall paint', 'quantity': 468, 'unit': 'sqft'},
                {'task_name': 'Install new ceiling paint', 'quantity': 168, 'unit': 'sqft'},
                {'task_name': 'Install new carpet', 'quantity': 168, 'unit': 'sqft'},  # Full area
                {'task_name': 'Install new baseboards', 'quantity': 52, 'unit': 'lf'},
                {'task_name': 'Prepare wall surfaces', 'quantity': 468, 'unit': 'sqft'},
                {'task_name': 'Prepare ceiling surfaces', 'quantity': 168, 'unit': 'sqft'},
                {'task_name': 'Prepare floor surfaces', 'quantity': 168, 'unit': 'sqft'},
                {'task_name': 'Final cleanup', 'quantity': 1, 'unit': 'item'}
            ]
            
            ground_truth = GroundTruthEstimate(
                room_name=f'Validation Bedroom {i+1}',
                expected_tasks=expected_tasks,
                expected_cost_range=(8500, 12500),
                expected_timeline_days=7,
                critical_requirements=[
                    'remove_replace_logic', 'demo_scope_calculation', 'proper_quantities'
                ],
                quality_benchmarks={'accuracy': 0.95, 'completeness': 0.95, 'consensus': 0.90}
            )
            
            validation_case = BenchmarkTestCase(
                test_id=f"validation_bedroom_{i+1:03d}",
                category='validation',
                difficulty_level='easy',
                description='Controlled validation case with known ground truth',
                input_data=input_data,
                ground_truth=ground_truth,
                validation_criteria={
                    'exact_task_count': 12,
                    'max_cost_variance': 0.1,
                    'required_demo_logic': True,
                    'consensus_threshold': 0.8
                },
                tags=['validation', 'controlled', 'ground_truth', 'bedroom'],
                created_timestamp=datetime.now().isoformat()
            )
            
            validation_cases.append(validation_case)
        
        return validation_cases
    
    def _generate_expected_tasks(self, room_type: str, materials: List[str], area: float, difficulty: str) -> List[Dict[str, Any]]:
        """Generate expected tasks for ground truth"""
        
        tasks = []
        
        for material in materials:
            # Basic removal and installation tasks
            tasks.append({
                'task_name': f'Remove existing {material.lower()}',
                'quantity': area if 'sqft' in material.lower() else random.randint(1, 10),
                'unit': 'sqft' if 'sqft' in material.lower() else 'item',
                'task_type': 'removal'
            })
            
            tasks.append({
                'task_name': f'Install new {material.lower()}',
                'quantity': area if 'sqft' in material.lower() else random.randint(1, 10),
                'unit': 'sqft' if 'sqft' in material.lower() else 'item',
                'task_type': 'installation'
            })
            
            # Add preparation tasks for medium/hard difficulty
            if difficulty in ['medium', 'hard', 'expert']:
                tasks.append({
                    'task_name': f'Prepare {material.lower()} surfaces',
                    'quantity': area if 'sqft' in material.lower() else 1,
                    'unit': 'sqft' if 'sqft' in material.lower() else 'item',
                    'task_type': 'preparation'
                })
        
        # Add room-specific tasks
        if room_type == 'kitchen':
            tasks.extend([
                {'task_name': 'Electrical rough-in', 'quantity': 1, 'unit': 'item', 'task_type': 'electrical'},
                {'task_name': 'Plumbing rough-in', 'quantity': 1, 'unit': 'item', 'task_type': 'plumbing'}
            ])
        
        if room_type == 'bathroom':
            tasks.extend([
                {'task_name': 'Waterproofing', 'quantity': area, 'unit': 'sqft', 'task_type': 'preparation'},
                {'task_name': 'Plumbing connections', 'quantity': 3, 'unit': 'item', 'task_type': 'plumbing'}
            ])
        
        return tasks
    
    def _generate_water_damage_tasks(self, materials: List[str], area: float, damage_level: str) -> List[Dict[str, Any]]:
        """Generate water damage specific tasks"""
        
        tasks = []
        
        # Standard water damage protocol tasks
        tasks.extend([
            {'task_name': 'Moisture assessment and testing', 'quantity': 1, 'unit': 'item', 'task_type': 'assessment'},
            {'task_name': 'Water extraction', 'quantity': area, 'unit': 'sqft', 'task_type': 'extraction'},
            {'task_name': 'Dehumidification setup', 'quantity': 1, 'unit': 'item', 'task_type': 'drying'},
            {'task_name': 'Antimicrobial treatment', 'quantity': area, 'unit': 'sqft', 'task_type': 'treatment'}
        ])
        
        # Material-specific removal and replacement
        for material in materials:
            tasks.extend([
                {'task_name': f'Remove water damaged {material.lower()}', 'quantity': area, 'unit': 'sqft', 'task_type': 'removal'},
                {'task_name': f'Install replacement {material.lower()}', 'quantity': area, 'unit': 'sqft', 'task_type': 'installation'}
            ])
        
        # Damage level specific tasks
        if damage_level in ['moderate', 'severe']:
            tasks.extend([
                {'task_name': 'Mold remediation', 'quantity': area * 0.5, 'unit': 'sqft', 'task_type': 'remediation'},
                {'task_name': 'Air quality testing', 'quantity': 1, 'unit': 'item', 'task_type': 'testing'}
            ])
        
        if damage_level == 'severe':
            tasks.extend([
                {'task_name': 'Structural assessment', 'quantity': 1, 'unit': 'item', 'task_type': 'assessment'},
                {'task_name': 'Insurance documentation', 'quantity': 1, 'unit': 'item', 'task_type': 'documentation'}
            ])
        
        return tasks
    
    def _calculate_cost_range(self, room_type: str, area: float, difficulty: str) -> Tuple[float, float]:
        """Calculate expected cost range based on room type and complexity"""
        
        # Base costs per square foot by room type (DMV area)
        base_costs = {
            'bedroom': (25, 45),
            'kitchen': (150, 300),
            'bathroom': (200, 400),
            'living_room': (30, 60)
        }
        
        base_min, base_max = base_costs.get(room_type, (30, 60))
        multiplier = self.complexity_multipliers[difficulty]
        
        cost_min = base_min * area * multiplier
        cost_max = base_max * area * multiplier
        
        return (cost_min, cost_max)
    
    # Additional generation methods for other categories...
    def _generate_high_end_renovation(self, case_id: int) -> BenchmarkTestCase:
        """Generate high-end renovation case with premium materials"""
        # Implementation similar to above but with luxury specifications
        pass
    
    def _generate_budget_constraint_case(self, case_id: int) -> BenchmarkTestCase:
        """Generate case with specific budget constraints"""
        # Implementation for budget-focused scenarios
        pass
    
    def _generate_accessibility_case(self, case_id: int) -> BenchmarkTestCase:
        """Generate ADA compliance case"""
        # Implementation for accessibility requirements
        pass
    
    def _generate_historic_property_case(self, case_id: int) -> BenchmarkTestCase:
        """Generate historic property restoration case"""
        # Implementation for historic property constraints
        pass
    
    def _generate_edge_case(self, case_id: int) -> BenchmarkTestCase:
        """Generate edge case for stress testing"""
        edge_case_types = [
            'missing_data', 'conflicting_data', 'extreme_dimensions',
            'unusual_materials', 'complex_geometry', 'regulatory_constraints'
        ]
        
        edge_type = random.choice(edge_case_types)
        
        # Generate edge case based on type
        if edge_type == 'missing_data':
            return self._generate_missing_data_case(case_id)
        elif edge_type == 'extreme_dimensions':
            return self._generate_extreme_dimensions_case(case_id)
        # ... implement other edge cases
        
        # Fallback to basic case
        return self._generate_single_room_basic(case_id)
    
    def _generate_missing_data_case(self, case_id: int) -> BenchmarkTestCase:
        """Generate case with missing critical data"""
        
        input_data = {
            'prompt': "Generate work scope with incomplete data",
            'data': [{
                'name': 'Incomplete Room',
                'materials': {'Paint': 'Existing'},  # Minimal materials
                'work_scope': {'Paint': 'Remove & Replace'},
                'measurements': {
                    'width': 12,
                    # Missing length and height
                },
                'demo_scope(already demo\'d)': {},
                'additional_notes': 'Limited information available'
            }],
            'expected_task_count': 3
        }
        
        expected_tasks = [
            {'task_name': 'Request additional measurements', 'quantity': 1, 'unit': 'item'},
            {'task_name': 'Estimate based on typical dimensions', 'quantity': 1, 'unit': 'item'},
            {'task_name': 'Paint walls', 'quantity': 300, 'unit': 'sqft'}  # Estimated
        ]
        
        ground_truth = GroundTruthEstimate(
            room_name='Incomplete Room',
            expected_tasks=expected_tasks,
            expected_cost_range=(1000, 5000),  # Wide range due to uncertainty
            expected_timeline_days=3,
            critical_requirements=['data_validation', 'assumption_documentation', 'uncertainty_handling'],
            quality_benchmarks={'accuracy': 0.60, 'completeness': 0.70, 'consensus': 0.50}
        )
        
        return BenchmarkTestCase(
            test_id=f"edge_missing_data_{case_id:03d}",
            category='edge_cases',
            difficulty_level='expert',
            description='Test case with missing critical measurement data',
            input_data=input_data,
            ground_truth=ground_truth,
            validation_criteria={
                'min_tasks': 2,
                'max_cost_variance': 0.6,
                'requires_assumptions': True,
                'consensus_threshold': 0.4
            },
            tags=['edge_case', 'missing_data', 'uncertainty', 'stress_test'],
            created_timestamp=datetime.now().isoformat()
        )
    
    def _generate_extreme_dimensions_case(self, case_id: int) -> BenchmarkTestCase:
        """Generate case with extreme room dimensions"""
        
        # Very large or very small dimensions
        if random.choice([True, False]):
            # Extremely large room
            width, length, height = 50, 80, 20
            area = width * length
            description = "Extremely large room renovation"
        else:
            # Extremely small room
            width, length, height = 4, 6, 7
            area = width * length
            description = "Extremely small room renovation"
        
        input_data = {
            'prompt': f"Generate work scope for {description}",
            'data': [{
                'name': 'Extreme Dimensions Room',
                'materials': {
                    'Paint - Walls': 'Existing',
                    'Flooring': 'Existing',
                    'Lighting': 'Existing'
                },
                'work_scope': {
                    'Paint - Walls': 'Remove & Replace',
                    'Flooring': 'Remove & Replace', 
                    'Lighting': 'Upgrade'
                },
                'measurements': {
                    'width': width,
                    'length': length,
                    'height': height,
                    'area_sqft': area
                },
                'demo_scope(already demo\'d)': {},
                'additional_notes': f'{description} with challenging dimensions'
            }],
            'expected_task_count': 8
        }
        
        # Adjust tasks for extreme dimensions
        expected_tasks = [
            {'task_name': 'Remove wall paint', 'quantity': 2*(width + length)*height, 'unit': 'sqft'},
            {'task_name': 'Remove flooring', 'quantity': area, 'unit': 'sqft'},
            {'task_name': 'Install wall paint', 'quantity': 2*(width + length)*height, 'unit': 'sqft'},
            {'task_name': 'Install flooring', 'quantity': area, 'unit': 'sqft'},
            {'task_name': 'Special handling for dimensions', 'quantity': 1, 'unit': 'item'}
        ]
        
        # Extreme cost adjustments
        if area > 2000:  # Large room
            cost_range = (area * 35, area * 65)
        else:  # Small room
            cost_range = (8000, 15000)  # Minimum costs still apply
        
        ground_truth = GroundTruthEstimate(
            room_name='Extreme Dimensions Room',
            expected_tasks=expected_tasks,
            expected_cost_range=cost_range,
            expected_timeline_days=random.randint(5, 30),
            critical_requirements=['dimension_validation', 'special_equipment', 'logistics_planning'],
            quality_benchmarks={'accuracy': 0.70, 'completeness': 0.80, 'consensus': 0.60}
        )
        
        return BenchmarkTestCase(
            test_id=f"edge_extreme_dims_{case_id:03d}",
            category='edge_cases',
            difficulty_level='expert',
            description=description,
            input_data=input_data,
            ground_truth=ground_truth,
            validation_criteria={
                'dimension_handling': True,
                'special_considerations': True,
                'consensus_threshold': 0.5
            },
            tags=['edge_case', 'extreme_dimensions', 'special_handling'],
            created_timestamp=datetime.now().isoformat()
        )
    
    def _save_dataset(self, dataset: List[BenchmarkTestCase], output_path: str):
        """Save dataset to file"""
        
        # Convert dataclasses to dictionaries
        dataset_dict = []
        for case in dataset:
            case_dict = asdict(case)
            dataset_dict.append(case_dict)
        
        # Create comprehensive dataset file
        output_data = {
            'metadata': {
                'total_cases': len(dataset),
                'categories': list(set(case.category for case in dataset)),
                'difficulty_levels': list(set(case.difficulty_level for case in dataset)),
                'generated_timestamp': datetime.now().isoformat(),
                'generator_version': '1.0.0'
            },
            'test_cases': dataset_dict
        }
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        self.logger.info(f"Dataset saved to {output_path}")
        
        # Create summary report
        self._generate_dataset_summary(dataset, output_path)
    
    def _generate_dataset_summary(self, dataset: List[BenchmarkTestCase], output_path: str):
        """Generate summary report of the dataset"""
        
        # Statistics
        category_counts = {}
        difficulty_counts = {}
        tag_counts = {}
        
        for case in dataset:
            category_counts[case.category] = category_counts.get(case.category, 0) + 1
            difficulty_counts[case.difficulty_level] = difficulty_counts.get(case.difficulty_level, 0) + 1
            for tag in case.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        summary = {
            'dataset_overview': {
                'total_cases': len(dataset),
                'categories': category_counts,
                'difficulty_levels': difficulty_counts,
                'most_common_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            },
            'quality_distribution': {
                'high_accuracy_cases': sum(1 for case in dataset if case.ground_truth.quality_benchmarks['accuracy'] >= 0.9),
                'medium_accuracy_cases': sum(1 for case in dataset if 0.7 <= case.ground_truth.quality_benchmarks['accuracy'] < 0.9),
                'challenging_cases': sum(1 for case in dataset if case.ground_truth.quality_benchmarks['accuracy'] < 0.7)
            },
            'cost_range_analysis': {
                'min_cost': min(case.ground_truth.expected_cost_range[0] for case in dataset),
                'max_cost': max(case.ground_truth.expected_cost_range[1] for case in dataset),
                'avg_cost_range': sum((case.ground_truth.expected_cost_range[0] + case.ground_truth.expected_cost_range[1]) / 2 for case in dataset) / len(dataset)
            }
        }
        
        # Save summary
        summary_path = output_path.replace('.json', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Dataset summary saved to {summary_path}")


# Utility function for easy dataset generation
def create_benchmark_dataset(output_dir: str = "benchmark_datasets", num_cases_per_category: int = 50):
    """Create a comprehensive benchmark dataset"""
    
    generator = BenchmarkDatasetGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = f"{output_dir}/construction_estimation_benchmark_{timestamp}.json"
    
    dataset = generator.generate_comprehensive_dataset(
        num_cases_per_category=num_cases_per_category,
        output_path=output_path
    )
    
    return dataset, output_path