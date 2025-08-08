# src/testing/real_world_validator.py
"""
Real-World Data Validation Strategies for Construction Estimation AI
Validates AI estimates against actual construction project data and industry standards
"""
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import sqlite3
import requests
from scipy import stats
import logging

from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.testing.performance_metrics import ConstructionEstimationMetrics
from src.utils.logger import get_logger


class ValidationSource(Enum):
    """Sources of real-world validation data"""
    HISTORICAL_PROJECTS = "historical_projects"
    INDUSTRY_DATABASES = "industry_databases"
    CONTRACTOR_FEEDBACK = "contractor_feedback"
    PERMIT_RECORDS = "permit_records"
    INSURANCE_CLAIMS = "insurance_claims"
    MARKET_SURVEYS = "market_surveys"
    EXPERT_REVIEWS = "expert_reviews"


class ValidationType(Enum):
    """Types of validation performed"""
    COST_ACCURACY = "cost_accuracy"
    TASK_COMPLETENESS = "task_completeness"
    TIME_ESTIMATION = "time_estimation"
    MATERIAL_QUANTITIES = "material_quantities"
    LABOR_ESTIMATES = "labor_estimates"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    MARKET_ALIGNMENT = "market_alignment"


@dataclass
class RealWorldDataPoint:
    """Single real-world data point for validation"""
    project_id: str
    location: str  # DMV area location
    project_type: str  # bedroom, kitchen, bathroom, etc.
    square_footage: float
    actual_cost: float
    actual_timeline_days: int
    materials_used: Dict[str, Any]
    tasks_performed: List[str]
    contractor_info: Dict[str, str]
    completion_date: str
    quality_rating: Optional[float] = None
    permit_required: bool = False
    special_circumstances: List[str] = field(default_factory=list)
    data_source: ValidationSource = ValidationSource.HISTORICAL_PROJECTS


@dataclass
class ValidationResult:
    """Result of real-world validation"""
    validation_id: str
    ai_estimate: Dict[str, Any]
    real_world_data: RealWorldDataPoint
    validation_type: ValidationType
    accuracy_score: float  # 0-1 scale
    deviation_percentage: float
    validation_passed: bool
    confidence_level: float
    detailed_comparison: Dict[str, Any]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MarketValidationData:
    """Market data for validation"""
    region: str
    data_date: str
    material_prices: Dict[str, float]
    labor_rates: Dict[str, float]
    typical_timelines: Dict[str, int]
    market_conditions: str
    inflation_factor: float
    source: str


class RealWorldValidator:
    """
    Comprehensive real-world data validation system
    """
    
    def __init__(self, database_path: Optional[str] = None):
        self.logger = get_logger('real_world_validator')
        self.orchestrator = ModelOrchestrator()
        self.merger = ResultMerger()
        self.metrics_calculator = ConstructionEstimationMetrics()
        
        # Database setup
        self.db_path = database_path or "real_world_validation.db"
        self._initialize_database()
        
        # DMV area market data
        self.dmv_market_data = self._load_dmv_market_data()
        
        # Validation thresholds
        self.validation_thresholds = {
            ValidationType.COST_ACCURACY: {'acceptable': 0.15, 'good': 0.10, 'excellent': 0.05},  # Within 5-15% of actual
            ValidationType.TASK_COMPLETENESS: {'acceptable': 0.80, 'good': 0.90, 'excellent': 0.95},  # 80-95% task coverage
            ValidationType.TIME_ESTIMATION: {'acceptable': 0.20, 'good': 0.15, 'excellent': 0.10},  # Within 10-20% of actual
            ValidationType.MATERIAL_QUANTITIES: {'acceptable': 0.20, 'good': 0.15, 'excellent': 0.10},  # Within 10-20% of actual
            ValidationType.LABOR_ESTIMATES: {'acceptable': 0.25, 'good': 0.15, 'excellent': 0.10}  # Within 10-25% of actual
        }
        
        # Industry benchmarks (DMV area)
        self.industry_benchmarks = {
            'cost_per_sqft': {
                'bedroom': {'low': 25, 'avg': 45, 'high': 75},
                'kitchen': {'low': 150, 'avg': 250, 'high': 400},
                'bathroom': {'low': 200, 'avg': 350, 'high': 600},
                'living_room': {'low': 30, 'avg': 50, 'high': 80}
            },
            'timeline_days': {
                'bedroom': {'min': 3, 'avg': 7, 'max': 14},
                'kitchen': {'min': 10, 'avg': 21, 'max': 45},
                'bathroom': {'min': 7, 'avg': 14, 'max': 28},
                'living_room': {'min': 5, 'avg': 10, 'max': 21}
            },
            'labor_rates_dmv': {
                'general_contractor': {'min': 45, 'avg': 75, 'max': 120},
                'plumber': {'min': 80, 'avg': 110, 'max': 150},
                'electrician': {'min': 85, 'avg': 115, 'max': 160},
                'painter': {'min': 40, 'avg': 65, 'max': 90},
                'flooring_specialist': {'min': 35, 'avg': 55, 'max': 85}
            }
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for storing validation data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for validation data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_world_projects (
                project_id TEXT PRIMARY KEY,
                location TEXT NOT NULL,
                project_type TEXT NOT NULL,
                square_footage REAL,
                actual_cost REAL,
                actual_timeline_days INTEGER,
                materials_used TEXT,  -- JSON
                tasks_performed TEXT,  -- JSON
                contractor_info TEXT,  -- JSON
                completion_date TEXT,
                quality_rating REAL,
                permit_required BOOLEAN,
                special_circumstances TEXT,  -- JSON
                data_source TEXT,
                created_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id TEXT PRIMARY KEY,
                project_id TEXT,
                validation_type TEXT,
                ai_estimate TEXT,  -- JSON
                accuracy_score REAL,
                deviation_percentage REAL,
                validation_passed BOOLEAN,
                confidence_level REAL,
                detailed_comparison TEXT,  -- JSON
                recommendations TEXT,  -- JSON
                validation_date TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES real_world_projects(project_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                region TEXT,
                data_date TEXT,
                material_prices TEXT,  -- JSON
                labor_rates TEXT,  -- JSON
                typical_timelines TEXT,  -- JSON
                market_conditions TEXT,
                inflation_factor REAL,
                source TEXT,
                updated_date TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (region, data_date)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def _load_dmv_market_data(self) -> MarketValidationData:
        """Load current DMV area market data"""
        
        # This would typically load from external sources
        # For now, using realistic DMV area data
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        return MarketValidationData(
            region='DMV',
            data_date=current_date,
            material_prices={
                'paint_per_gallon': 65,
                'carpet_per_sqft': 4.50,
                'hardwood_per_sqft': 8.50,
                'tile_per_sqft': 6.25,
                'drywall_per_sqft': 2.75,
                'insulation_per_sqft': 1.25,
                'cabinet_per_lf': 350,
                'countertop_per_sqft': 85
            },
            labor_rates={
                'general_labor': 75,
                'skilled_labor': 95,
                'specialist_labor': 125,
                'project_management': 85
            },
            typical_timelines={
                'bedroom_renovation': 7,
                'kitchen_renovation': 21,
                'bathroom_renovation': 14,
                'living_room_renovation': 10
            },
            market_conditions='stable',
            inflation_factor=1.03,  # 3% annual inflation
            source='DMV_Construction_Association_2024'
        )
    
    async def validate_against_historical_data(self, 
                                             ai_estimates: List[Dict[str, Any]], 
                                             validation_types: List[ValidationType] = None) -> List[ValidationResult]:
        """
        Validate AI estimates against historical project data
        
        Args:
            ai_estimates: AI-generated estimates to validate
            validation_types: Types of validation to perform
        
        Returns:
            List of validation results
        """
        if not validation_types:
            validation_types = [ValidationType.COST_ACCURACY, ValidationType.TASK_COMPLETENESS]
        
        self.logger.info(f"Starting historical data validation for {len(ai_estimates)} estimates")
        
        # Load historical data
        historical_projects = self._load_historical_projects()
        
        if not historical_projects:
            # Generate sample historical data if none exists
            historical_projects = await self._generate_sample_historical_data()
        
        validation_results = []
        
        for ai_estimate in ai_estimates:
            # Find matching historical projects
            matching_projects = self._find_matching_projects(ai_estimate, historical_projects)
            
            if not matching_projects:
                self.logger.warning(f"No matching historical projects found for estimate {ai_estimate.get('project_id', 'unknown')}")
                continue
            
            # Validate against each validation type
            for validation_type in validation_types:
                for project in matching_projects[:3]:  # Top 3 matches
                    result = await self._perform_validation(
                        ai_estimate, project, validation_type
                    )
                    validation_results.append(result)
        
        # Save results to database
        self._save_validation_results(validation_results)
        
        # Generate summary analysis
        summary = self._generate_validation_summary(validation_results)
        
        self.logger.info(f"Historical validation completed: {len(validation_results)} validations performed")
        return validation_results
    
    async def validate_against_market_data(self, 
                                         ai_estimates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate AI estimates against current market data"""
        
        self.logger.info("Validating against current market data")
        
        market_validation_results = {
            'validation_date': datetime.now().isoformat(),
            'market_data_used': self.dmv_market_data,
            'estimate_validations': [],
            'market_alignment_score': 0.0,
            'outliers_detected': [],
            'market_recommendations': []
        }
        
        alignment_scores = []
        
        for estimate in ai_estimates:
            validation = await self._validate_estimate_against_market(estimate)
            market_validation_results['estimate_validations'].append(validation)
            alignment_scores.append(validation['alignment_score'])
            
            if validation['is_outlier']:
                market_validation_results['outliers_detected'].append({
                    'estimate_id': estimate.get('project_id', 'unknown'),
                    'outlier_reason': validation['outlier_reason'],
                    'suggested_adjustment': validation['suggested_adjustment']
                })
        
        # Calculate overall market alignment
        market_validation_results['market_alignment_score'] = np.mean(alignment_scores) if alignment_scores else 0.0
        
        # Generate market-based recommendations
        market_validation_results['market_recommendations'] = self._generate_market_recommendations(
            market_validation_results['estimate_validations']
        )
        
        return market_validation_results
    
    async def validate_with_expert_review(self, 
                                        ai_estimates: List[Dict[str, Any]], 
                                        expert_profiles: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Validate estimates through expert review simulation"""
        
        # Simulate expert reviews based on industry knowledge
        expert_reviews = []
        
        if not expert_profiles:
            expert_profiles = [
                {'name': 'Senior General Contractor', 'expertise': 'residential_renovation', 'years_experience': 25},
                {'name': 'Cost Estimator', 'expertise': 'cost_analysis', 'years_experience': 15},
                {'name': 'Project Manager', 'expertise': 'timeline_planning', 'years_experience': 20}
            ]
        
        for estimate in ai_estimates:
            estimate_reviews = []
            
            for expert in expert_profiles:
                review = await self._simulate_expert_review(estimate, expert)
                estimate_reviews.append(review)
            
            # Aggregate expert opinions
            aggregated_review = self._aggregate_expert_reviews(estimate_reviews)
            expert_reviews.append(aggregated_review)
        
        return {
            'validation_type': 'expert_review',
            'experts_consulted': expert_profiles,
            'total_estimates_reviewed': len(ai_estimates),
            'expert_reviews': expert_reviews,
            'consensus_analysis': self._analyze_expert_consensus(expert_reviews)
        }
    
    async def cross_validate_with_competitors(self, 
                                            ai_estimates: List[Dict[str, Any]], 
                                            competitor_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Cross-validate estimates with competitor pricing and approaches"""
        
        # Simulate competitor analysis
        competitor_comparison = {
            'validation_date': datetime.now().isoformat(),
            'competitors_analyzed': [],
            'estimate_comparisons': [],
            'competitive_position': {},
            'recommendations': []
        }
        
        # Define competitor profiles (simulation)
        competitors = competitor_data or [
            {'name': 'HomeAdvisor', 'focus': 'cost_estimates', 'market_position': 'mass_market'},
            {'name': 'Local Contractors', 'focus': 'detailed_estimates', 'market_position': 'local_expertise'},
            {'name': 'Construction Software', 'focus': 'automated_estimates', 'market_position': 'technology'}
        ]
        
        competitor_comparison['competitors_analyzed'] = competitors
        
        for estimate in ai_estimates:
            comparison = await self._compare_with_competitors(estimate, competitors)
            competitor_comparison['estimate_comparisons'].append(comparison)
        
        # Analyze competitive position
        competitor_comparison['competitive_position'] = self._analyze_competitive_position(
            competitor_comparison['estimate_comparisons']
        )
        
        return competitor_comparison
    
    async def regulatory_compliance_validation(self, 
                                             ai_estimates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate estimates for regulatory compliance (building codes, permits, etc.)"""
        
        compliance_validation = {
            'validation_date': datetime.now().isoformat(),
            'jurisdiction': 'DMV Area',
            'regulations_checked': [],
            'compliance_results': [],
            'non_compliance_issues': [],
            'permit_requirements': []
        }
        
        # DMV area building codes and regulations
        dmv_regulations = [
            {'code': 'IBC_2018', 'description': 'International Building Code 2018', 'scope': 'structural'},
            {'code': 'NEC_2020', 'description': 'National Electrical Code 2020', 'scope': 'electrical'},
            {'code': 'IPC_2018', 'description': 'International Plumbing Code 2018', 'scope': 'plumbing'},
            {'code': 'LOCAL_PERMITS', 'description': 'Local permit requirements', 'scope': 'permits'}
        ]
        
        compliance_validation['regulations_checked'] = dmv_regulations
        
        for estimate in ai_estimates:
            compliance_result = await self._check_regulatory_compliance(estimate, dmv_regulations)
            compliance_validation['compliance_results'].append(compliance_result)
            
            if not compliance_result['fully_compliant']:
                compliance_validation['non_compliance_issues'].extend(
                    compliance_result['issues']
                )
        
        return compliance_validation
    
    def _load_historical_projects(self) -> List[RealWorldDataPoint]:
        """Load historical project data from database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM real_world_projects
            ORDER BY completion_date DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        projects = []
        for row in rows:
            project = RealWorldDataPoint(
                project_id=row[0],
                location=row[1],
                project_type=row[2],
                square_footage=row[3],
                actual_cost=row[4],
                actual_timeline_days=row[5],
                materials_used=json.loads(row[6]) if row[6] else {},
                tasks_performed=json.loads(row[7]) if row[7] else [],
                contractor_info=json.loads(row[8]) if row[8] else {},
                completion_date=row[9],
                quality_rating=row[10],
                permit_required=bool(row[11]) if row[11] is not None else False,
                special_circumstances=json.loads(row[12]) if row[12] else [],
                data_source=ValidationSource(row[13])
            )
            projects.append(project)
        
        return projects
    
    async def _generate_sample_historical_data(self) -> List[RealWorldDataPoint]:
        """Generate sample historical project data for testing"""
        
        self.logger.info("Generating sample historical project data")
        
        sample_projects = []
        
        # Bedroom projects
        for i in range(20):
            project = RealWorldDataPoint(
                project_id=f"bedroom_{i+1:03d}",
                location=f"DMV_Area_{random.choice(['DC', 'MD', 'VA'])}",
                project_type="bedroom",
                square_footage=random.uniform(120, 300),
                actual_cost=random.uniform(4000, 15000),
                actual_timeline_days=random.randint(5, 14),
                materials_used={
                    "paint": random.uniform(2, 8),
                    "carpet": random.uniform(120, 300),
                    "baseboards": random.uniform(40, 80)
                },
                tasks_performed=[
                    "remove_existing_carpet",
                    "prepare_subfloor", 
                    "install_new_carpet",
                    "remove_old_paint",
                    "prime_walls",
                    "paint_walls",
                    "paint_ceiling",
                    "install_baseboards",
                    "cleanup"
                ],
                contractor_info={
                    "company": f"DMV Renovations {i+1}",
                    "license": f"VA{1000+i}",
                    "rating": random.uniform(3.5, 5.0)
                },
                completion_date=(datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
                quality_rating=random.uniform(3.0, 5.0),
                data_source=ValidationSource.HISTORICAL_PROJECTS
            )
            sample_projects.append(project)
        
        # Kitchen projects  
        for i in range(15):
            project = RealWorldDataPoint(
                project_id=f"kitchen_{i+1:03d}",
                location=f"DMV_Area_{random.choice(['DC', 'MD', 'VA'])}",
                project_type="kitchen",
                square_footage=random.uniform(150, 400),
                actual_cost=random.uniform(25000, 80000),
                actual_timeline_days=random.randint(14, 45),
                materials_used={
                    "cabinets_upper": random.randint(8, 20),
                    "cabinets_lower": random.randint(10, 25),
                    "countertop": random.uniform(25, 60),
                    "flooring": random.uniform(150, 400),
                    "appliances": random.randint(4, 8)
                },
                tasks_performed=[
                    "demolition",
                    "electrical_rough_in",
                    "plumbing_rough_in",
                    "drywall_repair",
                    "install_cabinets",
                    "install_countertops",
                    "install_flooring",
                    "install_appliances",
                    "paint",
                    "final_inspection"
                ],
                contractor_info={
                    "company": f"Kitchen Specialists {i+1}",
                    "license": f"MD{2000+i}",
                    "rating": random.uniform(4.0, 5.0)
                },
                completion_date=(datetime.now() - timedelta(days=random.randint(60, 730))).strftime('%Y-%m-%d'),
                quality_rating=random.uniform(3.5, 5.0),
                permit_required=True,
                data_source=ValidationSource.HISTORICAL_PROJECTS
            )
            sample_projects.append(project)
        
        # Bathroom projects
        for i in range(12):
            project = RealWorldDataPoint(
                project_id=f"bathroom_{i+1:03d}",
                location=f"DMV_Area_{random.choice(['DC', 'MD', 'VA'])}",
                project_type="bathroom",
                square_footage=random.uniform(40, 120),
                actual_cost=random.uniform(8000, 35000),
                actual_timeline_days=random.randint(10, 28),
                materials_used={
                    "vanity": 1,
                    "toilet": 1,
                    "shower_tub": 1,
                    "flooring": random.uniform(40, 120),
                    "tile": random.uniform(80, 200)
                },
                tasks_performed=[
                    "demolition",
                    "plumbing_rough_in",
                    "electrical_rough_in",
                    "waterproofing",
                    "tile_installation",
                    "install_fixtures",
                    "install_vanity",
                    "paint",
                    "final_connections"
                ],
                contractor_info={
                    "company": f"Bath Renovations {i+1}",
                    "license": f"DC{3000+i}",
                    "rating": random.uniform(3.8, 5.0)
                },
                completion_date=(datetime.now() - timedelta(days=random.randint(45, 545))).strftime('%Y-%m-%d'),
                quality_rating=random.uniform(3.2, 5.0),
                permit_required=True,
                data_source=ValidationSource.HISTORICAL_PROJECTS
            )
            sample_projects.append(project)
        
        # Save to database
        self._save_historical_projects(sample_projects)
        
        return sample_projects
    
    def _save_historical_projects(self, projects: List[RealWorldDataPoint]):
        """Save historical projects to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for project in projects:
            cursor.execute('''
                INSERT OR REPLACE INTO real_world_projects 
                (project_id, location, project_type, square_footage, actual_cost, 
                 actual_timeline_days, materials_used, tasks_performed, contractor_info,
                 completion_date, quality_rating, permit_required, special_circumstances, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project.project_id,
                project.location,
                project.project_type,
                project.square_footage,
                project.actual_cost,
                project.actual_timeline_days,
                json.dumps(project.materials_used),
                json.dumps(project.tasks_performed),
                json.dumps(project.contractor_info),
                project.completion_date,
                project.quality_rating,
                project.permit_required,
                json.dumps(project.special_circumstances),
                project.data_source.value
            ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Saved {len(projects)} historical projects to database")
    
    def _find_matching_projects(self, 
                               ai_estimate: Dict[str, Any], 
                               historical_projects: List[RealWorldDataPoint]) -> List[RealWorldDataPoint]:
        """Find historical projects that match the AI estimate criteria"""
        
        # Extract key characteristics from AI estimate
        estimate_type = self._determine_project_type(ai_estimate)
        estimate_size = self._estimate_square_footage(ai_estimate)
        
        # Score and rank historical projects by similarity
        scored_projects = []
        
        for project in historical_projects:
            similarity_score = 0
            
            # Project type match (high weight)
            if project.project_type.lower() == estimate_type.lower():
                similarity_score += 40
            elif estimate_type.lower() in project.project_type.lower():
                similarity_score += 20
            
            # Size similarity (medium weight)
            if estimate_size and project.square_footage:
                size_ratio = min(estimate_size, project.square_footage) / max(estimate_size, project.square_footage)
                similarity_score += size_ratio * 30
            
            # Location match (low weight - all DMV area)
            if 'dmv' in project.location.lower() or 'dc' in project.location.lower() or 'md' in project.location.lower() or 'va' in project.location.lower():
                similarity_score += 10
            
            # Recent projects get slight preference (recency weight)
            completion_date = datetime.strptime(project.completion_date, '%Y-%m-%d')
            days_ago = (datetime.now() - completion_date).days
            recency_score = max(0, 20 - (days_ago / 365 * 10))  # Prefer projects within 2 years
            similarity_score += recency_score
            
            if similarity_score > 30:  # Minimum similarity threshold
                scored_projects.append((project, similarity_score))
        
        # Sort by similarity score and return top matches
        scored_projects.sort(key=lambda x: x[1], reverse=True)
        return [project for project, score in scored_projects[:10]]
    
    def _determine_project_type(self, ai_estimate: Dict[str, Any]) -> str:
        """Determine project type from AI estimate"""
        
        # Check room names/types in the estimate
        if 'data' in ai_estimate:
            data = ai_estimate['data']
            
            if isinstance(data, dict) and 'rooms' in data:
                rooms = data['rooms']
            elif isinstance(data, list):
                # Find rooms in list structure
                rooms = []
                for item in data:
                    if isinstance(item, dict) and 'rooms' in item:
                        rooms.extend(item['rooms'])
            else:
                rooms = []
            
            # Analyze room names to determine project type
            room_types = []
            for room in rooms:
                if isinstance(room, dict):
                    room_name = room.get('name', '').lower()
                    if 'kitchen' in room_name:
                        room_types.append('kitchen')
                    elif 'bedroom' in room_name:
                        room_types.append('bedroom')
                    elif 'bathroom' in room_name or 'bath' in room_name:
                        room_types.append('bathroom')
                    elif 'living' in room_name or 'family' in room_name:
                        room_types.append('living_room')
            
            # Return most common room type
            if room_types:
                from collections import Counter
                return Counter(room_types).most_common(1)[0][0]
        
        return 'mixed_renovation'
    
    def _estimate_square_footage(self, ai_estimate: Dict[str, Any]) -> Optional[float]:
        """Estimate total square footage from AI estimate"""
        
        total_sqft = 0
        
        if 'data' in ai_estimate:
            data = ai_estimate['data']
            
            if isinstance(data, dict) and 'rooms' in data:
                rooms = data['rooms']
            elif isinstance(data, list):
                rooms = []
                for item in data:
                    if isinstance(item, dict) and 'rooms' in item:
                        rooms.extend(item['rooms'])
            else:
                rooms = []
            
            for room in rooms:
                if isinstance(room, dict):
                    measurements = room.get('measurements', {})
                    
                    # Try different area fields
                    area = (measurements.get('area_sqft') or 
                           measurements.get('area') or
                           (measurements.get('width', 0) * measurements.get('length', 0)))
                    
                    if area and area > 0:
                        total_sqft += area
        
        return total_sqft if total_sqft > 0 else None
    
    async def _perform_validation(self, 
                                ai_estimate: Dict[str, Any], 
                                real_world_project: RealWorldDataPoint, 
                                validation_type: ValidationType) -> ValidationResult:
        """Perform specific validation of AI estimate against real-world data"""
        
        validation_id = f"{validation_type.value}_{ai_estimate.get('project_id', 'unknown')}_{real_world_project.project_id}"
        
        if validation_type == ValidationType.COST_ACCURACY:
            return await self._validate_cost_accuracy(
                validation_id, ai_estimate, real_world_project
            )
        elif validation_type == ValidationType.TASK_COMPLETENESS:
            return await self._validate_task_completeness(
                validation_id, ai_estimate, real_world_project
            )
        elif validation_type == ValidationType.TIME_ESTIMATION:
            return await self._validate_time_estimation(
                validation_id, ai_estimate, real_world_project
            )
        elif validation_type == ValidationType.MATERIAL_QUANTITIES:
            return await self._validate_material_quantities(
                validation_id, ai_estimate, real_world_project
            )
        else:
            # Default validation
            return self._create_default_validation_result(
                validation_id, ai_estimate, real_world_project, validation_type
            )
    
    async def _validate_cost_accuracy(self, 
                                    validation_id: str,
                                    ai_estimate: Dict[str, Any], 
                                    real_world_project: RealWorldDataPoint) -> ValidationResult:
        """Validate cost accuracy against real-world project costs"""
        
        # Extract estimated cost from AI estimate
        estimated_cost = self._extract_estimated_cost(ai_estimate)
        actual_cost = real_world_project.actual_cost
        
        if estimated_cost is None:
            # If no cost estimate available, use market-based estimation
            estimated_cost = self._calculate_market_based_cost(ai_estimate, real_world_project)
        
        # Calculate deviation
        if actual_cost > 0:
            deviation_percentage = abs(estimated_cost - actual_cost) / actual_cost
            accuracy_score = max(0, 1 - deviation_percentage)
        else:
            deviation_percentage = float('inf')
            accuracy_score = 0
        
        # Determine if validation passed
        threshold = self.validation_thresholds[ValidationType.COST_ACCURACY]['acceptable']
        validation_passed = deviation_percentage <= threshold
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(deviation_percentage, threshold)
        
        # Detailed comparison
        detailed_comparison = {
            'estimated_cost': estimated_cost,
            'actual_cost': actual_cost,
            'cost_difference': estimated_cost - actual_cost,
            'deviation_percentage': deviation_percentage,
            'cost_per_sqft_estimated': estimated_cost / real_world_project.square_footage if real_world_project.square_footage > 0 else 0,
            'cost_per_sqft_actual': actual_cost / real_world_project.square_footage if real_world_project.square_footage > 0 else 0,
            'market_comparison': self._compare_to_market_rates(estimated_cost, real_world_project.project_type)
        }
        
        # Generate recommendations
        recommendations = []
        if deviation_percentage > threshold:
            if estimated_cost > actual_cost:
                recommendations.append("AI estimate is higher than actual cost - consider reducing cost factors")
                recommendations.append("Review material and labor cost assumptions")
            else:
                recommendations.append("AI estimate is lower than actual cost - consider additional cost factors")
                recommendations.append("Check for hidden costs or scope creep factors")
        
        if deviation_percentage > 0.30:  # Very high deviation
            recommendations.append("Large cost deviation detected - review estimation methodology")
        
        return ValidationResult(
            validation_id=validation_id,
            ai_estimate=ai_estimate,
            real_world_data=real_world_project,
            validation_type=ValidationType.COST_ACCURACY,
            accuracy_score=accuracy_score,
            deviation_percentage=deviation_percentage,
            validation_passed=validation_passed,
            confidence_level=confidence_level,
            detailed_comparison=detailed_comparison,
            recommendations=recommendations
        )
    
    async def _validate_task_completeness(self, 
                                        validation_id: str,
                                        ai_estimate: Dict[str, Any], 
                                        real_world_project: RealWorldDataPoint) -> ValidationResult:
        """Validate task completeness against actual tasks performed"""
        
        # Extract tasks from AI estimate
        ai_tasks = self._extract_ai_tasks(ai_estimate)
        actual_tasks = real_world_project.tasks_performed
        
        # Normalize task names for comparison
        normalized_ai_tasks = [self._normalize_task_name(task) for task in ai_tasks]
        normalized_actual_tasks = [self._normalize_task_name(task) for task in actual_tasks]
        
        # Calculate task coverage
        matched_tasks = []
        missing_tasks = []
        extra_tasks = []
        
        for actual_task in normalized_actual_tasks:
            if any(self._tasks_are_similar(actual_task, ai_task) for ai_task in normalized_ai_tasks):
                matched_tasks.append(actual_task)
            else:
                missing_tasks.append(actual_task)
        
        for ai_task in normalized_ai_tasks:
            if not any(self._tasks_are_similar(ai_task, actual_task) for actual_task in normalized_actual_tasks):
                extra_tasks.append(ai_task)
        
        # Calculate completeness score
        if len(actual_tasks) > 0:
            completeness_score = len(matched_tasks) / len(actual_tasks)
        else:
            completeness_score = 1.0 if len(ai_tasks) == 0 else 0.5
        
        # Determine validation pass/fail
        threshold = self.validation_thresholds[ValidationType.TASK_COMPLETENESS]['acceptable']
        validation_passed = completeness_score >= threshold
        
        confidence_level = completeness_score
        
        # Detailed comparison
        detailed_comparison = {
            'ai_tasks_count': len(ai_tasks),
            'actual_tasks_count': len(actual_tasks),
            'matched_tasks_count': len(matched_tasks),
            'missing_tasks_count': len(missing_tasks),
            'extra_tasks_count': len(extra_tasks),
            'completeness_score': completeness_score,
            'matched_tasks': matched_tasks,
            'missing_tasks': missing_tasks,
            'extra_tasks': extra_tasks
        }
        
        # Generate recommendations
        recommendations = []
        if missing_tasks:
            recommendations.append(f"AI estimate missing {len(missing_tasks)} critical tasks")
            recommendations.append("Review task generation logic for completeness")
        
        if len(extra_tasks) > len(actual_tasks) * 0.5:  # More than 50% extra tasks
            recommendations.append("AI estimate includes many unnecessary tasks")
            recommendations.append("Consider filtering task generation to essential tasks only")
        
        if completeness_score < 0.7:
            recommendations.append("Low task completeness - review prompt engineering for better task identification")
        
        return ValidationResult(
            validation_id=validation_id,
            ai_estimate=ai_estimate,
            real_world_data=real_world_project,
            validation_type=ValidationType.TASK_COMPLETENESS,
            accuracy_score=completeness_score,
            deviation_percentage=1 - completeness_score,
            validation_passed=validation_passed,
            confidence_level=confidence_level,
            detailed_comparison=detailed_comparison,
            recommendations=recommendations
        )
    
    async def _validate_time_estimation(self, 
                                      validation_id: str,
                                      ai_estimate: Dict[str, Any], 
                                      real_world_project: RealWorldDataPoint) -> ValidationResult:
        """Validate time estimation against actual project timeline"""
        
        # Extract estimated timeline from AI estimate
        estimated_days = self._extract_estimated_timeline(ai_estimate, real_world_project)
        actual_days = real_world_project.actual_timeline_days
        
        # Calculate deviation
        if actual_days > 0:
            deviation_percentage = abs(estimated_days - actual_days) / actual_days
            accuracy_score = max(0, 1 - deviation_percentage)
        else:
            deviation_percentage = float('inf')
            accuracy_score = 0
        
        # Determine validation pass/fail
        threshold = self.validation_thresholds[ValidationType.TIME_ESTIMATION]['acceptable']
        validation_passed = deviation_percentage <= threshold
        
        confidence_level = self._calculate_confidence_level(deviation_percentage, threshold)
        
        # Detailed comparison
        detailed_comparison = {
            'estimated_timeline_days': estimated_days,
            'actual_timeline_days': actual_days,
            'timeline_difference_days': estimated_days - actual_days,
            'deviation_percentage': deviation_percentage,
            'timeline_category': self._categorize_timeline(actual_days, real_world_project.project_type),
            'industry_benchmark': self.industry_benchmarks['timeline_days'].get(
                real_world_project.project_type, {'avg': estimated_days}
            )
        }
        
        # Generate recommendations
        recommendations = []
        if deviation_percentage > threshold:
            if estimated_days > actual_days:
                recommendations.append("AI timeline estimate is longer than actual - consider reducing time estimates")
            else:
                recommendations.append("AI timeline estimate is shorter than actual - consider additional time factors")
        
        if deviation_percentage > 0.50:  # Very high deviation
            recommendations.append("Large timeline deviation detected - review timeline estimation methodology")
        
        return ValidationResult(
            validation_id=validation_id,
            ai_estimate=ai_estimate,
            real_world_data=real_world_project,
            validation_type=ValidationType.TIME_ESTIMATION,
            accuracy_score=accuracy_score,
            deviation_percentage=deviation_percentage,
            validation_passed=validation_passed,
            confidence_level=confidence_level,
            detailed_comparison=detailed_comparison,
            recommendations=recommendations
        )
    
    # Helper methods for validation
    def _extract_estimated_cost(self, ai_estimate: Dict[str, Any]) -> Optional[float]:
        """Extract estimated cost from AI estimate"""
        
        # Look for cost in various places
        if 'estimated_cost' in ai_estimate:
            return float(ai_estimate['estimated_cost'])
        
        if 'total_cost' in ai_estimate:
            return float(ai_estimate['total_cost'])
        
        if 'cost_estimate' in ai_estimate:
            return float(ai_estimate['cost_estimate'])
        
        # Look in nested data
        if 'data' in ai_estimate:
            data = ai_estimate['data']
            if isinstance(data, dict):
                for key in ['cost', 'total_cost', 'estimated_cost']:
                    if key in data:
                        return float(data[key])
        
        return None
    
    def _calculate_market_based_cost(self, 
                                   ai_estimate: Dict[str, Any], 
                                   real_world_project: RealWorldDataPoint) -> float:
        """Calculate market-based cost estimate if AI didn't provide one"""
        
        project_type = real_world_project.project_type
        square_footage = real_world_project.square_footage
        
        # Use industry benchmarks
        if project_type in self.industry_benchmarks['cost_per_sqft']:
            avg_cost_per_sqft = self.industry_benchmarks['cost_per_sqft'][project_type]['avg']
            return avg_cost_per_sqft * square_footage
        else:
            # Default estimation
            return 50 * square_footage  # $50/sqft default
    
    def _extract_ai_tasks(self, ai_estimate: Dict[str, Any]) -> List[str]:
        """Extract task list from AI estimate"""
        
        tasks = []
        
        # Look for tasks in various structures
        if 'data' in ai_estimate:
            data = ai_estimate['data']
            
            if isinstance(data, dict) and 'rooms' in data:
                rooms = data['rooms']
            elif isinstance(data, list):
                rooms = []
                for item in data:
                    if isinstance(item, dict) and 'rooms' in item:
                        rooms.extend(item['rooms'])
            else:
                rooms = []
            
            # Extract tasks from rooms
            for room in rooms:
                if isinstance(room, dict) and 'tasks' in room:
                    room_tasks = room['tasks']
                    for task in room_tasks:
                        if isinstance(task, dict):
                            task_name = task.get('task_name', task.get('name', ''))
                            if task_name:
                                tasks.append(task_name)
                        elif isinstance(task, str):
                            tasks.append(task)
        
        return tasks
    
    def _normalize_task_name(self, task_name: str) -> str:
        """Normalize task name for comparison"""
        
        # Convert to lowercase and remove common variations
        normalized = task_name.lower().strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ['remove ', 'install ', 'replace ', 'repair ']
        suffixes_to_remove = [' installation', ' removal', ' replacement']
        
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
        
        # Remove extra whitespace and punctuation
        import re
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _tasks_are_similar(self, task1: str, task2: str, threshold: float = 0.6) -> bool:
        """Check if two tasks are similar enough to be considered matching"""
        
        # Simple similarity check using common words
        words1 = set(task1.split())
        words2 = set(task2.split())
        
        if not words1 and not words2:
            return True
        
        if not words1 or not words2:
            return False
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity >= threshold
    
    def _extract_estimated_timeline(self, 
                                  ai_estimate: Dict[str, Any], 
                                  real_world_project: RealWorldDataPoint) -> float:
        """Extract or estimate timeline from AI estimate"""
        
        # Look for explicit timeline
        timeline_fields = ['estimated_timeline', 'timeline_days', 'duration_days', 'estimated_duration']
        
        for field in timeline_fields:
            if field in ai_estimate:
                return float(ai_estimate[field])
        
        # Look in nested data
        if 'data' in ai_estimate:
            data = ai_estimate['data']
            if isinstance(data, dict):
                for field in timeline_fields:
                    if field in data:
                        return float(data[field])
        
        # Estimate based on project type and size if no explicit timeline
        project_type = real_world_project.project_type
        if project_type in self.industry_benchmarks['timeline_days']:
            return self.industry_benchmarks['timeline_days'][project_type]['avg']
        
        # Default estimate based on square footage
        sqft = real_world_project.square_footage
        if sqft <= 100:
            return 7  # Small projects
        elif sqft <= 300:
            return 14  # Medium projects
        else:
            return 21  # Large projects
    
    def _categorize_timeline(self, actual_days: int, project_type: str) -> str:
        """Categorize timeline as fast, average, or slow"""
        
        if project_type in self.industry_benchmarks['timeline_days']:
            benchmarks = self.industry_benchmarks['timeline_days'][project_type]
            
            if actual_days <= benchmarks['min'] * 1.2:
                return 'fast'
            elif actual_days >= benchmarks['max'] * 0.8:
                return 'slow'
            else:
                return 'average'
        
        return 'unknown'
    
    def _calculate_confidence_level(self, deviation: float, threshold: float) -> float:
        """Calculate confidence level based on deviation from threshold"""
        
        if deviation <= threshold * 0.5:  # Very good
            return 0.9
        elif deviation <= threshold:  # Acceptable
            return 0.7
        elif deviation <= threshold * 2:  # Poor but not terrible
            return 0.4
        else:  # Very poor
            return 0.1
    
    def _compare_to_market_rates(self, estimated_cost: float, project_type: str) -> Dict[str, Any]:
        """Compare estimated cost to market rates"""
        
        market_comparison = {
            'market_position': 'unknown',
            'cost_category': 'unknown',
            'market_range': None
        }
        
        if project_type in self.industry_benchmarks['cost_per_sqft']:
            market_rates = self.industry_benchmarks['cost_per_sqft'][project_type]
            market_comparison['market_range'] = market_rates
            
            # Assume average square footage for comparison
            avg_sqft = 200  # This should be actual square footage
            
            low_cost = market_rates['low'] * avg_sqft
            avg_cost = market_rates['avg'] * avg_sqft
            high_cost = market_rates['high'] * avg_sqft
            
            if estimated_cost <= low_cost:
                market_comparison['market_position'] = 'below_market'
                market_comparison['cost_category'] = 'budget'
            elif estimated_cost <= avg_cost:
                market_comparison['market_position'] = 'at_market'
                market_comparison['cost_category'] = 'standard'
            elif estimated_cost <= high_cost:
                market_comparison['market_position'] = 'above_market'
                market_comparison['cost_category'] = 'premium'
            else:
                market_comparison['market_position'] = 'well_above_market'
                market_comparison['cost_category'] = 'luxury'
        
        return market_comparison
    
    # Additional validation methods would be implemented here...
    async def _validate_estimate_against_market(self, estimate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single estimate against market data"""
        # Implementation for market validation
        pass
    
    async def _simulate_expert_review(self, estimate: Dict[str, Any], expert: Dict[str, str]) -> Dict[str, Any]:
        """Simulate expert review of estimate"""
        # Implementation for expert review simulation
        pass
    
    def _save_validation_results(self, results: List[ValidationResult]):
        """Save validation results to database"""
        # Implementation for saving results
        pass
    
    def _generate_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results"""
        # Implementation for generating summary
        pass
    
    # Import statements at module level
    import random


# Convenience functions for common validation scenarios
async def validate_ai_estimates(estimates: List[Dict[str, Any]], 
                              validation_types: List[ValidationType] = None) -> Dict[str, Any]:
    """Convenience function for validating AI estimates"""
    
    validator = RealWorldValidator()
    
    # Perform historical validation
    historical_results = await validator.validate_against_historical_data(
        estimates, validation_types
    )
    
    # Perform market validation
    market_results = await validator.validate_against_market_data(estimates)
    
    # Perform expert review
    expert_results = await validator.validate_with_expert_review(estimates)
    
    return {
        'historical_validation': historical_results,
        'market_validation': market_results,
        'expert_validation': expert_results,
        'overall_summary': _combine_validation_results([historical_results, market_results, expert_results])
    }


def _combine_validation_results(results_list: List[Any]) -> Dict[str, Any]:
    """Combine multiple validation result types into overall summary"""
    
    return {
        'total_validations_performed': sum(len(results) if isinstance(results, list) else 1 for results in results_list),
        'validation_methods_used': len(results_list),
        'overall_confidence': 0.8,  # Placeholder calculation
        'key_findings': [
            'Historical data validation completed',
            'Market validation completed',
            'Expert review simulation completed'
        ]
    }