# src/validators/response_validator.py
"""
Comprehensive Response Validation System for AI Estimation Pipeline
Validates AI model responses, ensures data quality, and provides error recovery
"""

import re
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    CRITICAL = "critical"  # Must fix, blocks processing
    HIGH = "high"         # Should fix, affects quality
    MEDIUM = "medium"     # May fix, minor issues
    LOW = "low"          # Informational
    

class QualityLevel(Enum):
    """Response quality levels"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 70-89
    ACCEPTABLE = "acceptable"  # 50-69
    POOR = "poor"           # 30-49
    FAILED = "failed"       # 0-29


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: ValidationSeverity
    category: str
    field: str
    message: str
    auto_fixable: bool = False
    fixed: bool = False
    
    
@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    quality_score: float  # 0-100
    quality_level: QualityLevel
    total_issues: int
    critical_issues: int
    high_issues: int
    auto_fixed: int
    issues: List[ValidationIssue] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no critical issues and quality score >= 50)"""
        return self.critical_issues == 0 and self.quality_score >= 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'quality_score': self.quality_score,
            'quality_level': self.quality_level.value,
            'total_issues': self.total_issues,
            'critical_issues': self.critical_issues,
            'high_issues': self.high_issues,
            'auto_fixed': self.auto_fixed,
            'issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category,
                    'field': issue.field,
                    'message': issue.message,
                    'auto_fixable': issue.auto_fixable,
                    'fixed': issue.fixed
                }
                for issue in self.issues
            ],
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        return (
            f"Quality: {self.quality_level.value} ({self.quality_score:.1f}/100)\n"
            f"Issues: {self.total_issues} total "
            f"({self.critical_issues} critical, {self.high_issues} high)\n"
            f"Auto-fixed: {self.auto_fixed} issues\n"
            f"Processing time: {self.processing_time:.3f}s"
        )


class ResponseStructureValidator:
    """Validates AI response structure and format"""
    
    def __init__(self):
        self.room_name_patterns = [
            r'^\*+(.+?)\*+$',  # **Kitchen** -> Kitchen
            r'^Room\s*\d+$',    # Room 1, Room 2
            r'^Unknown.*$',     # Unknown, Unknown Room
            r'^\s*$'           # Empty
        ]
        
        self.required_room_fields = {'room_name', 'tasks'}
        self.required_task_fields = {'task_name', 'task_type', 'quantity', 'unit'}
        
    def validate_response_structure(self, response_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate overall response structure"""
        issues = []
        
        # Check for rooms array
        if 'rooms' not in response_data:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="structure",
                field="rooms",
                message="Missing 'rooms' field in response",
                auto_fixable=False
            ))
            return issues
            
        rooms = response_data.get('rooms', [])
        
        # Check if rooms is a list
        if not isinstance(rooms, list):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="structure",
                field="rooms",
                message=f"'rooms' must be a list, got {type(rooms).__name__}",
                auto_fixable=False
            ))
            return issues
            
        # Check if rooms is empty
        if len(rooms) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="structure",
                field="rooms",
                message="Empty rooms array",
                auto_fixable=False
            ))
            
        # Validate each room
        for i, room in enumerate(rooms):
            room_issues = self._validate_room_structure(room, i)
            issues.extend(room_issues)
            
        return issues
    
    def _validate_room_structure(self, room: Dict[str, Any], index: int) -> List[ValidationIssue]:
        """Validate individual room structure"""
        issues = []
        
        # Check room is dict
        if not isinstance(room, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="structure",
                field=f"rooms[{index}]",
                message=f"Room must be a dictionary, got {type(room).__name__}",
                auto_fixable=False
            ))
            return issues
            
        # Check required fields
        for field in self.required_room_fields:
            if field not in room:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="structure",
                    field=f"rooms[{index}].{field}",
                    message=f"Missing required field '{field}'",
                    auto_fixable=field == 'room_name'  # Can auto-generate room name
                ))
                
        # Validate room name
        room_name = room.get('room_name', '')
        name_issues = self._validate_room_name(room_name, index)
        issues.extend(name_issues)
        
        # Validate tasks
        tasks = room.get('tasks', [])
        if not isinstance(tasks, list):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="structure",
                field=f"rooms[{index}].tasks",
                message=f"Tasks must be a list, got {type(tasks).__name__}",
                auto_fixable=False
            ))
        else:
            for j, task in enumerate(tasks):
                task_issues = self._validate_task_structure(task, index, j)
                issues.extend(task_issues)
                
        return issues
    
    def _validate_room_name(self, room_name: str, index: int) -> List[ValidationIssue]:
        """Validate room name is not placeholder"""
        issues = []
        
        # Check for empty
        if not room_name or not room_name.strip():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="room_name",
                field=f"rooms[{index}].room_name",
                message="Empty room name",
                auto_fixable=True
            ))
            return issues
            
        # Check for placeholder patterns
        for pattern in self.room_name_patterns:
            if re.match(pattern, room_name):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="room_name",
                    field=f"rooms[{index}].room_name",
                    message=f"Room name appears to be placeholder: '{room_name}'",
                    auto_fixable=True
                ))
                break
                
        # Check for asterisks
        if '**' in room_name:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                category="room_name",
                field=f"rooms[{index}].room_name",
                message=f"Room name contains asterisks: '{room_name}'",
                auto_fixable=True
            ))
            
        return issues
    
    def _validate_task_structure(self, task: Dict[str, Any], room_idx: int, task_idx: int) -> List[ValidationIssue]:
        """Validate individual task structure"""
        issues = []
        
        if not isinstance(task, dict):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="structure",
                field=f"rooms[{room_idx}].tasks[{task_idx}]",
                message=f"Task must be a dictionary, got {type(task).__name__}",
                auto_fixable=False
            ))
            return issues
            
        # Check required fields
        for field in self.required_task_fields:
            if field not in task:
                severity = ValidationSeverity.HIGH if field in {'task_name', 'task_type'} else ValidationSeverity.MEDIUM
                issues.append(ValidationIssue(
                    severity=severity,
                    category="structure",
                    field=f"rooms[{room_idx}].tasks[{task_idx}].{field}",
                    message=f"Missing required field '{field}'",
                    auto_fixable=field == 'unit'  # Can auto-determine unit
                ))
                
        # Validate quantity
        quantity = task.get('quantity')
        if quantity is not None:
            if not isinstance(quantity, (int, float)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="data_type",
                    field=f"rooms[{room_idx}].tasks[{task_idx}].quantity",
                    message=f"Quantity must be numeric, got {type(quantity).__name__}",
                    auto_fixable=True
                ))
            elif quantity < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="data_validation",
                    field=f"rooms[{room_idx}].tasks[{task_idx}].quantity",
                    message=f"Negative quantity: {quantity}",
                    auto_fixable=True
                ))
                
        return issues
    
    def fix_room_names(self, response_data: Dict[str, Any], original_data: Dict[str, Any]) -> int:
        """Attempt to fix room names using original data"""
        fixed_count = 0
        
        # Extract room names from original data
        original_rooms = self._extract_original_room_names(original_data)
        
        rooms = response_data.get('rooms', [])
        for i, room in enumerate(rooms):
            room_name = room.get('room_name', '')
            
            # Check if needs fixing
            needs_fix = False
            for pattern in self.room_name_patterns:
                if re.match(pattern, room_name):
                    needs_fix = True
                    break
                    
            if '**' in room_name:
                needs_fix = True
                
            if needs_fix:
                # Try to extract clean name
                clean_name = self._clean_room_name(room_name)
                
                # Try to match with original rooms
                if i < len(original_rooms):
                    room['room_name'] = original_rooms[i]
                    fixed_count += 1
                elif clean_name:
                    room['room_name'] = clean_name
                    fixed_count += 1
                else:
                    room['room_name'] = f"Room {i+1}"
                    fixed_count += 1
                    
        return fixed_count
    
    def _clean_room_name(self, room_name: str) -> str:
        """Clean room name from asterisks and other artifacts"""
        # Remove asterisks
        clean = re.sub(r'\*+', '', room_name).strip()
        
        # If result is empty or generic, return empty
        if not clean or re.match(r'^Room\s*\d*$', clean) or clean.lower() == 'unknown':
            return ''
            
        return clean
    
    def _extract_original_room_names(self, original_data: Any) -> List[str]:
        """Extract room names from original input data"""
        room_names = []
        
        if isinstance(original_data, dict):
            # Check for rooms in data
            if 'rooms' in original_data:
                for room in original_data['rooms']:
                    if isinstance(room, dict) and 'name' in room:
                        room_names.append(room['name'])
                        
            # Check for floors structure
            elif 'floors' in original_data:
                for floor in original_data['floors']:
                    if isinstance(floor, dict) and 'rooms' in floor:
                        for room in floor['rooms']:
                            if isinstance(room, dict) and 'name' in room:
                                room_names.append(room['name'])
                                
        elif isinstance(original_data, list):
            # Array format [project_info, floor1, floor2, ...]
            for item in original_data[1:]:  # Skip project info
                if isinstance(item, dict) and 'rooms' in item:
                    for room in item['rooms']:
                        if isinstance(room, dict) and 'name' in room:
                            room_names.append(room['name'])
                            
        return room_names


class DataIntegrityValidator:
    """Validates data integrity and business logic"""
    
    def __init__(self):
        self.task_type_mapping = {
            'removal': ['remove', 'demolish', 'tear out', 'strip'],
            'installation': ['install', 'replace', 'new', 'apply'],
            'protection': ['protect', 'cover', 'seal', 'wrap'],
            'detach': ['detach', 'disconnect', 'remove temporarily'],
            'reset': ['reset', 'reconnect', 'reinstall']
        }
        
        self.unit_ranges = {
            'sqft': (0, 10000),  # Square feet
            'lf': (0, 1000),      # Linear feet
            'sy': (0, 1500),      # Square yards
            'item': (0, 100),     # Items
            'hour': (0, 500),     # Hours
            'each': (0, 100)      # Each
        }
        
    def validate_data_integrity(self, response_data: Dict[str, Any], 
                               original_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate data integrity and business logic"""
        issues = []
        
        rooms = response_data.get('rooms', [])
        
        for i, room in enumerate(rooms):
            # Validate Remove & Replace logic
            rr_issues = self._validate_remove_replace_logic(room, i)
            issues.extend(rr_issues)
            
            # Validate measurements consistency
            measure_issues = self._validate_measurements(room, i, original_data)
            issues.extend(measure_issues)
            
            # Validate task relationships
            task_issues = self._validate_task_relationships(room, i)
            issues.extend(task_issues)
            
            # Validate quantity ranges
            qty_issues = self._validate_quantity_ranges(room, i)
            issues.extend(qty_issues)
            
        return issues
    
    def _validate_remove_replace_logic(self, room: Dict[str, Any], index: int) -> List[ValidationIssue]:
        """Validate Remove & Replace task pairing"""
        issues = []
        tasks = room.get('tasks', [])
        
        # Group tasks by material category
        removal_tasks = {}
        installation_tasks = {}
        
        for task in tasks:
            task_type = task.get('task_type', '')
            material = task.get('material_category', '')
            
            if task_type == 'removal':
                removal_tasks[material] = task
            elif task_type == 'installation':
                installation_tasks[material] = task
                
        # Check for orphaned installations
        for material, install_task in installation_tasks.items():
            if material and material not in removal_tasks:
                # Check if this is a repair/patch (doesn't need removal)
                task_name = install_task.get('task_name', '').lower()
                if 'patch' not in task_name and 'repair' not in task_name:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="business_logic",
                        field=f"rooms[{index}].tasks",
                        message=f"Installation task for '{material}' without corresponding removal",
                        auto_fixable=False
                    ))
                    
        # Check for removal without installation
        for material, removal_task in removal_tasks.items():
            if material and material not in installation_tasks:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category="business_logic",
                    field=f"rooms[{index}].tasks",
                    message=f"Removal task for '{material}' without installation (may be intentional)",
                    auto_fixable=False
                ))
                
        return issues
    
    def _validate_measurements(self, room: Dict[str, Any], index: int, 
                              original_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate measurements are properly used"""
        issues = []
        
        # Extract original measurements for this room
        original_measurements = self._extract_room_measurements(
            room.get('room_name', ''), 
            original_data
        )
        
        if not original_measurements:
            return issues
            
        tasks = room.get('tasks', [])
        
        # Check flooring tasks against floor area
        floor_area = original_measurements.get('floor_area_sqft', 0)
        if floor_area > 0:
            flooring_tasks = [t for t in tasks if 'flooring' in t.get('material_category', '').lower()]
            for task in flooring_tasks:
                quantity = task.get('quantity', 0)
                if quantity > floor_area * 1.2:  # Allow 20% variance
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="measurement_consistency",
                        field=f"rooms[{index}].tasks",
                        message=f"Flooring quantity ({quantity}) exceeds room area ({floor_area}) by >20%",
                        auto_fixable=False
                    ))
                    
        return issues
    
    def _validate_task_relationships(self, room: Dict[str, Any], index: int) -> List[ValidationIssue]:
        """Validate logical relationships between tasks"""
        issues = []
        tasks = room.get('tasks', [])
        
        # Check for detach without reset
        detach_items = set()
        reset_items = set()
        
        for task in tasks:
            task_type = task.get('task_type', '')
            task_name = task.get('task_name', '')
            
            if task_type == 'detach':
                # Extract item being detached
                item = self._extract_item_from_task(task_name)
                if item:
                    detach_items.add(item.lower())
            elif task_type == 'reset':
                # Extract item being reset
                item = self._extract_item_from_task(task_name)
                if item:
                    reset_items.add(item.lower())
                    
        # Check for mismatched detach/reset
        detached_not_reset = detach_items - reset_items
        for item in detached_not_reset:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                category="business_logic",
                field=f"rooms[{index}].tasks",
                message=f"Item '{item}' is detached but not reset",
                auto_fixable=True
            ))
            
        return issues
    
    def _validate_quantity_ranges(self, room: Dict[str, Any], index: int) -> List[ValidationIssue]:
        """Validate quantities are within reasonable ranges"""
        issues = []
        tasks = room.get('tasks', [])
        
        for j, task in enumerate(tasks):
            quantity = task.get('quantity', 0)
            unit = task.get('unit', '')
            
            if unit in self.unit_ranges:
                min_val, max_val = self.unit_ranges[unit]
                if quantity < min_val or quantity > max_val:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="data_validation",
                        field=f"rooms[{index}].tasks[{j}].quantity",
                        message=f"Quantity {quantity} {unit} outside expected range ({min_val}-{max_val})",
                        auto_fixable=False
                    ))
                    
        return issues
    
    def _extract_room_measurements(self, room_name: str, original_data: Any) -> Dict[str, Any]:
        """Extract measurements for a specific room from original data"""
        measurements = {}
        
        # Implementation would extract measurements from original data structure
        # This is a simplified version
        if isinstance(original_data, dict) and 'rooms' in original_data:
            for room in original_data.get('rooms', []):
                if room.get('name') == room_name:
                    measurements = room.get('measurements', {})
                    break
                    
        return measurements
    
    def _extract_item_from_task(self, task_name: str) -> Optional[str]:
        """Extract the item name from a task description"""
        # Simple extraction - would be more sophisticated in production
        task_lower = task_name.lower()
        
        # Remove common prefixes
        for prefix in ['detach', 'reset', 'remove', 'install', 'disconnect', 'reconnect']:
            if task_lower.startswith(prefix):
                item = task_name[len(prefix):].strip()
                # Remove leading articles
                for article in ['the', 'a', 'an']:
                    if item.lower().startswith(article + ' '):
                        item = item[len(article)+1:].strip()
                return item
                
        return None
    
    def fix_data_integrity_issues(self, response_data: Dict[str, Any], 
                                 issues: List[ValidationIssue]) -> int:
        """Attempt to fix data integrity issues"""
        fixed_count = 0
        
        for issue in issues:
            if not issue.auto_fixable or issue.fixed:
                continue
                
            # Fix missing reset tasks
            if 'not reset' in issue.message:
                fixed = self._add_missing_reset_task(response_data, issue)
                if fixed:
                    issue.fixed = True
                    fixed_count += 1
                    
            # Fix negative quantities
            elif 'Negative quantity' in issue.message:
                fixed = self._fix_negative_quantity(response_data, issue)
                if fixed:
                    issue.fixed = True
                    fixed_count += 1
                    
        return fixed_count
    
    def _add_missing_reset_task(self, response_data: Dict[str, Any], 
                               issue: ValidationIssue) -> bool:
        """Add missing reset task for detached item"""
        # Extract room index and item from issue
        match = re.search(r"rooms\[(\d+)\]", issue.field)
        if not match:
            return False
            
        room_idx = int(match.group(1))
        rooms = response_data.get('rooms', [])
        
        if room_idx >= len(rooms):
            return False
            
        # Extract item name from message
        match = re.search(r"Item '(.+?)' is detached", issue.message)
        if not match:
            return False
            
        item = match.group(1)
        
        # Add reset task
        room = rooms[room_idx]
        tasks = room.get('tasks', [])
        tasks.append({
            'task_id': f"auto_reset_{len(tasks)}",
            'task_name': f"Reset {item}",
            'task_type': 'reset',
            'quantity': 1,
            'unit': 'item',
            'notes': 'Auto-generated reset task'
        })
        
        return True
    
    def _fix_negative_quantity(self, response_data: Dict[str, Any], 
                              issue: ValidationIssue) -> bool:
        """Fix negative quantity by converting to positive"""
        # Extract indices from field
        match = re.search(r"rooms\[(\d+)\]\.tasks\[(\d+)\]", issue.field)
        if not match:
            return False
            
        room_idx = int(match.group(1))
        task_idx = int(match.group(2))
        
        rooms = response_data.get('rooms', [])
        if room_idx >= len(rooms):
            return False
            
        tasks = rooms[room_idx].get('tasks', [])
        if task_idx >= len(tasks):
            return False
            
        # Convert to positive
        task = tasks[task_idx]
        if 'quantity' in task and task['quantity'] < 0:
            task['quantity'] = abs(task['quantity'])
            return True
            
        return False


class ValidationOrchestrator:
    """Orchestrates all validation components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.structure_validator = ResponseStructureValidator()
        self.integrity_validator = DataIntegrityValidator()
        
        # Configuration
        self.auto_fix_enabled = self.config.get('auto_fix', True)
        self.min_quality_threshold = self.config.get('min_quality_threshold', 30.0)
        self.max_processing_time = self.config.get('max_processing_time', 5.0)
        
    def validate_response(self, response_data: Dict[str, Any], 
                         original_data: Dict[str, Any],
                         auto_fix: bool = None) -> Tuple[Dict[str, Any], ValidationReport]:
        """
        Comprehensive response validation with optional auto-fixing
        
        Args:
            response_data: AI model response
            original_data: Original input data
            auto_fix: Whether to attempt auto-fixing (overrides config)
            
        Returns:
            Tuple of (validated_response, validation_report)
        """
        start_time = time.time()
        
        if auto_fix is None:
            auto_fix = self.auto_fix_enabled
            
        # Initialize report
        all_issues = []
        auto_fixed_count = 0
        
        # Phase 1: Structure validation
        structure_issues = self.structure_validator.validate_response_structure(response_data)
        all_issues.extend(structure_issues)
        
        # Phase 2: Auto-fix structure issues if enabled
        if auto_fix and structure_issues:
            fixed = self.structure_validator.fix_room_names(response_data, original_data)
            auto_fixed_count += fixed
            
        # Phase 3: Data integrity validation
        integrity_issues = self.integrity_validator.validate_data_integrity(
            response_data, original_data
        )
        all_issues.extend(integrity_issues)
        
        # Phase 4: Auto-fix integrity issues if enabled
        if auto_fix and integrity_issues:
            fixed = self.integrity_validator.fix_data_integrity_issues(
                response_data, integrity_issues
            )
            auto_fixed_count += fixed
            
        # Phase 5: Validate minimum task count
        total_tasks = self._count_total_tasks(response_data)
        room_count = len(response_data.get('rooms', []))
        expected_min_tasks = room_count * 10  # Minimum 10 tasks per room
        
        if total_tasks == 0:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="EMPTY_RESPONSE",
                message=f"No tasks generated - Phase 1 requires at least {expected_min_tasks} tasks for {room_count} rooms",
                field="tasks",
                auto_fixable=False,
                fixed=False
            ))
        elif total_tasks < room_count * 5:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                category="INSUFFICIENT_TASKS",
                message=f"Only {total_tasks} tasks generated - expected at least {expected_min_tasks} for {room_count} rooms",
                field="tasks",
                auto_fixable=False,
                fixed=False
            ))
        elif total_tasks < expected_min_tasks:
            all_issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                category="LOW_TASK_COUNT",
                message=f"Only {total_tasks} tasks generated - optimal is {expected_min_tasks}-{room_count * 20} for {room_count} rooms",
                field="tasks",
                auto_fixable=False,
                fixed=False
            ))
            
        # Calculate quality score
        quality_score = self._calculate_quality_score(all_issues)
        quality_level = self._determine_quality_level(quality_score)
        
        # Count issue severities
        critical_count = sum(1 for i in all_issues if i.severity == ValidationSeverity.CRITICAL)
        high_count = sum(1 for i in all_issues if i.severity == ValidationSeverity.HIGH)
        
        # Create report
        report = ValidationReport(
            quality_score=quality_score,
            quality_level=quality_level,
            total_issues=len(all_issues),
            critical_issues=critical_count,
            high_issues=high_count,
            auto_fixed=auto_fixed_count,
            issues=all_issues,
            processing_time=time.time() - start_time,
            metadata={
                'auto_fix_enabled': auto_fix,
                'original_room_count': self._count_original_rooms(original_data),
                'validated_room_count': len(response_data.get('rooms', [])),
                'total_tasks': self._count_total_tasks(response_data)
            }
        )
        
        return response_data, report
    
    def _calculate_quality_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate quality score based on issues (0-100)"""
        if not issues:
            return 100.0
            
        # Start with perfect score
        score = 100.0
        
        # Deduct points based on severity
        severity_penalties = {
            ValidationSeverity.CRITICAL: 20,
            ValidationSeverity.HIGH: 10,
            ValidationSeverity.MEDIUM: 5,
            ValidationSeverity.LOW: 2
        }
        
        for issue in issues:
            if not issue.fixed:  # Don't penalize fixed issues
                penalty = severity_penalties.get(issue.severity, 0)
                score -= penalty
                
        # Ensure score stays in range
        return max(0.0, min(100.0, score))
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 70:
            return QualityLevel.GOOD
        elif score >= 50:
            return QualityLevel.ACCEPTABLE
        elif score >= 30:
            return QualityLevel.POOR
        else:
            return QualityLevel.FAILED
            
    def _count_original_rooms(self, original_data: Any) -> int:
        """Count rooms in original data"""
        count = 0
        
        if isinstance(original_data, dict):
            if 'rooms' in original_data:
                count = len(original_data['rooms'])
            elif 'floors' in original_data:
                for floor in original_data['floors']:
                    if 'rooms' in floor:
                        count += len(floor['rooms'])
                        
        elif isinstance(original_data, list):
            for item in original_data[1:]:  # Skip project info
                if isinstance(item, dict) and 'rooms' in item:
                    count += len(item['rooms'])
                    
        return count
    
    def _count_total_tasks(self, response_data: Dict[str, Any]) -> int:
        """Count total tasks in response"""
        total = 0
        for room in response_data.get('rooms', []):
            total += len(room.get('tasks', []))
        return total
    
    def validate_batch(self, responses: List[Dict[str, Any]], 
                       original_data: Dict[str, Any]) -> List[Tuple[Dict[str, Any], ValidationReport]]:
        """Validate multiple responses in batch"""
        results = []
        
        for response in responses:
            validated, report = self.validate_response(response, original_data)
            results.append((validated, report))
            
        return results
    
    def get_validation_summary(self, reports: List[ValidationReport]) -> Dict[str, Any]:
        """Get summary statistics from multiple validation reports"""
        if not reports:
            return {}
            
        avg_quality = sum(r.quality_score for r in reports) / len(reports)
        total_issues = sum(r.total_issues for r in reports)
        total_fixed = sum(r.auto_fixed for r in reports)
        
        quality_distribution = {}
        for level in QualityLevel:
            count = sum(1 for r in reports if r.quality_level == level)
            quality_distribution[level.value] = count
            
        return {
            'average_quality_score': avg_quality,
            'total_issues': total_issues,
            'total_auto_fixed': total_fixed,
            'quality_distribution': quality_distribution,
            'report_count': len(reports)
        }


# Convenience function for easy integration
def validate_model_response(response: Dict[str, Any], 
                           original_data: Dict[str, Any],
                           auto_fix: bool = True,
                           config: Dict[str, Any] = None) -> Tuple[Dict[str, Any], ValidationReport]:
    """
    Validate a single model response
    
    Args:
        response: AI model response
        original_data: Original input data
        auto_fix: Whether to attempt auto-fixing
        config: Optional configuration
        
    Returns:
        Tuple of (validated_response, validation_report)
    """
    orchestrator = ValidationOrchestrator(config)
    return orchestrator.validate_response(response, original_data, auto_fix)


# Integration with ModelOrchestrator
def integrate_with_model_orchestrator():
    """
    Example integration with existing ModelOrchestrator
    This would be added to model_interface.py
    """
    code = '''
    # In src/models/model_interface.py, add to ModelOrchestrator class:
    
    from src.validators.response_validator import ValidationOrchestrator
    
    class ModelOrchestrator:
        def __init__(self, config=None):
            # ... existing init code ...
            self.validation_orchestrator = ValidationOrchestrator(config)
            self.enable_validation = config.get('enable_validation', True) if config else True
            
        async def run_parallel(self, prompt, json_data, model_names=None, 
                              validate_responses=None, min_quality_threshold=None):
            """Run models in parallel with optional validation"""
            
            if validate_responses is None:
                validate_responses = self.enable_validation
                
            # ... existing parallel execution code ...
            
            if validate_responses:
                validated_results = []
                for result in results:
                    # Extract response data
                    response_data = self._extract_response_data(result.raw_response)
                    
                    # Validate and fix
                    validated_data, report = self.validation_orchestrator.validate_response(
                        response_data, json_data
                    )
                    
                    # Log validation results
                    self.logger.info(f"{result.model_name} validation: {report.get_summary()}")
                    
                    # Filter by quality threshold
                    if min_quality_threshold and report.quality_score < min_quality_threshold:
                        self.logger.warning(
                            f"Excluding {result.model_name} response due to low quality: "
                            f"{report.quality_score:.1f} < {min_quality_threshold}"
                        )
                        continue
                        
                    # Update result with validated data
                    result.room_estimates = validated_data.get('rooms', [])
                    result.validation_report = report
                    validated_results.append(result)
                    
                return validated_results
            
            return results
    '''
    return code


if __name__ == "__main__":
    # Example usage
    sample_response = {
        "rooms": [
            {
                "room_name": "**Kitchen**",
                "tasks": [
                    {
                        "task_name": "Remove flooring",
                        "task_type": "removal",
                        "quantity": -150,  # Negative quantity
                        "unit": "sqft"
                    }
                ]
            }
        ]
    }
    
    sample_original = {
        "rooms": [
            {"name": "Kitchen", "measurements": {"floor_area_sqft": 150}}
        ]
    }
    
    # Validate with auto-fixing
    validated, report = validate_model_response(sample_response, sample_original)
    
    print(report.get_summary())
    print(f"Fixed room name: {validated['rooms'][0]['room_name']}")
    print(f"Fixed quantity: {validated['rooms'][0]['tasks'][0]['quantity']}")