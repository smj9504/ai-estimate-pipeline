# src/validators/estimation_validator.py
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from src.models.data_models import (
    ProjectData, Room, MergedEstimate, ModelResponse,
    requires_remove_replace, is_high_ceiling
)
from src.utils.validation_utils import ValidationUtils

@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    is_valid: bool
    score: float  # 0-1
    issues: List[str]
    warnings: List[str]
    details: Dict[str, Any]

class RemoveReplaceValidator:
    """Remove & Replace 로직 검증기"""
    
    def validate_logic_application(self, room: Room, work_items: List[Dict[str, Any]]) -> ValidationResult:
        """Remove & Replace 로직 적용 검증"""
        issues = []
        warnings = []
        details = {}
        
        # 각 재료별 Remove & Replace 로직 확인
        materials = [
            ('Flooring', room.work_scope.Flooring),
            ('Baseboard', room.work_scope.Baseboard), 
            ('Quarter_Round', room.work_scope.Quarter_Round)
        ]
        
        validation_score = 0
        total_checks = 0
        
        for material_name, work_scope in materials:
            total_checks += 1
            
            if requires_remove_replace(work_scope):
                # Remove & Replace가 필요한 경우
                removal_task = self._find_removal_task(material_name, work_items)
                install_task = self._find_installation_task(material_name, work_items)
                
                if not removal_task:
                    issues.append(f"{material_name}: Remove & Replace 지정되었으나 제거 작업 누락")
                elif not install_task:
                    issues.append(f"{material_name}: Remove & Replace 지정되었으나 설치 작업 누락")
                else:
                    validation_score += 1
                    details[f"{material_name}_remove_replace"] = "올바름"
                
                # 이미 철거된 부분에 대한 중복 제거 작업 확인
                demo_conflict = self._check_demo_scope_conflict(material_name, room, work_items)
                if demo_conflict:
                    issues.append(f"{material_name}: 이미 철거된 부분에 대한 중복 제거 작업 발견")
            
            elif work_scope == "":
                # 작업이 필요 없는 경우
                unnecessary_work = self._find_unnecessary_work(material_name, work_items)
                if unnecessary_work:
                    warnings.append(f"{material_name}: 작업 범위가 비어있으나 관련 작업 발견")
                else:
                    validation_score += 1
            
            else:
                # 기타 작업 (Paint, Repair 등)
                validation_score += 1  # 기본적으로 통과
        
        final_score = validation_score / total_checks if total_checks > 0 else 0
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=final_score,
            issues=issues,
            warnings=warnings,
            details=details
        )
    
    def _find_removal_task(self, material: str, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """특정 재료의 제거 작업 찾기"""
        removal_keywords = ['remove', 'removal', 'tear out', 'demo', 'demolish']
        
        for item in work_items:
            task_name = item.get('task_name', '').lower()
            description = item.get('description', '').lower()
            
            if (material.lower() in task_name and 
                any(keyword in task_name or keyword in description for keyword in removal_keywords)):
                return item
        
        return None
    
    def _find_installation_task(self, material: str, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """특정 재료의 설치 작업 찾기"""
        install_keywords = ['install', 'installation', 'mount', 'place', 'apply']
        
        for item in work_items:
            task_name = item.get('task_name', '').lower()
            description = item.get('description', '').lower()
            
            if (material.lower() in task_name and 
                any(keyword in task_name or keyword in description for keyword in install_keywords)):
                return item
        
        return None
    
    def _find_unnecessary_work(self, material: str, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """불필요한 작업 찾기"""
        unnecessary_tasks = []
        
        for item in work_items:
            task_name = item.get('task_name', '').lower()
            if material.lower() in task_name:
                unnecessary_tasks.append(item)
        
        return unnecessary_tasks
    
    def _check_demo_scope_conflict(self, material: str, room: Room, 
                                 work_items: List[Dict[str, Any]]) -> bool:
        """이미 철거된 항목과의 충돌 확인"""
        # demo_scope에서 이미 철거된 항목 확인
        already_demoed = []
        if room.demo_scope.ceiling_drywall_sqft > 0:
            already_demoed.append('ceiling')
        if room.demo_scope.wall_drywall_sqft > 0:
            already_demoed.append('wall')
        
        # 해당 재료와 관련된 철거 작업이 이미 완료된 영역에 있는지 확인
        for demo_area in already_demoed:
            if (demo_area in material.lower() and 
                self._find_removal_task(material, work_items)):
                return True
        
        return False

class MeasurementValidator:
    """측정값 사용 정확성 검증기"""
    
    def validate_measurement_usage(self, room: Room, work_items: List[Dict[str, Any]]) -> ValidationResult:
        """측정값 사용 검증"""
        issues = []
        warnings = []
        details = {}
        
        validation_checks = []
        
        # 바닥 면적 vs 바닥 작업
        floor_check = self._validate_floor_work(room, work_items)
        validation_checks.append(floor_check)
        
        # 벽 면적 vs 벽 작업
        wall_check = self._validate_wall_work(room, work_items)
        validation_checks.append(wall_check)
        
        # 천장 면적 vs 천장 작업
        ceiling_check = self._validate_ceiling_work(room, work_items)
        validation_checks.append(ceiling_check)
        
        # 높은 천장 할증
        height_check = self._validate_height_premium(room, work_items)
        validation_checks.append(height_check)
        
        # 결과 집계
        valid_checks = sum(1 for check in validation_checks if check['valid'])
        total_checks = len(validation_checks)
        
        for check in validation_checks:
            if not check['valid']:
                issues.extend(check['issues'])
            warnings.extend(check['warnings'])
            details.update(check['details'])
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=valid_checks / total_checks if total_checks > 0 else 0,
            issues=issues,
            warnings=warnings,
            details=details
        )
    
    def _validate_floor_work(self, room: Room, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """바닥 작업 검증"""
        floor_area = room.measurements.floor_area_sqft
        floor_scope = room.work_scope.Flooring
        
        floor_tasks = [item for item in work_items 
                      if 'floor' in item.get('task_name', '').lower()]
        
        if floor_area > 0 and floor_scope and not floor_tasks:
            return {
                'valid': False,
                'issues': [f"바닥 면적 {floor_area} sq ft이나 바닥 작업 없음"],
                'warnings': [],
                'details': {'floor_area': floor_area, 'floor_tasks': len(floor_tasks)}
            }
        elif floor_area == 0 and floor_tasks:
            return {
                'valid': False,
                'issues': [],
                'warnings': [f"바닥 면적이 0이나 바닥 작업 {len(floor_tasks)}개 있음"],
                'details': {'floor_area': floor_area, 'floor_tasks': len(floor_tasks)}
            }
        else:
            return {
                'valid': True,
                'issues': [],
                'warnings': [],
                'details': {'floor_area': floor_area, 'floor_tasks': len(floor_tasks)}
            }
    
    def _validate_wall_work(self, room: Room, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """벽 작업 검증"""
        wall_area = room.measurements.wall_area_sqft
        wall_scope = room.work_scope.Wall
        
        wall_tasks = [item for item in work_items 
                     if 'wall' in item.get('task_name', '').lower()]
        
        if wall_area > 0 and wall_scope and not wall_tasks:
            return {
                'valid': False,
                'issues': [f"벽 면적 {wall_area} sq ft이나 벽 작업 없음"],
                'warnings': [],
                'details': {'wall_area': wall_area, 'wall_tasks': len(wall_tasks)}
            }
        elif wall_area == 0 and wall_tasks:
            return {
                'valid': False,
                'issues': [],
                'warnings': [f"벽 면적이 0이나 벽 작업 {len(wall_tasks)}개 있음"],
                'details': {'wall_area': wall_area, 'wall_tasks': len(wall_tasks)}
            }
        else:
            return {
                'valid': True,
                'issues': [],
                'warnings': [],
                'details': {'wall_area': wall_area, 'wall_tasks': len(wall_tasks)}
            }
    
    def _validate_ceiling_work(self, room: Room, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """천장 작업 검증"""
        ceiling_area = room.measurements.ceiling_area_sqft
        ceiling_scope = room.work_scope.Ceiling
        
        ceiling_tasks = [item for item in work_items 
                        if 'ceiling' in item.get('task_name', '').lower()]
        
        if ceiling_area > 0 and ceiling_scope and not ceiling_tasks:
            return {
                'valid': False,
                'issues': [f"천장 면적 {ceiling_area} sq ft이나 천장 작업 없음"],
                'warnings': [],
                'details': {'ceiling_area': ceiling_area, 'ceiling_tasks': len(ceiling_tasks)}
            }
        elif ceiling_area == 0 and ceiling_tasks:
            return {
                'valid': False,
                'issues': [],
                'warnings': [f"천장 면적이 0이나 천장 작업 {len(ceiling_tasks)}개 있음"],
                'details': {'ceiling_area': ceiling_area, 'ceiling_tasks': len(ceiling_tasks)}
            }
        else:
            return {
                'valid': True,
                'issues': [],
                'warnings': [],
                'details': {'ceiling_area': ceiling_area, 'ceiling_tasks': len(ceiling_tasks)}
            }
    
    def _validate_height_premium(self, room: Room, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """높은 천장 할증 검증"""
        height = room.measurements.height
        is_high = is_high_ceiling(height)
        
        premium_tasks = [item for item in work_items 
                        if 'high ceiling' in item.get('description', '').lower() or
                           'premium' in item.get('description', '').lower()]
        
        if is_high and not premium_tasks:
            return {
                'valid': False,
                'issues': [],
                'warnings': [f"높은 천장({height} ft)이나 할증 없음"],
                'details': {'height': height, 'premium_tasks': len(premium_tasks)}
            }
        elif not is_high and premium_tasks:
            return {
                'valid': False,
                'issues': [f"일반 천장({height} ft)이나 높은 천장 할증 적용"],
                'warnings': [],
                'details': {'height': height, 'premium_tasks': len(premium_tasks)}
            }
        else:
            return {
                'valid': True,
                'issues': [],
                'warnings': [],
                'details': {'height': height, 'premium_tasks': len(premium_tasks)}
            }

class ComprehensiveValidator:
    """종합 검증 시스템"""
    
    def __init__(self):
        self.remove_replace_validator = RemoveReplaceValidator()
        self.measurement_validator = MeasurementValidator()
        self.validation_utils = ValidationUtils()
    
    def validate_merged_estimate(self, merged_estimate: MergedEstimate, 
                                original_data: ProjectData) -> ValidationResult:
        """병합된 견적 전체 검증"""
        start_time = time.time()
        
        all_issues = []
        all_warnings = []
        room_validations = []
        overall_score = 0
        
        # 방별 검증
        for room_data in merged_estimate.rooms:
            room_name = room_data['name']
            
            # 원본 방 데이터 찾기
            original_room = self._find_original_room(room_name, original_data)
            if not original_room:
                all_warnings.append(f"원본 데이터에서 {room_name} 방을 찾을 수 없음")
                continue
            
            # 방별 검증 실행
            room_validation = self.validate_single_room(original_room, room_data['tasks'])
            room_validations.append({
                'room': room_name,
                'validation': room_validation
            })
            
            overall_score += room_validation.score
            all_issues.extend([f"{room_name}: {issue}" for issue in room_validation.issues])
            all_warnings.extend([f"{room_name}: {warning}" for warning in room_validation.warnings])
        
        # 전체 검증
        consistency_validation = self._validate_consistency(merged_estimate)
        all_issues.extend(consistency_validation.issues)
        all_warnings.extend(consistency_validation.warnings)
        
        # 최종 점수 계산
        room_count = len(room_validations)
        final_score = (overall_score / room_count if room_count > 0 else 0) * 0.8 + \
                     consistency_validation.score * 0.2
        
        processing_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(all_issues) == 0,
            score=final_score,
            issues=all_issues,
            warnings=all_warnings,
            details={
                'room_validations': room_validations,
                'consistency_validation': consistency_validation.details,
                'processing_time': processing_time,
                'total_rooms_validated': room_count
            }
        )
    
    def validate_single_room(self, room: Room, work_items: List[Dict[str, Any]]) -> ValidationResult:
        """단일 방 검증"""
        validations = []
        
        # Remove & Replace 로직 검증
        rr_validation = self.remove_replace_validator.validate_logic_application(room, work_items)
        validations.append(('remove_replace', rr_validation))
        
        # 측정값 사용 검증
        measurement_validation = self.measurement_validator.validate_measurement_usage(room, work_items)
        validations.append(('measurements', measurement_validation))
        
        # 추가 노트 검증 (보호, 분리/재설치)
        notes_validation = self._validate_additional_notes(room, work_items)
        validations.append(('additional_notes', notes_validation))
        
        # 결과 통합
        all_issues = []
        all_warnings = []
        total_score = 0
        
        for validation_type, validation in validations:
            all_issues.extend(validation.issues)
            all_warnings.extend(validation.warnings)
            total_score += validation.score
        
        return ValidationResult(
            is_valid=len(all_issues) == 0,
            score=total_score / len(validations) if validations else 0,
            issues=all_issues,
            warnings=all_warnings,
            details={validation_type: validation.details for validation_type, validation in validations}
        )
    
    def _validate_additional_notes(self, room: Room, work_items: List[Dict[str, Any]]) -> ValidationResult:
        """추가 노트 검증 (보호, 분리/재설치)"""
        issues = []
        warnings = []
        
        # 보호 작업 확인
        protection_needed = room.additional_notes.protection
        protection_tasks = [item for item in work_items 
                           if 'protect' in item.get('task_name', '').lower()]
        
        if protection_needed and not protection_tasks:
            warnings.append(f"보호 작업 필요({protection_needed})하나 관련 작업 없음")
        
        # 분리/재설치 작업 확인
        detach_reset_needed = room.additional_notes.detach_reset
        detach_reset_tasks = [item for item in work_items 
                             if 'detach' in item.get('task_name', '').lower() or
                                'reset' in item.get('task_name', '').lower()]
        
        if detach_reset_needed and not detach_reset_tasks:
            warnings.append(f"분리/재설치 필요({detach_reset_needed})하나 관련 작업 없음")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.5,  # 경고는 점수에 덜 영향
            issues=issues,
            warnings=warnings,
            details={
                'protection_needed': len(protection_needed),
                'protection_tasks': len(protection_tasks),
                'detach_reset_needed': len(detach_reset_needed),
                'detach_reset_tasks': len(detach_reset_tasks)
            }
        )
    
    def _validate_consistency(self, merged_estimate: MergedEstimate) -> ValidationResult:
        """전체 일관성 검증"""
        issues = []
        warnings = []
        
        # 메타데이터 일관성 확인
        if merged_estimate.overall_confidence < 0.5:
            warnings.append(f"전체 신뢰도가 낮음: {merged_estimate.overall_confidence:.2f}")
        
        if merged_estimate.metadata.manual_review_required:
            warnings.append("수동 검토가 필요한 항목들이 있음")
        
        # 통계적 일관성
        total_tasks = merged_estimate.total_work_items
        if total_tasks == 0:
            issues.append("전체 작업 항목이 0개")
        elif total_tasks > 1000:  # 비현실적으로 많은 작업
            warnings.append(f"작업 항목이 매우 많음: {total_tasks}개")
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=0.9 if len(issues) == 0 and len(warnings) <= 2 else 0.5,
            issues=issues,
            warnings=warnings,
            details={
                'confidence_score': merged_estimate.overall_confidence,
                'total_tasks': total_tasks,
                'requires_review': merged_estimate.metadata.manual_review_required
            }
        )
    
    def _find_original_room(self, room_name: str, project_data: ProjectData) -> Room:
        """원본 데이터에서 방 찾기"""
        for floor in project_data.floors:
            for room in floor.rooms:
                if room.name == room_name:
                    return room
        return None