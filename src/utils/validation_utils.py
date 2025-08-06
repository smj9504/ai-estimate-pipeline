# src/utils/validation_utils.py
"""
검증 유틸리티 함수들
"""
from typing import Dict, List, Any, Optional
import re
from difflib import SequenceMatcher


class ValidationUtils:
    """검증 관련 유틸리티 함수 모음"""
    
    @staticmethod
    def normalize_task_description(description: str) -> str:
        """작업 설명 정규화"""
        if not description:
            return ""
        normalized = description.lower()
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        return normalized.strip()
    
    @staticmethod
    def extract_material_type(description: str) -> Optional[str]:
        """작업 설명에서 재료 타입 추출"""
        description_lower = description.lower()
        
        material_keywords = {
            'drywall': ['drywall', 'sheetrock', 'gypsum', 'wall board'],
            'flooring': ['flooring', 'floor', 'carpet', 'tile', 'hardwood', 'laminate', 'vinyl'],
            'ceiling': ['ceiling', 'drop ceiling', 'suspended ceiling'],
            'baseboard': ['baseboard', 'base trim', 'skirting'],
            'door': ['door', 'door frame', 'doorway'],
            'window': ['window', 'window frame', 'glazing'],
            'insulation': ['insulation', 'fiberglass', 'foam'],
            'paint': ['paint', 'painting', 'primer', 'coating'],
            'electrical': ['electrical', 'wiring', 'outlet', 'switch', 'lighting'],
            'plumbing': ['plumbing', 'pipe', 'fixture', 'drain'],
            'hvac': ['hvac', 'ductwork', 'vent', 'air conditioning', 'heating']
        }
        
        for material_type, keywords in material_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return material_type
        return None
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """두 텍스트 간 유사도 계산"""
        if not text1 or not text2:
            return 0.0
        norm1 = ValidationUtils.normalize_task_description(text1)
        norm2 = ValidationUtils.normalize_task_description(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    @staticmethod
    def is_removal_task(description: str) -> bool:
        """제거 작업 여부 확인"""
        removal_keywords = [
            'remove', 'removal', 'demolish', 'demolition', 'tear out',
            'dispose', 'disposal', 'haul away', 'take out', 'strip'
        ]
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in removal_keywords)
    
    @staticmethod
    def is_installation_task(description: str) -> bool:
        """설치 작업 여부 확인"""
        installation_keywords = [
            'install', 'installation', 'replace', 'replacement', 'new',
            'apply', 'put in', 'set up', 'mount', 'attach', 'fit'
        ]
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in installation_keywords)
    
    @staticmethod
    def validate_quantity(quantity: Any) -> bool:
        """수량 유효성 검증"""
        try:
            if quantity is None:
                return False
            float_val = float(quantity)
            return float_val > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_unit(unit: str) -> bool:
        """단위 유효성 검증"""
        valid_units = [
            'sf', 'sq ft', 'square feet', 'sqft',
            'lf', 'linear feet', 'ln ft',
            'ea', 'each', 'unit', 'units',
            'cy', 'cubic yard', 'cubic yards',
            'cf', 'cubic feet', 'cubic ft',
            'hour', 'hours', 'hr', 'hrs',
            'day', 'days',
            'lot', 'ls', 'lump sum',
            'gal', 'gallon', 'gallons',
            'ton', 'tons',
            'bag', 'bags',
            'box', 'boxes'
        ]
        if not unit:
            return False
        return unit.lower() in valid_units
    
    @staticmethod
    def format_issue(category: str, message: str) -> str:
        """이슈 메시지 포맷팅"""
        return f"[{category}] {message}"
    
    @staticmethod
    def calculate_confidence_score(factors: Dict[str, float]) -> float:
        """신뢰도 점수 계산"""
        if not factors:
            return 0.0
        
        weights = {
            'logic_compliance': 0.25,
            'measurement_accuracy': 0.20,
            'completeness': 0.20,
            'consistency': 0.15,
            'consensus_level': 0.10,
            'outlier_ratio': 0.10
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for factor, score in factors.items():
            weight = weights.get(factor, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(1.0, weighted_sum / total_weight)
    
    @staticmethod
    def check_high_ceiling_premium(description: str, is_high_ceiling: bool) -> bool:
        """높은 천장 할증 적용 여부 확인"""
        if not is_high_ceiling:
            return False
        
        premium_required_keywords = [
            'wall', 'ceiling', 'paint', 'drywall', 'insulation',
            'scaffold', 'ladder', 'high work', 'overhead'
        ]
        
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in premium_required_keywords)