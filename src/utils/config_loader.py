# src/utils/config_loader.py
import yaml
import os
import time
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from src.models.data_models import AppConfig

class ConfigLoader:
    """애플리케이션 설정 로더"""
    
    def __init__(self, config_path: str = None):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.config_path = config_path or str(self.base_dir / "config" / "settings.yaml")
        self.env_path = str(self.base_dir / ".env")
        
        # 환경변수 로드
        load_dotenv(self.env_path)
        
    def load_config(self) -> AppConfig:
        """YAML 설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            return AppConfig(**config_data)
        
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            print("Using default configuration")
            return AppConfig()
        
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            print("Using default configuration")
            return AppConfig()
    
    def get_api_keys(self) -> Dict[str, str]:
        """환경변수에서 API 키 로드"""
        return {
            'openai': os.getenv('OPENAI_API_KEY', ''),
            'anthropic': os.getenv('ANTHROPIC_API_KEY', ''),
            'google': os.getenv('GOOGLE_API_KEY', '')
        }
    
    def get_debug_mode(self) -> bool:
        """디버그 모드 설정"""
        return os.getenv('DEBUG', 'False').lower() == 'true'


# src/utils/statistical_utils.py
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats

class StatisticalProcessor:
    """통계 처리 유틸리티"""
    
    @staticmethod
    def calculate_iqr_bounds(values: List[float], multiplier: float = 1.5) -> Tuple[float, float]:
        """IQR 방식으로 아웃라이어 경계값 계산"""
        if len(values) < 2:
            return float('-inf'), float('inf')
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return lower_bound, upper_bound
    
    @staticmethod
    def remove_outliers_iqr(values: List[float], multiplier: float = 1.5) -> List[float]:
        """IQR 방식으로 아웃라이어 제거"""
        if len(values) <= 2:
            return values
        
        lower_bound, upper_bound = StatisticalProcessor.calculate_iqr_bounds(values, multiplier)
        return [v for v in values if lower_bound <= v <= upper_bound]
    
    @staticmethod
    def calculate_z_scores(values: List[float]) -> List[float]:
        """Z-score 계산"""
        if len(values) <= 1:
            return [0.0] * len(values)
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return [0.0] * len(values)
        
        return [(v - mean) / std for v in values]
    
    @staticmethod
    def remove_outliers_zscore(values: List[float], threshold: float = 3.0) -> List[float]:
        """Z-score 방식으로 아웃라이어 제거"""
        if len(values) <= 2:
            return values
        
        z_scores = StatisticalProcessor.calculate_z_scores(values)
        return [values[i] for i, z in enumerate(z_scores) if abs(z) <= threshold]
    
    @staticmethod
    def weighted_average(values: List[float], weights: List[float]) -> float:
        """가중 평균 계산"""
        if not values or not weights or len(values) != len(weights):
            return 0.0
        
        if sum(weights) == 0:
            return np.mean(values)
        
        return np.average(values, weights=weights)
    
    @staticmethod
    def calculate_variance_metrics(values: List[float]) -> Dict[str, float]:
        """분산 메트릭스 계산"""
        if len(values) <= 1:
            return {
                'mean': values[0] if values else 0.0,
                'std': 0.0,
                'cv': 0.0,  # coefficient of variation
                'range': 0.0,
                'iqr': 0.0
            }
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        return {
            'mean': mean,
            'std': std,
            'cv': (std / mean * 100) if mean != 0 else 0.0,
            'range': max(values) - min(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25)
        }
    
    @staticmethod
    def calculate_consensus_score(agreement_matrix: List[List[bool]]) -> float:
        """합의 점수 계산 (0-1)"""
        if not agreement_matrix:
            return 0.0
        
        total_comparisons = 0
        agreements = 0
        
        for i in range(len(agreement_matrix)):
            for j in range(i + 1, len(agreement_matrix)):
                total_comparisons += len(agreement_matrix[i])
                agreements += sum(1 for k in range(len(agreement_matrix[i])) 
                                if agreement_matrix[i][k] == agreement_matrix[j][k])
        
        return agreements / total_comparisons if total_comparisons > 0 else 0.0


# src/utils/text_utils.py
import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher

class TextProcessor:
    """텍스트 처리 유틸리티"""
    
    @staticmethod
    def normalize_task_name(task: str) -> str:
        """작업명 정규화"""
        # 소문자 변환, 특수문자 제거, 공백 정리
        normalized = re.sub(r'[^\w\s]', '', task.lower())
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return normalized
    
    @staticmethod
    def find_similar_tasks(target: str, candidates: List[str], threshold: float = 0.8) -> List[Tuple[str, float]]:
        """유사한 작업명 찾기"""
        similarities = []
        target_normalized = TextProcessor.normalize_task_name(target)
        
        for candidate in candidates:
            candidate_normalized = TextProcessor.normalize_task_name(candidate)
            similarity = SequenceMatcher(None, target_normalized, candidate_normalized).ratio()
            
            if similarity >= threshold:
                similarities.append((candidate, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def create_task_mapping(task_lists: List[List[str]]) -> Dict[str, List[str]]:
        """여러 모델의 작업 리스트를 매핑"""
        all_tasks = []
        for task_list in task_lists:
            all_tasks.extend(task_list)
        
        unique_tasks = list(set(all_tasks))
        mapping = {}
        
        for task in unique_tasks:
            normalized = TextProcessor.normalize_task_name(task)
            if normalized not in mapping:
                mapping[normalized] = []
            mapping[normalized].append(task)
        
        return mapping
    
    @staticmethod
    def extract_quantity_from_text(text: str) -> Tuple[float, str]:
        """텍스트에서 수량과 단위 추출"""
        # 숫자와 단위 패턴 매칭
        pattern = r'(\d+\.?\d*)\s*(sq\s?ft|sqft|square\s?feet|lf|linear\s?feet|ea|each|pcs|pieces|sy|square\s?yards?)'
        
        match = re.search(pattern, text.lower())
        if match:
            quantity = float(match.group(1))
            unit = re.sub(r'\s+', '_', match.group(2).strip().lower())
            return quantity, unit
        
        return 0.0, ""
    
    @staticmethod
    def standardize_units(unit: str) -> str:
        """단위 표준화"""
        unit_mapping = {
            'sq_ft': ['sqft', 'square_feet', 'sq_feet'],
            'lf': ['linear_feet', 'lin_ft'],
            'sy': ['square_yards', 'sq_yards'],
            'ea': ['each', 'pcs', 'pieces']
        }
        
        unit_lower = unit.lower().replace(' ', '_')
        
        for standard, variants in unit_mapping.items():
            if unit_lower in variants or unit_lower == standard:
                return standard
        
        return unit_lower


# src/utils/validation_utils.py
from typing import Dict, List, Any, Tuple
from src.models.data_models import Room, WorkScope, requires_remove_replace, is_high_ceiling

class ValidationUtils:
    """검증 유틸리티"""
    
    @staticmethod
    def validate_remove_replace_logic(room: Room, work_items: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Remove & Replace 로직 검증"""
        results = {}
        
        # 각 재료별 Remove & Replace 로직 확인
        materials = ['Flooring', 'Baseboard', 'Quarter_Round']
        
        for material in materials:
            work_scope = getattr(room.work_scope, material, "")
            
            if requires_remove_replace(work_scope):
                # 제거 작업 포함 여부 확인
                removal_found = any(
                    material.lower() in item.get('task_name', '').lower() and 
                    'removal' in item.get('task_name', '').lower()
                    for item in work_items
                )
                
                # 설치 작업 포함 여부 확인
                installation_found = any(
                    material.lower() in item.get('task_name', '').lower() and 
                    'install' in item.get('task_name', '').lower()
                    for item in work_items
                )
                
                results[f"{material}_remove_replace"] = removal_found and installation_found
            else:
                results[f"{material}_remove_replace"] = True  # N/A인 경우 통과
        
        return results
    
    @staticmethod
    def validate_measurements_usage(room: Room, work_items: List[Dict[str, Any]]) -> bool:
        """측정값 사용 정확성 검증"""
        # 바닥 면적이 0이 아닌데 바닥 작업이 있는지 확인
        if room.measurements.floor_area_sqft > 0:
            flooring_work = any(
                'floor' in item.get('task_name', '').lower() for item in work_items
            )
            if not flooring_work and room.work_scope.Flooring:
                return False
        
        # 벽 면적이 0이 아닌데 벽 작업이 있는지 확인
        if room.measurements.wall_area_sqft > 0:
            wall_work = any(
                'wall' in item.get('task_name', '').lower() for item in work_items
            )
            if not wall_work and room.work_scope.Wall:
                return False
        
        return True
    
    @staticmethod
    def validate_high_ceiling_premium(room: Room, work_items: List[Dict[str, Any]]) -> bool:
        """높은 천장 할증 적용 검증"""
        if is_high_ceiling(room.measurements.height):
            # 높은 천장 할증이 언급되었는지 확인
            premium_mentioned = any(
                'high ceiling' in item.get('description', '').lower() or
                'premium' in item.get('description', '').lower()
                for item in work_items if 'ceiling' in item.get('task_name', '').lower() or
                'wall' in item.get('task_name', '').lower()
            )
            return premium_mentioned
        else:
            # 높은 천장이 아닌데 할증이 적용되었는지 확인 (잘못된 경우)
            incorrect_premium = any(
                'high ceiling' in item.get('description', '').lower()
                for item in work_items
            )
            return not incorrect_premium
    
    @staticmethod
    def validate_demo_scope_handling(room: Room, work_items: List[Dict[str, Any]]) -> bool:
        """이미 철거된 항목 처리 검증"""
        # 이미 철거된 항목에 대해 제거 비용이 계산되지 않았는지 확인
        demo_items = []
        
        if room.demo_scope.ceiling_drywall_sqft > 0:
            demo_items.append('ceiling_drywall')
        if room.demo_scope.wall_drywall_sqft > 0:
            demo_items.append('wall_drywall')
        
        for demo_item in demo_items:
            # 이미 철거된 항목에 대한 제거 작업이 포함되지 않았는지 확인
            demo_removal_found = any(
                demo_item.replace('_', ' ') in item.get('task_name', '').lower() and
                'removal' in item.get('task_name', '').lower()
                for item in work_items
            )
            
            if demo_removal_found:
                return False
        
        return True
    
    @staticmethod
    def generate_validation_report(room: Room, work_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """종합 검증 리포트 생성"""
        return {
            'room_name': room.name,
            'remove_replace_logic': ValidationUtils.validate_remove_replace_logic(room, work_items),
            'measurements_usage': ValidationUtils.validate_measurements_usage(room, work_items),
            'high_ceiling_premium': ValidationUtils.validate_high_ceiling_premium(room, work_items),
            'demo_scope_handling': ValidationUtils.validate_demo_scope_handling(room, work_items),
            'total_work_items': len(work_items),
            'has_protection_tasks': any('protection' in item.get('task_name', '').lower() 
                                      for item in work_items),
            'has_detach_reset_tasks': any('detach' in item.get('task_name', '').lower() or
                                        'reset' in item.get('task_name', '').lower()
                                        for item in work_items)
        }