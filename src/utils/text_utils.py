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
    
    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정리"""
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text
    
    @staticmethod
    def extract_room_name(text: str) -> str:
        """텍스트에서 방 이름 추출"""
        room_patterns = [
            r'(living\s+room|bedroom|bathroom|kitchen|dining\s+room|family\s+room|office)',
            r'(master\s+bedroom|guest\s+room|laundry\s+room|utility\s+room)',
            r'(\w+\s+room|\w+room)'
        ]
        
        text_lower = text.lower()
        for pattern in room_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).title()
        
        return ""
    
    @staticmethod
    def contains_keywords(text: str, keywords: List[str]) -> bool:
        """텍스트에 키워드들이 포함되어 있는지 확인"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """텍스트에서 숫자들 추출"""
        pattern = r'\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """두 텍스트 간 유사도 계산"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        # 간단한 문장 분리 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
        """특수문자 제거"""
        if keep_spaces:
            return re.sub(r'[^\w\s]', '', text)
        else:
            return re.sub(r'[^\w]', '', text)
    
    @staticmethod
    def abbreviate_text(text: str, max_length: int = 50) -> str:
        """텍스트 축약"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length-3] + "..."
    
    @staticmethod
    def extract_task_type(task_name: str) -> str:
        """작업 타입 추출 (removal, installation, painting 등)"""
        task_lower = task_name.lower()
        
        if any(keyword in task_lower for keyword in ['remove', 'tear', 'demo', 'strip']):
            return 'removal'
        elif any(keyword in task_lower for keyword in ['install', 'mount', 'place', 'apply']):
            return 'installation'  
        elif any(keyword in task_lower for keyword in ['paint', 'stain', 'finish']):
            return 'finishing'
        elif any(keyword in task_lower for keyword in ['repair', 'fix', 'patch']):
            return 'repair'
        elif any(keyword in task_lower for keyword in ['clean', 'wash', 'prep']):
            return 'preparation'
        else:
            return 'other'
    
    @staticmethod
    def group_similar_strings(strings: List[str], threshold: float = 0.8) -> Dict[str, List[str]]:
        """유사한 문자열들을 그룹핑"""
        groups = {}
        processed = set()
        
        for i, string in enumerate(strings):
            if i in processed:
                continue
                
            # 새 그룹 시작
            group_key = string
            group_members = [string]
            processed.add(i)
            
            # 유사한 문자열들 찾기
            for j, other_string in enumerate(strings[i+1:], i+1):
                if j in processed:
                    continue
                    
                similarity = TextProcessor.calculate_text_similarity(string, other_string)
                if similarity >= threshold:
                    group_members.append(other_string)
                    processed.add(j)
            
            groups[group_key] = group_members
        
        return groups