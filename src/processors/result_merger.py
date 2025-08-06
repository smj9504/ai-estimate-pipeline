# src/processors/result_merger.py
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict, Counter

from src.models.data_models import (
    ModelResponse, MergedEstimate, MergeMetadata, 
    ConfidenceLevel, calculate_confidence_level
)
from src.utils.statistical_utils import StatisticalProcessor
from src.utils.text_utils import TextProcessor
from src.utils.config_loader import ConfigLoader

class QualitativeMerger:
    """질적 데이터 병합 (작업 범위, 로직 적용)"""
    
    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor()
        self.consensus_threshold = config.consensus.minimum_agreement
    
    def merge_work_scopes(self, model_responses: List[ModelResponse]) -> Dict[str, Any]:
        """작업 범위 병합"""
        print(f"작업 범위 병합 시작: {len(model_responses)}개 모델")
        
        # 1. 모든 작업 항목 수집
        all_tasks = self._collect_all_tasks(model_responses)
        
        # 2. 유사 작업 그룹핑
        task_groups = self._group_similar_tasks(all_tasks)
        
        # 3. 합의 규칙 적용
        consensus_tasks = self._apply_consensus_rules(task_groups, len(model_responses))
        
        # 4. 방별로 정리
        room_based_tasks = self._organize_by_room(consensus_tasks)
        
        return {
            'merged_work_scope': room_based_tasks,
            'task_groups': task_groups,
            'consensus_level': self._calculate_consensus_level(task_groups, len(model_responses)),
            'outlier_tasks': self._identify_outlier_tasks(task_groups, len(model_responses))
        }
    
    def _collect_all_tasks(self, model_responses: List[ModelResponse]) -> List[Dict[str, Any]]:
        """모든 모델의 작업 항목 수집"""
        all_tasks = []
        
        for response in model_responses:
            for room in response.room_estimates:
                room_name = room.get('name', 'Unknown')
                tasks = room.get('tasks', [])
                
                for task in tasks:
                    task_item = {
                        'model': response.model_name,
                        'room': room_name,
                        'task_name': task.get('task_name', ''),
                        'description': task.get('description', ''),
                        'necessity': task.get('necessity', 'required'),
                        'quantity': task.get('quantity', 0),
                        'unit': task.get('unit', ''),
                        'reasoning': task.get('reasoning', '')
                    }
                    all_tasks.append(task_item)
        
        print(f"전체 작업 항목 수집: {len(all_tasks)}개")
        return all_tasks
    
    def _group_similar_tasks(self, all_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """유사한 작업들을 그룹핑"""
        task_groups = defaultdict(list)
        processed_tasks = set()
        
        for i, task in enumerate(all_tasks):
            if i in processed_tasks:
                continue
            
            # 현재 작업을 기준으로 그룹 생성
            task_name = task['task_name']
            room_name = task['room']
            group_key = f"{room_name}::{self.text_processor.normalize_task_name(task_name)}"
            
            # 현재 작업을 그룹에 추가
            task_groups[group_key].append(task)
            processed_tasks.add(i)
            
            # 유사한 작업들 찾아서 같은 그룹에 추가
            for j, other_task in enumerate(all_tasks):
                if j in processed_tasks or j <= i:
                    continue
                
                if (other_task['room'] == room_name and 
                    self._are_similar_tasks(task_name, other_task['task_name'])):
                    task_groups[group_key].append(other_task)
                    processed_tasks.add(j)
        
        print(f"작업 그룹핑 완료: {len(task_groups)}개 그룹")
        return dict(task_groups)
    
    def _are_similar_tasks(self, task1: str, task2: str, threshold: float = 0.8) -> bool:
        """두 작업이 유사한지 판단"""
        similar_tasks = self.text_processor.find_similar_tasks(task1, [task2], threshold)
        return len(similar_tasks) > 0
    
    def _apply_consensus_rules(self, task_groups: Dict[str, List[Dict[str, Any]]], 
                             total_models: int) -> List[Dict[str, Any]]:
        """합의 규칙 적용"""
        consensus_tasks = []
        
        for group_key, tasks in task_groups.items():
            model_count = len(set(task['model'] for task in tasks))
            
            # 합의 판정
            if model_count >= self.consensus_threshold:
                # 대표 작업 선택 (가장 상세한 설명을 가진 것)
                representative_task = max(tasks, key=lambda t: len(t.get('description', '')))
                
                # 합의 정보 추가
                representative_task['consensus_level'] = model_count / total_models
                representative_task['supporting_models'] = [t['model'] for t in tasks]
                representative_task['group_size'] = len(tasks)
                
                consensus_tasks.append(representative_task)
            
            elif model_count == 1 and self._is_safety_critical_task(tasks[0]):
                # 안전 관련 작업은 1개 모델만 제안해도 포함
                safety_task = tasks[0].copy()
                safety_task['consensus_level'] = 0.3  # 낮은 합의도
                safety_task['safety_critical'] = True
                safety_task['supporting_models'] = [safety_task['model']]
                safety_task['group_size'] = 1
                
                consensus_tasks.append(safety_task)
        
        print(f"합의된 작업: {len(consensus_tasks)}개")
        return consensus_tasks
    
    def _is_safety_critical_task(self, task: Dict[str, Any]) -> bool:
        """안전 관련 작업인지 판단"""
        safety_keywords = [
            'electrical', 'wiring', 'circuit', 'outlet',
            'plumbing', 'water', 'gas', 'ventilation',
            'structural', 'load bearing', 'support',
            'asbestos', 'lead', 'mold', 'hazard'
        ]
        
        task_text = f"{task.get('task_name', '')} {task.get('description', '')}".lower()
        return any(keyword in task_text for keyword in safety_keywords)
    
    def _organize_by_room(self, consensus_tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """방별로 작업 정리"""
        room_tasks = defaultdict(list)
        
        for task in consensus_tasks:
            room_name = task.get('room', 'Unknown')
            room_tasks[room_name].append(task)
        
        return dict(room_tasks)
    
    def _calculate_consensus_level(self, task_groups: Dict[str, List[Dict[str, Any]]], 
                                 total_models: int) -> float:
        """전체 합의 수준 계산"""
        if not task_groups:
            return 0.0
        
        consensus_scores = []
        for tasks in task_groups.values():
            model_count = len(set(task['model'] for task in tasks))
            consensus_scores.append(model_count / total_models)
        
        return np.mean(consensus_scores)
    
    def _identify_outlier_tasks(self, task_groups: Dict[str, List[Dict[str, Any]]], 
                              total_models: int) -> List[str]:
        """이상치 작업 식별"""
        outlier_tasks = []
        
        for group_key, tasks in task_groups.items():
            model_count = len(set(task['model'] for task in tasks))
            
            # 1개 모델만 제안하고, 안전 관련이 아닌 작업
            if model_count == 1 and not self._is_safety_critical_task(tasks[0]):
                outlier_tasks.append(group_key)
        
        return outlier_tasks

class QuantitativeMerger:
    """정량적 데이터 병합 (수량, 비용)"""
    
    def __init__(self, config):
        self.config = config
        self.stats = StatisticalProcessor()
        self.model_weights = config.model_weights.normalize()
    
    def merge_quantities(self, consensus_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """수량 데이터 병합"""
        print(f"수량 데이터 병합 시작: {len(consensus_tasks)}개 작업")
        
        merged_quantities = {}
        quantity_metadata = {}
        
        for task in consensus_tasks:
            task_key = f"{task.get('room', '')}::{task.get('task_name', '')}"
            
            # 같은 작업에 대한 여러 모델의 수량 데이터
            quantities = self._extract_quantities_for_task(task)
            
            if quantities:
                merged_qty, metadata = self._merge_quantity_values(quantities, task['supporting_models'])
                merged_quantities[task_key] = merged_qty
                quantity_metadata[task_key] = metadata
            else:
                # 수량 데이터가 없는 경우
                merged_quantities[task_key] = 0.0
                quantity_metadata[task_key] = {'confidence': 'low', 'method': 'no_data'}
        
        return {
            'merged_quantities': merged_quantities,
            'quantity_metadata': quantity_metadata,
            'overall_confidence': self._calculate_overall_quantity_confidence(quantity_metadata)
        }
    
    def _extract_quantities_for_task(self, task: Dict[str, Any]) -> List[Tuple[float, str]]:
        """작업에 대한 수량 데이터 추출"""
        # 이 예제에서는 단순화했지만, 실제로는 여러 모델의 동일 작업에 대한 수량을 추출
        quantity = task.get('quantity', 0)
        model = task.get('model', '')
        
        if quantity > 0:
            return [(quantity, model)]
        else:
            return []
    
    def _merge_quantity_values(self, quantities: List[Tuple[float, str]], 
                             supporting_models: List[str]) -> Tuple[float, Dict[str, Any]]:
        """수량 값들 병합"""
        values = [q[0] for q in quantities]
        models = [q[1] for q in quantities]
        
        if len(values) == 1:
            return values[0], {
                'confidence': 'medium',
                'method': 'single_value',
                'models': models,
                'variance': 0.0
            }
        
        # 아웃라이어 제거
        clean_values = self.stats.remove_outliers_iqr(values)
        
        if not clean_values:
            clean_values = values  # 모든 값이 아웃라이어인 경우 원래 값 사용
        
        # 가중 평균 계산
        weights = [self._get_model_weight(model) for model in models[:len(clean_values)]]
        merged_value = self.stats.weighted_average(clean_values, weights)
        
        # 안전 마진 적용
        variance_metrics = self.stats.calculate_variance_metrics(clean_values)
        cv = variance_metrics['cv']  # coefficient of variation
        
        if cv < 10:  # 낮은 분산
            safety_margin = self.config.safety_margins.low_variance
            confidence = 'high'
        else:  # 높은 분산
            safety_margin = self.config.safety_margins.high_variance
            confidence = 'low'
        
        final_value = merged_value * (1 + safety_margin)
        
        return final_value, {
            'confidence': confidence,
            'method': 'weighted_average',
            'models': models,
            'variance': cv,
            'safety_margin': safety_margin,
            'outliers_removed': len(values) - len(clean_values)
        }
    
    def _get_model_weight(self, model_name: str) -> float:
        """모델별 가중치 반환"""
        weight_mapping = {
            'gpt-4': self.model_weights.gpt4,
            'claude-3-sonnet': self.model_weights.claude,
            'gemini-pro': self.model_weights.gemini
        }
        return weight_mapping.get(model_name, 0.33)  # 기본값
    
    def _calculate_overall_quantity_confidence(self, quantity_metadata: Dict[str, Any]) -> float:
        """전체 수량 신뢰도 계산"""
        if not quantity_metadata:
            return 0.0
        
        confidence_scores = []
        for metadata in quantity_metadata.values():
            confidence_level = metadata.get('confidence', 'low')
            score = {'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(confidence_level, 0.3)
            confidence_scores.append(score)
        
        return np.mean(confidence_scores)

class ResultMerger:
    """전체 결과 병합 관리자"""
    
    def __init__(self, config=None):
        if config is None:
            # config가 없으면 새로 로드
            self.config_loader = ConfigLoader()
            self.config = self.config_loader.load_config()
        else:
            # config 객체가 제공되면 직접 사용
            self.config = config
            self.config_loader = ConfigLoader()
        
        self.qualitative_merger = QualitativeMerger(self.config)
        self.quantitative_merger = QuantitativeMerger(self.config)
    
    def merge_results(self, model_responses: List[ModelResponse]) -> MergedEstimate:
        """전체 결과 병합"""
        if not model_responses:
            return self._create_empty_result()
        
        print(f"결과 병합 시작: {len(model_responses)}개 모델 결과")
        
        start_time = time.time()
        
        # 1. 질적 데이터 병합 (작업 범위)
        qualitative_result = self.qualitative_merger.merge_work_scopes(model_responses)
        
        # 2. 정량적 데이터 병합 (수량)
        consensus_tasks = self._flatten_consensus_tasks(qualitative_result['merged_work_scope'])
        quantitative_result = self.quantitative_merger.merge_quantities(consensus_tasks)
        
        # 3. 최종 결과 통합
        merged_estimate = self._create_merged_estimate(
            model_responses,
            qualitative_result,
            quantitative_result,
            time.time() - start_time
        )
        
        print(f"결과 병합 완료: 신뢰도 {merged_estimate.overall_confidence:.2f}")
        return merged_estimate
    
    def _flatten_consensus_tasks(self, room_based_tasks: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """방별 작업을 플랫 리스트로 변환"""
        all_tasks = []
        for room_tasks in room_based_tasks.values():
            all_tasks.extend(room_tasks)
        return all_tasks
    
    def _create_merged_estimate(self, model_responses: List[ModelResponse],
                              qualitative_result: Dict[str, Any],
                              quantitative_result: Dict[str, Any],
                              processing_time: float) -> MergedEstimate:
        """최종 병합 결과 생성"""
        
        # 메타데이터 생성
        metadata = MergeMetadata(
            models_used=[r.model_name for r in model_responses],
            consensus_level=qualitative_result['consensus_level'],
            deviation_metrics={
                'task_groups': len(qualitative_result['task_groups']),
                'outlier_tasks': len(qualitative_result['outlier_tasks']),
                'quantity_confidence': quantitative_result['overall_confidence']
            },
            outlier_flags=qualitative_result['outlier_tasks'],
            processing_time_total=processing_time,
            confidence_level=calculate_confidence_level(qualitative_result['consensus_level']),
            manual_review_required=qualitative_result['consensus_level'] < 0.6,
            safety_margin_applied=self.config.safety_margins.low_variance  # 평균값
        )
        
        # 방별 정리된 결과
        rooms_data = []
        for room_name, tasks in qualitative_result['merged_work_scope'].items():
            room_data = {
                'name': room_name,
                'tasks': tasks,
                'task_count': len(tasks),
                'high_consensus_tasks': len([t for t in tasks if t.get('consensus_level', 0) >= 0.8]),
                'safety_critical_tasks': len([t for t in tasks if t.get('safety_critical', False)])
            }
            rooms_data.append(room_data)
        
        return MergedEstimate(
            project_info={'merged': True},  # 실제로는 원본 프로젝트 정보
            rooms=rooms_data,
            total_work_items=sum(len(tasks) for tasks in qualitative_result['merged_work_scope'].values()),
            overall_confidence=min(qualitative_result['consensus_level'], quantitative_result['overall_confidence']),
            metadata=metadata,
            summary_stats={
                'total_rooms': len(rooms_data),
                'models_used': len(model_responses),
                'consensus_tasks': sum(room_data['high_consensus_tasks'] for room_data in rooms_data),
                'safety_tasks': sum(room_data['safety_critical_tasks'] for room_data in rooms_data)
            }
        )
    
    def _create_empty_result(self) -> MergedEstimate:
        """빈 결과 생성"""
        return MergedEstimate(
            project_info={},
            rooms=[],
            total_work_items=0,
            overall_confidence=0.0,
            metadata=MergeMetadata(
                models_used=[],
                consensus_level=0.0,
                confidence_level=ConfidenceLevel.LOW,
                processing_time_total=0.0,
                manual_review_required=True
            )
        )