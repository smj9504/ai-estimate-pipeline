# src/phases/phase2_processor.py
"""
Phase 2: Quantity Survey
- Phase 1의 출력을 입력으로 받음
- 멀티모델 사용하여 정확한 수량 산출
- 작업별 상세 수량 및 단위 계산
"""
import json
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.prompt_manager import PromptManager
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.utils.logger import get_logger, log_phase_start, log_phase_end, log_error

class Phase2Processor:
    """
    Phase 2: Quantity Survey
    멀티모델을 사용하여 정확한 수량 산출 및 병합
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prompt_manager = PromptManager()
        self.orchestrator = ModelOrchestrator()
        self.merger = ResultMerger(config)
        self.logger = get_logger('phase2_processor')
    
    async def process(self,
                     phase1_output: Dict[str, Any],
                     models_to_use: List[str] = None,
                     project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 2 실행 - 멀티모델로 수량 산출
        
        Args:
            phase1_output: Phase 1의 출력 (Merge Measurement & Work Scope 결과)
            models_to_use: 사용할 AI 모델 리스트
            project_id: 프로젝트 ID
        
        Returns:
            상세 수량이 포함된 견적 데이터
        """
        start_time = time.time()
        log_phase_start(2, "Quantity Survey", models=models_to_use)
        self.logger.info(f"Phase 2 시작: Quantity Survey - 모델: {models_to_use}")
        
        try:
            # Phase 1 출력에서 데이터 추출
            if 'data' in phase1_output:
                input_data = phase1_output['data']
            elif 'merged_scope' in phase1_output:
                input_data = phase1_output['merged_scope']
            else:
                input_data = phase1_output
            
            # 기본 모델 설정
            if not models_to_use:
                models_to_use = ["gpt4", "claude", "gemini"]
            
            # 1. 프롬프트 로드 및 변수 치환
            prompt_variables = {
                'project_id': project_id or phase1_output.get('project_id', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'location': 'DMV area',
                'phase1_confidence': phase1_output.get('confidence_score', 'N/A')
            }
            
            base_prompt = self.prompt_manager.load_prompt_with_variables(
                phase_number=2,
                variables=prompt_variables
            )
            
            # 2. 멀티모델 병렬 실행
            self.logger.info(f"멀티모델 실행 중: {models_to_use}")
            model_results = await self.orchestrator.run_parallel(
                prompt=base_prompt,
                json_data=input_data,
                model_names=models_to_use
            )
            
            if not model_results:
                raise ValueError("모든 모델 실행이 실패했습니다")
            
            self.logger.info(f"{len(model_results)}개 모델 응답 수신")
            
            # 3. 수량 데이터 병합 (정량적 병합 중심)
            merged_result = await self._merge_quantity_results(model_results)
            
            # 4. 수량 검증 및 일관성 체크
            validation_result = await self._validate_quantities(merged_result, input_data)
            
            # 5. 최종 결과 구성
            result = {
                'phase': 2,
                'phase_name': 'Quantity Survey',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'models_responded': len(model_results),
                'project_id': prompt_variables['project_id'],
                'data': merged_result,
                'validation': validation_result,
                'confidence_score': self._calculate_confidence(merged_result, validation_result),
                'processing_time': sum(r.processing_time for r in model_results),
                'phase3_ready': validation_result.get('overall_valid', False),
                'success': True
            }
            
            duration = time.time() - start_time
            log_phase_end(2, "Quantity Survey", True, duration)
            self.logger.info(f"Phase 2 완료: 수량 산출 완료, Phase 3 준비 상태: {result['phase3_ready']}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            log_phase_end(2, "Quantity Survey", False, duration)
            log_error('phase2_processor', e, {
                'models': models_to_use,
                'project_id': project_id
            })
            self.logger.error(f"Phase 2 오류: {e}")
            return {
                'phase': 2,
                'phase_name': 'Quantity Survey',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'error': str(e),
                'success': False
            }
    
    async def _merge_quantity_results(self, model_results: List[Any]) -> Dict[str, Any]:
        """
        수량 중심의 결과 병합
        
        Args:
            model_results: 각 모델의 응답 리스트
        
        Returns:
            병합된 수량 데이터
        """
        # ResultMerger를 사용하되, 수량 데이터에 특화된 병합
        merged = self.merger.merge_results(model_results)
        
        # merged가 MergedEstimate 객체인 경우 처리
        if hasattr(merged, 'metadata'):
            consensus_level = merged.metadata.consensus_level if hasattr(merged.metadata, 'consensus_level') else 0
            confidence_score = merged.overall_confidence if hasattr(merged, 'overall_confidence') else 0
        else:
            consensus_level = 0
            confidence_score = 0
        
        # 수량 데이터 구조화
        quantity_data = {
            'rooms': [],
            'summary': {
                'total_items': 0,
                'consensus_level': consensus_level,
                'confidence_score': confidence_score
            }
        }
        
        # 각 방별 수량 정리
        rooms = merged.rooms if hasattr(merged, 'rooms') else []
        for room in rooms:
            room_quantity = {
                'name': room.get('name', 'Unknown'),
                'line_items': [],
                'measurements': room.get('measurements', {}),
                'totals': {
                    'labor_hours': 0,
                    'material_sqft': 0,
                    'disposal_cuyd': 0
                }
            }
            
            # 작업 항목별 수량 추출 - tasks 또는 work_items 처리
            tasks = room.get('tasks', []) or room.get('work_items', [])
            for task in tasks:
                line_item = {
                    'description': task.get('description', ''),
                    'quantity': task.get('quantity', 0),
                    'unit': task.get('unit', 'EA'),
                    'labor_hours': task.get('labor_hours', 0),
                    'material_needed': task.get('material_needed', 0),
                    'notes': task.get('notes', '')
                }
                room_quantity['line_items'].append(line_item)
                
                # 합계 계산
                room_quantity['totals']['labor_hours'] += line_item.get('labor_hours', 0)
                if 'sqft' in line_item.get('unit', '').lower():
                    room_quantity['totals']['material_sqft'] += line_item.get('quantity', 0)
            
            quantity_data['rooms'].append(room_quantity)
            quantity_data['summary']['total_items'] += len(room_quantity['line_items'])
        
        return quantity_data
    
    async def _validate_quantities(self, 
                                  quantity_data: Dict[str, Any],
                                  original_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        수량 데이터 검증
        
        Args:
            quantity_data: 병합된 수량 데이터
            original_data: 원본 데이터
        
        Returns:
            검증 결과
        """
        validation = {
            'quantity_consistency': {'valid': True, 'issues': []},
            'unit_accuracy': {'valid': True, 'issues': []},
            'completeness': {'valid': True, 'issues': []},
            'overall_valid': True
        }
        
        try:
            for room in quantity_data.get('rooms', []):
                room_name = room.get('name', 'Unknown')
                
                # 수량 일관성 체크
                measurements = room.get('measurements', {})
                floor_area = measurements.get('floor_area_sqft', 0)
                
                for item in room.get('line_items', []):
                    # 바닥재 수량이 실제 면적보다 큰지 체크
                    if 'flooring' in item.get('description', '').lower():
                        if item.get('quantity', 0) > floor_area * 1.1:  # 10% 여유 허용
                            validation['quantity_consistency']['issues'].append(
                                f"{room_name}: 바닥재 수량이 실제 면적보다 과도함"
                            )
                    
                    # 단위 검증
                    if not item.get('unit'):
                        validation['unit_accuracy']['issues'].append(
                            f"{room_name}: {item.get('description', 'Unknown')} - 단위 누락"
                        )
                
                # 필수 항목 확인
                if not room.get('line_items'):
                    validation['completeness']['issues'].append(
                        f"{room_name}: 작업 항목이 없음"
                    )
            
            # 전체 유효성 판단
            for category in validation.values():
                if isinstance(category, dict) and category.get('issues'):
                    category['valid'] = False
                    validation['overall_valid'] = False
            
        except Exception as e:
            validation['overall_valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def _calculate_confidence(self, 
                            quantity_data: Dict[str, Any],
                            validation: Dict[str, Any]) -> float:
        """
        수량 데이터의 신뢰도 계산
        
        Args:
            quantity_data: 수량 데이터
            validation: 검증 결과
        
        Returns:
            신뢰도 점수 (0-1)
        """
        base_confidence = quantity_data.get('summary', {}).get('confidence_score', 0.5)
        
        # 검증 결과에 따른 조정
        if validation.get('overall_valid'):
            confidence = base_confidence
        else:
            # 문제가 있을 때마다 신뢰도 감소
            issue_count = sum(
                len(v.get('issues', [])) 
                for v in validation.values() 
                if isinstance(v, dict)
            )
            confidence = max(0.3, base_confidence - (issue_count * 0.05))
        
        return min(1.0, confidence)
    
    def prepare_for_phase3(self, phase2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2 결과를 Phase 3 입력 형식으로 변환
        
        Args:
            phase2_result: Phase 2 출력
        
        Returns:
            Phase 3용 입력 데이터 (Market Research용)
        """
        if not phase2_result.get('success'):
            raise ValueError("Phase 2가 성공적으로 완료되지 않았습니다")
        
        # Phase 2의 수량 데이터를 Phase 3 형식으로 정리
        phase3_input = {
            'quantity_survey': phase2_result['data'],
            'project_id': phase2_result.get('project_id'),
            'location': 'DMV area',  # Market research를 위한 지역 정보
            'phase2_confidence': phase2_result.get('confidence_score'),
            'line_items_count': phase2_result['data'].get('summary', {}).get('total_items', 0),
            'metadata': {
                'phase2_timestamp': phase2_result.get('timestamp'),
                'phase2_models': phase2_result.get('models_used')
            }
        }
        
        return phase3_input