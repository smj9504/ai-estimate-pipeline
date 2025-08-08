# src/phases/phase1_processor.py
"""
Phase 1: Merge Measurement & Work Scope
- Phase 0의 출력을 입력으로 받음
- 멀티모델 사용하여 측정값과 작업 범위 병합
- Remove & Replace 로직 적용
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.prompt_manager import PromptManager
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.validators.estimation_validator import ComprehensiveValidator

class Phase1Processor:
    """
    Phase 1: Merge Measurement & Work Scope
    멀티모델을 사용하여 측정값과 작업 범위를 병합하고 검증
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prompt_manager = PromptManager()
        self.orchestrator = ModelOrchestrator()
        self.merger = ResultMerger(config)
        self.validator = ComprehensiveValidator()
    
    async def process(self,
                     phase0_output: Dict[str, Any],
                     models_to_use: List[str] = None,
                     project_id: Optional[str] = None,
                     prompt_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 1 실행 - 멀티모델로 측정값과 작업 범위 병합
        
        Args:
            phase0_output: Phase 0의 출력 (Generate Scope of Work 결과)
            models_to_use: 사용할 AI 모델 리스트
            project_id: 프로젝트 ID
            prompt_version: 사용할 프롬프트 버전 ('improved', 'fast', None=기본)
        
        Returns:
            병합된 측정값과 작업 범위 데이터
        """
        print(f"Phase 1 시작: Merge Measurement & Work Scope - 모델: {models_to_use}")
        
        try:
            # Phase 0 출력에서 실제 데이터 추출
            if 'data' in phase0_output:
                input_data = phase0_output['data']
            else:
                input_data = phase0_output
            
            # 기본 모델 설정
            if not models_to_use:
                models_to_use = ["gpt4", "claude", "gemini"]
            
            # 1. 프롬프트 로드 및 변수 치환
            prompt_variables = {
                'project_id': project_id or phase0_output.get('project_id', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'location': 'DMV area'
            }
            
            # 프롬프트 버전 결정 (설정 가능)
            # prompt_version: None(기본), 'improved', 'fast', 기타 커스텀 버전
            effective_version = prompt_version or self.config.get('prompt_version')
            
            if effective_version:
                print(f"프롬프트 버전: {effective_version}")
            else:
                print("프롬프트 버전: 기본 버전 사용")
            
            base_prompt = self.prompt_manager.load_prompt_with_variables(
                phase_number=1,
                variables=prompt_variables,
                version=effective_version  # 동적 버전 선택
            )
            
            # 2. 멀티모델 병렬 실행
            print(f"전체 처리 모드: 멀티모델 실행 중 - {models_to_use}")
            model_results = await self.orchestrator.run_parallel(
                prompt=base_prompt,
                json_data=input_data,
                model_names=models_to_use
            )
            
            if not model_results:
                raise ValueError("모든 모델 실행이 실패했습니다")
            
            print(f"{len(model_results)}개 모델 응답 수신")
            
            # 3. 결과 병합 (질적/정량적 병합)
            merged_result = self.merger.merge_results(model_results)
            
            # 4. Remove & Replace 로직 검증
            validation_result = await self._validate_remove_replace_logic(
                merged_result, 
                input_data
            )
            
            # 5. 최종 결과 구성
            result = {
                'phase': 1,
                'phase_name': 'Merge Measurement & Work Scope',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'models_responded': len(model_results),
                'project_id': prompt_variables['project_id'],
                'data': merged_result.model_dump(),
                'validation': validation_result,
                'confidence_score': merged_result.overall_confidence,
                'consensus_level': merged_result.metadata.consensus_level,
                'processing_time': sum(r.processing_time for r in model_results),
                'success': True
            }
            
            print(f"Phase 1 완료: 신뢰도 {merged_result.overall_confidence:.2f}")
            return result
            
        except Exception as e:
            print(f"Phase 1 오류: {e}")
            return {
                'phase': 1,
                'phase_name': 'Merge Measurement & Work Scope',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'error': str(e),
                'success': False
            }
    
    
    
    async def _validate_remove_replace_logic(self, 
                                            merged_result: Any,
                                            original_data: Any) -> Dict[str, Any]:
        """
        Remove & Replace 로직 검증
        
        Args:
            merged_result: 병합된 결과
            original_data: 원본 입력 데이터
        
        Returns:
            검증 결과
        """
        validation = {
            'remove_replace_logic': {'valid': True, 'issues': []},
            'measurements_accuracy': {'valid': True, 'issues': []},
            'special_tasks': {'valid': True, 'issues': []},
            'overall_valid': True
        }
        
        # 병합 결과가 dict인지 객체인지 확인
        if hasattr(merged_result, 'rooms'):
            rooms = merged_result.rooms
        elif isinstance(merged_result, dict) and 'rooms' in merged_result:
            rooms = merged_result['rooms']
        else:
            validation['overall_valid'] = False
            validation['error'] = 'Invalid merged result structure'
            return validation
        
        try:
            # Remove & Replace 로직 체크
            for room in rooms:
                # work_scope가 "Remove & Replace"인 항목 확인
                for material, work in room.get('work_scope', {}).items():
                    if 'Remove & Replace' in str(work):
                        # demo_scope에 이미 철거된 부분 확인
                        demo_scope = room.get('demo_scope(already demo\'d)', {})
                        
                        # 이미 철거된 부분은 제거 비용 제외되었는지 확인
                        if material in demo_scope and demo_scope[material] > 0:
                            # 작업 항목에서 이미 철거된 부분이 제외되었는지 체크
                            validation['remove_replace_logic']['issues'].append(
                                f"{room.get('name', 'Unknown')}: {material} - demo_scope 확인 필요"
                            )
                
                # 높은 천장 프리미엄 적용 확인 (9피트 초과)
                measurements = room.get('measurements', {})
                if measurements.get('height', 0) > 9:
                    # 벽과 천장 작업에 프리미엄이 적용되었는지 확인
                    validation['special_tasks']['issues'].append(
                        f"{room.get('name', 'Unknown')}: 높은 천장 프리미엄 적용 확인 필요"
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
    
    def prepare_for_phase2(self, phase1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1 결과를 Phase 2 입력 형식으로 변환
        
        Args:
            phase1_result: Phase 1 출력
        
        Returns:
            Phase 2용 입력 데이터
        """
        if not phase1_result.get('success'):
            raise ValueError("Phase 1이 성공적으로 완료되지 않았습니다")
        
        # Phase 1 데이터 구조 표준화
        phase1_data = phase1_result['data']
        
        # 표준 구조로 변환
        standardized_data = {
            'rooms': [],
            'overall_confidence': phase1_result.get('confidence_score', 0),
            'consensus_level': phase1_result.get('consensus_level', 0)
        }
        
        # rooms 데이터 추출 및 표준화
        if isinstance(phase1_data, dict):
            if 'rooms' in phase1_data:
                standardized_data['rooms'] = phase1_data['rooms']
            elif 'merged_work_scope' in phase1_data:
                # merged_work_scope를 rooms 형식으로 변환
                for room_name, tasks in phase1_data['merged_work_scope'].items():
                    room_data = {
                        'name': room_name,
                        'tasks': tasks if isinstance(tasks, list) else [],
                        'measurements': {},
                        'work_scope': {},
                        'materials': {}
                    }
                    standardized_data['rooms'].append(room_data)
        
        # Phase 2용 입력 데이터
        phase2_input = {
            'merged_scope': standardized_data,
            'project_id': phase1_result.get('project_id'),
            'phase1_confidence': phase1_result.get('confidence_score'),
            'phase1_validation': phase1_result.get('validation'),
            'metadata': {
                'phase1_timestamp': phase1_result.get('timestamp'),
                'phase1_models': phase1_result.get('models_used')
            }
        }
        
        return phase2_input