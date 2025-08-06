# src/phases/phase0_processor.py
"""
Phase 0: Generate Scope of Work
- 3가지 데이터(measurement, demolition_scope, intake_form)를 병합
- 단일 AI 모델 사용
- 최종 JSON 구조 생성
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.prompt_manager import PromptManager
from src.models.model_interface import ModelOrchestrator

class Phase0Processor:
    """
    Phase 0: Generate Scope of Work
    단일 모델을 사용하여 초기 데이터를 병합하고 표준 JSON 포맷 생성
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prompt_manager = PromptManager()
        self.orchestrator = ModelOrchestrator()
    
    async def process(self, 
                     measurement_data: Dict[str, Any],
                     demolition_scope_data: Dict[str, Any],
                     scope_of_work_intake_form: str,
                     model_to_use: str = "gpt4",
                     project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 0 실행 - 데이터 병합 및 JSON 생성
        
        Args:
            measurement_data: 측정 데이터 JSON
            demolition_scope_data: 철거 범위 데이터 JSON
            scope_of_work_intake_form: 작업 범위 입력 양식 텍스트
            model_to_use: 사용할 AI 모델 (기본: gpt4)
            project_id: 프로젝트 ID (템플릿 변수용)
        
        Returns:
            병합된 JSON 데이터
        """
        print(f"Phase 0 시작: Generate Scope of Work - 모델: {model_to_use}")
        
        try:
            # 1. 프롬프트 로드 및 변수 치환
            prompt_variables = {
                'project_id': project_id or f"PRJ-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'location': 'DMV area'
            }
            
            base_prompt = self.prompt_manager.load_prompt_with_variables(
                phase_number=0,
                variables=prompt_variables
            )
            
            # 2. 입력 데이터를 프롬프트에 포함
            # 프롬프트의 플레이스홀더를 실제 데이터로 치환
            full_prompt = base_prompt.replace(
                '{measurement_data}',
                json.dumps(measurement_data, indent=2, ensure_ascii=False)
            ).replace(
                '{demolition_scope_data}',
                json.dumps(demolition_scope_data, indent=2, ensure_ascii=False)
            ).replace(
                '{scope_of_work_intake_form}',
                scope_of_work_intake_form
            )
            
            # 3. 단일 모델 실행
            print(f"AI 모델 호출 중: {model_to_use}")
            
            # ModelOrchestrator의 run_single 메서드 사용 (없으면 추가 필요)
            # 일단 run_parallel을 단일 모델로 실행
            model_results = await self.orchestrator.run_parallel(
                prompt=full_prompt,
                json_data={},  # Phase 0은 프롬프트에 모든 데이터 포함
                models_to_use=[model_to_use]
            )
            
            if not model_results or len(model_results) == 0:
                raise ValueError(f"모델 {model_to_use} 실행 실패")
            
            # 4. 결과 파싱
            model_response = model_results[0]
            raw_response = model_response.raw_response
            
            # JSON 응답 추출
            merged_json = self._extract_json_from_response(raw_response)
            
            # 5. 검증 및 정리
            validated_json = self._validate_and_clean_output(merged_json)
            
            # 6. 메타데이터 추가
            result = {
                'phase': 0,
                'phase_name': 'Generate Scope of Work',
                'timestamp': datetime.now().isoformat(),
                'model_used': model_to_use,
                'project_id': prompt_variables['project_id'],
                'data': validated_json,
                'processing_time': model_response.processing_time,
                'success': True
            }
            
            print(f"Phase 0 완료: {len(validated_json)} 섹션 생성됨")
            return result
            
        except Exception as e:
            print(f"Phase 0 오류: {e}")
            return {
                'phase': 0,
                'phase_name': 'Generate Scope of Work',
                'timestamp': datetime.now().isoformat(),
                'model_used': model_to_use,
                'error': str(e),
                'success': False
            }
    
    def _extract_json_from_response(self, raw_response: str) -> Any:
        """
        AI 응답에서 JSON 추출
        
        Args:
            raw_response: AI 모델의 원시 응답
        
        Returns:
            파싱된 JSON 객체
        """
        try:
            # 직접 JSON 파싱 시도
            return json.loads(raw_response)
        except json.JSONDecodeError:
            # JSON 블록 찾기 (```json ... ``` 형태)
            import re
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, raw_response)
            
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            
            # [ 또는 { 로 시작하는 부분 찾기
            json_start_idx = raw_response.find('[')
            if json_start_idx == -1:
                json_start_idx = raw_response.find('{')
            
            if json_start_idx != -1:
                json_str = raw_response[json_start_idx:]
                # 마지막 ] 또는 } 찾기
                if json_str.startswith('['):
                    json_end_idx = json_str.rfind(']') + 1
                else:
                    json_end_idx = json_str.rfind('}') + 1
                
                json_str = json_str[:json_end_idx]
                return json.loads(json_str)
            
            raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")
    
    def _validate_and_clean_output(self, json_data: Any) -> Any:
        """
        생성된 JSON 검증 및 정리
        
        Args:
            json_data: 검증할 JSON 데이터
        
        Returns:
            검증되고 정리된 JSON 데이터
        """
        # 기본 구조 확인
        if not isinstance(json_data, list):
            raise ValueError("출력은 리스트 형태여야 합니다")
        
        if len(json_data) < 2:
            raise ValueError("최소 2개 섹션(jobsite info + floor data)이 필요합니다")
        
        # 첫 번째 요소는 jobsite 정보
        jobsite_info = json_data[0]
        required_jobsite_keys = ['Jobsite', 'occupancy', 'company']
        for key in required_jobsite_keys:
            if key not in jobsite_info:
                jobsite_info[key] = "" if key != 'company' else {}
        
        # 나머지 요소들은 floor 데이터
        for floor_data in json_data[1:]:
            if 'location' not in floor_data:
                raise ValueError("각 floor 데이터에는 'location' 키가 필요합니다")
            
            if 'rooms' not in floor_data:
                floor_data['rooms'] = []
            
            # 각 room 데이터 검증
            for room in floor_data['rooms']:
                self._validate_room_structure(room)
        
        return json_data
    
    def _validate_room_structure(self, room: Dict[str, Any]):
        """
        Room 데이터 구조 검증
        
        Args:
            room: 검증할 room 딕셔너리
        """
        # 필수 키 확인 및 기본값 설정
        required_sections = {
            'name': '',
            'material': {},
            'work_scope': {},
            'measurements': {},
            'demo_scope(already demo\'d)': {},
            'additional_notes': {'protection': [], 'detach_reset': []}
        }
        
        for key, default_value in required_sections.items():
            if key not in room:
                room[key] = default_value
        
        # measurements 필수 필드 확인
        measurement_fields = [
            'height', 'wall_area_sqft', 'ceiling_area_sqft', 
            'floor_area_sqft', 'walls_and_ceiling_area_sqft',
            'flooring_area_sy', 'ceiling_perimeter_lf', 
            'floor_perimeter_lf'
        ]
        
        for field in measurement_fields:
            if field not in room['measurements']:
                room['measurements'][field] = 0.00
        
        if 'openings' not in room['measurements']:
            room['measurements']['openings'] = []
    
    async def validate_input_data(self,
                                 measurement_data: Dict[str, Any],
                                 demolition_scope_data: Dict[str, Any],
                                 scope_of_work_intake_form: str) -> Dict[str, Any]:
        """
        입력 데이터 사전 검증
        
        Args:
            measurement_data: 측정 데이터
            demolition_scope_data: 철거 범위 데이터
            scope_of_work_intake_form: 작업 범위 양식
        
        Returns:
            검증 결과
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Measurement data 검증
        if not isinstance(measurement_data, (dict, list)):
            validation_result['valid'] = False
            validation_result['errors'].append("measurement_data는 dict 또는 list여야 합니다")
        
        # Demolition scope 검증
        if not isinstance(demolition_scope_data, dict):
            validation_result['valid'] = False
            validation_result['errors'].append("demolition_scope_data는 dict여야 합니다")
        
        # Intake form 검증
        if not scope_of_work_intake_form or not isinstance(scope_of_work_intake_form, str):
            validation_result['valid'] = False
            validation_result['errors'].append("scope_of_work_intake_form은 비어있지 않은 문자열이어야 합니다")
        
        # 데이터 일관성 체크
        if validation_result['valid']:
            # Room 이름 매칭 체크 등 추가 검증 가능
            pass
        
        return validation_result