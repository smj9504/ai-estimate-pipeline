# src/phases/phase0_processor.py
"""
Phase 0: Generate Scope of Work
- 3가지 데이터(measurement, demolition_scope, intake_form)를 병합
- 단일 AI 모델 사용
- 최종 JSON 구조 생성
"""
import json
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.prompt_manager import PromptManager
from src.models.model_interface import ModelOrchestrator, GPT4Interface, ClaudeInterface, GeminiInterface
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, log_phase_start, log_phase_end, log_error, log_json

class Phase0Processor:
    """
    Phase 0: Generate Scope of Work
    단일 모델을 사용하여 초기 데이터를 병합하고 표준 JSON 포맷 생성
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prompt_manager = PromptManager()
        self.orchestrator = ModelOrchestrator()
        self.logger = get_logger('phase0_processor')
        self.config_loader = ConfigLoader()
        self.available_models = {}
    
    async def process(self, 
                     measurement_data: Dict[str, Any],
                     demolition_scope_data: Dict[str, Any],
                     intake_form: str,
                     model_to_use: str = "gemini",  # Gemini를 기본값으로 변경 (빠른 응답)
                     project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 0 실행 - 데이터 병합 및 JSON 생성
        
        Args:
            measurement_data: 측정 데이터 JSON
            demolition_scope_data: 철거 범위 데이터 JSON
            intake_form: 작업 범위 입력 양식 텍스트
            model_to_use: 사용할 AI 모델 (기본: gemini, 옵션: gpt4, claude, gemini)
            project_id: 프로젝트 ID (템플릿 변수용)
        
        Returns:
            병합된 JSON 데이터
        """
        start_time = time.time()
        log_phase_start(0, "Generate Scope of Work", model=model_to_use)
        
        self.logger.info(f"입력 데이터 크기:")
        self.logger.info(f"  - Measurement: {len(json.dumps(measurement_data))} bytes")
        self.logger.info(f"  - Demolition: {len(json.dumps(demolition_scope_data))} bytes")
        self.logger.info(f"  - Intake Form: {len(intake_form)} characters")
        
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
            
            self.logger.debug(f"프롬프트 템플릿 로드 완료: {len(base_prompt)} characters")
            
            # 2. 입력 데이터를 프롬프트에 포함
            # 프롬프트의 플레이스홀더를 실제 데이터로 치환
            full_prompt = base_prompt
            
            # 각 플레이스홀더를 실제 데이터로 치환
            if '{measurement_data}' in full_prompt:
                full_prompt = full_prompt.replace(
                    '{measurement_data}',
                    json.dumps(measurement_data, indent=2, ensure_ascii=False)
                )
                self.logger.debug("measurement_data 치환 완료")
            else:
                self.logger.warning("{measurement_data} 플레이스홀더를 찾을 수 없습니다")
            
            if '{demolition_scope_data}' in full_prompt:
                full_prompt = full_prompt.replace(
                    '{demolition_scope_data}',
                    json.dumps(demolition_scope_data, indent=2, ensure_ascii=False)
                )
                self.logger.debug("demolition_scope_data 치환 완료")
            else:
                self.logger.warning("{demolition_scope_data} 플레이스홀더를 찾을 수 없습니다")
            
            if '{intake_form}' in full_prompt:
                full_prompt = full_prompt.replace(
                    '{intake_form}',
                    intake_form
                )
                self.logger.debug("intake_form 치환 완료")
            else:
                self.logger.warning("{intake_form} 플레이스홀더를 찾을 수 없습니다")
            
            self.logger.info(f"최종 프롬프트 크기: {len(full_prompt)} characters")
            
            # 3. 단일 모델 실행
            self.logger.info(f"AI 모델 호출 중: {model_to_use}")
            
            # 모델 선택 및 직접 호출 (더 빠른 응답을 위해)
            model_response = await self._call_single_model(
                model_name=model_to_use,
                prompt=full_prompt
            )
            
            if not model_response:
                raise ValueError(f"모델 {model_to_use} 실행 실패 - 응답 없음")
            
            # 4. 결과 파싱
            raw_response = model_response.raw_response
            
            self.logger.debug(f"모델 응답 크기: {len(raw_response)} characters")
            self.logger.debug(f"응답 처음 500자: {raw_response[:500]}...")
            
            # 에러 응답 체크
            if "Error:" in raw_response or "Request timed out" in raw_response:
                self.logger.error(f"모델 에러 응답: {raw_response}")
                raise ValueError(f"모델 응답 에러: {raw_response[:200]}")
            
            # JSON 응답 추출
            self.logger.info("JSON 추출 시도 중...")
            merged_json = self._extract_json_from_response(raw_response)
            
            self.logger.info(f"JSON 추출 성공: {type(merged_json)}")
            
            # 5. 검증 및 정리 (intake_form 전달하여 프로젝트 정보 추출)
            validated_json = self._validate_and_clean_output(merged_json, intake_form)
            
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
            
            duration = time.time() - start_time
            log_phase_end(0, "Generate Scope of Work", True, duration)
            self.logger.info(f"Phase 0 완료: {len(validated_json)} 섹션 생성됨")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            log_phase_end(0, "Generate Scope of Work", False, duration)
            log_error('phase0_processor', e, {
                'model': model_to_use,
                'project_id': project_id
            })
            
            return {
                'phase': 0,
                'phase_name': 'Generate Scope of Work',
                'timestamp': datetime.now().isoformat(),
                'model_used': model_to_use,
                'error': str(e),
                'success': False
            }
    
    async def _call_single_model(self, model_name: str, prompt: str) -> Any:
        """
        단일 모델 직접 호출 (오케스트레이터 대신 직접 호출로 속도 개선)
        
        Args:
            model_name: 모델 이름 (gpt4, claude, gemini)
            prompt: 프롬프트
            
        Returns:
            ModelResponse 객체
        """
        from src.models.data_models import ModelResponse
        import google.generativeai as genai
        
        try:
            if model_name.lower() == 'gemini':
                # Gemini 직접 호출 (가장 빠름)
                api_keys = self.config_loader.get_api_keys()
                api_key = api_keys.get('google')
                if not api_key:
                    raise ValueError("Google API 키가 설정되지 않았습니다.")
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')  # 빠른 모델
                
                self.logger.info("Gemini 1.5 Flash 모델 사용")
                
                # 동기 호출을 비동기로 래핑
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=3000,
                        temperature=0.1
                    )
                )
                
                return ModelResponse(
                    model_name='gemini-1.5-flash',
                    raw_response=response.text,
                    processing_time=0,
                    total_work_items=0,
                    room_estimates=[],
                    confidence_self_assessment=0.85
                )
                
            elif model_name.lower() == 'gpt4':
                # GPT-4 호출
                api_keys = self.config_loader.get_api_keys()
                api_key = api_keys.get('openai')
                if not api_key:
                    raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
                
                model = GPT4Interface(api_key)
                return await model.call_model(prompt, {})
                
            elif model_name.lower() == 'claude':
                # Claude 호출
                api_keys = self.config_loader.get_api_keys()
                api_key = api_keys.get('anthropic')
                if not api_key:
                    raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
                
                model = ClaudeInterface(api_key)
                return await model.call_model(prompt, {})
                
            else:
                # 지원하지 않는 모델인 경우 기존 오케스트레이터 사용
                self.logger.warning(f"알 수 없는 모델 {model_name}, 오케스트레이터 사용")
                model_results = await self.orchestrator.run_parallel(
                    prompt=prompt,
                    json_data={},
                    model_names=[model_name]
                )
                return model_results[0] if model_results else None
                
        except asyncio.TimeoutError:
            self.logger.error(f"{model_name} 타임아웃")
            return ModelResponse(
                model_name=model_name,
                raw_response="Error: Timeout",
                processing_time=0,
                total_work_items=0,
                room_estimates=[],
                confidence_self_assessment=0.0
            )
        except Exception as e:
            self.logger.error(f"{model_name} 호출 오류: {e}")
            return ModelResponse(
                model_name=model_name,
                raw_response=f"Error: {str(e)}",
                processing_time=0,
                total_work_items=0,
                room_estimates=[],
                confidence_self_assessment=0.0
            )
    
    def _extract_json_from_response(self, raw_response: str) -> Any:
        """
        AI 응답에서 JSON 추출
        
        Args:
            raw_response: AI 모델의 원시 응답
        
        Returns:
            파싱된 JSON 객체
        """
        import re
        
        self.logger.debug("JSON 추출 시작...")
        
        # 1. 직접 JSON 파싱 시도
        try:
            self.logger.debug("직접 JSON 파싱 시도")
            result = json.loads(raw_response)
            self.logger.debug("직접 JSON 파싱 성공")
            return result
        except json.JSONDecodeError as e:
            self.logger.debug(f"직접 파싱 실패: {e}")
        
        # 2. JSON 블록 찾기 (```json ... ``` 형태)
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, raw_response)
        
        if match:
            json_str = match.group(1)
            self.logger.debug(f"JSON 블록 발견: {len(json_str)} characters")
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.debug(f"JSON 블록 파싱 실패: {e}")
        
        # 3. [ 또는 { 로 시작하는 부분 찾기
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
            
            self.logger.debug(f"JSON 구조 발견 (위치 {json_start_idx}): {len(json_str)} characters")
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON 파싱 최종 실패: {e}")
                self.logger.debug(f"파싱 시도한 문자열 (처음 500자): {json_str[:500]}")
        
        self.logger.error("응답에서 JSON을 찾을 수 없음")
        self.logger.debug(f"전체 응답 (처음 1000자): {raw_response[:1000]}")
        raise ValueError("응답에서 유효한 JSON을 찾을 수 없습니다")
    
    def _extract_project_info_from_intake(self, intake_form: str) -> Dict[str, Any]:
        """
        Intake form 텍스트에서 프로젝트 정보 추출
        
        Args:
            intake_form: 작업 범위 입력 양식 텍스트
        
        Returns:
            추출된 프로젝트 정보
        """
        import re
        
        project_info = {
            'Jobsite': '',
            'occupancy': '',
            'company': {}
        }
        
        # Property Address 추출
        address_match = re.search(r'Property Address:\s*(.+?)(?:\n|$)', intake_form)
        if address_match:
            project_info['Jobsite'] = address_match.group(1).strip()
            self.logger.debug(f"Extracted Jobsite: {project_info['Jobsite']}")
        
        # Occupancy 추출
        occupancy_match = re.search(r'Occupancy:\s*(.+?)(?:\n|$)', intake_form)
        if occupancy_match:
            project_info['occupancy'] = occupancy_match.group(1).strip()
            self.logger.debug(f"Extracted Occupancy: {project_info['occupancy']}")
        
        return project_info
    
    def _validate_and_clean_output(self, json_data: Any, intake_form: str = None) -> Any:
        """
        생성된 JSON 검증 및 정리
        
        Args:
            json_data: 검증할 JSON 데이터
            intake_form: 원본 intake form 텍스트 (프로젝트 정보 추출용)
        
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
        
        # intake form에서 프로젝트 정보 추출 시도
        if intake_form and (not jobsite_info.get('Jobsite') or not jobsite_info.get('occupancy')):
            extracted_info = self._extract_project_info_from_intake(intake_form)
            if extracted_info['Jobsite'] and not jobsite_info.get('Jobsite'):
                jobsite_info['Jobsite'] = extracted_info['Jobsite']
            if extracted_info['occupancy'] and not jobsite_info.get('occupancy'):
                jobsite_info['occupancy'] = extracted_info['occupancy']
        
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