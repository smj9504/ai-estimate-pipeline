# src/models/model_interface.py
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import openai
from anthropic import Anthropic
import google.generativeai as genai

from src.models.data_models import ModelResponse
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, log_model_call, log_error

class AIModelInterface(ABC):
    """AI 모델 인터페이스 추상 클래스"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = 3
        self.timeout = 90  # 90초로 증가 (방별 처리 시 더 많은 시간 필요)
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """모델 호출 추상 메서드"""
        pass
    
    def _prepare_prompt(self, base_prompt: str, json_data: Dict[str, Any]) -> str:
        """프롬프트 준비"""
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        return f"{base_prompt}\n\n[JSON DATA]\n{json_str}"
    
    def _extract_response_data(self, raw_response: str) -> Dict[str, Any]:
        """응답에서 구조화된 데이터 추출 - Phase 1/2 지원"""
        try:
            # 응답이 비어있거나 에러인 경우 먼저 체크
            if not raw_response or len(raw_response.strip()) < 10:
                self.logger.warning(f"응답이 너무 짧거나 비어있음: {raw_response[:100]}")
                return {'work_items': [], 'rooms': [], 'parse_error': 'Empty response'}
            
            # 1. JSON 응답 처리 시도
            parsed_json = self._try_parse_json(raw_response)
            if parsed_json:
                result = self._process_structured_response(parsed_json)
                self.logger.info(f"JSON 파싱 성공: {len(result.get('rooms', []))} 방, {len(result.get('work_items', []))} 작업")
                return result
            
            # 2. 텍스트 응답에서 구조화된 데이터 추출
            result = self._parse_text_response(raw_response)
            self.logger.info(f"텍스트 파싱 완료: {len(result.get('rooms', []))} 방, {len(result.get('work_items', []))} 작업")
            return result
            
        except Exception as e:
            self.logger.error(f"응답 파싱 오류: {e}")
            self.logger.debug(f"원본 응답 (처음 500자): {raw_response[:500]}")
            return {
                'work_items': [],
                'rooms': [],
                'parse_error': str(e),
                'raw_text': raw_response
            }
    
    def _try_parse_json(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """JSON 파싱 시도 (여러 형태의 JSON 지원)"""
        response = raw_response.strip()
        
        # 직접 JSON인 경우
        if response.startswith('{') or response.startswith('['):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        
        # 코드 블록 안에 JSON이 있는 경우
        import re
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'`(\{.*?\})`',
            r'(\{[\s\S]*\})'  # 마지막 시도: 가장 큰 JSON 객체
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _process_structured_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """구조화된 JSON 응답 처리"""
        result = {
            'work_items': [],
            'rooms': [],
            'phase_info': {}
        }
        
        # Phase 정보 추출
        if 'phase' in data:
            result['phase_info'] = {
                'phase': data.get('phase'),
                'phase_name': data.get('phase_name', ''),
                'model_used': data.get('model_used', ''),
                'timestamp': data.get('timestamp', '')
            }
        
        # GPT-4 structured output 형식 (phase 포함)
        if 'phase' in data and data.get('phase') in ['phase1_work_scope', '1', 'phase1']:
            # 이미 올바른 구조화된 응답
            for room in data.get('rooms', []):
                processed_room = {
                    'name': room.get('room_name', '알 수 없는 방'),
                    'tasks': []
                }
                
                # tasks 배열 직접 처리
                for task in room.get('tasks', []):
                    normalized_task = {
                        'task_name': task.get('task_name', ''),
                        'description': task.get('notes', ''),
                        'necessity': 'required',
                        'quantity': task.get('quantity', 0.0),
                        'unit': task.get('unit', ''),
                        'room_name': room.get('room_name', ''),
                        'reasoning': task.get('reasoning', task.get('notes', '')),
                        'task_type': task.get('task_type', ''),
                        'material_category': task.get('material_category', '')
                    }
                    processed_room['tasks'].append(normalized_task)
                    result['work_items'].append(normalized_task)
                
                result['rooms'].append(processed_room)
        
        # 프로젝트 데이터 구조인 경우 (Phase 0/1 형태)
        elif 'data' in data and isinstance(data['data'], list):
            project_data = data['data']
            if len(project_data) > 1:
                # floors 데이터에서 rooms 추출
                for floor_data in project_data[1:]:
                    if 'rooms' in floor_data:
                        for room in floor_data['rooms']:
                            processed_room = self._process_room_data(room)
                            result['rooms'].append(processed_room)
                            result['work_items'].extend(processed_room.get('tasks', []))
        
        # 직접 rooms 배열인 경우
        elif 'rooms' in data:
            for room in data['rooms']:
                processed_room = self._process_room_data(room)
                result['rooms'].append(processed_room)
                result['work_items'].extend(processed_room.get('tasks', []))
        
        # work_items 직접 포함인 경우
        elif 'work_items' in data:
            result['work_items'] = data['work_items']
            # work_items에서 rooms 재구성
            result['rooms'] = self._group_work_items_by_room(data['work_items'])
        
        # 단일 작업 목록인 경우 (리스트 형태)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and ('task_name' in item or 'name' in item):
                    result['work_items'].append(self._normalize_work_item(item))
            result['rooms'] = self._group_work_items_by_room(result['work_items'])
        
        return result
    
    def _process_room_data(self, room_data: Dict[str, Any]) -> Dict[str, Any]:
        """방 데이터 처리 및 작업 항목 추출"""
        room_name = room_data.get('room_name') or room_data.get('name', '알 수 없는 방')
        
        processed_room = {
            'name': room_name,
            'material': room_data.get('material', {}),
            'work_scope': room_data.get('work_scope', {}),
            'measurements': room_data.get('measurements', {}),
            'demo_scope': room_data.get('demo_scope(already demo\'d)', {}),
            'additional_notes': room_data.get('additional_notes', {}),
            'tasks': []
        }
        
        # 먼저 tasks 필드가 직접 있는지 확인 (GPT-4 structured output)
        if 'tasks' in room_data and isinstance(room_data['tasks'], list):
            for task in room_data['tasks']:
                normalized_task = self._normalize_work_item(task)
                normalized_task['room'] = room_name
                processed_room['tasks'].append(normalized_task)
        
        # tasks가 없으면 work_scope에서 생성 시도
        elif room_data.get('work_scope'):
            tasks = self._extract_tasks_from_room(room_data, room_name)
            processed_room['tasks'] = tasks
        
        # Phase 2: 수량 정보가 포함된 경우 처리
        if 'quantity_estimates' in room_data:
            self._add_quantity_info(processed_room['tasks'], room_data['quantity_estimates'])
        
        return processed_room
    
    def _extract_tasks_from_room(self, room_data: Dict[str, Any], room_name: str) -> List[Dict[str, Any]]:
        """방 데이터에서 작업 항목 추출 (Remove & Replace 로직 적용)"""
        tasks = []
        work_scope = room_data.get('work_scope', {})
        measurements = room_data.get('measurements', {})
        demo_scope = room_data.get('demo_scope(already demo\'d)', {})
        
        # 각 작업 영역별로 처리
        scope_mappings = {
            'Flooring': 'floor_area_sqft',
            'Wall': 'wall_area_sqft', 
            'Ceiling': 'ceiling_area_sqft',
            'Baseboard': 'floor_perimeter_lf',
            'Quarter Round': 'floor_perimeter_lf'
        }
        
        for scope_type, area_key in scope_mappings.items():
            scope_value = work_scope.get(scope_type, '').strip()
            if not scope_value or scope_value.lower() in ['', 'n/a', 'none']:
                continue
            
            # 측정값 가져오기
            area_value = measurements.get(area_key, 0.0)
            unit = 'sqft' if 'area' in area_key else 'lf'
            
            # Remove & Replace 로직 적용
            if scope_value == "Remove & Replace":
                # 철거량 확인
                demo_amount = self._get_demo_amount(demo_scope, scope_type)
                remaining_area = max(0, area_value - demo_amount)
                
                # 제거 작업 (남은 부분만)
                if remaining_area > 0:
                    tasks.append({
                        'task_name': f'Remove existing {scope_type.lower()}',
                        'description': f'Remove existing {scope_type.lower()} material',
                        'necessity': 'required',
                        'quantity': remaining_area,
                        'unit': unit,
                        'room_name': room_name,
                        'reasoning': f'Remove & Replace scope - Demo already done: {demo_amount} {unit}'
                    })
                
                # 설치 작업 (전체 면적)
                tasks.append({
                    'task_name': f'Install new {scope_type.lower()}',
                    'description': f'Install new {scope_type.lower()} material',
                    'necessity': 'required',
                    'quantity': area_value,
                    'unit': unit,
                    'room_name': room_name,
                    'reasoning': f'Remove & Replace scope - Full area installation required'
                })
            
            elif scope_value in ["Paint", "Patch"]:
                # 페인트/패치 작업
                tasks.append({
                    'task_name': f'{scope_value} {scope_type.lower()}',
                    'description': f'{scope_value} {scope_type.lower()} surface',
                    'necessity': 'required',
                    'quantity': area_value,
                    'unit': unit,
                    'room_name': room_name,
                    'reasoning': f'{scope_value} work specified'
                })
        
        # 추가 작업 (보호, 분리/재설치 등)
        additional_tasks = self._extract_additional_tasks(room_data, room_name)
        tasks.extend(additional_tasks)
        
        return tasks
    
    def _get_demo_amount(self, demo_scope: Dict[str, Any], scope_type: str) -> float:
        """철거 완료된 수량 확인"""
        demo_mappings = {
            'Wall': 'Wall Drywall(sq_ft)',
            'Ceiling': 'Ceiling Drywall(sq_ft)'
        }
        
        demo_key = demo_mappings.get(scope_type)
        if demo_key:
            return demo_scope.get(demo_key, 0.0)
        return 0.0
    
    def _extract_additional_tasks(self, room_data: Dict[str, Any], room_name: str) -> List[Dict[str, Any]]:
        """추가 작업 항목 추출 (보호, 분리/재설치 등)"""
        tasks = []
        additional_notes = room_data.get('additional_notes', {})
        
        # 보호 작업
        protection_items = additional_notes.get('protection', [])
        for item in protection_items:
            tasks.append({
                'task_name': f'Protection: {item}',
                'description': f'Provide {item}',
                'necessity': 'required',
                'quantity': 1,
                'unit': 'item',
                'room_name': room_name,
                'reasoning': 'Protection requirement specified'
            })
        
        # 분리/재설치 작업
        detach_reset_items = additional_notes.get('detach_reset', [])
        for item in detach_reset_items:
            tasks.extend([
                {
                    'task_name': f'Detach {item}',
                    'description': f'Carefully detach {item}',
                    'necessity': 'required',
                    'quantity': 1,
                    'unit': 'item',
                    'room_name': room_name,
                    'reasoning': 'Detach/reset requirement specified'
                },
                {
                    'task_name': f'Reset {item}',
                    'description': f'Reinstall {item}',
                    'necessity': 'required',
                    'quantity': 1,
                    'unit': 'item',
                    'room_name': room_name,
                    'reasoning': 'Detach/reset requirement specified'
                }
            ])
        
        return tasks
    
    def _parse_text_response(self, raw_response: str) -> Dict[str, Any]:
        """텍스트 응답 파싱 (향상된 버전)"""
        lines = raw_response.split('\n')
        work_items = []
        current_room = ""
        current_context = {}
        
        # 패턴 인식을 위한 정규식
        import re
        room_pattern = re.compile(r'(?:room|kitchen|bathroom|bedroom|living|dining)\s*:?\s*([^,\n]*)', re.IGNORECASE)
        task_pattern = re.compile(r'^[\s]*[-*•]|^[\s]*\d+[\.]|^[\s]*\w+:')
        quantity_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(sqft|sq\s*ft|lf|sf|sy)', re.IGNORECASE)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 방 이름 감지
            room_match = room_pattern.search(line)
            if room_match:
                current_room = room_match.group(1).strip() or room_match.group(0).strip()
                continue
            
            # 작업 항목 감지
            if task_pattern.match(line):
                task_text = re.sub(r'^[\s]*[-*•\d\.]+\s*', '', line).strip()
                if not task_text:
                    continue
                
                # 수량 정보 추출
                quantity = 0.0
                unit = ""
                quantity_match = quantity_pattern.search(task_text)
                if quantity_match:
                    quantity = float(quantity_match.group(1))
                    unit = quantity_match.group(2).replace(' ', '').lower()
                
                work_item = {
                    'task_name': task_text,
                    'description': task_text,
                    'necessity': 'required',
                    'quantity': quantity,
                    'unit': unit,
                    'room_name': current_room,
                    'reasoning': 'Extracted from text response'
                }
                
                work_items.append(work_item)
        
        # 방별로 그룹화
        rooms = self._group_work_items_by_room(work_items)
        
        return {
            'work_items': work_items,
            'rooms': rooms
        }
    
    def _group_work_items_by_room(self, work_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """작업 항목을 방별로 그룹화"""
        room_groups = {}
        
        for item in work_items:
            room_name = item.get('room_name', '미분류')
            if room_name not in room_groups:
                room_groups[room_name] = {
                    'name': room_name,
                    'tasks': []
                }
            room_groups[room_name]['tasks'].append(item)
        
        return list(room_groups.values())
    
    def _normalize_work_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """작업 항목 정규화"""
        return {
            'task_name': item.get('task_name') or item.get('name', '알 수 없는 작업'),
            'description': item.get('description', item.get('task_name', item.get('name', ''))),
            'necessity': item.get('necessity', 'required'),
            'quantity': item.get('quantity', 0.0),
            'unit': item.get('unit', ''),
            'room_name': item.get('room_name', ''),
            'reasoning': item.get('reasoning', '')
        }
    
    def _add_quantity_info(self, tasks: List[Dict[str, Any]], quantity_estimates: Dict[str, Any]):
        """Phase 2 수량 정보 추가"""
        # 수량 정보가 있는 경우 기존 작업에 추가
        for task in tasks:
            task_name = task.get('task_name', '').lower()
            
            # 매칭되는 수량 정보 찾기
            for qty_key, qty_value in quantity_estimates.items():
                if any(keyword in task_name for keyword in qty_key.lower().split()):
                    if isinstance(qty_value, dict):
                        task['quantity'] = qty_value.get('quantity', task['quantity'])
                        task['unit'] = qty_value.get('unit', task['unit'])
                        task['cost_estimate'] = qty_value.get('cost', 0.0)
                    elif isinstance(qty_value, (int, float)):
                        task['quantity'] = qty_value
    
    def _log_model_response(self, model_name: str, raw_response: str, response_time: float):
        """AI 모델 응답을 파일과 콘솔에 로깅"""
        import os
        from datetime import datetime
        
        # 디렉토리 생성
        debug_dir = "ai_responses"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 파일명 생성
        clean_name = model_name.lower().replace('-', '_').replace(' ', '_')
        debug_file = f"{debug_dir}/{clean_name}_response_{timestamp}.txt"
        
        # 파일에 저장
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"{model_name} Response\n")
            f.write(f"처리 시간: {response_time:.2f}초\n")
            f.write(f"응답 크기: {len(raw_response)} characters\n")
            f.write("="*80 + "\n")
            f.write(raw_response)
            f.write("\n" + "="*80 + "\n")
            
            # JSON 파싱 시도
            try:
                import json
                parsed = self._try_parse_json(raw_response)
                if parsed:
                    f.write("\n[Parsed JSON Structure]\n")
                    f.write(json.dumps(parsed, indent=2, ensure_ascii=False)[:5000])  # 처음 5000자만
            except:
                pass
        
        # 콘솔에 요약 출력
        print(f"\n" + "="*80)
        print(f"📝 {model_name} AI 응답 수신")
        print(f"⏱️  처리 시간: {response_time:.2f}초")
        print(f"📊 응답 크기: {len(raw_response)} characters")
        print(f"💾 저장 위치: {debug_file}")
        
        # 응답 미리보기 (처음 500자)
        preview = raw_response[:500]
        if len(raw_response) > 500:
            preview += "... (truncated)"
        print(f"\n[응답 미리보기]\n{preview}")
        print("="*80 + "\n")
        
        # 작업 개수 확인
        try:
            data = self._extract_response_data(raw_response)
            work_items = data.get('work_items', [])
            rooms = data.get('rooms', [])
            
            total_tasks = len(work_items)
            if total_tasks == 0 and rooms:
                # rooms에서 tasks 계산
                for room in rooms:
                    total_tasks += len(room.get('tasks', []))
            
            if total_tasks == 0:
                print(f"⚠️  경고: {model_name}에서 작업이 생성되지 않았습니다!")
                self.logger.warning(f"{model_name} generated 0 tasks")
            else:
                print(f"✅ {model_name}: {total_tasks}개 작업 생성")
                self.logger.info(f"{model_name} generated {total_tasks} tasks")
        except Exception as e:
            print(f"⚠️  작업 개수 파싱 실패: {e}")
            self.logger.error(f"Failed to parse task count: {e}")

class GPT4Interface(AIModelInterface):
    """GPT-4 인터페이스"""
    
    def __init__(self, api_key: str, model_name: str = None):
        self.actual_model_name = model_name or "gpt-4o-mini"
        super().__init__(api_key, self.actual_model_name)
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.logger = get_logger('gpt4_interface')
        self._last_api_response = None  # Store raw API response for token tracking
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """GPT-4 모델 호출 with Structured Outputs"""
        start_time = time.time()
        
        try:
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            self.logger.info("GPT-4 API 호출 시작 (Structured Output Mode)")
            self.logger.debug(f"프롬프트 크기: {len(full_prompt)} characters")
            log_model_call(self.actual_model_name, len(full_prompt))
            
            # Structured Output JSON 스키마 정의 - Phase 1: Work Scope Only (NO COSTS)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "phase1_work_scope",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "phase": {"type": "string"},
                            "processing_timestamp": {"type": "string"},
                            "rooms": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "room_name": {"type": "string"},
                                        "room_id": {"type": "string"},
                                        "tasks": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "task_id": {"type": "string"},
                                                    "task_name": {"type": "string"},
                                                    "task_type": {
                                                        "type": "string",
                                                        "enum": ["removal", "installation", "protection", "detach", "reset", "preparation", "cleaning", "disposal", "finishing", "repair", "other"]
                                                    },
                                                    "material_category": {
                                                        "type": "string",
                                                        "enum": ["flooring", "wall", "ceiling", "baseboard", "other"]
                                                    },
                                                    "quantity": {"type": "number"},
                                                    "unit": {
                                                        "type": "string",
                                                        "enum": ["sqft", "lf", "sy", "item", "hour", "each"]
                                                    },
                                                    "notes": {"type": "string"},
                                                    "high_ceiling_premium_applied": {"type": "boolean"},
                                                    "demo_already_completed": {"type": "number"}
                                                },
                                                "required": ["task_id", "task_name", "task_type", "material_category", "quantity", "unit", "notes", "high_ceiling_premium_applied", "demo_already_completed"]
                                            }
                                        },
                                        "room_totals": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "total_tasks": {"type": "number"},
                                                "total_removal_tasks": {"type": "number"},
                                                "total_installation_tasks": {"type": "number"}
                                            },
                                            "required": ["total_tasks", "total_removal_tasks", "total_installation_tasks"]
                                        }
                                    },
                                    "required": ["room_name", "room_id", "tasks", "room_totals"]
                                }
                            },
                            "summary": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "total_rooms": {"type": "number"},
                                    "total_tasks": {"type": "number"},
                                    "has_high_ceiling_areas": {"type": "boolean"},
                                    "validation_status": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "remove_replace_logic_applied": {"type": "boolean"},
                                            "measurements_used": {"type": "boolean"},
                                            "special_tasks_included": {"type": "boolean"}
                                        },
                                        "required": ["remove_replace_logic_applied", "measurements_used", "special_tasks_included"]
                                    }
                                },
                                "required": ["total_rooms", "total_tasks", "has_high_ceiling_areas", "validation_status"]
                            }
                        },
                        "required": ["phase", "processing_timestamp", "rooms", "summary"]
                    }
                }
            }
            
            # Check if model supports structured outputs
            if self.actual_model_name in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo-preview"]:
                response = await self.client.chat.completions.create(
                    model=self.actual_model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a Senior Reconstruction Estimating Specialist. Your task is to:
1. Analyze room data and generate COMPREHENSIVE task lists (10-20 tasks per room minimum)
2. Apply Remove & Replace logic correctly (removal for remaining + installation for full area)
3. Include ALL necessary tasks: removal, disposal, preparation, installation, finishing, protection, cleanup
4. Use the EXACT JSON structure specified in the schema
5. Ensure every room has substantial tasks even for simple work scopes
6. NEVER return empty task arrays - every room MUST have multiple detailed tasks
7. ALWAYS fill the 'notes' field with detailed reasoning explaining WHY this task is needed
8. Include specific justification in notes like: "Required due to Remove & Replace scope", "Necessary for proper surface preparation", "Essential for safety compliance", etc."""
                        },
                        {"role": "user", "content": full_prompt}
                    ],
                    response_format=response_format,
                    max_tokens=4000,
                    temperature=0.1,
                    timeout=self.timeout
                )
                
                # Store raw response for token tracking
                self._last_api_response = response
            else:
                # Fallback for older models
                response = await self.client.chat.completions.create(
                    model=self.actual_model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a Senior Reconstruction Estimating Specialist. Analyze the data carefully and provide detailed work scope estimates. Return ONLY valid JSON matching the specified schema."
                        },
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.1,
                    timeout=self.timeout
                )
                
                # Store raw response for token tracking
                self._last_api_response = response
            
            raw_response = response.choices[0].message.content
            response_time = time.time() - start_time
            
            self.logger.info(f"GPT-4 응답 수신 (소요시간: {response_time:.2f}초)")
            self.logger.debug(f"응답 크기: {len(raw_response)} characters")
            
            # AI 응답 파일과 콘솔에 로깅
            self._log_model_response('GPT-4', raw_response, response_time)
            
            processed_data = self._extract_response_data(raw_response)
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=time.time() - start_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.85  # GPT-4 기본 신뢰도
            )
            
        except Exception as e:
            self.logger.error(f"GPT-4 호출 오류: {e}")
            log_error('gpt4_interface', e, {'prompt_length': len(full_prompt) if 'full_prompt' in locals() else 0})
            
            # 타임아웃이나 오류 시 None 반환 또는 명시적 에러 응답
            # ModelResponse 반환 시 실제 데이터가 없음을 명확히 표시
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class ClaudeInterface(AIModelInterface):
    """Claude 인터페이스"""
    
    def __init__(self, api_key: str, model_name: str = None):
        self.actual_model_name = model_name or "claude-3-5-sonnet-20241022"
        super().__init__(api_key, self.actual_model_name)
        self.client = Anthropic(api_key=api_key)
        self.logger = get_logger('claude_interface')
        self._last_api_response = None  # Store raw API response for token tracking
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """Claude 모델 호출"""
        start_time = time.time()
        
        try:
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            self.logger.info("Claude API 호출 시작")
            self.logger.debug(f"프롬프트 크기: {len(full_prompt)} characters")
            log_model_call(self.actual_model_name, len(full_prompt))
            
            # asyncio.to_thread를 사용해서 동기 API를 비동기로 실행
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.actual_model_name,
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Store raw response for token tracking
            self._last_api_response = response
            
            raw_response = response.content[0].text
            response_time = time.time() - start_time
            
            self.logger.info(f"Claude 응답 수신 (소요시간: {response_time:.2f}초)")
            self.logger.debug(f"응답 크기: {len(raw_response)} characters")
            
            # AI 응답 파일과 콘솔에 로깅
            self._log_model_response('Claude', raw_response, response_time)
            
            processed_data = self._extract_response_data(raw_response)
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=time.time() - start_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.88  # Claude 기본 신뢰도 (보수적)
            )
            
        except Exception as e:
            self.logger.error(f"Claude 호출 오류: {e}")
            log_error('claude_interface', e, {'prompt_length': len(full_prompt) if 'full_prompt' in locals() else 0})
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class GeminiInterface(AIModelInterface):
    """Gemini 인터페이스"""
    
    def __init__(self, api_key: str, model_name: str = None):
        self.actual_model_name = model_name or "gemini-1.5-flash"
        super().__init__(api_key, self.actual_model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.actual_model_name)
        self.logger = get_logger('gemini_interface')
        self._last_api_response = None  # Store raw API response for token tracking
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """Gemini 모델 호출"""
        start_time = time.time()
        
        try:
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            self.logger.info("Gemini API 호출 시작")
            self.logger.debug(f"프롬프트 크기: {len(full_prompt)} characters")
            log_model_call(self.actual_model_name, len(full_prompt))
            
            # asyncio.to_thread를 사용해서 동기 API를 비동기로 실행
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=3000,
                    temperature=0.1
                )
            )
            
            # Store raw response for token tracking
            self._last_api_response = response
            
            raw_response = response.text
            response_time = time.time() - start_time
            
            self.logger.info(f"Gemini 응답 수신 (소요시간: {response_time:.2f}초)")
            self.logger.debug(f"응답 크기: {len(raw_response)} characters")
            
            # AI 응답 파일과 콘솔에 로깅
            self._log_model_response('Gemini', raw_response, response_time)
            
            processed_data = self._extract_response_data(raw_response)
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=time.time() - start_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.80  # Gemini 기본 신뢰도
            )
            
        except Exception as e:
            self.logger.error(f"Gemini 호출 오류: {e}")
            log_error('gemini_interface', e, {'prompt_length': len(full_prompt) if 'full_prompt' in locals() else 0})
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class ModelOrchestrator:
    """Enhanced model orchestrator with integrated response validation and token tracking"""
    
    def __init__(self, enable_validation: bool = True, enable_tracking: bool = True):
        self.config_loader = ConfigLoader()
        self.api_keys = self.config_loader.get_api_keys()
        self.model_names = self.config_loader.get_model_names()
        self.logger = get_logger('model_orchestrator')
        
        # Token tracking setup
        self.enable_tracking = enable_tracking
        self.token_tracker = None
        if enable_tracking:
            try:
                from src.tracking.token_tracker import TokenTracker
                self.token_tracker = TokenTracker()
                self.logger.info("Token tracking enabled")
            except ImportError as e:
                self.logger.warning(f"Token tracking unavailable: {e}")
                self.enable_tracking = False
        
        # Response validation settings
        self.enable_validation = enable_validation
        self.validation_orchestrator = None
        if enable_validation:
            try:
                from src.validators.response_validator import ValidationOrchestrator
                self.validation_orchestrator = ValidationOrchestrator()
                self.logger.info("Response validation enabled")
            except ImportError as e:
                self.logger.warning(f"Response validation unavailable: {e}")
                self.enable_validation = False
        
        # 모델 인터페이스 초기화
        self.models = {}
        
        if self.api_keys['openai']:
            self.models['gpt4'] = GPT4Interface(self.api_keys['openai'], self.model_names['gpt4'])
            self.logger.info("GPT-4 모델 초기화 완료")
        
        if self.api_keys['anthropic']:
            self.models['claude'] = ClaudeInterface(self.api_keys['anthropic'], self.model_names['claude'])
            self.logger.info("Claude 모델 초기화 완료")
        
        if self.api_keys['google']:
            self.models['gemini'] = GeminiInterface(self.api_keys['google'], self.model_names['gemini'])
            self.logger.info("Gemini 모델 초기화 완료")
        
        self.logger.info(f"총 {len(self.models)}개 모델 사용 가능, 검증 {'활성화' if self.enable_validation else '비활성화'}")
    
    async def run_single_model(self, model_name: str, prompt: str, json_data: Dict[str, Any]) -> Optional[ModelResponse]:
        """단일 모델 실행"""
        if model_name not in self.models:
            self.logger.warning(f"모델 {model_name}을 사용할 수 없습니다. (API 키 확인)")
            return None
        
        try:
            self.logger.debug(f"모델 {model_name} 실행 시작")
            result = await self.models[model_name].call_model(prompt, json_data)
            self.logger.debug(f"모델 {model_name} 실행 완료")
            return result
        except Exception as e:
            self.logger.error(f"모델 {model_name} 실행 오류: {e}")
            log_error('model_orchestrator', e, {'model': model_name})
            return None
    
    async def run_parallel(self, prompt: str, json_data: Dict[str, Any], 
                          model_names: List[str] = None,
                          enable_validation: Optional[bool] = None,
                          min_quality_threshold: float = 30.0) -> List[ModelResponse]:
        """여러 모델 병렬 실행 with enhanced validation"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        if enable_validation is None:
            enable_validation = self.enable_validation
        
        # 사용 가능한 모델만 필터링
        available_models = [name for name in model_names if name in self.models]
        
        if not available_models:
            self.logger.error("사용 가능한 모델이 없습니다.")
            return []
        
        self.logger.info(f"모델 병렬 실행 시작: {available_models} (검증: {'ON' if enable_validation else 'OFF'})")
        
        # 병렬 실행
        tasks = [
            self.run_single_model(model_name, prompt, json_data)
            for model_name in available_models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 결과만 필터링 with enhanced validation
        successful_results = []
        validation_reports = []
        
        for i, result in enumerate(results):
            model_name = available_models[i]
            
            if isinstance(result, ModelResponse):
                # Enhanced validation if enabled
                if enable_validation and self.validation_orchestrator:
                    validated_response, validation_report = self._validate_response(
                        result, json_data, min_quality_threshold
                    )
                    validation_reports.append({
                        'model': model_name,
                        'report': validation_report
                    })
                    
                    if validation_report.is_valid and validation_report.quality_score >= min_quality_threshold:
                        successful_results.append(validated_response)
                        self.logger.info(
                            f"OK {model_name} - {validation_report.quality_level.value.upper()} "
                            f"({validation_report.quality_score:.1f}/100, {result.total_work_items}개 작업)"
                        )
                    else:
                        self.logger.warning(
                            f"FAIL {model_name} - 품질 기준 미달 "
                            f"({validation_report.quality_score:.1f}/100, "
                            f"{len(validation_report.issues)}개 이슈)"
                        )
                else:
                    # Fallback to basic validation (legacy behavior)
                    if result.total_work_items > 0 or (result.room_estimates and len(result.room_estimates) > 0):
                        successful_results.append(result)
                        self.logger.info(f"OK {model_name} 모델 성공 (작업 {result.total_work_items}개)")
                    else:
                        error_msg = result.raw_response[:200] if isinstance(result.raw_response, str) else "빈 응답"
                        self.logger.warning(f"FAIL {model_name} 모델 응답 비어있음: {error_msg}")
            
            elif isinstance(result, Exception):
                self.logger.error(f"FAIL {model_name} 모델 실행 중 예외: {result}")
        
        # Log validation summary
        if enable_validation and validation_reports:
            self._log_validation_summary(validation_reports)
        
        self.logger.info(f"모델 실행 완료: {len(successful_results)}/{len(available_models)} 성공")
        return successful_results
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.models.keys())
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """API 키 유효성 검증"""
        validation_results = {}
        
        for model_name, api_key in self.api_keys.items():
            is_valid = bool(api_key and api_key.strip())
            validation_results[model_name] = is_valid
            
            if is_valid:
                self.logger.debug(f"✓ {model_name} API 키 유효")
            else:
                self.logger.warning(f"✗ {model_name} API 키 없음")
        
        return validation_results
    
    def _validate_response(self, response: ModelResponse, 
                          original_data: Optional[Dict[str, Any]] = None,
                          min_quality_threshold: float = 30.0) -> Tuple[ModelResponse, Any]:
        """Validate and optionally fix model response"""
        try:
            # Extract response data for validation
            if hasattr(response, 'raw_response'):
                if isinstance(response.raw_response, dict):
                    response_data = response.raw_response
                else:
                    # Try to parse as JSON
                    try:
                        response_data = json.loads(response.raw_response) if isinstance(response.raw_response, str) else {}
                    except:
                        response_data = {'rooms': response.room_estimates}
            else:
                response_data = {'rooms': response.room_estimates}
                
            # Use the validation orchestrator from response_validator
            from src.validators.response_validator import validate_model_response
            validated_data, validation_report = validate_model_response(
                response_data, original_data, auto_fix=True
            )
            
            # Update response with validated data
            if 'rooms' in validated_data:
                response.room_estimates = validated_data['rooms']
                
            return response, validation_report
        except Exception as e:
            self.logger.error(f"Validation failed for {response.model_name}: {e}")
            # Return mock validation report for failed validation
            from src.validators.response_validator import ValidationReport, QualityLevel
            mock_report = ValidationReport(
                quality_score=min_quality_threshold,  # Minimum passing score
                quality_level=QualityLevel.ACCEPTABLE,
                total_issues=0,
                critical_issues=0,
                high_issues=0,
                auto_fixed=0,
                issues=[],
                processing_time=0.0,
                metadata={'validation_error': str(e)}
            )
            return response, mock_report
    
    def _log_validation_summary(self, validation_reports: List[Dict[str, Any]]) -> None:
        """Log summary of all validation reports"""
        if not validation_reports:
            return
        
        total_reports = len(validation_reports)
        valid_reports = sum(1 for r in validation_reports if r['report'].is_valid)
        
        # Calculate average quality score
        avg_quality = sum(r['report'].quality_score for r in validation_reports) / total_reports
        
        # Count quality levels
        quality_counts = {}
        total_issues = 0
        total_fixes = 0
        
        for r in validation_reports:
            report = r['report']
            quality_level = report.quality_level.value
            quality_counts[quality_level] = quality_counts.get(quality_level, 0) + 1
            total_issues += len(report.issues)
            total_fixes += report.auto_fixed
        
        self.logger.info(
            f"🔍 Validation Summary: {valid_reports}/{total_reports} valid, "
            f"avg quality: {avg_quality:.1f}/100"
        )
        
        if quality_counts:
            quality_summary = ", ".join([f"{level}: {count}" for level, count in quality_counts.items()])
            self.logger.info(f"📊 Quality distribution: {quality_summary}")
        
        if total_issues > 0:
            self.logger.info(f"⚠️  Total issues found: {total_issues}, auto-fixes applied: {total_fixes}")
    
    def get_validation_enabled(self) -> bool:
        """Check if validation is enabled and available"""
        return self.enable_validation and self.validation_orchestrator is not None
    
    def set_validation_enabled(self, enabled: bool) -> bool:
        """Enable or disable validation (returns actual state)"""
        if enabled and self.validation_orchestrator is None:
            try:
                from src.validators.response_validator import ValidationOrchestrator
                self.validation_orchestrator = ValidationOrchestrator()
                self.enable_validation = True
                self.logger.info("Response validation enabled")
            except ImportError as e:
                self.logger.warning(f"Cannot enable validation: {e}")
                self.enable_validation = False
        else:
            self.enable_validation = enabled
            self.logger.info(f"Response validation {'enabled' if enabled else 'disabled'}")
        
        return self.enable_validation