# src/models/model_interface.py
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import openai
from anthropic import Anthropic
import google.generativeai as genai

from src.models.data_models import ModelResponse
from src.utils.config_loader import ConfigLoader

class AIModelInterface(ABC):
    """AI 모델 인터페이스 추상 클래스"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = 3
        self.timeout = 30
    
    @abstractmethod
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """모델 호출 추상 메서드"""
        pass
    
    def _prepare_prompt(self, base_prompt: str, json_data: Dict[str, Any]) -> str:
        """프롬프트 준비"""
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        return f"{base_prompt}\n\n[JSON DATA]\n{json_str}"
    
    def _extract_response_data(self, raw_response: str) -> Dict[str, Any]:
        """응답에서 구조화된 데이터 추출"""
        try:
            # JSON 형태로 응답이 온 경우
            if raw_response.strip().startswith('{'):
                return json.loads(raw_response)
            
            # 텍스트 응답에서 작업 항목 추출 (간단한 파싱)
            lines = raw_response.split('\n')
            work_items = []
            current_room = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 방 이름 감지
                if any(keyword in line.lower() for keyword in ['room', 'kitchen', 'bathroom', 'bedroom']):
                    if ':' in line:
                        current_room = line.split(':')[0].strip()
                
                # 작업 항목 감지 (-, *, 1., 2. 등으로 시작)
                if any(line.startswith(prefix) for prefix in ['-', '*', '•']) or \
                   any(line.startswith(f"{i}.") for i in range(1, 20)):
                    
                    task_name = line.lstrip('-*•0123456789. ').strip()
                    if task_name:
                        work_items.append({
                            'task_name': task_name,
                            'room_name': current_room,
                            'description': task_name,
                            'necessity': 'required'  # 기본값
                        })
            
            return {
                'work_items': work_items,
                'rooms': [{'name': current_room, 'tasks': work_items}] if current_room else []
            }
            
        except Exception as e:
            print(f"응답 파싱 오류: {e}")
            return {
                'work_items': [],
                'rooms': [],
                'raw_text': raw_response
            }

class GPT4Interface(AIModelInterface):
    """GPT-4 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "gpt-4")
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """GPT-4 모델 호출"""
        start_time = time.time()
        
        try:
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a Senior Reconstruction Estimating Specialist. Analyze the data carefully and provide detailed work scope estimates."
                    },
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=3000,
                temperature=0.1,  # 일관성을 위해 낮은 temperature
                timeout=self.timeout
            )
            
            raw_response = response.choices[0].message.content
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
            print(f"GPT-4 호출 오류: {e}")
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
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "claude-3-sonnet")
        self.client = Anthropic(api_key=api_key)
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """Claude 모델 호출"""
        start_time = time.time()
        
        try:
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            # asyncio.to_thread를 사용해서 동기 API를 비동기로 실행
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            raw_response = response.content[0].text
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
            print(f"Claude 호출 오류: {e}")
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
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "gemini-pro")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """Gemini 모델 호출"""
        start_time = time.time()
        
        try:
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            # asyncio.to_thread를 사용해서 동기 API를 비동기로 실행
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=3000,
                    temperature=0.1
                )
            )
            
            raw_response = response.text
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
            print(f"Gemini 호출 오류: {e}")
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class ModelOrchestrator:
    """모델 호출 오케스트레이터"""
    
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.api_keys = self.config_loader.get_api_keys()
        
        # 모델 인터페이스 초기화
        self.models = {}
        
        if self.api_keys['openai']:
            self.models['gpt4'] = GPT4Interface(self.api_keys['openai'])
        
        if self.api_keys['anthropic']:
            self.models['claude'] = ClaudeInterface(self.api_keys['anthropic'])
        
        if self.api_keys['google']:
            self.models['gemini'] = GeminiInterface(self.api_keys['google'])
    
    async def run_single_model(self, model_name: str, prompt: str, json_data: Dict[str, Any]) -> Optional[ModelResponse]:
        """단일 모델 실행"""
        if model_name not in self.models:
            print(f"모델 {model_name}을 사용할 수 없습니다. (API 키 확인)")
            return None
        
        try:
            return await self.models[model_name].call_model(prompt, json_data)
        except Exception as e:
            print(f"모델 {model_name} 실행 오류: {e}")
            return None
    
    async def run_parallel(self, prompt: str, json_data: Dict[str, Any], 
                          model_names: List[str] = None) -> List[ModelResponse]:
        """여러 모델 병렬 실행"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        # 사용 가능한 모델만 필터링
        available_models = [name for name in model_names if name in self.models]
        
        if not available_models:
            print("사용 가능한 모델이 없습니다.")
            return []
        
        print(f"모델 실행 시작: {available_models}")
        
        # 병렬 실행
        tasks = [
            self.run_single_model(model_name, prompt, json_data)
            for model_name in available_models
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 성공한 결과만 필터링
        successful_results = []
        for result in results:
            if isinstance(result, ModelResponse):
                successful_results.append(result)
            elif isinstance(result, Exception):
                print(f"모델 실행 중 예외 발생: {result}")
        
        print(f"성공한 모델 수: {len(successful_results)}/{len(available_models)}")
        return successful_results
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.models.keys())
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """API 키 유효성 검증"""
        validation_results = {}
        
        for model_name, api_key in self.api_keys.items():
            validation_results[model_name] = bool(api_key and api_key.strip())
        
        return validation_results