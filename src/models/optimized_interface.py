# src/models/optimized_interface.py
"""
최적화된 AI 모델 인터페이스 - 타임아웃 문제 해결
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import hashlib

import openai
from anthropic import Anthropic
import google.generativeai as genai

from src.models.data_models import ModelResponse
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, log_model_call, log_error

class OptimizedModelInterface(ABC):
    """최적화된 AI 모델 인터페이스"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = 3
        self.timeout = 60  # 적당한 타임아웃
        self.chunk_size = 50  # 청킹 크기
        self.response_cache = {}  # 응답 캐시
        
    @abstractmethod
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """모델 호출 추상 메서드"""
        pass
    
    def _optimize_json_data(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 데이터 최적화 - 불필요한 데이터 제거"""
        optimized = {}
        
        # 필수 필드만 추출
        if 'jobsite' in json_data:
            optimized['jobsite'] = json_data['jobsite']
        
        if 'floors' in json_data:
            optimized['floors'] = []
            for floor in json_data['floors']:
                opt_floor = {
                    'location': floor.get('location', ''),
                    'rooms': []
                }
                
                for room in floor.get('rooms', []):
                    # 핵심 데이터만 추출
                    opt_room = {
                        'name': room.get('name', ''),
                        'measurements': {
                            'wall_area_sqft': room.get('measurements', {}).get('wall_area_sqft', 0),
                            'ceiling_area_sqft': room.get('measurements', {}).get('ceiling_area_sqft', 0),
                            'floor_area_sqft': room.get('measurements', {}).get('floor_area_sqft', 0),
                            'height': room.get('measurements', {}).get('height', 8.0)
                        }
                    }
                    
                    # demo_scope가 있는 경우만 추가
                    if 'demo_scope' in room and room['demo_scope']:
                        opt_room['demo_scope'] = self._summarize_demo_scope(room['demo_scope'])
                    
                    # material이 있는 경우만 추가
                    if 'material' in room and room['material']:
                        opt_room['material'] = self._summarize_materials(room['material'])
                    
                    opt_floor['rooms'].append(opt_room)
                
                optimized['floors'].append(opt_floor)
        
        return optimized
    
    def _summarize_demo_scope(self, demo_scope: Dict) -> Dict:
        """demo_scope 요약"""
        summary = {}
        for key, value in demo_scope.items():
            if isinstance(value, dict) and 'quantity' in value:
                # 수량과 상태만 포함
                summary[key] = {
                    'qty': value.get('quantity', 0),
                    'status': value.get('status', 'unknown')
                }
        return summary
    
    def _summarize_materials(self, materials: Dict) -> Dict:
        """material 정보 요약"""
        # 빈 값 제거
        return {k: v for k, v in materials.items() if v}
    
    def _create_concise_prompt(self, base_prompt: str, json_data: Dict[str, Any]) -> str:
        """간결한 프롬프트 생성"""
        # JSON 데이터 최적화
        optimized_data = self._optimize_json_data(json_data)
        
        # 간결한 JSON 문자열 생성 (들여쓰기 최소화)
        json_str = json.dumps(optimized_data, separators=(',', ':'), ensure_ascii=False)
        
        # 프롬프트 템플릿 최적화
        concise_prompt = f"""TASK: Generate reconstruction estimate.

KEY REQUIREMENTS:
1. Apply Remove & Replace strategy
2. Use provided measurements exactly
3. Include all necessary work items
4. Calculate quantities accurately

DATA:
{json_str}

OUTPUT: List work items with quantities for each room."""
        
        return concise_prompt
    
    def _get_cache_key(self, prompt: str, json_data: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        data_str = f"{prompt}{json.dumps(json_data, sort_keys=True)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """지수 백오프를 사용한 재시도"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt  # 지수 백오프
                print(f"재시도 {attempt + 1}/{self.max_retries}, {wait_time}초 대기...")
                await asyncio.sleep(wait_time)

class OptimizedGPT4Interface(OptimizedModelInterface):
    """최적화된 GPT-4 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "gpt-4")
        # 비동기 클라이언트 설정
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=self.timeout,
            max_retries=0  # 자체 재시도 로직 사용
        )
        self.logger = get_logger('optimized_gpt4')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """최적화된 GPT-4 호출"""
        start_time = time.time()
        
        # 캐시 확인
        cache_key = self._get_cache_key(prompt, json_data)
        if cache_key in self.response_cache:
            self.logger.info("캐시된 응답 사용")
            cached = self.response_cache[cache_key]
            cached['processing_time'] = 0.01  # 캐시 응답 시간
            return ModelResponse(**cached)
        
        try:
            # 간결한 프롬프트 생성
            concise_prompt = self._create_concise_prompt(prompt, json_data)
            
            self.logger.info(f"최적화된 프롬프트 크기: {len(concise_prompt)} characters (원본 대비 감소)")
            
            # 스트리밍 호출로 변경 (타임아웃 방지)
            async def make_request():
                stream = await self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",  # 더 빠른 모델 사용
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a construction estimator. Be concise and accurate."
                        },
                        {"role": "user", "content": concise_prompt}
                    ],
                    max_tokens=2000,  # 토큰 수 감소
                    temperature=0.1,
                    stream=True  # 스트리밍 활성화
                )
                
                # 스트리밍 응답 수집
                chunks = []
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)
                
                return ''.join(chunks)
            
            # 재시도 로직과 함께 호출
            raw_response = await self._retry_with_backoff(make_request)
            
            response_time = time.time() - start_time
            self.logger.info(f"GPT-4 응답 완료 (소요시간: {response_time:.2f}초)")
            
            # 응답 처리
            processed_data = self._extract_response_data(raw_response)
            
            result = ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=response_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.85
            )
            
            # 캐시 저장
            self.response_cache[cache_key] = result.dict()
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error("GPT-4 타임아웃 - 프롬프트가 너무 크거나 서버 응답이 느림")
            return self._create_error_response(start_time, "Timeout error")
            
        except Exception as e:
            self.logger.error(f"GPT-4 호출 오류: {e}")
            return self._create_error_response(start_time, str(e))
    
    def _extract_response_data(self, raw_response: str) -> Dict[str, Any]:
        """응답 데이터 추출"""
        # 기존 로직 재사용
        try:
            if raw_response.strip().startswith('{'):
                return json.loads(raw_response)
            
            # 텍스트 파싱
            lines = raw_response.split('\n')
            work_items = []
            current_room = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 방 감지
                if any(keyword in line.lower() for keyword in ['room', 'kitchen', 'bathroom']):
                    if ':' in line:
                        current_room = line.split(':')[0].strip()
                
                # 작업 항목 감지
                if line.startswith(('-', '*', '•')) or any(line.startswith(f"{i}.") for i in range(1, 20)):
                    task_name = line.lstrip('-*•0123456789. ').strip()
                    if task_name:
                        work_items.append({
                            'task_name': task_name,
                            'room_name': current_room,
                            'description': task_name,
                            'necessity': 'required'
                        })
            
            return {
                'work_items': work_items,
                'rooms': [{'name': current_room, 'tasks': work_items}] if current_room else []
            }
            
        except Exception as e:
            self.logger.error(f"응답 파싱 오류: {e}")
            return {'work_items': [], 'rooms': []}
    
    def _create_error_response(self, start_time: float, error_msg: str) -> ModelResponse:
        """에러 응답 생성"""
        return ModelResponse(
            model_name=self.model_name,
            room_estimates=[],
            processing_time=time.time() - start_time,
            total_work_items=0,
            raw_response=f"Error: {error_msg}",
            confidence_self_assessment=0.0
        )

class OptimizedClaudeInterface(OptimizedModelInterface):
    """최적화된 Claude 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "claude-3-sonnet")
        self.client = Anthropic(api_key=api_key)
        self.logger = get_logger('optimized_claude')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """최적화된 Claude 호출"""
        start_time = time.time()
        
        try:
            # 간결한 프롬프트 생성
            concise_prompt = self._create_concise_prompt(prompt, json_data)
            
            self.logger.info(f"최적화된 프롬프트 크기: {len(concise_prompt)} characters")
            
            # 비동기 호출
            async def make_request():
                return await asyncio.to_thread(
                    self.client.messages.create,
                    model="claude-3-haiku-20240307",  # 더 빠른 모델 사용
                    max_tokens=2000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": concise_prompt}]
                )
            
            response = await self._retry_with_backoff(make_request)
            raw_response = response.content[0].text
            
            response_time = time.time() - start_time
            self.logger.info(f"Claude 응답 완료 (소요시간: {response_time:.2f}초)")
            
            # 응답 처리
            processed_data = self._extract_response_data(raw_response)
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=response_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.88
            )
            
        except Exception as e:
            self.logger.error(f"Claude 호출 오류: {e}")
            return self._create_error_response(start_time, str(e))
    
    def _extract_response_data(self, raw_response: str) -> Dict[str, Any]:
        """응답 데이터 추출 - GPT4와 동일"""
        return OptimizedGPT4Interface._extract_response_data(self, raw_response)
    
    def _create_error_response(self, start_time: float, error_msg: str) -> ModelResponse:
        """에러 응답 생성"""
        return OptimizedGPT4Interface._create_error_response(self, start_time, error_msg)

class OptimizedGeminiInterface(OptimizedModelInterface):
    """최적화된 Gemini 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "gemini-pro")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # 더 빠른 모델
        self.logger = get_logger('optimized_gemini')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """최적화된 Gemini 호출"""
        start_time = time.time()
        
        try:
            # 간결한 프롬프트 생성
            concise_prompt = self._create_concise_prompt(prompt, json_data)
            
            self.logger.info(f"최적화된 프롬프트 크기: {len(concise_prompt)} characters")
            
            # 비동기 호출
            async def make_request():
                return await asyncio.to_thread(
                    self.model.generate_content,
                    concise_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2000,
                        temperature=0.1
                    )
                )
            
            response = await self._retry_with_backoff(make_request)
            raw_response = response.text
            
            response_time = time.time() - start_time
            self.logger.info(f"Gemini 응답 완료 (소요시간: {response_time:.2f}초)")
            
            # 응답 처리
            processed_data = self._extract_response_data(raw_response)
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=response_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.80
            )
            
        except Exception as e:
            self.logger.error(f"Gemini 호출 오류: {e}")
            return self._create_error_response(start_time, str(e))
    
    def _extract_response_data(self, raw_response: str) -> Dict[str, Any]:
        """응답 데이터 추출"""
        return OptimizedGPT4Interface._extract_response_data(self, raw_response)
    
    def _create_error_response(self, start_time: float, error_msg: str) -> ModelResponse:
        """에러 응답 생성"""
        return OptimizedGPT4Interface._create_error_response(self, start_time, error_msg)

class OptimizedModelOrchestrator:
    """최적화된 모델 오케스트레이터"""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        self.config = config or ConfigLoader()
        self.models = {}
        self.logger = get_logger('optimized_orchestrator')
        
    def initialize_models(self, model_names: List[str]) -> Dict[str, bool]:
        """모델 초기화"""
        status = {}
        
        for model_name in model_names:
            try:
                if model_name.lower() == 'gpt4':
                    api_key = self.config.get_api_key('openai')
                    if api_key:
                        self.models['gpt4'] = OptimizedGPT4Interface(api_key)
                        status['gpt4'] = True
                        self.logger.info("최적화된 GPT-4 모델 초기화 성공")
                    else:
                        status['gpt4'] = False
                        
                elif model_name.lower() == 'claude':
                    api_key = self.config.get_api_key('anthropic')
                    if api_key:
                        self.models['claude'] = OptimizedClaudeInterface(api_key)
                        status['claude'] = True
                        self.logger.info("최적화된 Claude 모델 초기화 성공")
                    else:
                        status['claude'] = False
                        
                elif model_name.lower() == 'gemini':
                    api_key = self.config.get_api_key('google')
                    if api_key:
                        self.models['gemini'] = OptimizedGeminiInterface(api_key)
                        status['gemini'] = True
                        self.logger.info("최적화된 Gemini 모델 초기화 성공")
                    else:
                        status['gemini'] = False
                        
            except Exception as e:
                status[model_name] = False
                self.logger.error(f"{model_name} 초기화 실패: {e}")
                
        return status
    
    async def call_models_parallel(self, prompt: str, json_data: Dict[str, Any]) -> List[ModelResponse]:
        """병렬 모델 호출 with 타임아웃 관리"""
        if not self.models:
            self.logger.error("초기화된 모델이 없습니다")
            return []
        
        self.logger.info(f"최적화된 병렬 실행 시작: {list(self.models.keys())}")
        
        # 각 모델에 대해 개별 타임아웃 설정
        tasks = []
        for model_name, model_interface in self.models.items():
            # 각 태스크에 타임아웃 적용
            task = asyncio.create_task(
                asyncio.wait_for(
                    model_interface.call_model(prompt, json_data),
                    timeout=90  # 90초 타임아웃
                )
            )
            tasks.append(task)
        
        # 모든 태스크 완료 대기 (실패한 것도 포함)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        valid_results = []
        for i, result in enumerate(results):
            model_name = list(self.models.keys())[i]
            
            if isinstance(result, asyncio.TimeoutError):
                self.logger.error(f"{model_name} 타임아웃")
                valid_results.append(ModelResponse(
                    model_name=model_name,
                    room_estimates=[],
                    processing_time=90,
                    total_work_items=0,
                    raw_response="Error: Timeout after 90 seconds",
                    confidence_self_assessment=0.0
                ))
            elif isinstance(result, Exception):
                self.logger.error(f"{model_name} 실행 실패: {result}")
                valid_results.append(ModelResponse(
                    model_name=model_name,
                    room_estimates=[],
                    processing_time=0,
                    total_work_items=0,
                    raw_response=f"Error: {str(result)}",
                    confidence_self_assessment=0.0
                ))
            else:
                valid_results.append(result)
                self.logger.info(f"{model_name} 완료: {result.total_work_items} 항목, {result.processing_time:.2f}초")
        
        return valid_results


# 테스트
if __name__ == "__main__":
    async def test_optimized():
        orchestrator = OptimizedModelOrchestrator()
        orchestrator.initialize_models(['gpt4', 'claude', 'gemini'])
        
        # 테스트 데이터
        test_prompt = "Generate reconstruction estimate"
        test_data = {
            "jobsite": "Test",
            "floors": [
                {
                    "location": "1st Floor",
                    "rooms": [
                        {
                            "name": "Living Room",
                            "measurements": {
                                "wall_area_sqft": 300,
                                "ceiling_area_sqft": 150,
                                "floor_area_sqft": 150
                            }
                        }
                    ]
                }
            ]
        }
        
        results = await orchestrator.call_models_parallel(test_prompt, test_data)
        for result in results:
            print(f"{result.model_name}: {result.processing_time:.2f}초")
    
    asyncio.run(test_optimized())