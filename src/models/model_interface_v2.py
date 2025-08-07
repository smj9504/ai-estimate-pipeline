# src/models/model_interface_v2.py
"""
개선된 AI 모델 인터페이스 - 진행 상황 모니터링 기능 포함
"""
import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime

import openai
from anthropic import Anthropic
import google.generativeai as genai

from src.models.data_models import ModelResponse
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger, log_model_call, log_error

class ModelStatus(Enum):
    """모델 호출 상태"""
    IDLE = "idle"
    PREPARING = "preparing"
    SENDING_REQUEST = "sending_request"
    WAITING_RESPONSE = "waiting_response"
    PROCESSING_RESPONSE = "processing_response"
    COMPLETED = "completed"
    ERROR = "error"

class ProgressCallback:
    """진행 상황 콜백 클래스"""
    def __init__(self, callback_fn: Optional[Callable] = None):
        self.callback_fn = callback_fn
        self.status_history = []
        
    def update(self, model_name: str, status: ModelStatus, message: str, progress: float = 0.0, details: Dict[str, Any] = None):
        """진행 상황 업데이트"""
        update_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'status': status.value,
            'message': message,
            'progress': progress,
            'details': details or {}
        }
        
        self.status_history.append(update_data)
        
        if self.callback_fn:
            self.callback_fn(update_data)
        
        # 콘솔 출력 (디버깅용)
        print(f"[{model_name}] {status.value}: {message} ({progress:.0f}%)")

class AIModelInterfaceV2(ABC):
    """개선된 AI 모델 인터페이스 추상 클래스"""
    
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = 3
        self.timeout = 30
        self.progress_callback = ProgressCallback()
        self.current_status = ModelStatus.IDLE
        
    def set_progress_callback(self, callback_fn: Callable):
        """진행 상황 콜백 함수 설정"""
        self.progress_callback.callback_fn = callback_fn
    
    def _update_progress(self, status: ModelStatus, message: str, progress: float = 0.0, details: Dict[str, Any] = None):
        """진행 상황 업데이트 헬퍼"""
        self.current_status = status
        self.progress_callback.update(self.model_name, status, message, progress, details)
    
    @abstractmethod
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """모델 호출 추상 메서드"""
        pass
    
    def _prepare_prompt(self, base_prompt: str, json_data: Dict[str, Any]) -> str:
        """프롬프트 준비"""
        self._update_progress(ModelStatus.PREPARING, "프롬프트 준비 중", 10)
        
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        full_prompt = f"{base_prompt}\n\n[JSON DATA]\n{json_str}"
        
        self._update_progress(
            ModelStatus.PREPARING, 
            "프롬프트 준비 완료", 
            20,
            {'prompt_size': len(full_prompt), 'json_size': len(json_str)}
        )
        
        return full_prompt
    
    def _extract_response_data(self, raw_response: str) -> Dict[str, Any]:
        """응답에서 구조화된 데이터 추출"""
        self._update_progress(ModelStatus.PROCESSING_RESPONSE, "응답 데이터 파싱 중", 80)
        
        try:
            # JSON 형태로 응답이 온 경우
            if raw_response.strip().startswith('{'):
                data = json.loads(raw_response)
                self._update_progress(
                    ModelStatus.PROCESSING_RESPONSE, 
                    "JSON 응답 파싱 완료", 
                    90,
                    {'response_type': 'json', 'items_count': len(data.get('work_items', []))}
                )
                return data
            
            # 텍스트 응답에서 작업 항목 추출
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
                
                # 작업 항목 감지
                if any(line.startswith(prefix) for prefix in ['-', '*', '•']) or \
                   any(line.startswith(f"{i}.") for i in range(1, 20)):
                    
                    task_name = line.lstrip('-*•0123456789. ').strip()
                    if task_name:
                        work_items.append({
                            'task_name': task_name,
                            'room_name': current_room,
                            'description': task_name,
                            'necessity': 'required'
                        })
            
            self._update_progress(
                ModelStatus.PROCESSING_RESPONSE, 
                "텍스트 응답 파싱 완료", 
                90,
                {'response_type': 'text', 'items_count': len(work_items)}
            )
            
            return {
                'work_items': work_items,
                'rooms': [{'name': current_room, 'tasks': work_items}] if current_room else []
            }
            
        except Exception as e:
            self._update_progress(
                ModelStatus.ERROR, 
                f"응답 파싱 오류: {e}", 
                90,
                {'error': str(e)}
            )
            return {
                'work_items': [],
                'rooms': [],
                'raw_text': raw_response
            }

class GPT4InterfaceV2(AIModelInterfaceV2):
    """개선된 GPT-4 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "gpt-4")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.logger = get_logger('gpt4_interface_v2')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """GPT-4 모델 호출 with 진행 상황 모니터링"""
        start_time = time.time()
        
        try:
            # 프롬프트 준비
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            # API 호출 시작
            self._update_progress(
                ModelStatus.SENDING_REQUEST, 
                "GPT-4 API 요청 전송 중", 
                30,
                {'model': 'gpt-4', 'max_tokens': 3000}
            )
            
            self.logger.info("GPT-4 API 호출 시작")
            log_model_call("gpt-4", len(full_prompt))
            
            # 스트리밍 옵션 사용 (진행 상황 추적 가능)
            stream = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a Senior Reconstruction Estimating Specialist. Analyze the data carefully and provide detailed work scope estimates."
                    },
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=3000,
                temperature=0.1,
                timeout=self.timeout,
                stream=True  # 스트리밍 활성화
            )
            
            # 스트리밍 응답 수집
            self._update_progress(ModelStatus.WAITING_RESPONSE, "GPT-4 응답 수신 중", 40)
            
            collected_messages = []
            chunk_count = 0
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    collected_messages.append(chunk.choices[0].delta.content)
                    chunk_count += 1
                    
                    # 진행률 계산 (40% ~ 70% 구간)
                    progress = 40 + min(30, (chunk_count / 100) * 30)
                    
                    if chunk_count % 10 == 0:  # 10개 청크마다 업데이트
                        self._update_progress(
                            ModelStatus.WAITING_RESPONSE, 
                            f"응답 수신 중 (청크: {chunk_count})", 
                            progress,
                            {'chunks_received': chunk_count, 'current_size': len(''.join(collected_messages))}
                        )
            
            raw_response = ''.join(collected_messages)
            response_time = time.time() - start_time
            
            self._update_progress(
                ModelStatus.WAITING_RESPONSE, 
                "응답 수신 완료", 
                70,
                {'total_chunks': chunk_count, 'response_size': len(raw_response), 'response_time': response_time}
            )
            
            self.logger.info(f"GPT-4 응답 수신 (소요시간: {response_time:.2f}초)")
            
            # 응답 처리
            processed_data = self._extract_response_data(raw_response)
            
            # 완료
            self._update_progress(
                ModelStatus.COMPLETED, 
                "처리 완료", 
                100,
                {
                    'total_time': response_time,
                    'work_items': len(processed_data.get('work_items', [])),
                    'rooms': len(processed_data.get('rooms', []))
                }
            )
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=response_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.85
            )
            
        except Exception as e:
            self._update_progress(
                ModelStatus.ERROR, 
                f"오류 발생: {str(e)}", 
                0,
                {'error': str(e), 'error_type': type(e).__name__}
            )
            
            self.logger.error(f"GPT-4 호출 오류: {e}")
            log_error('gpt4_interface_v2', e, {'prompt_length': len(full_prompt) if 'full_prompt' in locals() else 0})
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class ClaudeInterfaceV2(AIModelInterfaceV2):
    """개선된 Claude 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "claude-3-sonnet")
        self.client = Anthropic(api_key=api_key)
        self.logger = get_logger('claude_interface_v2')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """Claude 모델 호출 with 진행 상황 모니터링"""
        start_time = time.time()
        
        try:
            # 프롬프트 준비
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            # API 호출 시작
            self._update_progress(
                ModelStatus.SENDING_REQUEST, 
                "Claude API 요청 전송 중", 
                30,
                {'model': 'claude-3-sonnet', 'max_tokens': 3000}
            )
            
            self.logger.info("Claude API 호출 시작")
            log_model_call("claude-3-sonnet", len(full_prompt))
            
            # Claude는 스트리밍을 지원하지만 여기서는 단순 호출 사용
            self._update_progress(ModelStatus.WAITING_RESPONSE, "Claude 응답 대기 중", 40)
            
            # asyncio.to_thread를 사용한 비동기 실행
            response = await asyncio.to_thread(
                self.client.messages.create,
                model="claude-3-sonnet-20240229",
                max_tokens=3000,
                temperature=0.1,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            raw_response = response.content[0].text
            response_time = time.time() - start_time
            
            self._update_progress(
                ModelStatus.WAITING_RESPONSE, 
                "응답 수신 완료", 
                70,
                {'response_size': len(raw_response), 'response_time': response_time}
            )
            
            self.logger.info(f"Claude 응답 수신 (소요시간: {response_time:.2f}초)")
            
            # 응답 처리
            processed_data = self._extract_response_data(raw_response)
            
            # 완료
            self._update_progress(
                ModelStatus.COMPLETED, 
                "처리 완료", 
                100,
                {
                    'total_time': response_time,
                    'work_items': len(processed_data.get('work_items', [])),
                    'rooms': len(processed_data.get('rooms', []))
                }
            )
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=response_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.88
            )
            
        except Exception as e:
            self._update_progress(
                ModelStatus.ERROR, 
                f"오류 발생: {str(e)}", 
                0,
                {'error': str(e), 'error_type': type(e).__name__}
            )
            
            self.logger.error(f"Claude 호출 오류: {e}")
            log_error('claude_interface_v2', e, {'prompt_length': len(full_prompt) if 'full_prompt' in locals() else 0})
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class GeminiInterfaceV2(AIModelInterfaceV2):
    """개선된 Gemini 인터페이스"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "gemini-pro")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.logger = get_logger('gemini_interface_v2')
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any]) -> ModelResponse:
        """Gemini 모델 호출 with 진행 상황 모니터링"""
        start_time = time.time()
        
        try:
            # 프롬프트 준비
            full_prompt = self._prepare_prompt(prompt, json_data)
            
            # API 호출 시작
            self._update_progress(
                ModelStatus.SENDING_REQUEST, 
                "Gemini API 요청 전송 중", 
                30,
                {'model': 'gemini-pro', 'max_tokens': 3000}
            )
            
            self.logger.info("Gemini API 호출 시작")
            log_model_call("gemini-pro", len(full_prompt))
            
            self._update_progress(ModelStatus.WAITING_RESPONSE, "Gemini 응답 대기 중", 40)
            
            # asyncio.to_thread를 사용한 비동기 실행
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=3000,
                    temperature=0.1
                )
            )
            
            raw_response = response.text
            response_time = time.time() - start_time
            
            self._update_progress(
                ModelStatus.WAITING_RESPONSE, 
                "응답 수신 완료", 
                70,
                {'response_size': len(raw_response), 'response_time': response_time}
            )
            
            self.logger.info(f"Gemini 응답 수신 (소요시간: {response_time:.2f}초)")
            
            # 응답 처리
            processed_data = self._extract_response_data(raw_response)
            
            # 완료
            self._update_progress(
                ModelStatus.COMPLETED, 
                "처리 완료", 
                100,
                {
                    'total_time': response_time,
                    'work_items': len(processed_data.get('work_items', [])),
                    'rooms': len(processed_data.get('rooms', []))
                }
            )
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=processed_data.get('rooms', []),
                processing_time=response_time,
                total_work_items=len(processed_data.get('work_items', [])),
                raw_response=raw_response,
                confidence_self_assessment=0.80
            )
            
        except Exception as e:
            self._update_progress(
                ModelStatus.ERROR, 
                f"오류 발생: {str(e)}", 
                0,
                {'error': str(e), 'error_type': type(e).__name__}
            )
            
            self.logger.error(f"Gemini 호출 오류: {e}")
            log_error('gemini_interface_v2', e, {'prompt_length': len(full_prompt) if 'full_prompt' in locals() else 0})
            
            return ModelResponse(
                model_name=self.model_name,
                room_estimates=[],
                processing_time=time.time() - start_time,
                total_work_items=0,
                raw_response=f"Error: {str(e)}",
                confidence_self_assessment=0.0
            )

class ModelOrchestratorV2:
    """개선된 모델 오케스트레이터 - 진행 상황 추적 기능 포함"""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        self.config = config or ConfigLoader()
        self.models = {}
        self.logger = get_logger('model_orchestrator_v2')
        self.progress_history = {}
        
    def initialize_models(self, model_names: List[str]) -> Dict[str, bool]:
        """모델 초기화"""
        status = {}
        
        for model_name in model_names:
            try:
                if model_name.lower() == 'gpt4':
                    api_key = self.config.get_api_key('openai')
                    if api_key:
                        self.models['gpt4'] = GPT4InterfaceV2(api_key)
                        status['gpt4'] = True
                        self.logger.info("GPT-4 모델 초기화 성공")
                    else:
                        status['gpt4'] = False
                        self.logger.warning("OpenAI API 키 없음")
                        
                elif model_name.lower() == 'claude':
                    api_key = self.config.get_api_key('anthropic')
                    if api_key:
                        self.models['claude'] = ClaudeInterfaceV2(api_key)
                        status['claude'] = True
                        self.logger.info("Claude 모델 초기화 성공")
                    else:
                        status['claude'] = False
                        self.logger.warning("Anthropic API 키 없음")
                        
                elif model_name.lower() == 'gemini':
                    api_key = self.config.get_api_key('google')
                    if api_key:
                        self.models['gemini'] = GeminiInterfaceV2(api_key)
                        status['gemini'] = True
                        self.logger.info("Gemini 모델 초기화 성공")
                    else:
                        status['gemini'] = False
                        self.logger.warning("Google API 키 없음")
                        
            except Exception as e:
                status[model_name] = False
                self.logger.error(f"{model_name} 초기화 실패: {e}")
                
        return status
    
    def set_progress_callback(self, model_name: str, callback_fn: Callable):
        """특정 모델의 진행 상황 콜백 설정"""
        if model_name in self.models:
            self.models[model_name].set_progress_callback(callback_fn)
    
    def set_global_progress_callback(self, callback_fn: Callable):
        """모든 모델의 진행 상황 콜백 설정"""
        for model in self.models.values():
            model.set_progress_callback(callback_fn)
    
    async def call_models_parallel(self, prompt: str, json_data: Dict[str, Any]) -> List[ModelResponse]:
        """병렬로 모든 모델 호출 with 진행 상황 추적"""
        if not self.models:
            self.logger.error("초기화된 모델이 없습니다")
            return []
        
        self.logger.info(f"모델 병렬 실행 시작: {list(self.models.keys())}")
        
        # 각 모델별 진행 상황 초기화
        for model_name in self.models.keys():
            self.progress_history[model_name] = []
        
        # 병렬 실행
        tasks = []
        for model_name, model_interface in self.models.items():
            task = model_interface.call_model(prompt, json_data)
            tasks.append(task)
        
        # 모든 태스크 완료 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 처리
        valid_results = []
        for i, result in enumerate(results):
            model_name = list(self.models.keys())[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"{model_name} 실행 실패: {result}")
                # 실패한 경우 빈 응답 생성
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
                self.logger.info(f"{model_name} 실행 완료: {result.total_work_items} 작업 항목")
        
        return valid_results
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """전체 진행 상황 요약"""
        summary = {
            'models': {},
            'overall_status': 'idle',
            'timestamp': datetime.now().isoformat()
        }
        
        for model_name, model_interface in self.models.items():
            summary['models'][model_name] = {
                'status': model_interface.current_status.value,
                'history_count': len(model_interface.progress_callback.status_history)
            }
            
            # 마지막 상태 정보 추가
            if model_interface.progress_callback.status_history:
                last_status = model_interface.progress_callback.status_history[-1]
                summary['models'][model_name]['last_update'] = last_status
        
        # 전체 상태 결정
        all_statuses = [m.current_status for m in self.models.values()]
        if any(s == ModelStatus.ERROR for s in all_statuses):
            summary['overall_status'] = 'error'
        elif all(s == ModelStatus.COMPLETED for s in all_statuses):
            summary['overall_status'] = 'completed'
        elif any(s in [ModelStatus.SENDING_REQUEST, ModelStatus.WAITING_RESPONSE, ModelStatus.PROCESSING_RESPONSE] for s in all_statuses):
            summary['overall_status'] = 'processing'
        
        return summary


# 사용 예시
if __name__ == "__main__":
    async def test_with_progress():
        """진행 상황 모니터링 테스트"""
        
        # 진행 상황 콜백 함수
        def progress_callback(update_data):
            print(f"📊 Progress Update: {update_data}")
        
        # 오케스트레이터 초기화
        orchestrator = ModelOrchestratorV2()
        
        # 모델 초기화
        status = orchestrator.initialize_models(['gpt4', 'claude', 'gemini'])
        print(f"Model initialization status: {status}")
        
        # 전역 진행 상황 콜백 설정
        orchestrator.set_global_progress_callback(progress_callback)
        
        # 테스트 데이터
        test_prompt = "Analyze the following construction data..."
        test_data = {
            "jobsite": "Test Site",
            "floors": [
                {
                    "location": "1st Floor",
                    "rooms": [
                        {
                            "name": "Living Room",
                            "measurements": {
                                "wall_area_sqft": 300,
                                "ceiling_area_sqft": 150
                            }
                        }
                    ]
                }
            ]
        }
        
        # 모델 호출
        print("\n🚀 Starting parallel model execution with progress tracking...\n")
        results = await orchestrator.call_models_parallel(test_prompt, test_data)
        
        # 결과 출력
        print("\n📋 Results:")
        for result in results:
            print(f"  - {result.model_name}: {result.total_work_items} items, {result.processing_time:.2f}s")
        
        # 진행 상황 요약
        summary = orchestrator.get_progress_summary()
        print(f"\n📊 Progress Summary: {json.dumps(summary, indent=2)}")
    
    # 실행
    asyncio.run(test_with_progress())