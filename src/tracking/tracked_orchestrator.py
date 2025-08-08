# src/tracking/tracked_orchestrator.py
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from src.models.model_interface import ModelOrchestrator, ModelResponse
from src.tracking.token_tracker import TokenTracker
from src.tracking.response_parser import TokenResponseParser
from src.utils.logger import get_logger

class TrackedModelOrchestrator(ModelOrchestrator):
    """Model orchestrator with comprehensive token tracking"""
    
    def __init__(self, enable_validation: bool = True, enable_tracking: bool = True, 
                 phase: str = "unknown", session_id: Optional[str] = None):
        super().__init__(enable_validation)
        
        self.enable_tracking = enable_tracking
        self.current_phase = phase
        self.session_id = session_id or str(uuid.uuid4())
        self.logger = get_logger('tracked_orchestrator')
        
        # Initialize token tracking
        self.token_tracker = None
        self.response_parser = None
        
        if enable_tracking:
            try:
                self.token_tracker = TokenTracker()
                self.response_parser = TokenResponseParser()
                self.logger.info(f"Token tracking enabled for phase: {phase}")
            except Exception as e:
                self.logger.warning(f"Token tracking unavailable: {e}")
                self.enable_tracking = False
    
    def set_phase(self, phase: str):
        """Update current phase for tracking"""
        self.current_phase = phase
        self.logger.debug(f"Phase updated to: {phase}")
    
    def set_session_id(self, session_id: str):
        """Update session ID for tracking"""
        self.session_id = session_id
        self.logger.debug(f"Session ID updated to: {session_id}")
    
    async def run_single_model_tracked(self, 
                                     model_name: str, 
                                     prompt: str, 
                                     json_data: Dict[str, Any],
                                     request_id: Optional[str] = None) -> Optional[ModelResponse]:
        """Single model execution with comprehensive token tracking"""
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        if model_name not in self.models:
            self.logger.warning(f"모델 {model_name}을 사용할 수 없습니다. (API 키 확인)")
            return None
        
        start_time = time.time()
        success = True
        error_message = None
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            self.logger.debug(f"모델 {model_name} 실행 시작 (request_id: {request_id})")
            
            # Get the model interface
            model_interface = self.models[model_name]
            
            # Prepare full prompt for token estimation
            full_prompt = model_interface._prepare_prompt(prompt, json_data)
            
            # Call the model
            result = await model_interface.call_model(prompt, json_data)
            
            processing_time = time.time() - start_time
            
            # Extract token usage from the stored response
            if self.enable_tracking and hasattr(model_interface, '_last_api_response'):
                api_provider = self._get_api_provider_for_model(model_name)
                prompt_tokens, completion_tokens = self.response_parser.parse_response_by_provider(
                    api_provider, model_interface._last_api_response, full_prompt
                )
            else:
                # Fallback to estimation
                prompt_tokens = self.response_parser._estimate_tokens(full_prompt)
                completion_tokens = self.response_parser._estimate_tokens(
                    result.raw_response if result and hasattr(result, 'raw_response') else ""
                )
            
            # Check for model failure
            if result is None or (hasattr(result, 'confidence_self_assessment') 
                                and result.confidence_self_assessment == 0.0):
                success = False
                error_message = "Model returned empty or failed response"
            
            self.logger.debug(f"모델 {model_name} 실행 완료 - tokens: {prompt_tokens}/{completion_tokens}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            success = False
            error_message = str(e)
            
            # Still estimate prompt tokens for failed requests
            try:
                model_interface = self.models[model_name]
                full_prompt = model_interface._prepare_prompt(prompt, json_data)
                prompt_tokens = self.response_parser._estimate_tokens(full_prompt)
            except:
                prompt_tokens = self.response_parser._estimate_tokens(prompt)
            
            completion_tokens = 0
            
            self.logger.error(f"모델 {model_name} 실행 오류: {e}")
            
            # Re-raise the exception
            raise
        
        finally:
            # Track usage regardless of success/failure
            if self.enable_tracking and self.token_tracker:
                try:
                    actual_model_name = getattr(self.models[model_name], 'actual_model_name', model_name)
                    
                    self.token_tracker.track_usage(
                        model_name=actual_model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        processing_time=processing_time,
                        phase=self.current_phase,
                        session_id=self.session_id,
                        request_id=request_id,
                        success=success,
                        error_message=error_message
                    )
                    
                    # Real-time console output
                    self._display_usage_info(
                        model_name=actual_model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        processing_time=processing_time,
                        success=success,
                        request_id=request_id
                    )
                    
                except Exception as tracking_error:
                    self.logger.error(f"Failed to track usage for {model_name}: {tracking_error}")
    
    async def run_parallel_tracked(self, 
                                 prompt: str, 
                                 json_data: Dict[str, Any],
                                 model_names: List[str] = None,
                                 enable_validation: Optional[bool] = None,
                                 min_quality_threshold: float = 30.0) -> List[ModelResponse]:
        """Enhanced parallel execution with comprehensive tracking"""
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        # Filter available models
        available_models = [name for name in model_names if name in self.models]
        
        if not available_models:
            self.logger.error("사용 가능한 모델이 없습니다.")
            return []
        
        tracking_info = "ON" if self.enable_tracking else "OFF"
        validation_info = "ON" if (enable_validation if enable_validation is not None else self.enable_validation) else "OFF"
        
        self.logger.info(f"모델 병렬 실행 시작: {available_models} (추적: {tracking_info}, 검증: {validation_info})")
        
        # Display session info
        if self.enable_tracking:
            print(f"\n🔍 Session: {self.session_id} | Phase: {self.current_phase}")
            print(f"📊 Executing {len(available_models)} models in parallel...")
            print("-" * 80)
        
        # Create tracking tasks
        tasks = []
        request_ids = []
        
        for model_name in available_models:
            request_id = str(uuid.uuid4())
            request_ids.append(request_id)
            
            task = self.run_single_model_tracked(
                model_name, prompt, json_data, request_id
            )
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results with validation (use parent class logic)
        successful_results = []
        validation_reports = []
        
        enable_validation_final = enable_validation if enable_validation is not None else self.enable_validation
        
        for i, result in enumerate(results):
            model_name = available_models[i]
            
            if isinstance(result, ModelResponse):
                # Apply validation if enabled (using parent class method)
                if enable_validation_final and self.validation_orchestrator:
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
                            f"✓ {model_name} - {validation_report.quality_level.value.upper()} "
                            f"({validation_report.quality_score:.1f}/100, {result.total_work_items}개 작업)"
                        )
                    else:
                        self.logger.warning(
                            f"✗ {model_name} - 품질 기준 미달 "
                            f"({validation_report.quality_score:.1f}/100, "
                            f"{len(validation_report.issues)}개 이슈)"
                        )
                else:
                    # Basic validation
                    if result.total_work_items > 0 or (result.room_estimates and len(result.room_estimates) > 0):
                        successful_results.append(result)
                        self.logger.info(f"✓ {model_name} 모델 성공 (작업 {result.total_work_items}개)")
                    else:
                        error_msg = result.raw_response[:200] if isinstance(result.raw_response, str) else "빈 응답"
                        self.logger.warning(f"✗ {model_name} 모델 응답 비어있음: {error_msg}")
            
            elif isinstance(result, Exception):
                self.logger.error(f"✗ {model_name} 모델 실행 중 예외: {result}")
        
        # Log validation summary
        if enable_validation_final and validation_reports:
            self._log_validation_summary(validation_reports)
        
        # Display session summary
        if self.enable_tracking:
            await self._display_session_summary()
        
        self.logger.info(f"모델 실행 완료: {len(successful_results)}/{len(available_models)} 성공")
        return successful_results
    
    def _get_api_provider_for_model(self, model_name: str) -> str:
        """Get API provider for model name"""
        model_name_lower = model_name.lower()
        
        if 'gpt' in model_name_lower:
            return 'openai'
        elif 'claude' in model_name_lower:
            return 'anthropic'
        elif 'gemini' in model_name_lower:
            return 'google'
        else:
            return 'unknown'
    
    def _display_usage_info(self, model_name: str, prompt_tokens: int, completion_tokens: int,
                          processing_time: float, success: bool, request_id: str):
        """Display real-time token usage information"""
        if not self.enable_tracking:
            return
        
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost
        try:
            from src.tracking.token_tracker import TokenPricingManager
            cost = TokenPricingManager.calculate_cost(model_name, prompt_tokens, completion_tokens)
            cost_str = f"${cost:.6f}"
        except:
            cost_str = "N/A"
        
        # Status icon
        status_icon = "✅" if success else "❌"
        
        print(f"{status_icon} {model_name:25} | "
              f"{total_tokens:>6} tokens ({prompt_tokens:>4}→{completion_tokens:>4}) | "
              f"{cost_str:>10} | {processing_time:>6.2f}s")
    
    async def _display_session_summary(self):
        """Display session usage summary"""
        if not self.enable_tracking or not self.token_tracker:
            return
        
        try:
            # Get current session stats
            stats = self.token_tracker.get_usage_stats(days=1)
            
            if stats["summary"]["total_requests"] > 0:
                print("-" * 80)
                print(f"📊 Session Summary - {stats['summary']['total_requests']} requests")
                print(f"💰 Total cost: ${stats['summary']['total_cost']:.6f}")
                print(f"🎯 Success rate: {stats['summary']['success_rate']:.1%}")
                print(f"⏱️  Avg time: {stats['summary']['avg_processing_time']:.2f}s")
                
                if stats["breakdown"]["by_model"]:
                    print("\n📋 By Model:")
                    for model_key, data in stats["breakdown"]["by_model"].items():
                        print(f"  • {model_key}: {data['requests']} reqs, ${data['cost']:.6f}")
                
                print("=" * 80 + "\n")
        except Exception as e:
            self.logger.debug(f"Failed to display session summary: {e}")
    
    def get_session_usage_stats(self) -> Optional[Dict[str, Any]]:
        """Get usage statistics for current session"""
        if not self.enable_tracking or not self.token_tracker:
            return None
        
        try:
            return self.token_tracker.get_usage_stats(days=1)
        except Exception as e:
            self.logger.error(f"Failed to get session stats: {e}")
            return None
    
    def get_cost_projection(self) -> Optional[Dict[str, float]]:
        """Get cost projection based on recent usage"""
        if not self.enable_tracking or not self.token_tracker:
            return None
        
        try:
            return self.token_tracker.get_cost_projection()
        except Exception as e:
            self.logger.error(f"Failed to get cost projection: {e}")
            return None