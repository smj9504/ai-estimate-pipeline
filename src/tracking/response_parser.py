# src/tracking/response_parser.py
from typing import Dict, Any, Optional, Tuple
import json
import re
from src.utils.logger import get_logger

class TokenResponseParser:
    """Parses API responses to extract token usage information"""
    
    def __init__(self):
        self.logger = get_logger('token_response_parser')
    
    def parse_openai_response(self, response) -> Tuple[int, int]:
        """Parse OpenAI API response for token usage"""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                return prompt_tokens, completion_tokens
            else:
                self.logger.warning("OpenAI response missing usage information")
                return 0, 0
        except Exception as e:
            self.logger.error(f"Error parsing OpenAI response: {e}")
            return 0, 0
    
    def parse_anthropic_response(self, response, prompt_text: str) -> Tuple[int, int]:
        """Parse Anthropic API response for token usage"""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                input_tokens = getattr(usage, 'input_tokens', 0)
                output_tokens = getattr(usage, 'output_tokens', 0)
                return input_tokens, output_tokens
            else:
                # Fallback: estimate tokens based on text length
                prompt_tokens = self._estimate_tokens(prompt_text)
                completion_text = ""
                if hasattr(response, 'content') and response.content:
                    completion_text = response.content[0].text if response.content else ""
                completion_tokens = self._estimate_tokens(completion_text)
                
                self.logger.warning(f"Anthropic response missing usage info, estimated: {prompt_tokens}/{completion_tokens}")
                return prompt_tokens, completion_tokens
        except Exception as e:
            self.logger.error(f"Error parsing Anthropic response: {e}")
            # Fallback estimation
            prompt_tokens = self._estimate_tokens(prompt_text)
            return prompt_tokens, 0
    
    def parse_google_response(self, response, prompt_text: str) -> Tuple[int, int]:
        """Parse Google Gemini API response for token usage"""
        try:
            # Google's response structure varies, check for usage_metadata
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                prompt_tokens = getattr(usage, 'prompt_token_count', 0)
                completion_tokens = getattr(usage, 'candidates_token_count', 0)
                return prompt_tokens, completion_tokens
            else:
                # Fallback: estimate tokens
                prompt_tokens = self._estimate_tokens(prompt_text)
                completion_text = ""
                if hasattr(response, 'text'):
                    completion_text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            completion_text = candidate.content.parts[0].text
                
                completion_tokens = self._estimate_tokens(completion_text)
                
                self.logger.warning(f"Google response missing usage info, estimated: {prompt_tokens}/{completion_tokens}")
                return prompt_tokens, completion_tokens
        except Exception as e:
            self.logger.error(f"Error parsing Google response: {e}")
            # Fallback estimation
            prompt_tokens = self._estimate_tokens(prompt_text)
            return prompt_tokens, 0
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        if not text:
            return 0
        
        # Rough estimation: ~4 characters per token for English text
        # This is a conservative estimate that works reasonably well
        char_count = len(text)
        estimated_tokens = max(1, char_count // 4)
        
        # Account for JSON structure, special characters
        if '{' in text and '}' in text:
            # JSON content typically has more tokens per character
            estimated_tokens = int(estimated_tokens * 1.2)
        
        return estimated_tokens
    
    def parse_response_by_provider(self, 
                                 provider: str, 
                                 response: Any, 
                                 prompt_text: str = "") -> Tuple[int, int]:
        """Parse response based on API provider"""
        if provider == 'openai':
            return self.parse_openai_response(response)
        elif provider == 'anthropic':
            return self.parse_anthropic_response(response, prompt_text)
        elif provider == 'google':
            return self.parse_google_response(response, prompt_text)
        else:
            self.logger.warning(f"Unknown provider: {provider}, using text estimation")
            prompt_tokens = self._estimate_tokens(prompt_text)
            response_text = str(response) if response else ""
            completion_tokens = self._estimate_tokens(response_text)
            return prompt_tokens, completion_tokens

class TrackedModelInterface:
    """Wrapper for model interfaces that adds token tracking"""
    
    def __init__(self, model_interface, token_tracker, phase: str = "unknown"):
        self.model_interface = model_interface
        self.token_tracker = token_tracker
        self.phase = phase
        self.parser = TokenResponseParser()
        self.logger = get_logger(f'tracked_{model_interface.__class__.__name__.lower()}')
        
        # Determine API provider
        class_name = model_interface.__class__.__name__.lower()
        if 'gpt' in class_name:
            self.api_provider = 'openai'
        elif 'claude' in class_name:
            self.api_provider = 'anthropic'
        elif 'gemini' in class_name:
            self.api_provider = 'google'
        else:
            self.api_provider = 'unknown'
    
    async def call_model(self, prompt: str, json_data: Dict[str, Any], 
                        session_id: Optional[str] = None,
                        request_id: Optional[str] = None) -> Any:
        """Tracked model call with token usage monitoring"""
        
        import uuid
        import time
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        start_time = time.time()
        success = True
        error_message = None
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            # Prepare full prompt for token estimation
            full_prompt = self.model_interface._prepare_prompt(prompt, json_data)
            
            # Call the original model
            response = await self.model_interface.call_model(prompt, json_data)
            
            processing_time = time.time() - start_time
            
            # Extract token usage from the response
            if hasattr(self.model_interface, 'client') and self.model_interface.client:
                # For interfaces that store the raw API response
                if self.api_provider == 'openai' and hasattr(response, 'raw_response'):
                    # Try to get usage from stored response
                    if hasattr(self.model_interface, '_last_api_response'):
                        raw_response = self.model_interface._last_api_response
                        prompt_tokens, completion_tokens = self.parser.parse_openai_response(raw_response)
                    else:
                        # Fallback to estimation
                        prompt_tokens = self.parser._estimate_tokens(full_prompt)
                        completion_tokens = self.parser._estimate_tokens(
                            response.raw_response if hasattr(response, 'raw_response') else ""
                        )
                elif self.api_provider == 'anthropic':
                    if hasattr(self.model_interface, '_last_api_response'):
                        raw_response = self.model_interface._last_api_response
                        prompt_tokens, completion_tokens = self.parser.parse_anthropic_response(
                            raw_response, full_prompt
                        )
                    else:
                        prompt_tokens = self.parser._estimate_tokens(full_prompt)
                        completion_tokens = self.parser._estimate_tokens(
                            response.raw_response if hasattr(response, 'raw_response') else ""
                        )
                elif self.api_provider == 'google':
                    if hasattr(self.model_interface, '_last_api_response'):
                        raw_response = self.model_interface._last_api_response
                        prompt_tokens, completion_tokens = self.parser.parse_google_response(
                            raw_response, full_prompt
                        )
                    else:
                        prompt_tokens = self.parser._estimate_tokens(full_prompt)
                        completion_tokens = self.parser._estimate_tokens(
                            response.raw_response if hasattr(response, 'raw_response') else ""
                        )
                else:
                    # Fallback estimation for unknown providers
                    prompt_tokens = self.parser._estimate_tokens(full_prompt)
                    completion_tokens = self.parser._estimate_tokens(
                        response.raw_response if hasattr(response, 'raw_response') else ""
                    )
            else:
                # Pure estimation fallback
                prompt_tokens = self.parser._estimate_tokens(full_prompt)
                completion_tokens = self.parser._estimate_tokens(
                    response.raw_response if hasattr(response, 'raw_response') else ""
                )
            
            # Check for actual failure
            if hasattr(response, 'confidence_self_assessment') and response.confidence_self_assessment == 0.0:
                success = False
                error_message = "Model returned zero confidence"
            
        except Exception as e:
            processing_time = time.time() - start_time
            success = False
            error_message = str(e)
            
            # Still estimate prompt tokens for failed requests
            try:
                full_prompt = self.model_interface._prepare_prompt(prompt, json_data)
                prompt_tokens = self.parser._estimate_tokens(full_prompt)
            except:
                prompt_tokens = self.parser._estimate_tokens(prompt)
            
            completion_tokens = 0
            
            # Re-raise the exception
            raise
        
        finally:
            # Track usage regardless of success/failure
            try:
                self.token_tracker.track_usage(
                    model_name=self.model_interface.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    processing_time=processing_time,
                    phase=self.phase,
                    session_id=session_id,
                    request_id=request_id,
                    success=success,
                    error_message=error_message
                )
            except Exception as tracking_error:
                self.logger.error(f"Failed to track usage: {tracking_error}")
        
        return response