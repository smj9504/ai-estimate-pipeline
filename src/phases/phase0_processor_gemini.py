# src/phases/phase0_processor_gemini.py
"""
Phase 0 프로세서 - Gemini 단일 모델 사용
빠른 응답을 위해 Gemini 모델만 사용하는 간소화된 버전
"""
import json
import asyncio
from typing import Dict, Any, List
from pathlib import Path

import google.generativeai as genai
from src.utils.prompt_manager import PromptManager
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

class Phase0ProcessorGemini:
    """
    Phase 0: Gemini를 사용한 데이터 생성
    - Gemini는 응답이 빠르고 안정적임
    - 단일 모델로 타임아웃 문제 해결
    """
    
    def __init__(self):
        self.config = ConfigLoader()
        self.prompt_manager = PromptManager()
        self.logger = get_logger('phase0_gemini')
        
        # Gemini 설정
        api_key = self.config.get_api_key('google')
        if not api_key:
            raise ValueError("Google API 키가 설정되지 않았습니다.")
        
        genai.configure(api_key=api_key)
        # Gemini 1.5 Flash - 빠른 응답
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.logger.info("Gemini 모델 초기화 완료")
    
    def load_test_data(self) -> Dict[str, Any]:
        """테스트 데이터 로드"""
        test_data_path = Path("test_data")
        
        # Measurement 데이터 로드
        measurement_file = test_data_path / "sample_measurement.json"
        with open(measurement_file, 'r', encoding='utf-8') as f:
            measurement_data = json.load(f)
        
        # Demo scope 데이터 로드
        demo_file = test_data_path / "sample_demo.json"
        with open(demo_file, 'r', encoding='utf-8') as f:
            demo_data = json.load(f)
        
        # Intake form 로드
        intake_file = test_data_path / "sample_intake_form.txt"
        with open(intake_file, 'r', encoding='utf-8') as f:
            intake_form = f.read()
        
        return {
            'measurement_data': measurement_data,
            'demolition_scope_data': demo_data,
            'intake_form': intake_form
        }
    
    def prepare_prompt(self, data: Dict[str, Any]) -> str:
        """Gemini용 최적화된 프롬프트 준비"""
        # 간결한 프롬프트 템플릿
        prompt_template = """You are a construction estimation specialist. Generate a detailed scope of work based on the provided data.

TASK: Combine measurement, demolition, and intake form data into a unified work scope.

KEY REQUIREMENTS:
1. Apply "Remove & Replace" strategy
2. Use exact measurements from provided data
3. Include all necessary reconstruction tasks
4. Be specific with quantities and units

MEASUREMENT DATA:
{measurement_json}

DEMOLITION SCOPE DATA:
{demo_json}

INTAKE FORM:
{intake_text}

OUTPUT FORMAT:
For each room, list:
- Room name and location
- Demolition tasks (if not already done)
- Installation tasks with quantities
- Materials needed
- Special considerations

Be concise but comprehensive. Focus on actionable work items."""
        
        # 데이터를 간결한 JSON으로 변환
        measurement_json = json.dumps(data['measurement_data'], separators=(',', ':'))
        demo_json = json.dumps(data['demolition_scope_data'], separators=(',', ':'))
        
        prompt = prompt_template.format(
            measurement_json=measurement_json,
            demo_json=demo_json,
            intake_text=data['intake_form'][:1000]  # 인테이크 폼 길이 제한
        )
        
        self.logger.info(f"프롬프트 크기: {len(prompt)} characters")
        return prompt
    
    async def call_gemini_async(self, prompt: str) -> str:
        """Gemini 비동기 호출"""
        try:
            self.logger.info("Gemini API 호출 시작")
            
            # 비동기로 실행
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=3000,
                    temperature=0.1,
                    candidate_count=1
                ),
                safety_settings={
                    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                }
            )
            
            if response.text:
                self.logger.info(f"Gemini 응답 수신 완료: {len(response.text)} characters")
                return response.text
            else:
                self.logger.error("Gemini 응답이 비어있습니다")
                return "ERROR: Empty response from Gemini"
                
        except Exception as e:
            self.logger.error(f"Gemini API 호출 오류: {e}")
            return f"ERROR: {str(e)}"
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """Gemini 응답 파싱"""
        try:
            # JSON 형식인 경우
            if response_text.strip().startswith('{'):
                return json.loads(response_text)
            
            # 텍스트 형식인 경우 파싱
            result = {
                'floors': [],
                'raw_response': response_text
            }
            
            lines = response_text.split('\n')
            current_floor = None
            current_room = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 층 감지
                if 'floor' in line.lower() or 'basement' in line.lower():
                    if current_floor and current_room:
                        current_floor['rooms'].append(current_room)
                    if current_floor:
                        result['floors'].append(current_floor)
                    
                    current_floor = {
                        'location': line,
                        'rooms': []
                    }
                    current_room = None
                
                # 방 감지
                elif any(room_type in line.lower() for room_type in ['room', 'kitchen', 'bathroom', 'bedroom']):
                    if current_room and current_floor:
                        current_floor['rooms'].append(current_room)
                    
                    current_room = {
                        'name': line.replace(':', '').strip(),
                        'tasks': []
                    }
                
                # 작업 항목 감지
                elif line.startswith(('-', '*', '•')) or any(line.startswith(f"{i}.") for i in range(1, 20)):
                    if current_room:
                        task = line.lstrip('-*•0123456789. ').strip()
                        current_room['tasks'].append(task)
            
            # 마지막 항목 추가
            if current_room and current_floor:
                current_floor['rooms'].append(current_room)
            if current_floor:
                result['floors'].append(current_floor)
            
            return result
            
        except Exception as e:
            self.logger.error(f"응답 파싱 오류: {e}")
            return {
                'error': str(e),
                'raw_response': response_text
            }
    
    async def process(self, custom_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Phase 0 처리 - Gemini 단일 모델 사용"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("Phase 0 처리 시작 (Gemini 단일 모델)")
            self.logger.info("=" * 50)
            
            # 데이터 로드
            if custom_data:
                data = custom_data
                self.logger.info("커스텀 데이터 사용")
            else:
                data = self.load_test_data()
                self.logger.info("테스트 데이터 로드 완료")
            
            # 프롬프트 준비
            prompt = self.prepare_prompt(data)
            
            # Gemini 호출 (타임아웃 60초)
            try:
                response = await asyncio.wait_for(
                    self.call_gemini_async(prompt),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                self.logger.error("Gemini 타임아웃 (60초)")
                return {
                    'success': False,
                    'error': 'Gemini timeout after 60 seconds',
                    'model': 'gemini-1.5-flash'
                }
            
            # 응답 파싱
            parsed_result = self.parse_response(response)
            
            # 결과 구성
            result = {
                'success': True,
                'phase': 0,
                'model': 'gemini-1.5-flash',
                'result': parsed_result,
                'metadata': {
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    'floors_count': len(parsed_result.get('floors', [])),
                    'total_rooms': sum(len(floor.get('rooms', [])) 
                                     for floor in parsed_result.get('floors', []))
                }
            }
            
            self.logger.info(f"Phase 0 처리 완료: {result['metadata']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Phase 0 처리 오류: {e}")
            return {
                'success': False,
                'phase': 0,
                'error': str(e),
                'model': 'gemini-1.5-flash'
            }
    
    def save_result(self, result: Dict[str, Any], output_path: str = "output/phase0_result_gemini.json"):
        """결과 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"결과 저장 완료: {output_file}")


# 실행 코드
async def main():
    """메인 실행 함수"""
    processor = Phase0ProcessorGemini()
    result = await processor.process()
    
    if result['success']:
        processor.save_result(result)
        print("\n✅ Phase 0 처리 성공 (Gemini)")
        print(f"   - 모델: {result['model']}")
        print(f"   - 층 수: {result['metadata']['floors_count']}")
        print(f"   - 총 방 수: {result['metadata']['total_rooms']}")
    else:
        print(f"\n❌ Phase 0 처리 실패: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())