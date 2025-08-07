# src/utils/prompt_manager.py
"""
프롬프트 관리 시스템
- 템플릿 변수 지원
- 버전 관리 기능
- 프롬프트 검증
"""
import os
import re
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime
import json

class PromptManager:
    """
    Phase별 프롬프트 관리 클래스
    - 파일 기반 프롬프트 로드
    - 템플릿 변수 치환
    - 버전 관리 지원
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Args:
            prompts_dir: 프롬프트 파일들이 저장된 디렉토리 경로
        """
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"프롬프트 디렉토리를 찾을 수 없습니다: {prompts_dir}")
        
        # 프롬프트 캐시 (성능 향상)
        self._prompt_cache = {}
        
        # 기본 템플릿 변수
        self.default_variables = {
            'location': 'DMV area',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat()
        }
    
    def load_prompt(self, phase_number: int, version: Optional[str] = None) -> str:
        """
        Phase별 프롬프트 로드
        
        Args:
            phase_number: Phase 번호 (0-6)
            version: 프롬프트 버전 (예: 'v2', 'v3') - None이면 기본 버전 사용
        
        Returns:
            프롬프트 텍스트
        """
        # 파일명 구성
        if version:
            prompt_filename = f"phase{phase_number}_prompt_{version}.txt"
        else:
            prompt_filename = f"phase{phase_number}_prompt.txt"
        
        prompt_path = self.prompts_dir / prompt_filename
        
        # 캐시 확인
        cache_key = f"{phase_number}_{version or 'default'}"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        # 파일 존재 확인
        if not prompt_path.exists():
            # 버전이 지정된 경우 기본 버전으로 폴백
            if version:
                print(f"버전 {version} 프롬프트를 찾을 수 없습니다. 기본 버전으로 시도합니다.")
                return self.load_prompt(phase_number, None)
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
        
        # 프롬프트 로드
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        # 캐시에 저장
        self._prompt_cache[cache_key] = prompt_text
        
        return prompt_text
    
    def load_system_prompt(self) -> str:
        """
        시스템 프롬프트 로드
        
        Returns:
            시스템 프롬프트 텍스트
        """
        system_prompt_path = self.prompts_dir / "system_prompt.txt"
        
        if not system_prompt_path.exists():
            # 시스템 프롬프트가 없으면 빈 문자열 반환
            return ""
        
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_prompt_with_variables(self, 
                                  phase_number: int,
                                  variables: Optional[Dict[str, Any]] = None,
                                  version: Optional[str] = None,
                                  include_system_prompt: bool = True) -> str:
        """
        템플릿 변수를 치환하여 프롬프트 로드
        
        Args:
            phase_number: Phase 번호
            variables: 치환할 변수 딕셔너리
            version: 프롬프트 버전
            include_system_prompt: 시스템 프롬프트 포함 여부
        
        Returns:
            변수가 치환된 프롬프트 텍스트
        
        Example:
            프롬프트에 {project_id}, {client_name} 같은 변수 포함 시
            variables={'project_id': 'PRJ001', 'client_name': 'John Doe'}로 치환
        """
        # Phase별 프롬프트 로드
        prompt_text = self.load_prompt(phase_number, version)
        
        # 시스템 프롬프트 포함 옵션
        if include_system_prompt:
            system_prompt = self.load_system_prompt()
            if system_prompt:
                # Phase 정보로 시스템 프롬프트 변수 치환
                phase_descriptions = {
                    0: "Generate Scope of Work Data",
                    1: "Merge Measurement & Work Scope",
                    2: "Quantity Survey",
                    3: "Market Research",
                    4: "Timeline & Disposal Calculation",
                    5: "Final Estimate Completion",
                    6: "Formatting to JSON"
                }
                
                phase_instruction = {
                    0: "Combine measurement, demolition, and intake form data into a unified JSON format.",
                    1: "Merge measurements with work scope requirements. Apply Remove & Replace logic.",
                    2: "Calculate detailed quantities for all work items based on measurements and scope.",
                    3: "Research current material and labor costs specific to the DMV area.",
                    4: "Estimate project timeline and calculate disposal costs.",
                    5: "Compile comprehensive estimate with all costs and details.",
                    6: "Format final estimate to client-required JSON structure."
                }
                
                system_prompt = system_prompt.replace(
                    "{phase_number}", str(phase_number)
                ).replace(
                    "{phase_description}", phase_descriptions.get(phase_number, f"Phase {phase_number}")
                ).replace(
                    "{phase_specific_instruction}", phase_instruction.get(phase_number, "")
                )
                
                # 시스템 프롬프트를 Phase 프롬프트 앞에 추가
                prompt_text = system_prompt + "\n\n" + prompt_text
        
        # 변수 병합 (기본값 + 사용자 제공 값)
        all_variables = {**self.default_variables}
        if variables:
            all_variables.update(variables)
        
        # 템플릿 변수 치환 ({variable_name} 형태)
        for key, value in all_variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt_text:
                prompt_text = prompt_text.replace(placeholder, str(value))
        
        # 치환되지 않은 변수 검출 (디버깅용)
        unresolved = self._find_unresolved_variables(prompt_text)
        if unresolved:
            print(f"경고: 치환되지 않은 변수가 있습니다: {unresolved}")
        
        return prompt_text
    
    def _find_unresolved_variables(self, text: str) -> List[str]:
        """
        치환되지 않은 템플릿 변수 찾기
        
        Args:
            text: 검사할 텍스트
        
        Returns:
            치환되지 않은 변수 이름 리스트
        """
        # JSON 데이터 블록을 먼저 제거 (중괄호 내용이 JSON 형태인 경우)
        # JSON 블록은 여러 줄에 걸쳐 있고, 들여쓰기가 있으며, : 또는 [ 를 포함
        temp_text = text
        
        # JSON 블록 패턴 제거 (여러 줄 JSON 데이터)
        json_block_pattern = r'\{[^{}]*[\[\]":,\n\t][^{}]*\}'
        temp_text = re.sub(json_block_pattern, '', temp_text, flags=re.MULTILINE | re.DOTALL)
        
        # 템플릿 변수 패턴 찾기 (단순한 {variable_name} 형태만)
        # 변수명은 알파벳, 숫자, 언더스코어만 허용
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        matches = re.findall(pattern, temp_text)
        
        # JSON 데이터 플레이스홀더와 시스템 프롬프트 변수는 제외
        excluded = [
            'measurement_data', 
            'demolition_scope_data', 
            'intake_form',
            'phase_number',
            'phase_description',
            'phase_specific_instruction'
        ]
        
        # 중복 제거 및 제외 항목 필터링
        unique_matches = list(set(m for m in matches if m not in excluded))
        return unique_matches
    
    def list_available_prompts(self) -> Dict[int, List[str]]:
        """
        사용 가능한 모든 프롬프트 목록 조회
        
        Returns:
            {phase_number: [versions]} 형태의 딕셔너리
        """
        available = {}
        
        for prompt_file in self.prompts_dir.glob("phase*_prompt*.txt"):
            filename = prompt_file.name
            
            # Phase 번호 추출
            phase_match = re.match(r'phase(\d+)_prompt', filename)
            if not phase_match:
                continue
            
            phase_number = int(phase_match.group(1))
            
            # 버전 추출
            version_match = re.match(r'phase\d+_prompt_(.+)\.txt', filename)
            version = version_match.group(1) if version_match else 'default'
            
            if phase_number not in available:
                available[phase_number] = []
            available[phase_number].append(version)
        
        return available
    
    def validate_prompt(self, phase_number: int, version: Optional[str] = None) -> Dict[str, Any]:
        """
        프롬프트 유효성 검증
        
        Args:
            phase_number: Phase 번호
            version: 프롬프트 버전
        
        Returns:
            검증 결과 딕셔너리
        """
        try:
            prompt_text = self.load_prompt(phase_number, version)
            
            # 검증 항목들
            validation_result = {
                'valid': True,
                'phase': phase_number,
                'version': version or 'default',
                'length': len(prompt_text),
                'has_instructions': 'Instructions' in prompt_text or 'instructions' in prompt_text,
                'has_output_format': 'Output' in prompt_text or 'output' in prompt_text,
                'template_variables': self._find_unresolved_variables(prompt_text),
                'errors': []
            }
            
            # 최소 길이 체크
            if validation_result['length'] < 100:
                validation_result['valid'] = False
                validation_result['errors'].append("프롬프트가 너무 짧습니다 (100자 미만)")
            
            # 필수 섹션 체크
            if not validation_result['has_instructions']:
                validation_result['errors'].append("Instructions 섹션이 없습니다")
            
            # Phase별 특정 요구사항 체크
            if phase_number == 0:
                # Phase 0은 데이터 병합 관련 키워드 필요
                required_keywords = ['measurement_data', 'demolition_scope_data', 'intake_form']
                for keyword in required_keywords:
                    if keyword not in prompt_text:
                        validation_result['errors'].append(f"필수 키워드 누락: {keyword}")
            
            elif phase_number in [1, 2]:
                # Phase 1, 2는 Remove & Replace 로직 필요
                if 'Remove & Replace' not in prompt_text and 'Remove and Replace' not in prompt_text:
                    validation_result['errors'].append("Remove & Replace 전략이 명시되지 않았습니다")
            
            if validation_result['errors']:
                validation_result['valid'] = False
            
            return validation_result
            
        except FileNotFoundError as e:
            return {
                'valid': False,
                'phase': phase_number,
                'version': version or 'default',
                'errors': [str(e)]
            }
    
    def save_prompt_version(self, phase_number: int, prompt_text: str, version: str) -> str:
        """
        프롬프트를 새 버전으로 저장
        
        Args:
            phase_number: Phase 번호
            prompt_text: 저장할 프롬프트 텍스트
            version: 버전 이름 (예: 'v2', 'test', '20240101')
        
        Returns:
            저장된 파일 경로
        """
        filename = f"phase{phase_number}_prompt_{version}.txt"
        file_path = self.prompts_dir / filename
        
        # 기존 파일 존재 확인
        if file_path.exists():
            raise ValueError(f"이미 존재하는 버전입니다: {version}")
        
        # 프롬프트 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        # 캐시 무효화
        cache_key = f"{phase_number}_{version}"
        if cache_key in self._prompt_cache:
            del self._prompt_cache[cache_key]
        
        print(f"프롬프트 버전 저장 완료: {file_path}")
        return str(file_path)
    
    def get_prompt_info(self, phase_number: int, version: Optional[str] = None) -> Dict[str, Any]:
        """
        프롬프트 메타데이터 조회
        
        Args:
            phase_number: Phase 번호
            version: 프롬프트 버전
        
        Returns:
            프롬프트 정보 딕셔너리
        """
        prompt_filename = f"phase{phase_number}_prompt_{version}.txt" if version else f"phase{phase_number}_prompt.txt"
        prompt_path = self.prompts_dir / prompt_filename
        
        if not prompt_path.exists():
            return {'error': 'Prompt not found'}
        
        stats = prompt_path.stat()
        prompt_text = self.load_prompt(phase_number, version)
        
        return {
            'phase': phase_number,
            'version': version or 'default',
            'file_path': str(prompt_path),
            'file_size': stats.st_size,
            'modified_time': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'character_count': len(prompt_text),
            'line_count': prompt_text.count('\n') + 1,
            'template_variables': self._find_unresolved_variables(prompt_text),
            'validation': self.validate_prompt(phase_number, version)
        }
    
    def clear_cache(self):
        """프롬프트 캐시 클리어"""
        self._prompt_cache.clear()
        print("프롬프트 캐시가 클리어되었습니다.")


# 사용 예시
if __name__ == "__main__":
    # PromptManager 초기화
    manager = PromptManager()
    
    # Phase 0 프롬프트 로드
    prompt = manager.load_prompt(0)
    print(f"Phase 0 프롬프트 길이: {len(prompt)}")
    
    # 템플릿 변수와 함께 로드
    prompt_with_vars = manager.load_prompt_with_variables(
        phase_number=0,
        variables={
            'project_id': 'PRJ-2024-001',
            'client_name': 'ABC Construction',
            'location': 'Washington DC'
        }
    )
    
    # 사용 가능한 프롬프트 목록
    available = manager.list_available_prompts()
    print(f"사용 가능한 프롬프트: {available}")
    
    # 프롬프트 검증
    validation = manager.validate_prompt(0)
    print(f"Phase 0 프롬프트 검증: {validation}")