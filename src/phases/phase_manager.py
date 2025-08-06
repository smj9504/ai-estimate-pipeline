# src/phases/phase_manager.py
"""
Phase 관리 시스템 - 7단계 파이프라인 오케스트레이션
Phase 0-6 순차 실행 및 사용자 확인/수정 기능 포함
"""
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from src.phases.phase0_processor import Phase0Processor
from src.phases.phase1_processor import Phase1Processor
from src.phases.phase2_processor import Phase2Processor
from src.utils.config_loader import ConfigLoader

class PhaseResult:
    """Phase 실행 결과 저장 클래스"""
    def __init__(self, phase_number: int, input_data: Dict, output_data: Dict, 
                 metadata: Dict = None, user_approved: bool = False):
        self.phase_number = phase_number
        self.input_data = input_data
        self.output_data = output_data
        self.metadata = metadata or {}
        self.user_approved = user_approved
        self.timestamp = datetime.now().isoformat()
        self.session_id = str(uuid.uuid4())
    
    def to_dict(self):
        return {
            'phase_number': self.phase_number,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'metadata': self.metadata,
            'user_approved': self.user_approved,
            'timestamp': self.timestamp,
            'session_id': self.session_id
        }

class PhaseManager:
    """
    Multi-Phase Pipeline 관리자
    - Phase 0-6 순차 실행
    - Phase 간 데이터 전달
    - 사용자 확인 포인트 관리
    - 세션 상태 저장/복원
    """
    
    PHASE_DEFINITIONS = {
        0: {
            'name': 'generate_scope_of_work',
            'description': 'Generate Scope of Work (Single Model)',
            'processor_class': Phase0Processor,
            'requires_review': True,
            'multi_model': False
        },
        1: {
            'name': 'merge_measurement_work_scope',
            'description': 'Merge Measurement & Work Scope (Multi-Model)',
            'processor_class': Phase1Processor,
            'requires_review': True,
            'multi_model': True
        },
        2: {
            'name': 'quantity_survey',
            'description': 'Quantity Survey (Multi-Model)',
            'processor_class': Phase2Processor,
            'requires_review': True,
            'multi_model': True
        },
        3: {
            'name': 'market_research',
            'description': 'Market Research & Pricing',
            'processor_class': None,  # TODO: Implement Phase3Processor
            'requires_review': True,
            'multi_model': True
        },
        4: {
            'name': 'timeline_disposal',
            'description': 'Timeline & Disposal Calculation',
            'processor_class': None,  # TODO: Implement Phase4Processor
            'requires_review': False,
            'multi_model': True
        },
        5: {
            'name': 'final_estimate',
            'description': 'Final Estimate Completion',
            'processor_class': None,  # TODO: Implement Phase5Processor
            'requires_review': True,
            'multi_model': True
        },
        6: {
            'name': 'format_to_json',
            'description': 'Format to Final JSON',
            'processor_class': None,  # TODO: Implement Phase6Processor
            'requires_review': False,
            'multi_model': False
        }
    }
    
    def __init__(self):
        self.config = ConfigLoader().load_config()
        self.phase_results = {}  # session_id -> {phase_num: PhaseResult}
        self.active_sessions = {}  # session_id -> current_phase
        
        # Phase 프로세서 인스턴스 생성
        self.processors = {}
        for phase_num, definition in self.PHASE_DEFINITIONS.items():
            if definition['processor_class']:
                self.processors[phase_num] = definition['processor_class'](self.config)
    
    async def start_pipeline(self, 
                            initial_data: Dict[str, Any],
                            start_phase: int = 0,
                            model_to_use: str = "gpt4") -> str:
        """
        파이프라인 시작 - 세션 생성 및 Phase 0 실행
        
        Args:
            initial_data: 초기 입력 데이터
            start_phase: 시작할 Phase 번호 (기본: 0)
            model_to_use: Phase 0에서 사용할 모델 (단일 모델)
        
        Returns:
            session_id: 파이프라인 세션 식별자
        """
        session_id = str(uuid.uuid4())
        self.phase_results[session_id] = {}
        self.active_sessions[session_id] = start_phase
        
        print(f"파이프라인 시작 - 세션 ID: {session_id}")
        
        # Phase 0 실행 (단일 모델)
        if start_phase == 0:
            result = await self.execute_phase(
                session_id=session_id,
                phase_number=0,
                input_data=initial_data,
                model_to_use=model_to_use
            )
        else:
            # 다른 Phase부터 시작하는 경우
            result = await self.execute_phase(
                session_id=session_id,
                phase_number=start_phase,
                input_data=initial_data,
                models_to_use=["gpt4", "claude", "gemini"]
            )
        
        return session_id
    
    async def execute_phase(self, 
                          session_id: str,
                          phase_number: int,
                          input_data: Dict[str, Any] = None,
                          model_to_use: str = None,
                          models_to_use: List[str] = None) -> PhaseResult:
        """
        특정 Phase 실행
        
        Args:
            session_id: 파이프라인 세션 ID
            phase_number: 실행할 Phase 번호
            input_data: 입력 데이터 (None이면 이전 Phase 결과 사용)
            model_to_use: Phase 0용 단일 모델
            models_to_use: Phase 1-6용 멀티모델 리스트
        """
        print(f"Phase {phase_number} 실행 시작 - 세션: {session_id}")
        
        # Phase 정의 확인
        if phase_number not in self.PHASE_DEFINITIONS:
            raise ValueError(f"유효하지 않은 Phase 번호: {phase_number}")
        
        phase_def = self.PHASE_DEFINITIONS[phase_number]
        
        # 입력 데이터 결정
        if input_data is None and phase_number > 0:
            # 이전 Phase 결과를 입력으로 사용
            prev_result = self.phase_results[session_id].get(phase_number - 1)
            if not prev_result:
                raise ValueError(f"이전 Phase {phase_number - 1} 결과가 없습니다.")
            if phase_def['requires_review'] and not prev_result.user_approved:
                raise ValueError(f"Phase {phase_number - 1} 결과가 아직 승인되지 않았습니다.")
            input_data = prev_result.output_data
        
        # Phase 프로세서 가져오기
        processor = self.processors.get(phase_number)
        if not processor:
            raise NotImplementedError(f"Phase {phase_number} 프로세서가 아직 구현되지 않았습니다.")
        
        # Phase 실행
        try:
            if phase_number == 0:
                # Phase 0: 단일 모델 사용
                if not isinstance(input_data, dict) or 'measurement_data' not in input_data:
                    raise ValueError("Phase 0에는 measurement_data, demolition_scope_data, scope_of_work_intake_form이 필요합니다")
                
                output_data = await processor.process(
                    measurement_data=input_data['measurement_data'],
                    demolition_scope_data=input_data['demolition_scope_data'],
                    scope_of_work_intake_form=input_data['scope_of_work_intake_form'],
                    model_to_use=model_to_use or "gpt4",
                    project_id=input_data.get('project_id')
                )
            else:
                # Phase 1-6: 멀티모델 사용
                if not models_to_use:
                    models_to_use = ["gpt4", "claude", "gemini"]
                
                output_data = await processor.process(
                    phase0_output=input_data if phase_number == 1 else None,
                    phase1_output=input_data if phase_number == 2 else None,
                    models_to_use=models_to_use,
                    project_id=input_data.get('project_id')
                )
        
        except Exception as e:
            print(f"Phase {phase_number} 실행 오류: {e}")
            output_data = {
                'phase': phase_number,
                'error': str(e),
                'success': False
            }
        
        # 결과 저장
        result = PhaseResult(
            phase_number=phase_number,
            input_data=input_data,
            output_data=output_data,
            metadata={
                'phase_name': phase_def['name'],
                'description': phase_def['description'],
                'requires_review': phase_def['requires_review'],
                'multi_model': phase_def['multi_model'],
                'models_used': models_to_use if phase_def['multi_model'] else [model_to_use]
            },
            user_approved=False  # 초기값은 미승인
        )
        
        self.phase_results[session_id][phase_number] = result
        self.active_sessions[session_id] = phase_number
        
        print(f"Phase {phase_number} 완료 - 사용자 확인 대기중: {phase_def['requires_review']}")
        return result
    
    def get_phase_result(self, session_id: str, phase_number: int) -> Optional[PhaseResult]:
        """특정 Phase 결과 조회"""
        if session_id not in self.phase_results:
            return None
        return self.phase_results[session_id].get(phase_number)
    
    def approve_phase_result(self, session_id: str, phase_number: int, 
                            modified_data: Dict[str, Any] = None) -> bool:
        """
        Phase 결과 승인 (사용자 확인/수정 후)
        
        Args:
            session_id: 세션 ID
            phase_number: Phase 번호
            modified_data: 사용자가 수정한 데이터 (있는 경우)
        """
        result = self.get_phase_result(session_id, phase_number)
        if not result:
            return False
        
        # 수정사항이 있으면 적용
        if modified_data:
            result.output_data = modified_data
            result.metadata['user_modified'] = True
            result.metadata['modification_time'] = datetime.now().isoformat()
        
        # 승인 처리
        result.user_approved = True
        print(f"Phase {phase_number} 결과 승인됨 - 세션: {session_id}")
        return True
    
    async def continue_to_next_phase(self, 
                                    session_id: str,
                                    models_to_use: List[str] = None) -> Optional[PhaseResult]:
        """
        다음 Phase로 진행
        현재 Phase가 승인되었는지 확인 후 진행
        """
        current_phase = self.active_sessions.get(session_id)
        if current_phase is None:
            raise ValueError(f"세션 {session_id}를 찾을 수 없습니다.")
        
        # 현재 Phase 승인 확인
        current_result = self.get_phase_result(session_id, current_phase)
        phase_def = self.PHASE_DEFINITIONS[current_phase]
        
        if phase_def['requires_review'] and not current_result.user_approved:
            raise ValueError(f"Phase {current_phase} 결과가 아직 승인되지 않았습니다.")
        
        # 다음 Phase 확인
        next_phase = current_phase + 1
        if next_phase > 6:
            print("모든 Phase가 완료되었습니다.")
            return None
        
        # 다음 Phase 실행
        next_phase_def = self.PHASE_DEFINITIONS[next_phase]
        
        if next_phase_def['multi_model']:
            return await self.execute_phase(
                session_id=session_id,
                phase_number=next_phase,
                models_to_use=models_to_use
            )
        else:
            return await self.execute_phase(
                session_id=session_id,
                phase_number=next_phase,
                model_to_use="gpt4"  # 단일 모델 Phase
            )
    
    def get_pipeline_status(self, session_id: str) -> Dict[str, Any]:
        """파이프라인 진행 상태 조회"""
        if session_id not in self.phase_results:
            return {'error': 'Session not found'}
        
        completed_phases = []
        pending_phases = []
        
        for phase_num in range(7):  # Phase 0-6
            result = self.phase_results[session_id].get(phase_num)
            phase_def = self.PHASE_DEFINITIONS[phase_num]
            
            if result:
                completed_phases.append({
                    'phase': phase_num,
                    'name': phase_def['name'],
                    'description': phase_def['description'],
                    'approved': result.user_approved,
                    'timestamp': result.timestamp,
                    'success': result.output_data.get('success', False)
                })
            else:
                pending_phases.append({
                    'phase': phase_num,
                    'name': phase_def['name'],
                    'description': phase_def['description']
                })
        
        return {
            'session_id': session_id,
            'current_phase': self.active_sessions.get(session_id),
            'completed_phases': completed_phases,
            'pending_phases': pending_phases,
            'progress_percentage': (len(completed_phases) / 7) * 100,
            'can_continue': self._can_continue_to_next(session_id)
        }
    
    def _can_continue_to_next(self, session_id: str) -> bool:
        """다음 Phase로 진행 가능한지 확인"""
        current_phase = self.active_sessions.get(session_id)
        if current_phase is None:
            return False
        
        if current_phase >= 6:
            return False  # 마지막 Phase
        
        current_result = self.get_phase_result(session_id, current_phase)
        if not current_result:
            return False
        
        phase_def = self.PHASE_DEFINITIONS[current_phase]
        if phase_def['requires_review'] and not current_result.user_approved:
            return False
        
        return current_result.output_data.get('success', False)
    
    def save_session(self, session_id: str, file_path: str = None) -> str:
        """세션 상태를 파일로 저장"""
        if session_id not in self.phase_results:
            raise ValueError(f"세션 {session_id}를 찾을 수 없습니다.")
        
        if not file_path:
            sessions_dir = Path("sessions")
            sessions_dir.mkdir(exist_ok=True)
            file_path = sessions_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            'session_id': session_id,
            'current_phase': self.active_sessions.get(session_id),
            'phase_results': {
                phase: result.to_dict() 
                for phase, result in self.phase_results[session_id].items()
            },
            'saved_at': datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        print(f"세션 저장 완료: {file_path}")
        return str(file_path)
    
    def load_session(self, file_path: str) -> str:
        """저장된 세션 복원"""
        with open(file_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        session_id = session_data['session_id']
        self.active_sessions[session_id] = session_data['current_phase']
        
        # PhaseResult 객체 복원
        self.phase_results[session_id] = {}
        for phase_str, result_dict in session_data['phase_results'].items():
            phase_num = int(phase_str)
            self.phase_results[session_id][phase_num] = PhaseResult(
                phase_number=result_dict['phase_number'],
                input_data=result_dict['input_data'],
                output_data=result_dict['output_data'],
                metadata=result_dict['metadata'],
                user_approved=result_dict['user_approved']
            )
        
        print(f"세션 복원 완료: {session_id}")
        return session_id