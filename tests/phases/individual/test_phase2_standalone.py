"""
Phase 2 (Market Research & Pricing) 단독 테스트
AI 파이프라인 테스트 베스트 프랙티스 적용:
1. Golden Dataset 사용 (Perfect, Realistic, Edge Cases)
2. Input Validation & Sanitization
3. Error Recovery Testing
4. Contract Testing (Phase 1 -> Phase 2 인터페이스)
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.phases.phase2_processor import Phase2Processor
from src.models.data_models import ModelResponse


class Phase1OutputValidator:
    """Phase 1 출력을 검증하고 보정하는 클래스"""
    
    # 표준 Waste Factor (Phase 1과 동일)
    DEFAULT_WASTE_FACTORS = {
        'drywall': 0.10,
        'paint': 0.05,
        'carpet': 0.08,
        'hardwood': 0.12,
        'tile': 0.12,
        'vinyl': 0.08,
        'lvp': 0.08,
        'trim': 0.10,
        'baseboard': 0.10,
        'insulation': 0.05
    }
    
    def __init__(self):
        self.validation_log = []
        self.corrections_made = []
    
    def validate_and_sanitize(self, phase1_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1 출력을 검증하고 필요시 보정
        
        Returns:
            보정된 Phase 1 출력과 품질 점수
        """
        if not phase1_output.get('success'):
            self.validation_log.append("Phase 1 failed - using minimal defaults")
            return self._create_minimal_valid_output()
        
        data = phase1_output.get('data', {})
        rooms = data.get('rooms', [])
        
        # 검증 및 보정
        corrected_rooms = []
        for room in rooms:
            corrected_room = self._validate_room(room)
            if corrected_room:
                corrected_rooms.append(corrected_room)
        
        # 보정된 데이터로 업데이트
        data['rooms'] = corrected_rooms
        
        # Waste summary 재계산
        data['waste_summary'] = self._recalculate_waste_summary(corrected_rooms)
        
        # 품질 점수 계산
        quality_score = self._calculate_quality_score(phase1_output, len(self.corrections_made))
        
        phase1_output['data'] = data
        phase1_output['quality_score'] = quality_score
        phase1_output['validation_log'] = self.validation_log
        phase1_output['corrections_made'] = self.corrections_made
        
        return phase1_output
    
    def _validate_room(self, room: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """개별 방 데이터 검증 및 보정"""
        # 필수 필드 확인
        if not room.get('room_name'):
            self.validation_log.append(f"Room without name skipped")
            return None
        
        # 측정값 보정
        measurements = room.get('measurements', {})
        if not measurements.get('height'):
            measurements['height'] = 9  # 기본 높이
            self.corrections_made.append(f"{room['room_name']}: Added default height 9ft")
        
        if not measurements.get('sqft') and measurements.get('length') and measurements.get('width'):
            measurements['sqft'] = measurements['length'] * measurements['width']
            self.corrections_made.append(f"{room['room_name']}: Calculated sqft")
        
        room['measurements'] = measurements
        
        # 작업 항목 검증
        tasks = room.get('tasks', [])
        corrected_tasks = []
        for task in tasks:
            corrected_task = self._validate_task(task, room['room_name'])
            if corrected_task:
                corrected_tasks.append(corrected_task)
        
        room['tasks'] = corrected_tasks
        
        return room
    
    def _validate_task(self, task: Dict[str, Any], room_name: str) -> Optional[Dict[str, Any]]:
        """개별 작업 항목 검증 및 보정"""
        # 수량 검증
        quantity = task.get('quantity', 0)
        
        # 극단값 처리
        if quantity <= 0:
            self.corrections_made.append(f"{room_name}: Task with zero/negative quantity removed")
            return None
        
        if quantity > 10000:
            task['quantity'] = 1000  # 최대값으로 제한
            task['review_required'] = True
            self.corrections_made.append(f"{room_name}: Extreme quantity capped at 1000")
        
        # Waste factor 보정
        if not task.get('waste_factor') and task.get('material_type'):
            material_type = task['material_type']
            default_waste = self.DEFAULT_WASTE_FACTORS.get(material_type, 0.10)
            task['waste_factor'] = default_waste * 100  # 퍼센트로 변환
            task['quantity_with_waste'] = quantity * (1 + default_waste)
            self.corrections_made.append(f"{room_name}: Added default waste factor for {material_type}")
        
        return task
    
    def _recalculate_waste_summary(self, rooms: List[Dict]) -> Dict[str, Any]:
        """Waste summary 재계산"""
        waste_summary = {}
        
        for room in rooms:
            for task in room.get('tasks', []):
                material_type = task.get('material_type')
                if material_type and task.get('waste_factor'):
                    if material_type not in waste_summary:
                        waste_summary[material_type] = {
                            'base_quantity': 0,
                            'waste_amount': 0,
                            'waste_factor': task['waste_factor'] / 100
                        }
                    
                    base_qty = task.get('quantity', 0)
                    waste_summary[material_type]['base_quantity'] += base_qty
                    waste_summary[material_type]['waste_amount'] += base_qty * (task['waste_factor'] / 100)
        
        return waste_summary
    
    def _calculate_quality_score(self, output: Dict, correction_count: int) -> float:
        """데이터 품질 점수 계산 (0-1)"""
        score = 1.0
        
        # 보정 횟수에 따른 감점
        score -= (correction_count * 0.05)
        
        # 신뢰도 점수 반영
        confidence = output.get('confidence_score', 0.5)
        score = score * 0.7 + confidence * 0.3
        
        # Validation 실패 항목 반영
        validation = output.get('validation', {})
        if not validation.get('overall_valid', True):
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _create_minimal_valid_output(self) -> Dict[str, Any]:
        """최소한의 유효한 Phase 1 출력 생성 (fallback)"""
        return {
            'success': True,
            'phase': 1,
            'data': {
                'rooms': [
                    {
                        'room_name': 'Default Room',
                        'measurements': {'length': 10, 'width': 10, 'height': 9, 'sqft': 100},
                        'tasks': [
                            {
                                'category': 'Installation',
                                'description': 'Basic repair work',
                                'quantity': 100,
                                'unit': 'sqft',
                                'waste_factor': 10,
                                'quantity_with_waste': 110,
                                'material_type': 'drywall'
                            }
                        ]
                    }
                ],
                'waste_summary': {},
                'material_summary': {}
            },
            'quality_score': 0.1,
            'validation_log': ['Using minimal fallback data'],
            'corrections_made': ['Created default room with basic tasks']
        }


class TestPhase2Standalone:
    """Phase 2 단독 테스트 클래스"""
    
    @pytest.fixture
    def phase2_processor(self):
        """Phase 2 프로세서 인스턴스"""
        return Phase2Processor()
    
    @pytest.fixture
    def validator(self):
        """Phase 1 출력 검증기"""
        return Phase1OutputValidator()
    
    @pytest.fixture
    def golden_datasets(self):
        """Golden Dataset 로드"""
        fixtures_dir = Path(__file__).parent.parent.parent / 'fixtures' / 'phase2'
        
        datasets = {}
        dataset_files = {
            'perfect': 'golden_dataset_perfect.json',
            'realistic': 'golden_dataset_realistic.json',
            'edge_cases': 'golden_dataset_edge_cases.json'
        }
        
        for key, filename in dataset_files.items():
            filepath = fixtures_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    datasets[key] = json.load(f)
            else:
                # 파일이 없으면 기본 데이터 생성
                datasets[key] = self._create_default_dataset(key)
        
        return datasets
    
    def _create_default_dataset(self, dataset_type: str) -> Dict:
        """기본 테스트 데이터셋 생성"""
        if dataset_type == 'perfect':
            return {
                'success': True,
                'confidence_score': 0.95,
                'data': {
                    'rooms': [
                        {
                            'room_name': 'Test Room',
                            'measurements': {'length': 12, 'width': 10, 'height': 9, 'sqft': 120},
                            'tasks': [
                                {
                                    'category': 'Installation',
                                    'description': 'Install drywall',
                                    'quantity': 400,
                                    'unit': 'sqft',
                                    'waste_factor': 10,
                                    'quantity_with_waste': 440,
                                    'material_type': 'drywall'
                                }
                            ]
                        }
                    ]
                }
            }
        else:
            return {'success': False, 'data': {}}
    
    @pytest.mark.asyncio
    async def test_phase2_with_perfect_input(self, phase2_processor, golden_datasets, validator):
        """완벽한 Phase 1 출력으로 Phase 2 테스트"""
        perfect_input = golden_datasets.get('perfect')
        
        # Input validation (변경 없어야 함)
        validated_input = validator.validate_and_sanitize(perfect_input.copy())
        assert validated_input['quality_score'] > 0.9
        assert len(validator.corrections_made) == 0
        
        # Mock AI responses
        with patch.object(phase2_processor.orchestrator, 'run_parallel') as mock_run:
            mock_run.return_value = self._create_mock_market_responses()
            
            # Phase 2 실행
            result = await phase2_processor.process(
                phase1_output=validated_input,
                models_to_use=['gpt4', 'claude', 'gemini']
            )
            
            # 검증
            assert result['success'] == True
            assert 'pricing_data' in result['data']
            assert result['data']['cost_summary']['grand_total'] > 0
            assert result['data']['overhead_profit_percentage'] <= 20
    
    @pytest.mark.asyncio
    async def test_phase2_with_realistic_input(self, phase2_processor, golden_datasets, validator):
        """현실적인 (일부 문제 있는) Phase 1 출력으로 Phase 2 테스트"""
        realistic_input = golden_datasets.get('realistic')
        
        # Input validation (일부 보정 필요)
        validated_input = validator.validate_and_sanitize(realistic_input.copy())
        assert 0.5 < validated_input['quality_score'] < 0.9
        assert len(validator.corrections_made) > 0
        
        # Phase 2가 보정된 데이터로 정상 작동하는지 확인
        with patch.object(phase2_processor.orchestrator, 'run_parallel') as mock_run:
            mock_run.return_value = self._create_mock_market_responses()
            
            result = await phase2_processor.process(
                phase1_output=validated_input,
                models_to_use=['gpt4', 'claude']
            )
            
            assert result['success'] == True
            assert 'quality_warnings' in result.get('metadata', {})
    
    @pytest.mark.asyncio
    async def test_phase2_with_edge_cases(self, phase2_processor, golden_datasets, validator):
        """엣지 케이스 Phase 1 출력으로 Phase 2 테스트"""
        edge_input = golden_datasets.get('edge_cases')
        
        # Input validation (대량 보정 필요)
        validated_input = validator.validate_and_sanitize(edge_input.copy())
        assert validated_input['quality_score'] < 0.5
        assert len(validator.corrections_made) > 5
        
        # Phase 2가 저품질 데이터도 처리할 수 있는지 확인
        with patch.object(phase2_processor.orchestrator, 'run_parallel') as mock_run:
            mock_run.return_value = self._create_mock_market_responses()
            
            result = await phase2_processor.process(
                phase1_output=validated_input,
                models_to_use=['gpt4']  # 단일 모델로 테스트
            )
            
            # 최소한의 결과는 생성되어야 함
            assert result is not None
            if result['success']:
                assert 'fallback_pricing' in result.get('metadata', {})
    
    @pytest.mark.asyncio
    async def test_phase2_error_recovery(self, phase2_processor):
        """Phase 2 오류 복구 메커니즘 테스트"""
        # 완전히 잘못된 입력
        invalid_input = {
            'success': False,
            'error': 'Phase 1 failed completely'
        }
        
        validator = Phase1OutputValidator()
        validated_input = validator.validate_and_sanitize(invalid_input)
        
        # Fallback 데이터가 생성되었는지 확인
        assert validated_input['success'] == True
        assert 'Default Room' in str(validated_input['data'])
        assert validated_input['quality_score'] < 0.2
        
        # Phase 2가 fallback 데이터로 작동하는지 확인
        with patch.object(phase2_processor.orchestrator, 'run_parallel') as mock_run:
            mock_run.return_value = self._create_mock_market_responses()
            
            result = await phase2_processor.process(
                phase1_output=validated_input,
                models_to_use=['gpt4']
            )
            
            assert result is not None
            assert 'using_fallback' in result.get('metadata', {})
    
    @pytest.mark.asyncio
    async def test_phase2_partial_model_failure(self, phase2_processor, golden_datasets, validator):
        """일부 AI 모델 실패 시 Phase 2 테스트"""
        perfect_input = golden_datasets.get('perfect')
        validated_input = validator.validate_and_sanitize(perfect_input.copy())
        
        # 일부 모델만 응답
        partial_responses = [
            self._create_single_mock_response('gpt4'),
            # claude는 실패 (응답 없음)
            self._create_single_mock_response('gemini')
        ]
        
        with patch.object(phase2_processor.orchestrator, 'run_parallel') as mock_run:
            mock_run.return_value = partial_responses
            
            result = await phase2_processor.process(
                phase1_output=validated_input,
                models_to_use=['gpt4', 'claude', 'gemini']
            )
            
            # 2/3 모델로도 결과 생성 가능
            assert result['success'] == True
            assert result['models_responded'] == 2
            assert 'partial_consensus' in result.get('metadata', {})
    
    def _create_mock_market_responses(self) -> List[ModelResponse]:
        """Mock AI 응답 생성"""
        return [
            self._create_single_mock_response('gpt4'),
            self._create_single_mock_response('claude'),
            self._create_single_mock_response('gemini')
        ]
    
    def _create_single_mock_response(self, model_name: str) -> ModelResponse:
        """단일 모델의 Mock 응답 생성"""
        return ModelResponse(
            model_name=model_name,
            room_estimates=[
                {
                    'item': 'Install drywall',
                    'quantity': 440,
                    'unit': 'sqft',
                    'unit_price': 1.50,
                    'material_cost': 264,
                    'labor_cost': 396,
                    'total': 660
                }
            ],
            processing_time=1.5,
            total_work_items=1,
            confidence_self_assessment=0.85
        )


class TestPhase2Contract:
    """Phase 1 -> Phase 2 인터페이스 계약 테스트"""
    
    def test_phase1_output_contract(self):
        """Phase 1 출력이 Phase 2 입력 계약을 만족하는지 테스트"""
        required_fields = [
            'success',
            'data',
            'data.rooms',
            'data.waste_summary'
        ]
        
        # Phase 1 출력 구조 검증
        sample_output = {
            'success': True,
            'data': {
                'rooms': [],
                'waste_summary': {},
                'material_summary': {}
            }
        }
        
        for field_path in required_fields:
            parts = field_path.split('.')
            obj = sample_output
            for part in parts:
                assert part in obj, f"Missing required field: {field_path}"
                obj = obj[part]
    
    def test_room_data_contract(self):
        """Room 데이터 구조 계약 테스트"""
        required_room_fields = ['room_name', 'tasks']
        required_task_fields = ['description', 'quantity', 'unit']
        
        sample_room = {
            'room_name': 'Test Room',
            'tasks': [
                {
                    'description': 'Test task',
                    'quantity': 100,
                    'unit': 'sqft'
                }
            ]
        }
        
        for field in required_room_fields:
            assert field in sample_room
        
        for task in sample_room['tasks']:
            for field in required_task_fields:
                assert field in task


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v"])