# tests/test_model_interface.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json

from src.models.model_interface import ModelOrchestrator, GPT4Interface, ClaudeInterface
from src.models.data_models import ModelResponse

@pytest.fixture
def sample_json_data():
    """테스트용 샘플 JSON 데이터"""
    return [
        {
            "Jobsite": "Test Project",
            "occupancy": "Single Family",
            "company": {}
        },
        {
            "location": "1st Floor",
            "rooms": [
                {
                    "name": "Living Room",
                    "material": {
                        "Floor": "Hardwood",
                        "wall": "Drywall",
                        "ceiling": "Drywall",
                        "Baseboard": "Wood",
                        "Quarter Round": "Wood"
                    },
                    "work_scope": {
                        "Flooring": "Remove & Replace",
                        "Wall": "Paint",
                        "Ceiling": "Paint",
                        "Baseboard": "Remove & Replace",
                        "Quarter Round": "",
                        "Paint Scope": "Full Room"
                    },
                    "measurements": {
                        "height": 9.5,
                        "wall_area_sqft": 400.0,
                        "ceiling_area_sqft": 200.0,
                        "floor_area_sqft": 200.0,
                        "walls_and_ceiling_area_sqft": 600.0,
                        "flooring_area_sy": 22.22,
                        "ceiling_perimeter_lf": 60.0,
                        "floor_perimeter_lf": 60.0,
                        "openings": []
                    },
                    "demo_scope(already demo'd)": {
                        "Ceiling Drywall(sq_ft)": 50,
                        "Wall Drywall(sq_ft)": 100
                    },
                    "additional_notes": {
                        "protection": ["Floor protection required"],
                        "detach_reset": ["Light fixtures", "Ceiling fans"]
                    }
                }
            ]
        }
    ]

@pytest.fixture
def sample_prompt():
    """테스트용 프롬프트"""
    return """You are a Senior Reconstruction Estimating Specialist in the DMV area. 
    Generate a detailed reconstruction estimate by analyzing the provided JSON data."""

class TestModelOrchestrator:
    """ModelOrchestrator 테스트"""
    
    @pytest.fixture
    def orchestrator(self):
        """테스트용 오케스트레이터"""
        with patch('src.models.model_interface.ConfigLoader') as mock_loader:
            mock_loader.return_value.get_api_keys.return_value = {
                'openai': 'test-openai-key',
                'anthropic': 'test-anthropic-key',
                'google': 'test-google-key'
            }
            return ModelOrchestrator()
    
    def test_initialization(self, orchestrator):
        """초기화 테스트"""
        assert orchestrator is not None
        assert hasattr(orchestrator, 'models')
    
    def test_get_available_models(self, orchestrator):
        """사용 가능한 모델 목록 테스트"""
        models = orchestrator.get_available_models()
        assert isinstance(models, list)
        # API 키가 설정된 경우 모델들이 있어야 함
    
    def test_validate_api_keys(self, orchestrator):
        """API 키 검증 테스트"""
        validation_results = orchestrator.validate_api_keys()
        assert isinstance(validation_results, dict)
        assert 'openai' in validation_results
        assert 'anthropic' in validation_results
        assert 'google' in validation_results
    
    @pytest.mark.asyncio
    async def test_run_parallel_no_models(self):
        """모델이 없을 때 병렬 실행 테스트"""
        with patch('src.models.model_interface.ConfigLoader') as mock_loader:
            mock_loader.return_value.get_api_keys.return_value = {
                'openai': '',
                'anthropic': '', 
                'google': ''
            }
            orchestrator = ModelOrchestrator()
            
            results = await orchestrator.run_parallel("test prompt", {})
            assert results == []

class TestGPT4Interface:
    """GPT4Interface 테스트"""
    
    @pytest.fixture
    def gpt4_interface(self):
        """테스트용 GPT4 인터페이스"""
        return GPT4Interface("test-api-key")
    
    def test_prepare_prompt(self, gpt4_interface, sample_prompt, sample_json_data):
        """프롬프트 준비 테스트"""
        full_prompt = gpt4_interface._prepare_prompt(sample_prompt, sample_json_data)
        assert sample_prompt in full_prompt
        assert "[JSON DATA]" in full_prompt
        assert "Living Room" in full_prompt
    
    def test_extract_response_data_json(self, gpt4_interface):
        """JSON 응답 데이터 추출 테스트"""
        json_response = '{"rooms": [{"name": "Living Room", "tasks": ["flooring", "paint"]}]}'
        extracted = gpt4_interface._extract_response_data(json_response)
        
        assert "rooms" in extracted
        assert len(extracted["rooms"]) == 1
        assert extracted["rooms"][0]["name"] == "Living Room"
    
    def test_extract_response_data_text(self, gpt4_interface):
        """텍스트 응답 데이터 추출 테스트"""
        text_response = """
        Living Room:
        - Remove existing flooring
        - Install new hardwood flooring
        - Paint walls
        """
        extracted = gpt4_interface._extract_response_data(text_response)
        
        assert "work_items" in extracted
        assert len(extracted["work_items"]) > 0
        assert any("flooring" in item['task_name'].lower() for item in extracted["work_items"])

    @pytest.mark.asyncio
    async def test_call_model_success(self, gpt4_interface, sample_prompt, sample_json_data):
        """모델 호출 성공 테스트"""
        with patch.object(gpt4_interface, 'client') as mock_client:
            # Mock 응답 설정
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Living Room:\n- Remove flooring\n- Install flooring"
            
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            
            result = await gpt4_interface.call_model(sample_prompt, sample_json_data)
            
            assert isinstance(result, ModelResponse)
            assert result.model_name == "gpt-4"
            assert result.processing_time > 0
            assert result.confidence_self_assessment > 0


# tests/test_result_merger.py
import pytest
import numpy as np
from unittest.mock import Mock

from src.processors.result_merger import ResultMerger, QualitativeMerger, QuantitativeMerger
from src.models.data_models import ModelResponse, MergedEstimate, AppConfig

@pytest.fixture
def mock_config():
    """테스트용 설정"""
    return AppConfig()

@pytest.fixture
def sample_model_responses():
    """테스트용 모델 응답들"""
    return [
        ModelResponse(
            model_name="gpt-4",
            room_estimates=[
                {
                    "name": "Living Room",
                    "tasks": [
                        {
                            "task_name": "Remove existing flooring",
                            "description": "Remove hardwood flooring",
                            "necessity": "required",
                            "quantity": 200,
                            "unit": "sq_ft"
                        },
                        {
                            "task_name": "Install new flooring",
                            "description": "Install hardwood flooring",
                            "necessity": "required", 
                            "quantity": 200,
                            "unit": "sq_ft"
                        }
                    ]
                }
            ],
            processing_time=2.1,
            total_work_items=2,
            confidence_self_assessment=0.85
        ),
        ModelResponse(
            model_name="claude-3-sonnet",
            room_estimates=[
                {
                    "name": "Living Room",
                    "tasks": [
                        {
                            "task_name": "Flooring removal",
                            "description": "Remove existing hardwood",
                            "necessity": "required",
                            "quantity": 198,
                            "unit": "sq_ft"
                        },
                        {
                            "task_name": "Flooring installation", 
                            "description": "Install new hardwood",
                            "necessity": "required",
                            "quantity": 200,
                            "unit": "sq_ft"
                        },
                        {
                            "task_name": "Floor protection",
                            "description": "Protect surrounding areas",
                            "necessity": "required",
                            "quantity": 1,
                            "unit": "ea"
                        }
                    ]
                }
            ],
            processing_time=2.3,
            total_work_items=3,
            confidence_self_assessment=0.88
        ),
        ModelResponse(
            model_name="gemini-pro",
            room_estimates=[
                {
                    "name": "Living Room",
                    "tasks": [
                        {
                            "task_name": "Remove flooring",
                            "description": "Remove old flooring materials",
                            "necessity": "required",
                            "quantity": 202,
                            "unit": "sq_ft"
                        },
                        {
                            "task_name": "Install flooring",
                            "description": "Install replacement flooring",
                            "necessity": "required",
                            "quantity": 200,
                            "unit": "sq_ft"
                        }
                    ]
                }
            ],
            processing_time=1.9,
            total_work_items=2,
            confidence_self_assessment=0.80
        )
    ]

class TestQualitativeMerger:
    """QualitativeMerger 테스트"""
    
    @pytest.fixture
    def qualitative_merger(self, mock_config):
        """테스트용 질적 병합기"""
        return QualitativeMerger(mock_config)
    
    def test_collect_all_tasks(self, qualitative_merger, sample_model_responses):
        """모든 작업 수집 테스트"""
        tasks = qualitative_merger._collect_all_tasks(sample_model_responses)
        
        assert len(tasks) == 7  # 2 + 3 + 2 = 7개 작업
        assert all('model' in task for task in tasks)
        assert all('room' in task for task in tasks)
        assert all('task_name' in task for task in tasks)
    
    def test_group_similar_tasks(self, qualitative_merger, sample_model_responses):
        """유사 작업 그룹핑 테스트"""
        tasks = qualitative_merger._collect_all_tasks(sample_model_responses)
        groups = qualitative_merger._group_similar_tasks(tasks)
        
        # 유사한 작업들이 그룹핑되었는지 확인
        assert len(groups) <= len(tasks)  # 그룹 수는 원래 작업 수보다 작거나 같음
        
        # 플로어링 제거/설치 작업들이 각각 그룹핑되었는지 확인
        group_keys = list(groups.keys())
        removal_groups = [key for key in group_keys if 'remove' in key or 'removal' in key]
        install_groups = [key for key in group_keys if 'install' in key]
        
        assert len(removal_groups) >= 1
        assert len(install_groups) >= 1
    
    def test_apply_consensus_rules(self, qualitative_merger, sample_model_responses):
        """합의 규칙 적용 테스트"""
        tasks = qualitative_merger._collect_all_tasks(sample_model_responses)
        groups = qualitative_merger._group_similar_tasks(tasks)
        consensus_tasks = qualitative_merger._apply_consensus_rules(groups, len(sample_model_responses))
        
        # 합의된 작업들이 있어야 함
        assert len(consensus_tasks) > 0
        
        # 각 합의 작업에 메타데이터가 있는지 확인
        for task in consensus_tasks:
            assert 'consensus_level' in task
            assert 'supporting_models' in task
            assert task['consensus_level'] > 0

    def test_merge_work_scopes(self, qualitative_merger, sample_model_responses):
        """작업 범위 병합 테스트"""
        result = qualitative_merger.merge_work_scopes(sample_model_responses)
        
        assert 'merged_work_scope' in result
        assert 'consensus_level' in result
        assert 'outlier_tasks' in result
        
        # Living Room 작업들이 병합되었는지 확인
        merged_scope = result['merged_work_scope']
        assert 'Living Room' in merged_scope
        assert len(merged_scope['Living Room']) > 0

class TestQuantitativeMerger:
    """QuantitativeMerger 테스트"""
    
    @pytest.fixture
    def quantitative_merger(self, mock_config):
        """테스트용 정량적 병합기"""
        return QuantitativeMerger(mock_config)
    
    def test_merge_quantity_values_single(self, quantitative_merger):
        """단일 수량 값 병합 테스트"""
        quantities = [(200.0, 'gpt-4')]
        supporting_models = ['gpt-4']
        
        merged_value, metadata = quantitative_merger._merge_quantity_values(quantities, supporting_models)
        
        assert merged_value == 200.0
        assert metadata['confidence'] == 'medium'
        assert metadata['method'] == 'single_value'
    
    def test_merge_quantity_values_multiple(self, quantitative_merger):
        """다중 수량 값 병합 테스트"""
        quantities = [(198.0, 'claude'), (200.0, 'gpt-4'), (202.0, 'gemini')]
        supporting_models = ['claude', 'gpt-4', 'gemini']
        
        merged_value, metadata = quantitative_merger._merge_quantity_values(quantities, supporting_models)
        
        assert merged_value > 0
        assert metadata['confidence'] in ['high', 'medium', 'low']
        assert metadata['method'] == 'weighted_average'
        assert 'variance' in metadata
    
    def test_merge_quantities(self, quantitative_merger):
        """수량 병합 전체 테스트"""
        consensus_tasks = [
            {
                'room': 'Living Room',
                'task_name': 'Remove flooring',
                'quantity': 200,
                'supporting_models': ['gpt-4', 'claude'],
                'model': 'gpt-4'
            }
        ]
        
        result = quantitative_merger.merge_quantities(consensus_tasks)
        
        assert 'merged_quantities' in result
        assert 'quantity_metadata' in result
        assert 'overall_confidence' in result

class TestResultMerger:
    """ResultMerger 통합 테스트"""
    
    @pytest.fixture
    def result_merger(self):
        """테스트용 결과 병합기"""
        with patch('src.processors.result_merger.ConfigLoader') as mock_loader:
            mock_loader.return_value.load_config.return_value = AppConfig()
            return ResultMerger()
    
    def test_merge_results_empty(self, result_merger):
        """빈 결과 병합 테스트"""
        result = result_merger.merge_results([])
        
        assert isinstance(result, MergedEstimate)
        assert result.total_work_items == 0
        assert result.overall_confidence == 0.0
    
    def test_merge_results_success(self, result_merger, sample_model_responses):
        """성공적인 결과 병합 테스트"""
        with patch('time.time', side_effect=[0, 1.5]):  # 처리 시간 1.5초 모킹
            result = result_merger.merge_results(sample_model_responses)
        
        assert isinstance(result, MergedEstimate)
        assert result.total_work_items > 0
        assert result.overall_confidence > 0
        assert len(result.rooms) > 0
        assert result.metadata.models_used == ['gpt-4', 'claude-3-sonnet', 'gemini-pro']
        assert result.metadata.processing_time_total == 1.5


# tests/test_validation_system.py
import pytest
from src.validators.estimation_validator import (
    RemoveReplaceValidator, MeasurementValidator, ComprehensiveValidator,
    ValidationResult
)
from src.models.data_models import Room, WorkScope, Measurements, Materials, DemoScope, AdditionalNotes

@pytest.fixture
def sample_room():
    """테스트용 방 데이터"""
    return Room(
        name="Living Room",
        material=Materials(
            Floor="Hardwood",
            wall="Drywall", 
            ceiling="Drywall",
            Baseboard="Wood",
            Quarter_Round="Wood"
        ),
        work_scope=WorkScope(
            Flooring="Remove & Replace",
            Wall="Paint",
            Ceiling="Paint",
            Baseboard="Remove & Replace",
            Quarter_Round="",
            Paint_Scope="Full Room"
        ),
        measurements=Measurements(
            height=9.5,
            wall_area_sqft=400.0,
            ceiling_area_sqft=200.0,
            floor_area_sqft=200.0,
            walls_and_ceiling_area_sqft=600.0,
            flooring_area_sy=22.22,
            ceiling_perimeter_lf=60.0,
            floor_perimeter_lf=60.0,
            openings=[]
        ),
        demo_scope=DemoScope(
            ceiling_drywall_sqft=50,
            wall_drywall_sqft=100
        ),
        additional_notes=AdditionalNotes(
            protection=["Floor protection required"],
            detach_reset=["Light fixtures", "Ceiling fans"]
        )
    )

@pytest.fixture
def valid_work_items():
    """유효한 작업 항목들"""
    return [
        {
            'task_name': 'Remove existing flooring',
            'description': 'Remove hardwood flooring from living room',
            'necessity': 'required',
            'quantity': 200,
            'unit': 'sq_ft'
        },
        {
            'task_name': 'Install new flooring',
            'description': 'Install new hardwood flooring',
            'necessity': 'required',
            'quantity': 200,
            'unit': 'sq_ft'
        },
        {
            'task_name': 'Remove existing baseboard',
            'description': 'Remove wood baseboard',
            'necessity': 'required',
            'quantity': 60,
            'unit': 'lf'
        },
        {
            'task_name': 'Install new baseboard',
            'description': 'Install new wood baseboard with high ceiling premium',
            'necessity': 'required',
            'quantity': 60,
            'unit': 'lf'
        },
        {
            'task_name': 'Paint walls',
            'description': 'Paint all wall surfaces',
            'necessity': 'required',
            'quantity': 400,
            'unit': 'sq_ft'
        },
        {
            'task_name': 'Floor protection',
            'description': 'Protect adjacent areas during work',
            'necessity': 'required',
            'quantity': 1,
            'unit': 'ea'
        },
        {
            'task_name': 'Detach light fixtures',
            'description': 'Carefully remove and reset light fixtures',
            'necessity': 'required',
            'quantity': 3,
            'unit': 'ea'
        }
    ]

class TestRemoveReplaceValidator:
    """Remove & Replace 검증기 테스트"""
    
    @pytest.fixture
    def validator(self):
        """테스트용 검증기"""
        return RemoveReplaceValidator()
    
    def test_validate_logic_application_valid(self, validator, sample_room, valid_work_items):
        """유효한 Remove & Replace 로직 검증"""
        result = validator.validate_logic_application(sample_room, valid_work_items)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.score > 0.8
        assert len(result.issues) == 0
    
    def test_validate_logic_application_missing_removal(self, validator, sample_room):
        """제거 작업 누락 검증"""
        invalid_work_items = [
            {
                'task_name': 'Install new flooring',
                'description': 'Install new hardwood flooring',
                'necessity': 'required'
            }
        ]
        
        result = validator.validate_logic_application(sample_room, invalid_work_items)
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert any('제거 작업 누락' in issue for issue in result.issues)
    
    def test_validate_logic_application_missing_installation(self, validator, sample_room):
        """설치 작업 누락 검증"""
        invalid_work_items = [
            {
                'task_name': 'Remove existing flooring',
                'description': 'Remove hardwood flooring',
                'necessity': 'required'
            }
        ]
        
        result = validator.validate_logic_application(sample_room, invalid_work_items)
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert any('설치 작업 누락' in issue for issue in result.issues)
    
    def test_find_removal_task(self, validator, valid_work_items):
        """제거 작업 찾기 테스트"""
        removal_task = validator._find_removal_task('flooring', valid_work_items)
        
        assert removal_task is not None
        assert 'remove' in removal_task['task_name'].lower()
        assert 'flooring' in removal_task['task_name'].lower()
    
    def test_find_installation_task(self, validator, valid_work_items):
        """설치 작업 찾기 테스트"""
        install_task = validator._find_installation_task('flooring', valid_work_items)
        
        assert install_task is not None
        assert 'install' in install_task['task_name'].lower()
        assert 'flooring' in install_task['task_name'].lower()

class TestMeasurementValidator:
    """측정값 검증기 테스트"""
    
    @pytest.fixture
    def validator(self):
        """테스트용 검증기"""
        return MeasurementValidator()
    
    def test_validate_measurement_usage_valid(self, validator, sample_room, valid_work_items):
        """유효한 측정값 사용 검증"""
        result = validator.validate_measurement_usage(sample_room, valid_work_items)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.score > 0.8
    
    def test_validate_floor_work_missing(self, validator):
        """바닥 작업 누락 검증"""
        room_with_floor = Room(
            name="Test Room",
            material=Materials(),
            work_scope=WorkScope(Flooring="Remove & Replace"),
            measurements=Measurements(floor_area_sqft=200.0),
            demo_scope=DemoScope(),
            additional_notes=AdditionalNotes()
        )
        
        work_items_without_floor = [
            {
                'task_name': 'Paint walls',
                'description': 'Paint all walls'
            }
        ]
        
        result = validator.validate_measurement_usage(room_with_floor, work_items_without_floor)
        
        assert not result.is_valid
        assert len(result.issues) > 0
        assert any('바닥 면적' in issue and '바닥 작업 없음' in issue for issue in result.issues)

class TestComprehensiveValidator:
    """종합 검증기 테스트"""
    
    @pytest.fixture
    def validator(self):
        """테스트용 종합 검증기"""
        return ComprehensiveValidator()
    
    def test_validate_single_room_valid(self, validator, sample_room, valid_work_items):
        """단일 방 유효 검증"""
        result = validator.validate_single_room(sample_room, valid_work_items)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert result.score > 0.7  # 전체적으로 높은 점수
        assert 'remove_replace' in result.details
        assert 'measurements' in result.details
        assert 'additional_notes' in result.details

# tests/test_integration.py
import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock

from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.validators.estimation_validator import ComprehensiveValidator
from src.models.data_models import ProjectData

class TestIntegration:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, sample_json_data):
        """전체 파이프라인 모킹 테스트"""
        
        # 1. 모델 오케스트레이터 모킹
        with patch('src.models.model_interface.ConfigLoader') as mock_config_loader:
            mock_config_loader.return_value.get_api_keys.return_value = {
                'openai': 'test-key', 'anthropic': 'test-key', 'google': 'test-key'
            }
            
            orchestrator = ModelOrchestrator()
            
            # 모델 응답 모킹
            mock_responses = [
                Mock(
                    model_name='gpt-4',
                    room_estimates=[{
                        'name': 'Living Room',
                        'tasks': [
                            {'task_name': 'Remove flooring', 'description': 'Remove old flooring'},
                            {'task_name': 'Install flooring', 'description': 'Install new flooring'}
                        ]
                    }],
                    processing_time=2.0,
                    total_work_items=2,
                    confidence_self_assessment=0.85
                )
            ]
            
            with patch.object(orchestrator, 'run_parallel', return_value=mock_responses):
                # 2. 모델 실행
                model_results = await orchestrator.run_parallel(
                    "Test prompt", 
                    sample_json_data,
                    ['gpt4']
                )
                
                assert len(model_results) == 1
                
                # 3. 결과 병합
                with patch('src.processors.result_merger.ConfigLoader'):
                    merger = ResultMerger()
                    merged_result = merger.merge_results(model_results)
                
                assert merged_result.total_work_items > 0
                assert merged_result.overall_confidence > 0
                
                # 4. 검증
                validator = ComprehensiveValidator()
                project_data = ProjectData.from_json_list(sample_json_data)
                
                # 검증은 원본 데이터가 필요하므로 간단히 체크만
                assert project_data.floors[0].rooms[0].name == "Living Room"

    def test_data_model_validation(self, sample_json_data):
        """데이터 모델 검증 테스트"""
        # JSON 데이터에서 ProjectData 객체 생성
        project_data = ProjectData.from_json_list(sample_json_data)
        
        assert project_data.jobsite_info.Jobsite == "Test Project"
        assert len(project_data.floors) == 1
        assert len(project_data.floors[0].rooms) == 1
        
        room = project_data.floors[0].rooms[0]
        assert room.name == "Living Room"
        assert room.measurements.height == 9.5
        assert room.work_scope.Flooring == "Remove & Replace"

# pytest 실행을 위한 설정
if __name__ == "__main__":
    pytest.main(["-v", __file__])