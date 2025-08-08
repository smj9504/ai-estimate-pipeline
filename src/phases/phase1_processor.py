# src/phases/phase1_processor.py
"""
Phase 1: Work Scope Determination and Quantity Calculation (통합)
- Phase 0의 출력을 입력으로 받음
- 멀티모델 사용하여 작업 범위 결정 및 수량 계산
- Remove & Replace 로직 적용
- Waste factor 적용 및 명시
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.prompt_manager import PromptManager
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger
from src.validators.estimation_validator import ComprehensiveValidator

class Phase1Processor:
    """
    Phase 1: Work Scope Determination and Quantity Calculation (통합)
    멀티모델을 사용하여 작업 범위를 결정하고 수량을 계산하며 waste factor를 적용
    """
    
    # 표준 Waste Factor 정의 (건설 도메인 지식)
    WASTE_FACTORS = {
        'drywall': 0.10,  # 10% waste
        'trim': 0.10,     # 10% waste
        'baseboard': 0.10,  # 10% waste
        'paint': 0.05,    # 5% waste (coverage based)
        'carpet': 0.08,   # 8% waste
        'hardwood': 0.12, # 12% waste
        'tile': 0.12,     # 12% waste
        'vinyl': 0.08,    # 8% waste
        'lvp': 0.08,      # 8% waste (Luxury Vinyl Plank)
        'insulation': 0.05,  # 5% waste
        'electrical': 0.05,  # 5% waste
        'plumbing': 0.05,    # 5% waste
        'default': 0.10   # 10% default waste factor
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prompt_manager = PromptManager()
        self.orchestrator = ModelOrchestrator()
        self.merger = None  # ResultMerger는 process() 메서드에서 원본 데이터와 함께 생성
        self.validator = ComprehensiveValidator()
    
    async def process(self,
                     phase0_output: Dict[str, Any],
                     models_to_use: List[str] = None,
                     project_id: Optional[str] = None,
                     prompt_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 1 실행 - 멀티모델로 작업 범위 결정 및 수량 계산 (통합)
        
        Args:
            phase0_output: Phase 0의 출력 (Generate Scope of Work 결과)
            models_to_use: 사용할 AI 모델 리스트
            project_id: 프로젝트 ID
            prompt_version: 사용할 프롬프트 버전 ('integrated', 'improved', 'fast', None=기본)
        
        Returns:
            작업 범위 및 수량이 계산된 데이터 (waste factor 포함)
        """
        print(f"Phase 1 시작: Work Scope & Quantity Calculation (통합) - 모델: {models_to_use}")
        
        try:
            # Phase 0 출력에서 실제 데이터 추출
            if 'data' in phase0_output:
                input_data = phase0_output['data']
            else:
                input_data = phase0_output
            
            # 기본 모델 설정
            if not models_to_use:
                models_to_use = ["gpt4", "claude", "gemini"]
            
            # 1. 프롬프트 로드 및 변수 치환
            prompt_variables = {
                'project_id': project_id or phase0_output.get('project_id', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'location': 'DMV area'
            }
            
            # 프롬프트 버전 결정
            effective_version = prompt_version or self.config.get('prompt_version')
            
            if effective_version:
                print(f"프롬프트 버전: {effective_version}")
            else:
                print("프롬프트 버전: 기본 버전 사용")
            
            # 프롬프트 로드 (이제 기본이 통합 버전)
            base_prompt = self.prompt_manager.load_prompt_with_variables(
                phase_number=1,
                variables=prompt_variables,
                version=effective_version
            )
            
            # 2. 멀티모델 병렬 실행
            print(f"통합 처리 모드: 멀티모델 실행 중 - {models_to_use}")
            model_results = await self.orchestrator.run_parallel(
                prompt=base_prompt,
                json_data=input_data,
                model_names=models_to_use,
                prompt_version=effective_version
            )
            
            if not model_results:
                raise ValueError("모든 모델 실행이 실패했습니다")
            
            print(f"{len(model_results)}개 모델 응답 수신")
            
            # 3. 원본 데이터와 함께 ResultMerger 생성 및 결과 병합
            self.merger = ResultMerger(self.config, original_data=input_data)
            merged_result = self.merger.merge_results(model_results)
            
            # 4. Waste Factor 적용 및 검증
            enhanced_result = await self._apply_waste_factors(merged_result)
            
            # 5. Remove & Replace 로직 및 수량 검증
            validation_result = await self._validate_integrated_logic(
                enhanced_result, 
                input_data
            )
            
            # 6. 최종 결과 구성
            result = {
                'phase': 1,
                'phase_name': 'Work Scope & Quantity Calculation (Integrated)',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'models_responded': len(model_results),
                'project_id': prompt_variables['project_id'],
                'prompt_version': effective_version,
                'data': enhanced_result if isinstance(enhanced_result, dict) else enhanced_result.model_dump(),
                'validation': validation_result,
                'confidence_score': merged_result.overall_confidence if hasattr(merged_result, 'overall_confidence') else 0.85,
                'consensus_level': merged_result.metadata.consensus_level if hasattr(merged_result, 'metadata') else 'high',
                'processing_time': sum(r.processing_time for r in model_results),
                'waste_factors_applied': True,
                'success': True
            }
            
            print(f"Phase 1 완료: 작업 범위 및 수량 계산 완료 (waste factor 포함)")
            return result
            
        except Exception as e:
            print(f"Phase 1 오류: {e}")
            return {
                'phase': 1,
                'phase_name': 'Work Scope & Quantity Calculation (Integrated)',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'error': str(e),
                'success': False
            }
    
    async def _apply_waste_factors(self, merged_result: Any) -> Dict[str, Any]:
        """
        Waste Factor를 적용하여 최종 수량 계산
        
        Args:
            merged_result: 병합된 결과
        
        Returns:
            Waste factor가 적용된 결과
        """
        # 결과 구조 확인
        if hasattr(merged_result, 'rooms'):
            rooms = merged_result.rooms
            result_dict = merged_result.model_dump() if hasattr(merged_result, 'model_dump') else dict(merged_result)
        elif isinstance(merged_result, dict) and 'rooms' in merged_result:
            rooms = merged_result['rooms']
            result_dict = merged_result.copy()
        else:
            return merged_result
        
        # 각 방의 작업에 waste factor 적용
        enhanced_rooms = []
        total_waste_summary = {}
        
        for room in rooms:
            enhanced_room = room.copy() if isinstance(room, dict) else room
            
            # tasks가 있는 경우 waste factor 적용
            if 'tasks' in enhanced_room:
                enhanced_tasks = []
                for task in enhanced_room['tasks']:
                    enhanced_task = task.copy() if isinstance(task, dict) else task
                    
                    # 재료 카테고리 식별
                    material_type = self._identify_material_type(
                        enhanced_task.get('description', ''),
                        enhanced_task.get('category', '')
                    )
                    
                    # Waste factor 적용
                    if material_type and enhanced_task.get('quantity'):
                        waste_factor = self.WASTE_FACTORS.get(material_type, self.WASTE_FACTORS['default'])
                        base_quantity = enhanced_task['quantity']
                        
                        # Waste 포함 수량 계산
                        enhanced_task['waste_factor'] = waste_factor * 100  # 퍼센트로 표시
                        enhanced_task['quantity_with_waste'] = base_quantity * (1 + waste_factor)
                        enhanced_task['waste_amount'] = base_quantity * waste_factor
                        
                        # Waste summary 업데이트
                        if material_type not in total_waste_summary:
                            total_waste_summary[material_type] = {
                                'base_quantity': 0,
                                'waste_amount': 0,
                                'waste_factor': waste_factor
                            }
                        total_waste_summary[material_type]['base_quantity'] += base_quantity
                        total_waste_summary[material_type]['waste_amount'] += enhanced_task['waste_amount']
                    
                    enhanced_tasks.append(enhanced_task)
                
                enhanced_room['tasks'] = enhanced_tasks
            
            enhanced_rooms.append(enhanced_room)
        
        # 결과 업데이트
        result_dict['rooms'] = enhanced_rooms
        result_dict['waste_summary'] = total_waste_summary
        result_dict['waste_factors_applied'] = True
        
        return result_dict
    
    def _identify_material_type(self, description: str, category: str) -> Optional[str]:
        """
        작업 설명에서 재료 타입 식별
        
        Args:
            description: 작업 설명
            category: 작업 카테고리
        
        Returns:
            재료 타입 (waste factor 키)
        """
        description_lower = description.lower()
        category_lower = category.lower()
        
        # 재료 타입 매핑
        material_mapping = {
            'drywall': ['drywall', 'sheetrock', 'gypsum'],
            'paint': ['paint', 'primer', 'coating'],
            'trim': ['trim', 'molding', 'crown'],
            'baseboard': ['baseboard', 'base board'],
            'carpet': ['carpet'],
            'hardwood': ['hardwood', 'wood floor'],
            'tile': ['tile', 'ceramic', 'porcelain'],
            'vinyl': ['vinyl', 'lvt'],
            'lvp': ['lvp', 'luxury vinyl'],
            'insulation': ['insulation', 'fiberglass'],
            'electrical': ['electrical', 'wiring', 'outlet'],
            'plumbing': ['plumbing', 'pipe', 'fixture']
        }
        
        # 재료 타입 찾기
        for material_type, keywords in material_mapping.items():
            for keyword in keywords:
                if keyword in description_lower or keyword in category_lower:
                    return material_type
        
        # Installation 카테고리는 기본 waste factor 적용
        if 'installation' in category_lower:
            return 'default'
        
        return None
    
    async def _validate_integrated_logic(self, 
                                        enhanced_result: Dict[str, Any],
                                        original_data: Any) -> Dict[str, Any]:
        """
        통합된 로직 검증 (Remove & Replace + 수량 + Waste Factor)
        
        Args:
            enhanced_result: Waste factor가 적용된 결과
            original_data: 원본 입력 데이터
        
        Returns:
            검증 결과
        """
        validation = {
            'remove_replace_logic': {'valid': True, 'issues': []},
            'quantity_accuracy': {'valid': True, 'issues': []},
            'waste_factors': {'valid': True, 'issues': []},
            'measurements_accuracy': {'valid': True, 'issues': []},
            'special_tasks': {'valid': True, 'issues': []},
            'overall_valid': True
        }
        
        rooms = enhanced_result.get('rooms', [])
        
        try:
            for room in rooms:
                room_name = room.get('name', 'Unknown')
                
                # 1. Remove & Replace 로직 체크
                for material, work in room.get('work_scope', {}).items():
                    if 'Remove & Replace' in str(work):
                        demo_scope = room.get('demo_scope(already demo\'d)', {})
                        if material in demo_scope and demo_scope[material] > 0:
                            validation['remove_replace_logic']['issues'].append(
                                f"{room_name}: {material} - demo_scope 고려 확인 필요"
                            )
                
                # 2. 수량 정확성 체크
                measurements = room.get('measurements', {})
                room_sqft = measurements.get('sqft', 0)
                
                for task in room.get('tasks', []):
                    if task.get('unit') == 'sqft' and task.get('quantity'):
                        # 수량이 방 면적보다 비정상적으로 큰 경우
                        if task['quantity'] > room_sqft * 3:  # 벽 + 천장 고려
                            validation['quantity_accuracy']['issues'].append(
                                f"{room_name}: {task.get('description')} - 수량 과다"
                            )
                    
                    # 3. Waste Factor 적용 확인
                    if task.get('quantity') and not task.get('waste_factor'):
                        validation['waste_factors']['issues'].append(
                            f"{room_name}: {task.get('description')} - Waste factor 누락"
                        )
                
                # 4. 높은 천장 프리미엄 확인 (9피트 초과)
                if measurements.get('height', 0) > 9:
                    validation['special_tasks']['issues'].append(
                        f"{room_name}: 높은 천장 프리미엄 적용 확인 필요 ({measurements.get('height')}ft)"
                    )
            
            # 5. Waste Summary 검증
            if 'waste_summary' not in enhanced_result:
                validation['waste_factors']['issues'].append("전체 waste summary 누락")
            
            # 전체 유효성 판단
            for category in validation.values():
                if isinstance(category, dict) and category.get('issues'):
                    category['valid'] = False
                    validation['overall_valid'] = False
            
        except Exception as e:
            validation['overall_valid'] = False
            validation['error'] = str(e)
        
        return validation
    
    def prepare_for_phase2(self, phase1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1 결과를 Phase 2 (Market Research) 입력 형식으로 변환
        
        Args:
            phase1_result: Phase 1 출력 (작업 범위 및 수량 포함)
        
        Returns:
            Phase 2 (Market Research)용 입력 데이터
        """
        if not phase1_result.get('success'):
            raise ValueError("Phase 1이 성공적으로 완료되지 않았습니다")
        
        phase1_data = phase1_result['data']
        
        # Phase 2 (Market Research)를 위한 데이터 준비
        market_research_input = {
            'rooms_with_quantities': phase1_data.get('rooms', []),
            'waste_summary': phase1_data.get('waste_summary', {}),
            'project_info': {
                'project_id': phase1_result.get('project_id'),
                'location': 'DMV area',
                'timestamp': phase1_result.get('timestamp')
            },
            'materials_to_price': self._extract_materials_for_pricing(phase1_data),
            'labor_requirements': self._extract_labor_requirements(phase1_data),
            'metadata': {
                'phase1_confidence': phase1_result.get('confidence_score'),
                'phase1_validation': phase1_result.get('validation'),
                'waste_factors_applied': phase1_result.get('waste_factors_applied', False)
            }
        }
        
        return market_research_input
    
    def _extract_materials_for_pricing(self, phase1_data: Dict[str, Any]) -> List[Dict]:
        """
        가격 조사가 필요한 재료 목록 추출
        """
        materials = []
        seen = set()
        
        for room in phase1_data.get('rooms', []):
            for task in room.get('tasks', []):
                if task.get('category') == 'Installation':
                    material_key = f"{task.get('description')}_{task.get('unit')}"
                    if material_key not in seen:
                        materials.append({
                            'description': task.get('description'),
                            'total_quantity': task.get('quantity_with_waste', task.get('quantity', 0)),
                            'unit': task.get('unit'),
                            'waste_included': task.get('waste_factor') is not None
                        })
                        seen.add(material_key)
        
        return materials
    
    def _extract_labor_requirements(self, phase1_data: Dict[str, Any]) -> List[Dict]:
        """
        인건비 산정을 위한 작업 요구사항 추출
        """
        labor_tasks = []
        
        for room in phase1_data.get('rooms', []):
            for task in room.get('tasks', []):
                if task.get('category') in ['Installation', 'Finishing', 'Demolition']:
                    labor_tasks.append({
                        'room': room.get('name'),
                        'task': task.get('description'),
                        'quantity': task.get('quantity_with_waste', task.get('quantity', 0)),
                        'unit': task.get('unit'),
                        'category': task.get('category'),
                        'complexity': 'high' if room.get('measurements', {}).get('height', 0) > 9 else 'standard'
                    })
        
        return labor_tasks