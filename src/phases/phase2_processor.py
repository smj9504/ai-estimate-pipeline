# src/phases/phase2_processor.py
"""
Phase 2: Market Research & Pricing
- Phase 1의 수량 데이터를 받아 시장 가격 조사
- 보험 최적화 가격 전략 (60-75th percentile)
- Material/Labor 분리 및 세금 계산
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.prompt_manager import PromptManager
from src.models.model_interface import ModelOrchestrator
from src.processors.result_merger import ResultMerger


class Phase2Processor:
    """
    Phase 2: Market Research & Pricing
    Phase 1의 작업 범위와 수량을 기반으로 시장 가격 조사 및 비용 산정
    """
    
    # DMV 지역 표준 단가 (fallback용)
    DEFAULT_UNIT_PRICES = {
        'drywall': {'unit': 'sqft', 'price': 1.75, 'material_ratio': 0.35},
        'paint': {'unit': 'gallon', 'price': 35.00, 'material_ratio': 0.25},
        'carpet': {'unit': 'sqft', 'price': 3.50, 'material_ratio': 0.55},
        'hardwood': {'unit': 'sqft', 'price': 8.00, 'material_ratio': 0.65},
        'tile': {'unit': 'sqft', 'price': 6.00, 'material_ratio': 0.50},
        'vinyl': {'unit': 'sqft', 'price': 3.00, 'material_ratio': 0.60},
        'lvp': {'unit': 'sqft', 'price': 4.50, 'material_ratio': 0.60},
        'trim': {'unit': 'lf', 'price': 4.00, 'material_ratio': 0.50},
        'baseboard': {'unit': 'lf', 'price': 4.50, 'material_ratio': 0.50},
        'insulation': {'unit': 'sqft', 'price': 1.25, 'material_ratio': 0.40}
    }
    
    # 세금 설정 (Virginia 기준)
    TAX_SETTINGS = {
        'VA': {'material_tax': 0.053, 'labor_tax': 0.0},  # Virginia: 5.3% material tax, no labor tax
        'MD': {'material_tax': 0.06, 'labor_tax': 0.0},   # Maryland: 6% material tax
        'DC': {'material_tax': 0.06, 'labor_tax': 0.0}    # DC: 6% material tax
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.prompt_manager = PromptManager()
        self.orchestrator = ModelOrchestrator()
        self.location = self.config.get('location', 'VA')  # 기본 Virginia
    
    async def process(self,
                     phase1_output: Dict[str, Any],
                     models_to_use: List[str] = None,
                     project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Phase 2 실행 - 시장 가격 조사 및 비용 산정
        
        Args:
            phase1_output: Phase 1의 출력 (작업 범위 및 수량 포함)
            models_to_use: 사용할 AI 모델 리스트
            project_id: 프로젝트 ID
        
        Returns:
            가격 정보가 포함된 견적 데이터
        """
        print(f"Phase 2 시작: Market Research & Pricing - 모델: {models_to_use}")
        
        try:
            # Phase 1 데이터 검증
            if not phase1_output.get('success'):
                raise ValueError("Phase 1이 성공적으로 완료되지 않았습니다")
            
            phase1_data = phase1_output.get('data', {})
            quality_score = phase1_output.get('quality_score', 1.0)
            
            # 기본 모델 설정
            if not models_to_use:
                models_to_use = ["gpt4", "claude", "gemini"]
            
            # 1. Phase 1 데이터를 Phase 2 입력 형식으로 변환
            market_research_input = self._prepare_market_research_input(phase1_data)
            
            # 2. 프롬프트 로드
            prompt_variables = {
                'project_id': project_id or phase1_output.get('project_id', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'location': self.config.get('location', 'DMV area')
            }
            
            base_prompt = self.prompt_manager.load_prompt_with_variables(
                phase_number=2,
                variables=prompt_variables,
                version=None
            )
            
            # 3. 멀티모델 병렬 실행 (품질이 낮으면 단일 모델 사용)
            if quality_score < 0.5:
                print(f"낮은 품질 점수 ({quality_score:.2f}) - 단일 모델 사용")
                models_to_use = [models_to_use[0]]
            
            print(f"Market Research 실행 중 - {models_to_use}")
            model_results = await self.orchestrator.run_parallel(
                prompt=base_prompt,
                json_data=market_research_input,
                model_names=models_to_use
            )
            
            # 4. 결과 병합 또는 fallback 사용
            if not model_results:
                print("모든 모델 실패 - Fallback 가격 사용")
                pricing_data = self._generate_fallback_pricing(phase1_data)
            else:
                print(f"{len(model_results)}개 모델 응답 수신")
                pricing_data = self._merge_pricing_results(model_results, phase1_data)
            
            # 5. 세금 계산 및 O&P 적용
            final_pricing = self._calculate_final_costs(pricing_data)
            
            # 6. 결과 구성
            result = {
                'phase': 2,
                'phase_name': 'Market Research & Pricing',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'models_responded': len(model_results) if model_results else 0,
                'project_id': prompt_variables['project_id'],
                'location': self.location,
                'data': {
                    'pricing_data': pricing_data,
                    'cost_summary': final_pricing['cost_summary'],
                    'line_items': final_pricing['line_items'],
                    'tax_calculation': final_pricing['tax_calculation'],
                    'overhead_profit_percentage': final_pricing['overhead_profit_percentage']
                },
                'metadata': {
                    'phase1_quality_score': quality_score,
                    'using_fallback': len(model_results) == 0 if model_results is not None else True,
                    'partial_consensus': len(model_results) < len(models_to_use) if model_results else False,
                    'quality_warnings': phase1_output.get('validation_log', [])
                },
                'success': True
            }
            
            print(f"Phase 2 완료: 총 비용 ${final_pricing['cost_summary']['grand_total']:,.2f}")
            return result
            
        except Exception as e:
            print(f"Phase 2 오류: {e}")
            return {
                'phase': 2,
                'phase_name': 'Market Research & Pricing',
                'timestamp': datetime.now().isoformat(),
                'models_used': models_to_use,
                'error': str(e),
                'success': False
            }
    
    def _prepare_market_research_input(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1 데이터를 Phase 2 입력 형식으로 변환"""
        rooms_with_quantities = []
        materials_to_price = {}
        
        for room in phase1_data.get('rooms', []):
            room_data = {
                'room_name': room.get('room_name', 'Unknown'),
                'room_type': room.get('room_type', 'Unknown'),
                'measurements': room.get('measurements', {}),
                'tasks': []
            }
            
            for task in room.get('tasks', []):
                # 수량이 있는 작업만 포함
                if task.get('quantity_with_waste') or task.get('quantity'):
                    task_data = {
                        'description': task.get('description', ''),
                        'category': task.get('category', ''),
                        'quantity': task.get('quantity_with_waste', task.get('quantity', 0)),
                        'unit': task.get('unit', ''),
                        'material_type': task.get('material_type', ''),
                        'waste_included': task.get('waste_factor') is not None
                    }
                    room_data['tasks'].append(task_data)
                    
                    # 재료별 총량 집계
                    material_key = f"{task.get('material_type', 'unknown')}_{task.get('unit', '')}"
                    if material_key not in materials_to_price:
                        materials_to_price[material_key] = {
                            'description': task.get('description', ''),
                            'total_quantity': 0,
                            'unit': task.get('unit', ''),
                            'material_type': task.get('material_type', '')
                        }
                    materials_to_price[material_key]['total_quantity'] += task_data['quantity']
            
            rooms_with_quantities.append(room_data)
        
        return {
            'rooms_with_quantities': rooms_with_quantities,
            'materials_to_price': list(materials_to_price.values()),
            'waste_summary': phase1_data.get('waste_summary', {}),
            'project_info': {
                'location': self.location,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _generate_fallback_pricing(self, phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI 모델 실패 시 기본 가격 생성"""
        line_items = []
        
        for room in phase1_data.get('rooms', []):
            for task in room.get('tasks', []):
                material_type = task.get('material_type', 'default')
                quantity = task.get('quantity_with_waste', task.get('quantity', 0))
                unit = task.get('unit', 'sqft')
                
                # 기본 단가 조회
                default_price = self.DEFAULT_UNIT_PRICES.get(material_type, {})
                unit_price = default_price.get('price', 10.0)
                material_ratio = default_price.get('material_ratio', 0.5)
                
                total_cost = quantity * unit_price
                material_cost = total_cost * material_ratio
                labor_cost = total_cost * (1 - material_ratio)
                
                line_items.append({
                    'room': room.get('room_name', 'Unknown'),
                    'description': task.get('description', ''),
                    'quantity': quantity,
                    'unit': unit,
                    'unit_price': unit_price,
                    'material_cost': material_cost,
                    'labor_cost': labor_cost,
                    'total_cost': total_cost,
                    'source': 'fallback_pricing'
                })
        
        return {
            'line_items': line_items,
            'using_fallback': True
        }
    
    def _merge_pricing_results(self, model_results: List[Any], phase1_data: Dict[str, Any]) -> Dict[str, Any]:
        """여러 모델의 가격 결과 병합"""
        # 간단한 평균 계산 (실제로는 더 복잡한 병합 로직 필요)
        merged_items = []
        
        # 첫 번째 모델 결과를 기준으로 사용 (실제로는 모든 결과 병합 필요)
        if model_results and hasattr(model_results[0], 'room_estimates'):
            for estimate in model_results[0].room_estimates:
                merged_items.append({
                    'description': estimate.get('item', ''),
                    'quantity': estimate.get('quantity', 0),
                    'unit': estimate.get('unit', ''),
                    'unit_price': estimate.get('unit_price', 0),
                    'material_cost': estimate.get('material_cost', 0),
                    'labor_cost': estimate.get('labor_cost', 0),
                    'total_cost': estimate.get('total', 0),
                    'source': 'ai_consensus'
                })
        
        return {
            'line_items': merged_items,
            'using_fallback': False
        }
    
    def _calculate_final_costs(self, pricing_data: Dict[str, Any]) -> Dict[str, Any]:
        """최종 비용 계산 (세금, O&P 포함)"""
        line_items = pricing_data.get('line_items', [])
        
        # 집계
        total_material = sum(item.get('material_cost', 0) for item in line_items)
        total_labor = sum(item.get('labor_cost', 0) for item in line_items)
        
        # 세금 계산 (재료비에만 적용)
        tax_rate = self.TAX_SETTINGS[self.location]['material_tax']
        material_tax = total_material * tax_rate
        
        # Direct costs
        direct_costs = total_material + total_labor + material_tax
        
        # Overhead & Profit (최대 20%)
        overhead = direct_costs * 0.10  # 10% overhead
        subtotal_with_overhead = direct_costs + overhead
        profit = subtotal_with_overhead * 0.0909  # ~9.09% profit (총 ~20% O&P)
        
        overhead_profit_total = overhead + profit
        overhead_profit_percentage = (overhead_profit_total / direct_costs) * 100
        
        # 20% 초과 방지
        if overhead_profit_percentage > 20:
            overhead_profit_total = direct_costs * 0.20
            overhead = direct_costs * 0.10
            profit = direct_costs * 0.10
            overhead_profit_percentage = 20.0
        
        grand_total = direct_costs + overhead_profit_total
        
        return {
            'line_items': line_items,
            'tax_calculation': {
                'material_cost': total_material,
                'tax_rate': tax_rate,
                'tax_amount': material_tax
            },
            'cost_summary': {
                'total_material': total_material,
                'total_labor': total_labor,
                'total_tax': material_tax,
                'direct_costs': direct_costs,
                'overhead': overhead,
                'profit': profit,
                'overhead_profit_total': overhead_profit_total,
                'grand_total': grand_total
            },
            'overhead_profit_percentage': overhead_profit_percentage
        }