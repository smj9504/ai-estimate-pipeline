"""
Phase 1 독립 테스트 스크립트
Phase 0을 건너뛰고 Phase 1만 테스트할 수 있도록 함
"""
import asyncio
import json
import yaml
from pathlib import Path
from datetime import datetime
from src.phases.phase1_processor import Phase1Processor

# Phase 0 결과 예시 (수동으로 수정 가능)
SAMPLE_PHASE0_OUTPUT = {
    "phase": 0,
    "phase_name": "Generate Scope of Work",
    "timestamp": datetime.now().isoformat(),
    "project_id": "TEST_PROJECT_001",
    "success": True,
    "data": [
        {
            "name": "Bedroom 1",
            "materials": {
                "Paint - Walls": "Existing",
                "Paint - Ceiling": "Existing",
                "Carpet": "Existing",
                "Baseboards": "Existing",
                "Door": "Existing",
                "Window": "Existing"
            },
            "work_scope": {
                "Paint - Walls": "Remove & Replace",
                "Paint - Ceiling": "Remove & Replace",
                "Carpet": "Remove & Replace",
                "Baseboards": "Remove & Replace",
                "Door": "Repair",
                "Window": "Clean"
            },
            "measurements": {
                "width": 12,
                "length": 14,
                "height": 8,
                "windows": 2,
                "doors": 1
            },
            "demo_scope(already demo'd)": {
                "Carpet": 168  # 이미 철거된 카펫 면적
            },
            "additional_notes": "Water damage on walls, needs priming before painting"
        },
        {
            "name": "Kitchen",
            "materials": {
                "Cabinets - Upper": "Existing",
                "Cabinets - Lower": "Existing",
                "Countertop": "Granite",
                "Flooring": "Tile",
                "Paint - Walls": "Existing",
                "Paint - Ceiling": "Existing",
                "Appliances": "Standard"
            },
            "work_scope": {
                "Cabinets - Upper": "Remove & Replace",
                "Cabinets - Lower": "Remove & Replace",
                "Countertop": "Remove & Replace",
                "Flooring": "Clean & Seal",
                "Paint - Walls": "Remove & Replace",
                "Paint - Ceiling": "Remove & Replace",
                "Appliances": "Remove & Replace"
            },
            "measurements": {
                "width": 10,
                "length": 12,
                "height": 10,  # 높은 천장
                "linear_feet_upper": 12,
                "linear_feet_lower": 14,
                "countertop_sf": 35
            },
            "demo_scope(already demo'd)": {
                "Cabinets - Upper": 6,  # 이미 철거된 상부 캐비닛 개수
                "Countertop": 10  # 이미 철거된 카운터탑 면적
            },
            "additional_notes": "High ceiling requires scaffolding for painting"
        },
        {
            "name": "Bathroom",
            "materials": {
                "Vanity": "Existing",
                "Toilet": "Standard",
                "Tub/Shower": "Fiberglass",
                "Flooring": "Vinyl",
                "Paint - Walls": "Existing",
                "Paint - Ceiling": "Existing",
                "Mirror": "Existing"
            },
            "work_scope": {
                "Vanity": "Remove & Replace",
                "Toilet": "Remove & Replace",
                "Tub/Shower": "Clean & Recaulk",
                "Flooring": "Remove & Replace",
                "Paint - Walls": "Remove & Replace",
                "Paint - Ceiling": "Remove & Replace",
                "Mirror": "Replace"
            },
            "measurements": {
                "width": 5,
                "length": 8,
                "height": 8,
                "vanity_size": 36  # inches
            },
            "demo_scope(already demo'd)": {
                "Flooring": 40  # 이미 철거된 바닥재 면적
            },
            "additional_notes": "Moisture damage requires waterproofing treatment"
        }
    ]
}

async def test_phase1_standalone():
    """Phase 1 독립 실행 테스트"""
    
    print("=" * 80)
    print("Phase 1 독립 테스트 시작")
    print("=" * 80)
    
    # Phase 0 출력 파일 확인 (있으면 사용, 없으면 샘플 사용)
    phase0_file = Path("output/phase0_result.json")
    
    if phase0_file.exists():
        print(f"\n기존 Phase 0 결과 파일 발견: {phase0_file}")
        use_existing = input("이 파일을 사용하시겠습니까? (y/n, Enter는 y): ").strip().lower()
        
        if use_existing != 'n':  # 기본값을 'y'로 변경
            with open(phase0_file, 'r', encoding='utf-8') as f:
                phase0_output = json.load(f)
            print("기존 Phase 0 결과를 로드했습니다.")
            
            # 데이터 통계 출력
            if 'data' in phase0_output and isinstance(phase0_output['data'], list):
                total_rooms = 0
                for item in phase0_output['data'][1:]:  # 첫 번째는 jobsite_info
                    if 'rooms' in item:
                        total_rooms += len(item['rooms'])
                print(f"  - 총 {len(phase0_output['data']) - 1}개 층")
                print(f"  - 총 {total_rooms}개 방")
        else:
            phase0_output = SAMPLE_PHASE0_OUTPUT
            print("샘플 Phase 0 데이터를 사용합니다.")
    else:
        phase0_output = SAMPLE_PHASE0_OUTPUT
        print("샘플 Phase 0 데이터를 사용합니다.")
    
    # 사용할 모델 선택
    print("\n사용할 AI 모델을 선택하세요:")
    print("1. GPT-4만 사용")
    print("2. Claude만 사용")
    print("3. Gemini만 사용")
    print("4. 모든 모델 사용 (기본)")
    print("5. GPT-4 + Claude")
    print("6. GPT-4 + Gemini")
    print("7. Claude + Gemini")
    
    choice = input("\n선택 (1-7, Enter는 4): ").strip()
    
    model_configs = {
        '1': ["gpt4"],
        '2': ["claude"],
        '3': ["gemini"],
        '4': ["gpt4", "claude", "gemini"],
        '5': ["gpt4", "claude"],
        '6': ["gpt4", "gemini"],
        '7': ["claude", "gemini"],
        '': ["gpt4", "claude", "gemini"]  # 기본값
    }
    
    models_to_use = model_configs.get(choice, ["gpt4", "claude", "gemini"])
    
    print(f"\n선택된 모델: {models_to_use}")
    
    # Validation 모드 선택
    print("\n검증 수준을 선택하세요:")
    print("1. Strict - 필수 작업만 포함 (철거, 설치, 구조, 수리)")
    print("2. Balanced - 필수 작업 + 안전 관련 작업 (기본)")
    print("3. Lenient - 모든 유효한 작업 포함")
    
    val_choice = input("\n선택 (1-3, Enter는 2): ").strip()
    
    validation_modes = {
        '1': 'strict',
        '2': 'balanced',
        '3': 'lenient',
        '': 'balanced'  # 기본값
    }
    
    validation_mode = validation_modes.get(val_choice, 'balanced')
    
    # 설정 파일 임시 수정
    config_path = Path("config/settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['validation']['mode'] = validation_mode
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    print(f"검증 모드: {validation_mode}")
    
    # 방별 처리 옵션
    process_by_room = input("\n방별로 개별 처리하시겠습니까? (y/n, Enter는 y): ").strip().lower()
    process_by_room = process_by_room != 'n'  # 기본값은 True
    
    print(f"방별 처리: {'활성화' if process_by_room else '비활성화'}")
    
    # Phase 1 프로세서 초기화
    processor = Phase1Processor()
    
    try:
        # Phase 1 실행
        print("\n" + "=" * 80)
        print("Phase 1 실행 중...")
        print("=" * 80)
        
        result = await processor.process(
            phase0_output=phase0_output,
            models_to_use=models_to_use,
            project_id=phase0_output.get('project_id', 'TEST_001'),
            process_by_room=process_by_room
        )
        
        # 결과 저장
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"phase1_result_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n결과가 저장되었습니다: {output_file}")
        
        # 결과 요약 출력
        if result.get('success'):
            print("\n" + "=" * 80)
            print("Phase 1 성공")
            print("=" * 80)
            print(f"신뢰도 점수: {result.get('confidence_score', 0):.2f}")
            print(f"합의 수준: {result.get('consensus_level', 0):.2f}")
            print(f"응답한 모델 수: {result.get('models_responded', 0)}/{len(models_to_use)}")
            
            # 검증 결과 출력
            validation = result.get('validation', {})
            if validation:
                print("\n검증 결과:")
                print(f"  - Remove & Replace 로직: {'✓' if validation.get('remove_replace_logic', {}).get('valid') else '✗'}")
                print(f"  - 측정값 정확도: {'✓' if validation.get('measurements_accuracy', {}).get('valid') else '✗'}")
                print(f"  - 특수 작업: {'✓' if validation.get('special_tasks', {}).get('valid') else '✗'}")
                print(f"  - 전체 유효성: {'✓' if validation.get('overall_valid') else '✗'}")
                
                # 검증 이슈 출력
                for category, info in validation.items():
                    if isinstance(info, dict) and info.get('issues'):
                        print(f"\n  {category} 이슈:")
                        for issue in info['issues']:
                            print(f"    - {issue}")
            
            # Phase 2 준비
            print("\n" + "=" * 80)
            print("Phase 2 입력 데이터 준비")
            print("=" * 80)
            
            phase2_input = processor.prepare_for_phase2(result)
            phase2_file = output_dir / f"phase2_input_{timestamp}.json"
            
            with open(phase2_file, 'w', encoding='utf-8') as f:
                json.dump(phase2_input, f, indent=2, ensure_ascii=False)
            
            print(f"Phase 2 입력 데이터 저장: {phase2_file}")
            print("\nPhase 2를 실행하려면 test_phase2_standalone.py를 사용하세요.")
            
        else:
            print("\n" + "=" * 80)
            print("Phase 1 실패")
            print("=" * 80)
            print(f"오류: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("Phase 1 독립 테스트 스크립트")
    print("Phase 0을 건너뛰고 Phase 1만 테스트합니다")
    print("=" * 80)
    
    # 환경 변수 확인
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = {
        'OpenAI': os.getenv('OPENAI_API_KEY'),
        'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'Google': os.getenv('GOOGLE_API_KEY')
    }
    
    print("\nAPI 키 상태:")
    for name, key in api_keys.items():
        status = "✓ 설정됨" if key else "✗ 없음"
        print(f"  {name}: {status}")
    
    if not any(api_keys.values()):
        print("\n경고: API 키가 설정되지 않았습니다.")
        print(".env 파일에 API 키를 설정해주세요.")
        return
    
    # Phase 1 테스트 실행
    asyncio.run(test_phase1_standalone())

if __name__ == "__main__":
    main()