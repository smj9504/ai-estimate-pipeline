"""
Gemini 모델로 Phase 0 테스트
"""
import asyncio
import json
from pathlib import Path
from src.phases.phase0_processor import Phase0Processor

async def test_phase0_with_gemini():
    """Gemini 모델로 Phase 0 테스트"""
    
    print("=" * 60)
    print("Phase 0 테스트 - Gemini 모델 사용")
    print("=" * 60)
    
    # Phase 0 프로세서 초기화
    processor = Phase0Processor()
    
    # 테스트 데이터 로드
    test_data_path = Path("test_data")
    
    # 1. Measurement 데이터
    with open(test_data_path / "sample_measurement.json", 'r', encoding='utf-8') as f:
        measurement_data = json.load(f)
    print(f"[OK] Measurement 데이터 로드: {len(json.dumps(measurement_data))} bytes")
    
    # 2. Demolition scope 데이터
    with open(test_data_path / "sample_demo.json", 'r', encoding='utf-8') as f:
        demo_data = json.load(f)
    print(f"[OK] Demolition 데이터 로드: {len(json.dumps(demo_data))} bytes")
    
    # 3. Intake form
    with open(test_data_path / "sample_intake_form.txt", 'r', encoding='utf-8') as f:
        intake_form = f.read()
    print(f"[OK] Intake form 로드: {len(intake_form)} characters")
    
    print("\n" + "-" * 60)
    print("Gemini 모델 호출 중...")
    print("-" * 60)
    
    try:
        # Gemini 모델로 Phase 0 실행
        result = await processor.process(
            measurement_data=measurement_data,
            demolition_scope_data=demo_data,
            intake_form=intake_form,
            model_to_use="gemini",  # Gemini 사용
            project_id="TEST-GEMINI-001"
        )
        
        if result.get('success'):
            print("\n[SUCCESS] Phase 0 성공!")
            print(f"   - 모델: {result.get('model_used')}")
            print(f"   - 처리 시간: {result.get('processing_time', 0):.2f}초")
            
            # 결과 저장
            output_path = Path("output") / "phase0_gemini_result.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"   - 결과 저장: {output_path}")
            
            # 결과 요약 출력
            if 'data' in result:
                data = result['data']
                if isinstance(data, list) and len(data) > 0:
                    floors = data
                elif isinstance(data, dict) and 'floors' in data:
                    floors = data['floors']
                else:
                    floors = []
                
                print(f"\n[SUMMARY] 결과 요약:")
                print(f"   - 층 수: {len(floors)}")
                
                for floor in floors:
                    location = floor.get('location', 'Unknown')
                    rooms = floor.get('rooms', [])
                    print(f"   - {location}: {len(rooms)}개 방")
                    
                    for room in rooms[:3]:  # 처음 3개 방만 표시
                        room_name = room.get('name', 'Unknown')
                        print(f"      * {room_name}")
                    
                    if len(rooms) > 3:
                        print(f"      ... 외 {len(rooms) - 3}개")
        else:
            print(f"\n[ERROR] Phase 0 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(test_phase0_with_gemini())