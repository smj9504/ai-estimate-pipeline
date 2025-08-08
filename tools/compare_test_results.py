#!/usr/bin/env python
"""
Phase 1 테스트 결과 비교 도구
여러 테스트 조건으로 생성된 결과를 비교 분석
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any

def parse_filename(filename: str) -> Dict[str, str]:
    """
    파일명에서 테스트 조건 추출
    예: phase1_GCM_BAL_ROOM_SAMPLE_20250808_120000.json
    """
    parts = filename.replace('.json', '').split('_')
    
    if len(parts) >= 6:
        return {
            'models': parts[1],  # G=GPT4, C=Claude, M=Gemini
            'validation': parts[2],  # STR=Strict, BAL=Balanced, LEN=Lenient
            'processing': parts[3],  # ROOM=방별, BATCH=일괄
            'input': parts[4],  # SAMPLE 또는 타임스탬프
            'timestamp': f"{parts[5]}_{parts[6]}" if len(parts) > 6 else parts[5]
        }
    return {}

def decode_models(model_str: str) -> List[str]:
    """모델 약어를 풀네임으로 변환"""
    mapping = {'G': 'GPT-4', 'C': 'Claude', 'M': 'Gemini'}
    return [mapping.get(c, c) for c in model_str]

def load_result(filepath: Path) -> Dict[str, Any]:
    """결과 파일 로드 및 주요 지표 추출"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 주요 지표 추출
    metrics = {
        'confidence_score': data.get('confidence_score', 0),
        'consensus_level': data.get('consensus_level', 0),
        'models_responded': data.get('models_responded', 0),
        'success': data.get('success', False),
        'processing_time': data.get('processing_time', 0),
        'total_tasks': 0,
        'total_rooms': 0,
        'validation_passed': False
    }
    
    # 작업 수 계산
    if 'data' in data and isinstance(data['data'], dict):
        if 'rooms' in data['data']:
            metrics['total_rooms'] = len(data['data']['rooms'])
            for room in data['data']['rooms']:
                if 'tasks' in room:
                    metrics['total_tasks'] += len(room['tasks'])
    
    # 검증 결과
    if 'validation' in data:
        metrics['validation_passed'] = data['validation'].get('overall_valid', False)
    
    # 테스트 구성 정보 (있으면)
    if 'test_config' in data:
        metrics['test_config'] = data['test_config']
    
    return metrics

def compare_results():
    """모든 Phase 1 결과 파일 비교"""
    
    output_dir = Path("output")
    result_files = sorted(output_dir.glob("phase1_*.json"), reverse=True)
    
    if not result_files:
        print("비교할 결과 파일이 없습니다.")
        return
    
    print(f"\n총 {len(result_files)}개의 결과 파일 발견")
    print("=" * 100)
    
    # 결과 수집
    results = []
    for filepath in result_files[:20]:  # 최신 20개만
        filename = filepath.name
        config = parse_filename(filename)
        
        if not config:
            continue
            
        try:
            metrics = load_result(filepath)
            
            result = {
                'file': filename,
                'models': ' + '.join(decode_models(config['models'])),
                'validation': config['validation'],
                'processing': config['processing'],
                'input': config['input'],
                'timestamp': config['timestamp'],
                **metrics
            }
            results.append(result)
        except Exception as e:
            print(f"파일 로드 실패: {filename} - {e}")
    
    if not results:
        print("분석할 결과가 없습니다.")
        return
    
    # DataFrame으로 변환
    df = pd.DataFrame(results)
    
    # 컬럼 순서 정리
    columns_order = [
        'timestamp', 'models', 'validation', 'processing', 'input',
        'confidence_score', 'consensus_level', 'total_tasks', 'total_rooms',
        'validation_passed', 'processing_time', 'models_responded'
    ]
    
    # 존재하는 컬럼만 선택
    columns_to_show = [col for col in columns_order if col in df.columns]
    df = df[columns_to_show]
    
    # 1. 전체 결과 요약
    print("\n[전체 결과 요약]")
    print(df.to_string(index=False))
    
    # 2. 모델 조합별 평균 성능
    print("\n\n[모델 조합별 평균 성능]")
    print("-" * 80)
    model_stats = df.groupby('models').agg({
        'confidence_score': 'mean',
        'consensus_level': 'mean',
        'total_tasks': 'mean',
        'processing_time': 'mean'
    }).round(2)
    print(model_stats)
    
    # 3. 검증 모드별 비교
    print("\n\n[검증 모드별 비교]")
    print("-" * 80)
    validation_stats = df.groupby('validation').agg({
        'confidence_score': 'mean',
        'total_tasks': 'mean',
        'validation_passed': lambda x: f"{sum(x)}/{len(x)}"
    }).round(2)
    print(validation_stats)
    
    # 4. 처리 방식별 비교 (ROOM vs BATCH)
    print("\n\n[처리 방식별 비교]")
    print("-" * 80)
    processing_stats = df.groupby('processing').agg({
        'confidence_score': 'mean',
        'total_tasks': 'mean',
        'processing_time': 'mean'
    }).round(2)
    print(processing_stats)
    
    # 5. 최고/최저 성능
    print("\n\n[최고/최저 성능]")
    print("-" * 80)
    
    if len(df) > 0:
        # 최고 신뢰도
        best_confidence = df.loc[df['confidence_score'].idxmax()]
        print(f"최고 신뢰도: {best_confidence['confidence_score']:.2f}")
        print(f"  - 모델: {best_confidence['models']}")
        print(f"  - 검증: {best_confidence['validation']}")
        print(f"  - 처리: {best_confidence['processing']}")
        
        print()
        
        # 최다 작업 생성
        most_tasks = df.loc[df['total_tasks'].idxmax()]
        print(f"최다 작업 생성: {most_tasks['total_tasks']} tasks")
        print(f"  - 모델: {most_tasks['models']}")
        print(f"  - 검증: {most_tasks['validation']}")
        print(f"  - 처리: {most_tasks['processing']}")
    
    # 6. 상세 비교를 위한 파일 선택
    print("\n\n[상세 비교]")
    print("-" * 80)
    print("두 결과를 선택하여 상세 비교하려면 번호를 입력하세요:")
    
    for i, row in df.head(10).iterrows():
        print(f"{i}: {row['timestamp']} - {row['models']} ({row['validation']}, {row['processing']})")
    
    try:
        choice1 = int(input("\n첫 번째 파일 번호 (Enter로 건너뛰기): ").strip())
        choice2 = int(input("두 번째 파일 번호: ").strip())
        
        if choice1 in df.index and choice2 in df.index:
            compare_two_results(
                output_dir / df.loc[choice1, 'file'],
                output_dir / df.loc[choice2, 'file']
            )
    except (ValueError, KeyError):
        print("상세 비교를 건너뜁니다.")

def compare_two_results(file1: Path, file2: Path):
    """두 결과 파일 상세 비교"""
    
    print("\n\n" + "=" * 100)
    print("상세 비교")
    print("=" * 100)
    
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 방별 작업 수 비교
    print("\n[방별 작업 수 비교]")
    print("-" * 50)
    
    rooms1 = {}
    rooms2 = {}
    
    if 'data' in data1 and 'rooms' in data1['data']:
        for room in data1['data']['rooms']:
            room_name = room.get('name', 'Unknown')
            rooms1[room_name] = len(room.get('tasks', []))
    
    if 'data' in data2 and 'rooms' in data2['data']:
        for room in data2['data']['rooms']:
            room_name = room.get('name', 'Unknown')
            rooms2[room_name] = len(room.get('tasks', []))
    
    all_rooms = set(rooms1.keys()) | set(rooms2.keys())
    
    print(f"{'Room':<20} {'File 1 Tasks':<15} {'File 2 Tasks':<15} {'Difference':<10}")
    print("-" * 60)
    
    for room in sorted(all_rooms):
        tasks1 = rooms1.get(room, 0)
        tasks2 = rooms2.get(room, 0)
        diff = tasks2 - tasks1
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{room:<20} {tasks1:<15} {tasks2:<15} {diff_str:<10}")
    
    print("-" * 60)
    print(f"{'Total':<20} {sum(rooms1.values()):<15} {sum(rooms2.values()):<15}")

def main():
    """메인 실행 함수"""
    print("=" * 100)
    print("Phase 1 테스트 결과 비교 도구")
    print("=" * 100)
    
    compare_results()

if __name__ == "__main__":
    main()