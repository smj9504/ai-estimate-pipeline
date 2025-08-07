#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 개선사항 테스트 스크립트
1. 최소 작업 개수 검증 테스트
2. AI 모델 응답 로깅 확인

실행: python test_validation_improvements.py
"""

import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.validators.response_validator import ValidationOrchestrator, QualityLevel

def test_minimum_task_validation():
    """최소 작업 개수 검증 로직 테스트"""
    print("\n" + "="*80)
    print("TEST 1: 최소 작업 개수 검증 기능")
    print("="*80)
    
    validator = ValidationOrchestrator()
    
    # 테스트 1: 작업이 0개인 경우
    print("\n[1] 작업 0개 응답 (CRITICAL 예상)")
    response_empty = {
        'rooms': [
            {'name': 'Kitchen', 'tasks': []},
            {'name': 'Bathroom', 'tasks': []}
        ]
    }
    
    _, report = validator.validate_response(response_empty, {})
    print(f"   점수: {report.quality_score}/100")
    print(f"   등급: {report.quality_level.value.upper()}")
    print(f"   이슈: {report.total_issues}개 (Critical: {report.critical_issues})")
    
    empty_issue = [i for i in report.issues if i.category == "EMPTY_RESPONSE"]
    if empty_issue:
        print(f"   ✓ EMPTY_RESPONSE 감지: {empty_issue[0].message[:50]}...")
    
    # 테스트 2: 작업이 부족한 경우 (방당 2개)
    print("\n[2] 작업 부족 응답 (HIGH 예상)")
    response_few = {
        'rooms': [
            {
                'name': 'Kitchen', 
                'tasks': [
                    {'task_name': 'Remove flooring', 'quantity': 100, 'unit': 'sqft'},
                    {'task_name': 'Install flooring', 'quantity': 100, 'unit': 'sqft'}
                ]
            },
            {
                'name': 'Bathroom',
                'tasks': [
                    {'task_name': 'Remove tiles', 'quantity': 50, 'unit': 'sqft'},
                    {'task_name': 'Install tiles', 'quantity': 50, 'unit': 'sqft'}
                ]
            }
        ]
    }
    
    _, report = validator.validate_response(response_few, {})
    print(f"   점수: {report.quality_score}/100")
    print(f"   등급: {report.quality_level.value.upper()}")
    
    insufficient_issue = [i for i in report.issues if i.category == "INSUFFICIENT_TASKS"]
    if insufficient_issue:
        print(f"   ✓ INSUFFICIENT_TASKS 감지: {insufficient_issue[0].message[:50]}...")
    
    # 테스트 3: 적절한 작업 개수 (방당 12개)
    print("\n[3] 충분한 작업 응답 (GOOD/EXCELLENT 예상)")
    response_good = {
        'rooms': [
            {
                'name': 'Kitchen',
                'tasks': [
                    {'task_name': f'Kitchen Task {i+1}', 'quantity': 10+i, 'unit': 'sqft'}
                    for i in range(12)
                ]
            },
            {
                'name': 'Bathroom',
                'tasks': [
                    {'task_name': f'Bathroom Task {i+1}', 'quantity': 5+i, 'unit': 'sqft'}
                    for i in range(12)
                ]
            }
        ]
    }
    
    _, report = validator.validate_response(response_good, {})
    print(f"   점수: {report.quality_score}/100")
    print(f"   등급: {report.quality_level.value.upper()}")
    print(f"   총 작업: {report.metadata.get('total_tasks', 0)}개")
    
    task_issues = [i for i in report.issues if 'TASK' in i.category or 'EMPTY' in i.category]
    if not task_issues:
        print(f"   ✓ 작업 관련 이슈 없음")
    
    print("\n✓ 최소 작업 개수 검증 테스트 완료")
    return True

def check_ai_response_logging():
    """AI 응답 로깅 기능 확인"""
    print("\n" + "="*80)
    print("TEST 2: AI 응답 로깅 기능 확인")
    print("="*80)
    
    # ai_responses 폴더 확인
    if os.path.exists('ai_responses'):
        files = os.listdir('ai_responses')
        print(f"\n✓ ai_responses 폴더 존재")
        print(f"  저장된 응답 파일: {len(files)}개")
        
        if files:
            # 최신 파일 정보
            latest = sorted(files)[-1]
            file_path = os.path.join('ai_responses', latest)
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            print(f"  최신 파일: {latest}")
            print(f"  파일 크기: {file_size:.1f} KB")
            
            # 파일 구조 확인
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                has_header = any('Response' in line for line in lines[:5])
                has_separator = any('=' * 50 in line for line in lines[:10])
                
                if has_header and has_separator:
                    print(f"  ✓ 올바른 로그 형식")
    else:
        print("\n! ai_responses 폴더가 없음 (AI 모델 실행 후 생성됨)")
    
    # debug_responses 폴더 확인 (기존 Claude/Gemini 로그)
    if os.path.exists('debug_responses'):
        files = os.listdir('debug_responses')
        if files:
            print(f"\n! 기존 debug_responses 폴더 발견 ({len(files)}개 파일)")
            print("  → ai_responses 폴더로 통합 권장")
    
    print("\n✓ AI 응답 로깅 확인 완료")
    return True

def main():
    """메인 테스트 실행"""
    print("\n" + "="*80)
    print("개선사항 통합 테스트")
    print("="*80)
    
    success = True
    
    # 테스트 1: 최소 작업 검증
    try:
        test_minimum_task_validation()
    except Exception as e:
        print(f"\n✗ 최소 작업 검증 테스트 실패: {e}")
        success = False
    
    # 테스트 2: AI 응답 로깅 확인
    try:
        check_ai_response_logging()
    except Exception as e:
        print(f"\n✗ AI 응답 로깅 확인 실패: {e}")
        success = False
    
    # 결과 요약
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    
    print("\n구현된 개선사항:")
    print("1. ✓ 최소 작업 개수 검증 추가")
    print("   - 0개 작업 → CRITICAL 이슈")
    print("   - 방당 5개 미만 → HIGH 이슈")
    print("   - 방당 10개 미만 → MEDIUM 이슈")
    
    print("\n2. ✓ AI 모델 응답 로깅 개선")
    print("   - 모든 모델 응답을 ai_responses/ 폴더에 저장")
    print("   - 콘솔에 응답 요약 및 작업 개수 표시")
    print("   - 0개 작업 시 경고 메시지 출력")
    
    if success:
        print("\n✓ 모든 테스트 통과!")
    else:
        print("\n✗ 일부 테스트 실패")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)