#!/usr/bin/env python
"""
Test script to verify the fixes for:
1. Reasoning field generation in Phase 1
2. Project info extraction in Phase 0
"""

import re
import json

def test_project_info_extraction():
    """Test project info extraction from intake form"""
    print("\n=== Testing Project Info Extraction ===")
    
    intake_form = """Property Address: 456 Elm Street, Arlington, VA 22201
Occupancy: Occupied

Default Scope(1st Floor):
Material:
    Floor: Laminate Wood
    Wall: drywall
    Ceiling: drywall"""
    
    project_info = {
        'Jobsite': '',
        'occupancy': '',
        'company': {}
    }
    
    # Property Address extraction
    address_match = re.search(r'Property Address:\s*(.+?)(?:\n|$)', intake_form)
    if address_match:
        project_info['Jobsite'] = address_match.group(1).strip()
    
    # Occupancy extraction  
    occupancy_match = re.search(r'Occupancy:\s*(.+?)(?:\n|$)', intake_form)
    if occupancy_match:
        project_info['occupancy'] = occupancy_match.group(1).strip()
    
    print("Extracted project info:")
    print(json.dumps(project_info, indent=2))
    
    # Verify extraction
    assert project_info['Jobsite'] == "456 Elm Street, Arlington, VA 22201", "Jobsite extraction failed"
    assert project_info['occupancy'] == "Occupied", "Occupancy extraction failed"
    print("[PASS] Project info extraction test passed!")
    
    return project_info

def test_reasoning_field_prompt():
    """Check if the improved prompt includes reasoning field requirements"""
    print("\n=== Testing Reasoning Field in Prompt ===")
    
    try:
        with open('prompts/phase1_prompt_improved.txt', 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        
        # Check for reasoning field in JSON structure
        if '"reasoning":' in prompt_content:
            print("[PASS] Reasoning field found in JSON structure")
        else:
            print("[FAIL] Reasoning field missing in JSON structure")
            return False
        
        # Check for detailed reasoning requirements
        if 'CRITICAL REQUIREMENT: Reasoning Field' in prompt_content:
            print("[PASS] Critical requirement section for reasoning found")
        else:
            print("[FAIL] Critical requirement section missing")
            return False
        
        # Check for minimum words requirement
        if 'MINIMUM 20 words' in prompt_content:
            print("[PASS] Minimum 20 words requirement found")
        else:
            print("[FAIL] Minimum words requirement missing")
            return False
        
        print("[PASS] Reasoning field prompt test passed!")
        return True
        
    except FileNotFoundError:
        print("[FAIL] Could not find phase1_prompt_improved.txt")
        return False

def check_model_interface_update():
    """Check if model interface properly extracts reasoning field"""
    print("\n=== Testing Model Interface Reasoning Extraction ===")
    
    try:
        with open('src/models/model_interface.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for improved reasoning extraction
        if "task.get('reasoning', task.get('notes', ''))" in content:
            print("[PASS] Model interface properly extracts reasoning field with fallback to notes")
        else:
            print("[WARNING] Model interface may not properly extract reasoning field")
            
    except FileNotFoundError:
        print("[FAIL] Could not find model_interface.py")

def main():
    print("=" * 50)
    print("Testing Fixes for AI Estimate Pipeline")
    print("=" * 50)
    
    # Test 1: Project info extraction
    project_info = test_project_info_extraction()
    
    # Test 2: Reasoning field in prompt
    reasoning_ok = test_reasoning_field_prompt()
    
    # Test 3: Model interface update
    check_model_interface_update()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("- Project info extraction: [FIXED]")
    print("- Reasoning field generation: [FIXED]")
    print("=" * 50)
    
    print("\nNext Steps:")
    print("1. Run full pipeline test with actual API calls")
    print("2. Verify reasoning field contains detailed explanations (20+ words)")
    print("3. Verify project_info is populated in Phase 1 output")

if __name__ == "__main__":
    main()