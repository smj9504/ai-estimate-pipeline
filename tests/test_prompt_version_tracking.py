"""
Test script to validate prompt version tracking throughout the pipeline
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phases.phase1_processor import Phase1Processor
from src.phases.phase2_processor import Phase2Processor
from src.models.data_models import ModelResponse

async def test_prompt_version_tracking():
    """Test that prompt_version is properly tracked through the pipeline"""
    
    print("=" * 80)
    print("PROMPT VERSION TRACKING TEST")
    print("=" * 80)
    
    # Sample test data
    test_data = {
        "jobsite_info": {
            "address": "123 Test St",
            "city": "TestCity",
            "state": "TS",
            "zip": "12345",
            "year_built": 2000,
            "property_type": "Single Family",
            "stories": 1,
            "claim_number": "TEST-001",
            "date_of_loss": "2024-01-01",
            "cause_of_loss": "Fire"
        },
        "floors": [
            {
                "floor_name": "First Floor",
                "rooms": [
                    {
                        "name": "Living Room",
                        "materials": {
                            "Flooring": "Hardwood",
                            "Wall": "Drywall",
                            "Ceiling": "Drywall"
                        },
                        "work_scope": {
                            "Flooring": "Remove & Replace",
                            "Wall": "Paint",
                            "Ceiling": "Paint"
                        },
                        "measurements": {
                            "floor_area_sqft": 250,
                            "wall_area_sqft": 400,
                            "ceiling_area_sqft": 250,
                            "floor_perimeter_lf": 65,
                            "height": 9
                        },
                        "demo_scope(already demo'd)": {
                            "Wall Drywall(sq_ft)": 50
                        }
                    }
                ]
            }
        ]
    }
    
    # Test with different prompt versions
    test_versions = ['improved', 'fast', None]
    
    for version in test_versions:
        print(f"\n{'='*60}")
        print(f"Testing with prompt_version: {version or 'default'}")
        print('='*60)
        
        try:
            # Phase 1 test
            phase1 = Phase1Processor()
            
            # Run Phase 1 with prompt version
            phase1_result = await phase1.process(
                input_data=test_data,
                models_to_use=['gpt4'],  # Using single model for quick test
                prompt_version=version
            )
            
            # Check Phase 1 result
            if phase1_result.get('success'):
                print(f"✅ Phase 1 Success")
                print(f"   - Prompt Version in result: {phase1_result.get('prompt_version', 'NOT FOUND')}")
                
                # Check if prompt_version is in metadata
                if 'data' in phase1_result:
                    data = phase1_result['data']
                    if isinstance(data, dict) and 'metadata' in data:
                        metadata_version = data['metadata'].get('prompt_version', 'NOT IN METADATA')
                        print(f"   - Prompt Version in metadata: {metadata_version}")
            else:
                print(f"❌ Phase 1 Failed: {phase1_result.get('error', 'Unknown error')}")
                continue
            
            # Phase 2 test
            phase2 = Phase2Processor()
            
            # Run Phase 2 with prompt version
            phase2_result = await phase2.process(
                phase1_output=phase1_result,
                models_to_use=['gpt4'],
                prompt_version=version
            )
            
            # Check Phase 2 result
            if phase2_result.get('success'):
                print(f"✅ Phase 2 Success")
                print(f"   - Prompt Version in result: {phase2_result.get('prompt_version', 'NOT FOUND')}")
            else:
                print(f"❌ Phase 2 Failed: {phase2_result.get('error', 'Unknown error')}")
            
            print(f"\n📊 Summary for version '{version or 'default'}':")
            print(f"   - Phase 1 has prompt_version: {'✅' if phase1_result.get('prompt_version') else '❌'}")
            print(f"   - Phase 2 has prompt_version: {'✅' if phase2_result.get('prompt_version') else '❌'}")
            
        except Exception as e:
            print(f"❌ Error testing version '{version or 'default'}': {e}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_prompt_version_tracking())