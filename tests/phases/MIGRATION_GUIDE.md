# Migration Guide: From Legacy Tests to Phase Testing Framework

This guide helps transition from the old standalone test files to the new organized phase testing framework.

## 🔄 What Changed

### Old System (Legacy)
- `test_phase1_standalone.py` in root directory
- Manual test data creation
- Limited configuration options
- Hardcoded model selection
- Results saved with complex naming schemes

### New System (Phase Testing Framework)
- Organized `tests/phases/` directory structure
- Reusable test configurations
- Command-line interface with presets
- Flexible model and validation combinations
- Consistent result format and analysis

## 🚀 Quick Migration

### Before (Legacy)
```bash
# Old way - run the standalone script
python test_phase1_standalone.py
```

### After (New Framework)
```bash
# New way - use the CLI
python run_phase_tests.py single --phase 1

# Or directly
python -m tests.phases.cli single --phase 1
```

## 📋 Feature Mapping

### Test Configurations

#### Legacy Configuration Prompts:
```python
# Old interactive prompts
choice = input("선택 (1-7, Enter는 4): ").strip()  # Model selection
val_choice = input("선택 (1-3, Enter는 2): ").strip()  # Validation mode
process_by_room = input("방별로 개별 처리하시겠습니까?").strip().lower()
```

#### New Configuration System:
```bash
# Command-line arguments
python run_phase_tests.py single --phase 1 --models gpt4 claude --validation-mode balanced

# Or config files
python run_phase_tests.py single --phase 1 --config fast_test
```

### Model Selection

| Legacy Choice | New Equivalent |
|---------------|----------------|
| 1. GPT-4만 | `--models gpt4` |
| 2. Claude만 | `--models claude` |
| 3. Gemini만 | `--models gemini` |
| 4. 모든 모델 (기본) | `--models gpt4 claude gemini` |
| 5. GPT-4 + Claude | `--models gpt4 claude` |
| 6. GPT-4 + Gemini | `--models gpt4 gemini` |
| 7. Claude + Gemini | `--models claude gemini` |

### Validation Modes

| Legacy Choice | New Equivalent |
|---------------|----------------|
| 1. Strict | `--validation-mode strict` |
| 2. Balanced (기본) | `--validation-mode balanced` |
| 3. Lenient | `--validation-mode lenient` |

## 📁 File Organization Migration

### Legacy Files Moved:
- `test_phase1_standalone.py` → `tests/phases/legacy_test_phase1_standalone.py`

### New Structure:
```
tests/phases/
├── individual/
│   ├── test_phase0.py    # New Phase 0 implementation
│   ├── test_phase1.py    # Replaces legacy standalone
│   └── test_phase2.py    # New Phase 2 implementation
├── integration/
│   └── test_phase_pipeline.py  # Multi-phase testing
├── configs/
│   ├── default_single_phase.yaml
│   ├── fast_test.yaml
│   └── comprehensive.yaml
└── fixtures/
    ├── sample_measurement.json
    ├── sample_demo.json
    └── sample_intake_form.json
```

## 🔧 Converting Legacy Scripts

### Step 1: Identify the Phase
```python
# Legacy script typically focused on one phase
# e.g., test_phase1_standalone.py → Phase 1
```

### Step 2: Convert to New CLI Command
```bash
# Replace manual script execution with CLI command
python run_phase_tests.py single --phase 1 \
  --models gpt4 claude gemini \
  --validation-mode balanced \
  --test-name "converted_legacy_test"
```

### Step 3: Use Configuration Files for Complex Setups
```yaml
# Create tests/phases/configs/my_legacy_config.yaml
phase_numbers: [1]
models: ["gpt4", "claude", "gemini"]
validation_mode: "balanced"
process_by_room: true
timeout_seconds: 300
test_name: "legacy_conversion"
description: "Converted from legacy test_phase1_standalone.py"
```

### Step 4: Run with New Configuration
```bash
python run_phase_tests.py single --config my_legacy_config
```

## 📊 Result Format Changes

### Legacy Output Files:
```
output/phase1_GCM_BAL_ROOM_SAMPLE_20250808_123456.json
```

### New Output Files:
```
test_outputs/legacy_conversion_GPT_CLA_GEM_balanced_20250808_123456.json
```

### Result Structure Improvements:
```json
{
  "test_metadata": {
    "phase_number": 1,
    "phase_name": "Merge Measurement & Work Scope",
    "test_config": { ... },
    "timestamp": "20250808_123456"
  },
  "test_result": {
    "phase_number": 1,
    "success": true,
    "execution_time": 15.2,
    "confidence_score": 0.85,
    "consensus_level": 0.78,
    "models_responded": 3,
    "total_models": 3,
    "validation_results": { ... },
    "output_data": { ... }
  },
  "analysis": {
    "performance_metrics": { ... },
    "quality_metrics": { ... },
    "recommendations": [ ... ]
  }
}
```

## 🆕 New Capabilities Not in Legacy

### 1. Pipeline Testing
Run multiple phases in sequence:
```bash
python run_phase_tests.py pipeline --phases 0 1 2
```

### 2. Comparison Testing
Compare different configurations:
```bash
python run_phase_tests.py compare --phase 1 --compare-type models
python run_phase_tests.py compare --phase 1 --compare-type validation
```

### 3. Predefined Scenarios
```bash
python run_phase_tests.py scenario validation_comparison
python run_phase_tests.py scenario performance --phase 1
```

### 4. Result Management
```bash
python run_phase_tests.py list-results
python run_phase_tests.py list-configs
```

## ⚠️ Breaking Changes

### 1. Import Paths
```python
# Legacy
from src.phases.phase1_processor import Phase1Processor

# New (internal to framework)
from tests.phases.individual.test_phase1 import Phase1Test
```

### 2. Configuration Loading
```python
# Legacy - manual YAML handling
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# New - automatic config management
config = PhaseTestBase().load_test_config("fast_test")
```

### 3. Result Analysis
```python
# Legacy - manual result inspection
if result.get('success'):
    print(f"신뢰도 점수: {result.get('confidence_score', 0):.2f}")

# New - automatic analysis and reporting
analysis = phase_test.analyze_results(result)
```

## 🔄 Gradual Migration Strategy

### Phase 1: Keep Legacy, Add New
- Keep existing `test_phase1_standalone.py` working
- Start using new CLI for new tests
- Compare results to ensure consistency

### Phase 2: Transition Scripts
- Convert complex test scenarios to configuration files
- Update documentation and team training
- Establish new testing workflows

### Phase 3: Full Migration
- Archive legacy scripts
- Update CI/CD to use new framework
- Remove old test infrastructure

## 🚨 Common Migration Issues

### Issue 1: API Keys
**Legacy**: Loaded from `.env` in script
**Solution**: API keys still loaded from `.env` automatically

### Issue 2: Test Data
**Legacy**: Hardcoded or manual file loading
**Solution**: Use fixture system or pipeline input from previous phases

### Issue 3: Custom Analysis
**Legacy**: Manual result processing
**Solution**: Extend `PhaseTestBase` for custom analysis

### Issue 4: Batch Processing
**Legacy**: `process_by_room` parameter
**Solution**: Use `--config` with `process_by_room: false` setting

## 📞 Getting Help

### For Migration Issues:
1. Check `tests/phases/legacy_test_phase1_standalone.py` for reference
2. Compare old and new result formats
3. Use `--verbose` flag for detailed output
4. Review `tests/phases/README.md` for complete documentation

### For New Features:
1. Explore predefined scenarios: `python run_phase_tests.py scenario --help`
2. Try comparison testing: `python run_phase_tests.py compare --help`
3. Set up pipeline testing: `python run_phase_tests.py pipeline --help`

### Debug Commands:
```bash
# Compare legacy vs new results
python run_phase_tests.py single --phase 1 --verbose --test-name "migration_test"

# List available configurations
python run_phase_tests.py list-configs

# Check recent results
python run_phase_tests.py list-results --limit 5
```

---

**Migration Support**: For questions or issues during migration, refer to the team documentation or create an issue with specific details about the legacy functionality that needs to be preserved.