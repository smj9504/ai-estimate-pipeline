# Phase Testing Framework

A comprehensive testing system for the AI Estimate Pipeline phases, designed for scalability, flexibility, and ease of maintenance.

## 📁 Directory Structure

```
tests/phases/
├── README.md                    # This documentation
├── __init__.py                  # Package initialization
├── base.py                      # Base classes and utilities
├── orchestrator.py              # Test orchestration and coordination
├── cli.py                       # Command-line interface
├── legacy_test_phase1_standalone.py  # Moved legacy test
├── individual/                  # Individual phase test implementations
│   ├── __init__.py
│   ├── test_phase0.py          # Phase 0: Generate Scope of Work
│   ├── test_phase1.py          # Phase 1: Merge Measurement & Work Scope  
│   └── test_phase2.py          # Phase 2: Quantity Survey
├── integration/                 # Integration tests for phase combinations
│   ├── __init__.py
│   └── test_phase_pipeline.py  # Pipeline flow testing
├── configs/                     # Test configuration files
│   ├── default_single_phase.yaml
│   ├── fast_test.yaml
│   ├── comprehensive.yaml
│   ├── validation_comparison.yaml
│   └── model_comparison.yaml
└── fixtures/                    # Test data fixtures
    ├── sample_measurement.json
    ├── sample_demo.json
    └── sample_intake_form.json
```

## 🚀 Quick Start

### Running Individual Phase Tests

```bash
# Run Phase 1 with default settings
python -m tests.phases.cli single --phase 1

# Run Phase 0 with single model for speed
python -m tests.phases.cli single --phase 0 --models gpt4

# Run with custom validation mode
python -m tests.phases.cli single --phase 1 --validation-mode strict
```

### Running Pipeline Tests

```bash
# Run full pipeline (Phase 0 → 1 → 2)
python -m tests.phases.cli pipeline --phases 0 1 2

# Run partial pipeline with specific models
python -m tests.phases.cli pipeline --phases 1 2 --models gpt4 claude

# Continue even if phases fail
python -m tests.phases.cli pipeline --phases 0 1 2 --continue-on-failure
```

### Running Comparison Tests

```bash
# Compare different model combinations for Phase 1
python -m tests.phases.cli compare --phase 1 --compare-type models

# Compare validation modes
python -m tests.phases.cli compare --phase 1 --compare-type validation

# Custom model comparison
python -m tests.phases.cli compare --phase 2 --models gpt4 claude
```

### Running Predefined Scenarios

```bash
# Validation mode comparison scenario
python -m tests.phases.cli scenario validation_comparison

# Performance comparison for specific phase  
python -m tests.phases.cli scenario performance --phase 1

# Full pipeline scenario
python -m tests.phases.cli scenario full_pipeline
```

### Utility Commands

```bash
# List available test configurations
python -m tests.phases.cli list-configs

# List recent test results
python -m tests.phases.cli list-results --limit 5

# Verbose output
python -m tests.phases.cli single --phase 1 --verbose

# Quiet mode (minimal output)
python -m tests.phases.cli single --phase 1 --quiet
```

## 🏗️ Architecture Overview

### Core Components

#### 1. **PhaseTestBase** (`base.py`)
Abstract base class providing:
- Test configuration management
- Input data preparation and validation
- Result analysis and insights
- Output file management
- Common testing utilities

#### 2. **PhaseTestOrchestrator** (`orchestrator.py`)
Manages complex test scenarios:
- Single phase execution
- Multi-phase pipeline execution
- Comparison testing across configurations
- Test scenario orchestration
- Session management and result tracking

#### 3. **Individual Phase Tests** (`individual/`)
Specific implementations for each phase:
- **Phase0Test**: Generate Scope of Work testing
- **Phase1Test**: Merge Measurement & Work Scope testing
- **Phase2Test**: Quantity Survey testing

Each test handles:
- Phase-specific input data preparation
- Processor initialization and execution
- Result validation and analysis
- Performance metrics collection

#### 4. **Integration Tests** (`integration/`)
End-to-end testing:
- Pipeline flow validation
- Data compatibility between phases
- Error handling and recovery
- Performance characteristics

## 📊 Test Configuration System

### Configuration Files (`configs/`)

**default_single_phase.yaml**: Standard single phase testing
```yaml
phase_numbers: [1]
models: ["gpt4", "claude", "gemini"]
validation_mode: "balanced"
process_by_room: true
timeout_seconds: 300
```

**fast_test.yaml**: Quick testing with minimal resources
```yaml
phase_numbers: [1]
models: ["gpt4"]  # Single model for speed
process_by_room: false  # Batch processing
timeout_seconds: 120
```

**comprehensive.yaml**: Thorough testing with strict validation
```yaml
phase_numbers: [0, 1, 2]
models: ["gpt4", "claude", "gemini"]
validation_mode: "strict"
timeout_seconds: 600
```

### Custom Configuration

Create custom YAML files in `tests/phases/configs/` with:
- `phase_numbers`: List of phases to test
- `models`: AI models to use
- `validation_mode`: "strict", "balanced", or "lenient"
- `process_by_room`: Boolean for room-by-room processing
- `timeout_seconds`: Per-phase timeout
- `test_name`: Custom identifier
- `description`: Human-readable description

## 🧪 Test Types and Scenarios

### 1. **Single Phase Tests**
Test individual phases in isolation:
- Input data validation
- Processor functionality
- Output quality assessment
- Performance metrics
- Error handling

### 2. **Pipeline Tests**
Test phase sequences:
- Data flow between phases
- Error propagation
- Overall pipeline performance
- Intermediate result validation

### 3. **Comparison Tests**
Compare different configurations:
- **Model Comparison**: Different AI model combinations
- **Validation Comparison**: Strict vs. balanced vs. lenient validation
- **Performance Comparison**: Execution time and resource usage

### 4. **Integration Tests**
End-to-end validation:
- Full system functionality
- Data compatibility
- Error recovery
- Performance characteristics

## 📈 Result Analysis and Reporting

### Test Results Include:
- **Success/Failure Status**: Overall test outcome
- **Performance Metrics**: Execution time, model response rates
- **Quality Metrics**: Confidence scores, consensus levels
- **Validation Results**: Business logic validation outcomes
- **Error Information**: Detailed error messages and stack traces

### Result Files:
- **Individual Tests**: `test_outputs/phase{N}_{config}_{timestamp}.json`
- **Pipeline Sessions**: `test_outputs/sessions/session_{id}_{timestamp}.json`
- **Comparisons**: `test_outputs/comparisons/comparison_phase{N}_{timestamp}.json`
- **Scenarios**: `test_outputs/scenarios/scenario_{name}_{timestamp}.json`

### Analysis Features:
- Automatic best configuration identification
- Performance trend analysis
- Validation issue categorization
- Recommendation generation
- Historical comparison

## 🔧 Extensibility and Customization

### Adding New Phases

1. **Create Phase Test Class**:
```python
# tests/phases/individual/test_phase3.py
from ..base import PhaseTestBase, PhaseTestConfig, PhaseTestResult

class Phase3Test(PhaseTestBase):
    @property
    def phase_number(self) -> int:
        return 3
    
    @property 
    def phase_name(self) -> str:
        return "Market Research"
    
    async def prepare_test_data(self, test_config: PhaseTestConfig):
        # Implement data preparation
        pass
    
    async def execute_phase(self, input_data, test_config):
        # Implement phase execution
        pass
```

2. **Register with Orchestrator**:
```python
# In cli.py or test scripts
orchestrator.register_phase_test(Phase3Test())
```

### Custom Test Scenarios

```python
# Create custom scenarios
from tests.phases.orchestrator import PhaseTestScenario
from tests.phases.base import PhaseTestConfig

custom_scenario = PhaseTestScenario(
    name="custom_validation_test",
    description="Custom validation testing scenario",
    configs=[
        PhaseTestConfig(
            phase_numbers=[1, 2],
            models=["gpt4", "claude"],
            validation_mode="strict",
            test_name="strict_multi_phase"
        )
    ],
    tags=["custom", "validation"]
)

result = await orchestrator.run_test_scenario(custom_scenario)
```

### Custom Analysis

```python
# Extend PhaseTestBase for custom analysis
class CustomPhaseTest(Phase1Test):
    def analyze_results(self, result: PhaseTestResult):
        analysis = super().analyze_results(result)
        
        # Add custom analysis
        analysis['custom_metrics'] = {
            'complexity_score': self.calculate_complexity(result),
            'efficiency_rating': self.rate_efficiency(result)
        }
        
        return analysis
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **API Key Configuration**
```bash
# Check API keys are set
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
python -c "import os; print('Anthropic:', bool(os.getenv('ANTHROPIC_API_KEY')))"
python -c "import os; print('Google:', bool(os.getenv('GOOGLE_API_KEY')))"
```

#### 2. **Missing Dependencies**
```bash
# Install required packages
pip install -r requirements.txt
pip install pytest pyyaml
```

#### 3. **Phase 0 Input Data**
Ensure Phase 0 has proper input data:
- `measurement_data`: List of floors with room dimensions
- `demolition_scope_data`: Already completed demolition items
- `intake_form`: Project description and requirements

#### 4. **Pipeline Data Flow**
For pipeline tests, ensure each phase produces compatible output:
- Phase 0 → Phase 1: Structured room and work scope data
- Phase 1 → Phase 2: Validated measurements and work items
- Phase 2 → Phase 3: Quantities and specifications

### Debug Mode

```bash
# Run with verbose output for debugging
python -m tests.phases.cli single --phase 1 --verbose

# Check recent results for errors
python -m tests.phases.cli list-results --limit 3

# Use Python debugger
python -m pdb -m tests.phases.cli single --phase 1
```

### Performance Issues

```bash
# Use single model for faster testing
python -m tests.phases.cli single --phase 1 --models gpt4

# Disable room-by-room processing
python -m tests.phases.cli single --phase 1 --models gpt4 --config fast_test

# Reduce timeout for faster failure detection
python -m tests.phases.cli single --phase 1 --timeout 60
```

## 🔮 Future Enhancements

### Planned Features
1. **Parallel Phase Execution**: Run independent phases simultaneously
2. **Test Data Generation**: Automated generation of varied test scenarios
3. **Performance Benchmarking**: Automated performance regression testing
4. **Web Dashboard**: Browser-based test management and results visualization
5. **CI/CD Integration**: Automated testing in deployment pipelines
6. **Machine Learning Analysis**: Pattern recognition in test results

### Phase Expansion
- **Phase 3**: Market Research implementation
- **Phase 4**: Timeline & Disposal Calculation
- **Phase 5**: Final Estimate Completion
- **Phase 6**: JSON Formatting

### Advanced Testing
- **Load Testing**: High-volume concurrent test execution
- **Chaos Testing**: Fault injection and error recovery testing
- **A/B Testing**: Model performance comparison over time
- **Regression Testing**: Automated detection of performance degradation

## 📝 Contributing

### Adding New Tests
1. Create test class inheriting from `PhaseTestBase`
2. Implement required abstract methods
3. Add test configurations if needed
4. Register with orchestrator
5. Update documentation

### Best Practices
- Use descriptive test names and descriptions
- Include comprehensive error handling
- Validate input data thoroughly
- Provide meaningful result analysis
- Document any special requirements

### Code Style
- Follow existing patterns and naming conventions
- Use type hints for better code clarity
- Include docstrings for all public methods
- Handle exceptions gracefully
- Log important events and errors

---

For more information or assistance, refer to the individual module documentation or contact the development team.