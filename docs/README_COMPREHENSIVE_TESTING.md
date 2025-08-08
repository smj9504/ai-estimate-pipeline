# Comprehensive AI Model Combination Testing Framework

This document describes the comprehensive testing framework for AI model combinations in the estimation pipeline project.

## Overview

The testing framework has been completely refactored to:
- ✅ Use actual test data instead of dummy fixtures
- ✅ Support comprehensive model combination testing
- ✅ Provide detailed performance analysis and reporting
- ✅ Generate comparison reports in multiple formats
- ✅ Enable systematic testing of different configurations

## Architecture

### Core Components

#### 1. Test Data Loader (`tests/utils/test_data_loader.py`)
**Purpose**: Centralized access to actual test data files
- Loads demo data from `test_data/sample_demo.json`
- Loads measurement data from `test_data/sample_measurement.json`  
- Loads intake form from `test_data/sample_intake_form.txt`
- Combines data into ProjectData format for end-to-end testing
- Replaces old dummy fixtures with real project data

**Usage**:
```python
from tests.utils.test_data_loader import get_test_data_loader

loader = get_test_data_loader()
combined_data = loader.create_combined_project_data()
test_dataset = loader.get_test_dataset()
```

#### 2. Model Combination Tester (`tests/model_combinations/combination_tester.py`)
**Purpose**: Execute AI model combination tests
- Supports single and multi-model testing
- Parallel and sequential processing modes
- Comprehensive result tracking with metrics
- Automatic output saving and validation

**Features**:
- Real model API integration or mocking support
- Timeout and retry handling
- Performance metrics collection
- Quality scoring and validation
- Error handling and reporting

#### 3. Test Matrix Generator (`tests/model_combinations/test_matrix.py`)
**Purpose**: Generate all possible test configurations
- **Model Types**: GPT-4, Claude, Gemini
- **Validation Modes**: Strict, Balanced, Lenient
- **Processing Modes**: Parallel, Sequential, Single Model
- **Combination Types**: Single model, pairs, all models

**Test Matrix Options**:
```python
# All possible combinations (21 configurations)
comprehensive_configs = matrix.generate_all_configurations()

# Essential subset (7 configurations)  
essential_configs = matrix.generate_essential_configurations()

# Performance comparison (10 configurations)
performance_configs = matrix.generate_performance_comparison_matrix()
```

#### 4. Performance Analyzer (`tests/model_combinations/performance_analyzer.py`)
**Purpose**: Analyze test results and calculate performance metrics
- Success rates and reliability metrics
- Quality scores and confidence levels
- Execution time and cost analysis
- Statistical analysis with standard deviations
- Model comparison and optimization recommendations

#### 5. Comparison Reporter (`tests/model_combinations/comparison_reporter.py`)
**Purpose**: Generate comprehensive reports in multiple formats
- JSON reports with detailed metrics
- HTML reports with visualizations
- Excel exports for external analysis
- Executive summaries and recommendations

## Test Types

### 1. Single Model Tests
Test individual AI models in isolation:
```python
single_configs = [
    TestConfiguration(models=[ModelType.GPT4], validation_mode=ValidationMode.BALANCED),
    TestConfiguration(models=[ModelType.CLAUDE], validation_mode=ValidationMode.BALANCED), 
    TestConfiguration(models=[ModelType.GEMINI], validation_mode=ValidationMode.BALANCED)
]
```

### 2. Model Combination Tests  
Test pairs and groups of models:
```python
pair_configs = [
    TestConfiguration(models=[ModelType.GPT4, ModelType.CLAUDE]),
    TestConfiguration(models=[ModelType.GPT4, ModelType.GEMINI]),
    TestConfiguration(models=[ModelType.CLAUDE, ModelType.GEMINI])
]

all_models_config = TestConfiguration(
    models=[ModelType.GPT4, ModelType.CLAUDE, ModelType.GEMINI]
)
```

### 3. Validation Mode Comparison
Compare different validation strictness levels:
```python
validation_configs = [
    TestConfiguration(models=[ModelType.GPT4], validation_mode=ValidationMode.STRICT),
    TestConfiguration(models=[ModelType.GPT4], validation_mode=ValidationMode.BALANCED),
    TestConfiguration(models=[ModelType.GPT4], validation_mode=ValidationMode.LENIENT)
]
```

## Usage

### Command Line Interface

Run tests using the test runner:
```bash
# Run essential configurations (recommended for regular testing)
python -m tests.model_combinations.test_runner --test-type essential

# Run comprehensive tests (all combinations)
python -m tests.model_combinations.test_runner --test-type comprehensive

# Run performance comparison
python -m tests.model_combinations.test_runner --test-type performance

# Quick test (5 configurations)
python -m tests.model_combinations.test_runner --test-type quick

# Custom configuration
python -m tests.model_combinations.test_runner \
    --models gpt4 claude \
    --validation-mode balanced \
    --max-concurrent 2 \
    --output-dir custom_results
```

### Programmatic Usage

```python
import asyncio
from tests.model_combinations.test_runner import ModelCombinationTestRunner

async def run_model_tests():
    # Initialize runner
    runner = ModelCombinationTestRunner(
        config_file="tests/model_combinations/config_template.yaml",
        output_directory="test_results"
    )
    
    # Run tests and get analysis
    analysis = await runner.run_and_analyze("essential")
    return analysis

# Run the tests
results = asyncio.run(run_model_tests())
```

### Pytest Integration

```python
# Run comprehensive test suite
pytest tests/test_comprehensive_model_combinations.py -v

# Run specific test categories
pytest tests/test_comprehensive_model_combinations.py::TestModelCombinations -v

# Run integration tests
pytest tests/test_comprehensive_model_combinations.py::TestIntegration -v
```

## Configuration

### Configuration File (`config_template.yaml`)
```yaml
models:
  - "gpt4"
  - "claude" 
  - "gemini"

validation_modes:
  - "balanced"

processing_modes:
  - "parallel"

include_single_model: true
include_multi_model: true
max_concurrent_tests: 3
timeout_seconds: 300
retry_attempts: 2
save_outputs: true
```

## Output Structure

### Directory Layout
```
test_outputs/
├── combinations/           # Individual test results
│   ├── single_gpt4_balanced_parallel/
│   │   ├── result.json
│   │   ├── model_responses.json
│   │   └── merged_estimate.json
│   ├── pair_gpt4_claude_balanced_parallel/
│   └── ...
├── reports/               # Comparison reports
│   ├── model_combination_report_20250808_143022.json
│   ├── model_combination_report_20250808_143022.html
│   └── model_combination_analysis_20250808_143022.xlsx
└── comprehensive_tests/   # Framework test outputs
```

### Report Contents

#### JSON Report
- Executive summary with key metrics
- Detailed performance analysis
- Model comparison data
- Top performers by different criteria
- Comprehensive recommendations

#### HTML Report  
- Visual dashboard with charts
- Interactive result tables
- Performance highlights
- Executive summary
- Testing metadata

#### Excel Export
- Summary sheet with key metrics
- Detailed results with all test data
- Performance metrics by configuration
- Recommendations and insights

## Performance Metrics

### Quality Metrics
- **Overall Confidence**: Average confidence across models
- **Consensus Level**: Agreement between models
- **Validation Score**: Business logic validation results
- **Quality Score**: Composite quality metric (0-1)

### Performance Metrics
- **Execution Time**: Average time per test
- **Success Rate**: Percentage of successful tests
- **Model Response Rate**: Models that responded successfully
- **Cost Efficiency**: Quality score per dollar spent

### Reliability Metrics  
- **Error Rate**: Percentage of failed tests
- **Model Availability**: Individual model success rates
- **Standard Deviations**: Consistency measurements

## Best Practices

### 1. Test Selection
- **Daily Testing**: Use "essential" test type (7 configurations, ~10 minutes)
- **Weekly Testing**: Use "performance" test type (10 configurations, ~15 minutes)
- **Release Testing**: Use "comprehensive" test type (21 configurations, ~30 minutes)
- **Debug Testing**: Use "quick" test type (5 configurations, ~5 minutes)

### 2. Resource Management
- Set `max_concurrent_tests` based on API rate limits
- Use appropriate `timeout_seconds` for your models
- Enable `save_outputs` for debugging and analysis

### 3. Analysis
- Review HTML reports for quick insights
- Use Excel exports for detailed analysis
- Check console output for immediate feedback
- Monitor trends over time using consistent configurations

### 4. Error Handling
- Tests continue even if individual configurations fail
- Failed tests are reported with error details
- Partial results are still analyzed and reported
- Retry logic handles transient failures

## Migration from Old Tests

### Removed Components
- ✅ `tests/phases/fixtures/` - Dummy JSON fixtures
- ✅ Hardcoded sample data in test files
- ✅ Manual fixture loading in individual tests

### Updated Components
- ✅ `tests/phases/base.py` - Now uses centralized test data loader
- ✅ `tests/test_model_interface.py` - Uses actual project data
- ✅ All phase tests - Load real data instead of dummy fixtures

### Backward Compatibility
- Old `load_fixture()` method still works
- Automatic fallback to real test data
- Existing test structure preserved
- New functionality added alongside existing tests

## Troubleshooting

### Common Issues

1. **Missing Test Data Files**
   ```
   FileNotFoundError: Test data directory not found
   ```
   - Ensure `test_data/` directory exists in project root
   - Verify all required files are present

2. **API Key Issues**
   ```
   No models available for testing
   ```
   - Check API keys in `.env` file
   - Verify model interface configuration

3. **Memory Issues with Large Tests**
   - Reduce `max_concurrent_tests`
   - Use "essential" or "quick" test types
   - Enable output saving and clear cache

4. **Test Timeout Issues**
   - Increase `timeout_seconds` in configuration
   - Check network connectivity
   - Verify model API availability

### Debug Mode
```bash
# Enable verbose logging
python -m tests.model_combinations.test_runner --test-type quick --verbose

# Check individual components
python -c "from tests.utils.test_data_loader import get_test_data_loader; print(get_test_data_loader().get_test_dataset())"
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: AI Model Testing
on: [push, pull_request]
jobs:
  test-models:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run essential model tests
      run: python -m tests.model_combinations.test_runner --test-type essential
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
    - name: Upload test reports
      uses: actions/upload-artifact@v2
      with:
        name: model-test-reports
        path: test_outputs/reports/
```

## Future Enhancements

### Planned Features
1. **Real-time Monitoring**: Live dashboard for test execution
2. **Historical Tracking**: Trend analysis across test runs
3. **Auto-scaling**: Dynamic test parallelization
4. **Alert System**: Notifications for performance degradation
5. **A/B Testing**: Systematic model performance comparison
6. **Custom Metrics**: Domain-specific quality measures

### Extensibility
- Add new model types by extending `ModelType` enum
- Create custom validation modes in `ValidationMode`
- Implement specialized analyzers for domain-specific metrics
- Add new report formats by extending `ComparisonReporter`

This comprehensive testing framework provides the foundation for systematic AI model evaluation and continuous improvement of the estimation pipeline.