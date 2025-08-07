#!/bin/bash
# Test runner hook for AI Estimate Pipeline

echo "========================================="
echo "AI Estimate Pipeline - Test Suite"
echo "========================================="

# Activate conda environment
source activate ai-estimate 2>/dev/null || conda activate ai-estimate

# Check environment
echo "Environment: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run different test categories
echo "1. Running unit tests..."
python -m pytest tests/test_model_interface.py -v --tb=short
UNIT_RESULT=$?

echo ""
echo "2. Running integration tests..."
python -m pytest tests/ -v --tb=short -k "not test_model_interface"
INTEGRATION_RESULT=$?

echo ""
echo "3. Running validation checks..."
python -c "
from src.validators.estimation_validator import ComprehensiveValidator
from src.utils.config_loader import ConfigLoader
config = ConfigLoader().load_config()
validator = ComprehensiveValidator(config)
print('✅ Validator initialization OK')
"
VALIDATION_RESULT=$?

echo ""
echo "4. Running API connection tests..."
python -c "
import os
apis = {
    'OpenAI': os.getenv('OPENAI_API_KEY'),
    'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'Google': os.getenv('GOOGLE_API_KEY')
}
for name, key in apis.items():
    status = '✅' if key else '❌'
    print(f'{status} {name} API key {"configured" if key else "missing"}')
"

echo ""
echo "========================================="
echo "Test Results Summary:"
echo "========================================="

if [ $UNIT_RESULT -eq 0 ]; then
    echo "✅ Unit tests: PASSED"
else
    echo "❌ Unit tests: FAILED"
fi

if [ $INTEGRATION_RESULT -eq 0 ]; then
    echo "✅ Integration tests: PASSED"
else
    echo "❌ Integration tests: FAILED"
fi

if [ $VALIDATION_RESULT -eq 0 ]; then
    echo "✅ Validation checks: PASSED"
else
    echo "❌ Validation checks: FAILED"
fi

# Overall result
if [ $UNIT_RESULT -eq 0 ] && [ $INTEGRATION_RESULT -eq 0 ] && [ $VALIDATION_RESULT -eq 0 ]; then
    echo ""
    echo "🎉 All tests passed!"
    exit 0
else
    echo ""
    echo "⚠️ Some tests failed. Please review the output above."
    exit 1
fi