#!/bin/bash
# Pre-commit hook for AI Estimate Pipeline

echo "Running pre-commit checks..."

# 1. Check Python syntax
echo "Checking Python syntax..."
conda run -n ai-estimate python -m py_compile src/**/*.py
if [ $? -ne 0 ]; then
    echo "❌ Python syntax errors found"
    exit 1
fi

# 2. Run Black formatter (check mode)
echo "Checking code formatting..."
conda run -n ai-estimate black --check src/ tests/ 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Code formatting issues found. Run: black src/ tests/"
    # Don't fail, just warn
fi

# 3. Run flake8 linter
echo "Running linter..."
conda run -n ai-estimate flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Linting issues found"
    # Don't fail, just warn
fi

# 4. Check for API keys in code
echo "Checking for exposed API keys..."
grep -r "sk-" src/ 2>/dev/null
if [ $? -eq 0 ]; then
    echo "❌ Potential API key found in code!"
    exit 1
fi

grep -r "OPENAI_API_KEY\|ANTHROPIC_API_KEY\|GOOGLE_API_KEY" src/ --include="*.py" | grep -v "os.getenv\|os.environ" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "❌ Hard-coded API key reference found!"
    exit 1
fi

# 5. Check for large files
echo "Checking file sizes..."
find . -type f -size +10M | grep -v ".git" | grep -v "venv" | grep -v "__pycache__"
if [ $? -eq 0 ]; then
    echo "⚠️ Large files detected (>10MB)"
fi

# 6. Verify critical imports
echo "Verifying critical imports..."
conda run -n ai-estimate python -c "
try:
    from src.models.model_interface import ModelOrchestrator
    from src.processors.result_merger import ResultMerger
    from src.phases.phase_manager import PhaseManager
    print('✅ Critical imports OK')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "✅ Pre-commit checks completed"