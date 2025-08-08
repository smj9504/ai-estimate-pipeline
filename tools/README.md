# Tools Directory

This directory contains development and utility tools for the AI Estimate Pipeline.

## 📁 File Structure

### Development Tools
- **test_fixes.py** - Script for fixing and updating test cases
- **compare_test_results.py** - Compare and analyze test results across different runs
- **install_tracking.py** - Track and manage package installations and dependencies

## 🛠️ Tool Descriptions

### test_fixes.py
Utility for updating and fixing test cases after code changes.
```bash
python tools/test_fixes.py
```

### compare_test_results.py
Compare results from different test runs to identify improvements or regressions.
```bash
python tools/compare_test_results.py [result_file1] [result_file2]
```

### install_tracking.py
Track package installations and ensure all dependencies are properly managed.
```bash
python tools/install_tracking.py
```

## 📝 Usage Guidelines

These tools are meant for development and maintenance purposes:
- Run from the project root directory
- Ensure virtual environment is activated before use
- Check tool-specific documentation within each file for detailed options

## ⚠️ Caution

These are development tools and should not be used in production environments.