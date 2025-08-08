# Prompt Versioning System Guide

## Overview
This document describes the prompt versioning system implemented for the AI Estimate Pipeline project. The system allows for improved, enhanced, and specialized versions of prompts while maintaining backward compatibility.

## File Structure

```
prompts/
├── phase0_prompt.txt              # Basic Phase 0 prompt
├── phase0_prompt_improved.txt     # Enhanced Phase 0 prompt
├── phase1_prompt.txt              # Basic Phase 1 prompt  
├── phase1_prompt_improved.txt     # Enhanced Phase 1 prompt
├── phase2_prompt.txt              # Basic Phase 2 prompt
├── phase2_prompt_improved.txt     # Enhanced Phase 2 prompt
├── system_prompt.txt              # System-wide prompt template
└── [future phases...]
```

## Naming Convention

### Standard Format
- **Basic Version**: `phase{N}_prompt.txt` (default version)
- **Versioned**: `phase{N}_prompt_{version}.txt`

### Version Types
- **`improved`**: Enhanced version with better structure, clarity, and instructions
- **`v2`, `v3`, etc.**: Numbered versions for iterative improvements
- **`test`**: Experimental versions for testing
- **`specialized`**: Task-specific optimized versions
- **Custom names**: Project-specific versions (e.g., `phase1_prompt_dmv_specific.txt`)

## Version Management Usage

### Loading Prompts in Code
```python
from src.utils.prompt_manager import PromptManager

manager = PromptManager()

# Load basic version (default)
prompt = manager.load_prompt(phase_number=1)

# Load improved version
prompt = manager.load_prompt(phase_number=1, version='improved')

# Load with variables
prompt = manager.load_prompt_with_variables(
    phase_number=1,
    version='improved',
    variables={'project_id': 'PRJ001', 'location': 'DMV area'}
)
```

### Automatic Fallback System
If a requested version doesn't exist, the system automatically falls back to the default version:
```
Request: phase1_prompt_improved.txt
├── File exists? Yes → Load improved version
└── File exists? No → Load phase1_prompt.txt (default)
```

## Current Improved Versions

### Phase 0: Generate Scope of Work Data (Improved)
**Enhancements**:
- ✅ Enhanced data validation and quality control
- ✅ Advanced jobsite information processing  
- ✅ Comprehensive room data integration
- ✅ Professional error handling and documentation
- ✅ Hierarchical material/work scope application logic

### Phase 1: Merge Measurement & Work Scope (Improved)  
**Enhancements**:
- ✅ Detailed Remove & Replace strategy implementation
- ✅ Advanced calculation rules with high ceiling premiums
- ✅ Comprehensive validation checklist framework
- ✅ Professional estimation standards and best practices
- ✅ Enhanced output structure with task classification

### Phase 2: Quantity Survey (Improved)
**Enhancements**:
- ✅ Advanced technical specifications system
- ✅ Comprehensive material-specific waste calculations
- ✅ Professional-grade paint and coating systems
- ✅ Automated protection requirements calculation
- ✅ Enhanced quality control and validation framework

## Key Improvements in Enhanced Versions

### 1. Structure & Organization
- **Clear sections** with headings and subsections
- **Context sections** explaining phase purpose and importance
- **Step-by-step instructions** with detailed explanations
- **Professional formatting** with consistent styling

### 2. Technical Accuracy
- **Industry-standard terminology** and specifications
- **Detailed calculation methods** with examples
- **Quality control checklists** for validation
- **Error handling procedures** and fallback logic

### 3. Enhanced Instructions
- **Comprehensive examples** for complex scenarios
- **Edge case handling** with specific guidance
- **Cross-referencing** between related tasks
- **Validation requirements** at each step

### 4. Professional Standards
- **DMV area expertise** with local considerations
- **15-20 years experience level** perspective
- **Industry best practices** integration
- **Quality assurance measures** throughout

## Implementation Status

| Phase | Basic Version | Improved Version | Status |
|-------|---------------|------------------|---------|
| 0     | ✅ Available  | ✅ Complete     | Active |
| 1     | ✅ Available  | ✅ Complete     | Active |  
| 2     | ✅ Available  | ✅ Complete     | Active |
| 3-6   | 🚧 Future     | 🚧 Future       | Planned |

## Creating New Versions

### Step 1: Analyze Current Version
- Review existing prompt effectiveness
- Identify areas for improvement
- Gather feedback from test results

### Step 2: Design Improvements  
- Enhanced structure and organization
- Better instruction clarity
- Additional validation requirements
- Professional terminology updates

### Step 3: Create New Version
```python
# Using PromptManager
manager = PromptManager()
improved_prompt_text = """..."""  # Your improved prompt

# Save as new version
file_path = manager.save_prompt_version(
    phase_number=1,
    prompt_text=improved_prompt_text,
    version='improved_v2'
)
```

### Step 4: Test & Validate
- Run phase tests with new prompt version
- Compare results with basic version
- Validate improvement metrics
- Document changes and benefits

## Version Comparison & Testing

### Available Tools
```python
# List all available versions
available = manager.list_available_prompts()
print(available)  # {0: ['default', 'improved'], 1: ['default', 'improved'], ...}

# Get prompt information
info = manager.get_prompt_info(phase_number=1, version='improved')
print(info['character_count'], info['validation'])

# Validate prompt
validation = manager.validate_prompt(phase_number=1, version='improved')
print(validation['valid'], validation['errors'])
```

### A/B Testing Process
1. **Run tests with basic version** - Record metrics
2. **Run tests with improved version** - Compare results  
3. **Analyze performance differences** - Quality, speed, accuracy
4. **Document improvements** - Quantifiable benefits

## Best Practices

### 1. Backward Compatibility
- Always maintain basic versions as fallback
- Test new versions thoroughly before deployment
- Document breaking changes if any

### 2. Version Naming
- Use descriptive version names
- Follow consistent naming conventions
- Include creation date for tracking

### 3. Documentation
- Document changes made in each version
- Record performance improvements
- Maintain change log for major updates

### 4. Quality Control
- Validate all prompts before production use
- Test with multiple scenarios
- Monitor performance metrics continuously

## Troubleshooting

### Common Issues

**"버전 {version} 프롬프트를 찾을 수 없습니다"**
- Solution: Check file exists with correct naming convention
- Fallback: System automatically tries default version

**Prompt validation fails**
- Check required keywords for each phase
- Verify minimum length requirements (>100 characters)
- Ensure proper structure and formatting

**Performance regression with new version**
- Compare metrics with previous version
- Review and revise improvements
- Consider rolling back to stable version

### Debug Tools
```python
# Clear cache if needed
manager.clear_cache()

# Get detailed prompt info
info = manager.get_prompt_info(1, 'improved')

# Validate prompt structure
validation = manager.validate_prompt(1, 'improved') 
```

## Future Enhancements

### Planned Features
- **Phase 3-6 improved versions** when base phases implemented
- **Domain-specific versions** (DMV area, commercial, residential)
- **Performance optimization** versions for speed vs quality tradeoffs
- **Multi-language support** versions

### Version Strategy
- Regular review and updates quarterly
- Community feedback integration
- Performance-driven improvements
- Industry standard updates

---

## Quick Reference

**Load improved prompt**: `manager.load_prompt(phase=1, version='improved')`
**Create new version**: `manager.save_prompt_version(phase=1, text=prompt, version='v2')`
**List versions**: `manager.list_available_prompts()`
**Validate prompt**: `manager.validate_prompt(phase=1, version='improved')`

---

*This versioning system ensures continuous improvement of AI prompts while maintaining system stability and backward compatibility.*