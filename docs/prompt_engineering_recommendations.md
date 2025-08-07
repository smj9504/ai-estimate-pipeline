# Prompt Engineering Recommendations for Multi-Model AI Estimation System

## Executive Summary

This document provides specific recommendations to improve prompt engineering for consistent structured output across GPT-4, Claude, and Gemini models in the construction estimation pipeline.

## 1. Structured Output Formatting (JSON Schema)

### Current Issue
The prompts lack explicit JSON schema definitions, leading to unpredictable output formats.

### Solution: Explicit JSON Schema with Examples

```json
{
  "instruction": "Return ONLY a valid JSON object with this exact structure",
  "schema": {
    "rooms": [
      {
        "room_name": "string - REQUIRED",
        "tasks": [
          {
            "task_name": "string",
            "quantity": "number",
            "unit": "string enum: sqft|lf|item"
          }
        ]
      }
    ]
  }
}
```

### Implementation Strategy
1. **Define schema at the beginning** of the prompt
2. **Provide a complete example** of expected output
3. **Use JSON code blocks** with proper formatting
4. **Explicitly state**: "Return ONLY the JSON object, no explanatory text"

## 2. Clear Instruction Patterns for Multi-Model Compatibility

### Model-Specific Adaptations

#### GPT-4 Optimizations
```python
gpt4_additions = {
    "response_format": {"type": "json_object"},  # Use API parameter
    "system_message": "You are a JSON-only response assistant",
    "instruction_style": "step-by-step numbered lists"
}
```

#### Claude Optimizations
```python
claude_additions = {
    "xml_tags": True,  # Claude responds well to XML structure
    "thinking_tags": "<analysis>...</analysis>",
    "output_tags": "<json_output>...</json_output>"
}
```

#### Gemini Optimizations
```python
gemini_additions = {
    "structured_prompting": True,
    "section_headers": "## INPUT\n## PROCESSING\n## OUTPUT",
    "explicit_typing": "Ensure all numbers are numeric, not strings"
}
```

### Universal Patterns That Work Across All Models

```text
1. Role Definition: "You are a [specific role] that outputs JSON"
2. Task Clarity: "Your task is to [specific action] and return JSON"
3. Schema First: Show the expected structure before instructions
4. Validation Rules: List what makes output valid/invalid
5. Examples: Provide input→output examples
6. Constraints: "MUST", "NEVER", "ALWAYS" for critical rules
```

## 3. Better Data Extraction Instructions

### Current Issue: Ambiguous Extraction Rules

### Improved Pattern:

```text
## Data Extraction Rules

EXTRACT the following from each room object:
- room_name: Use the EXACT value from room.name field (NEVER use placeholders)
- measurements: Extract numeric values from room.measurements object
  - floor_area_sqft: Parse as float
  - wall_area_sqft: Parse as float
  - ceiling_area_sqft: Parse as float
  - floor_perimeter_lf: Parse as float
  - height: Parse as float (check if > 9 for premium)

CALCULATE work items using these formulas:
- If work_scope[material] = "Remove & Replace":
  - removal_quantity = measurements[area] - demo_scope[material]
  - installation_quantity = measurements[area]

VALIDATE extracted data:
- All room names must be non-empty strings
- All quantities must be positive numbers
- Units must match: sqft for areas, lf for perimeters
```

## 4. Removing Ambiguous Elements

### Problems to Remove:

1. **Checkboxes in Prompts**
   - Remove: ✅ Has the logic been applied?
   - Replace with: "validation_status": {"logic_applied": true/false}

2. **Placeholder Text**
   - Remove: [PASTE FULL JSON DATA HERE]
   - Replace with: Direct data injection in API call

3. **Ambiguous Field Names**
   - Remove: demo_scope(already demo'd)
   - Replace with: demo_scope or already_demolished

4. **Mixed Formatting**
   - Remove: Markdown headers mixed with instructions
   - Replace with: Consistent structure throughout

## 5. Example Output Format

### Provide Complete, Realistic Examples:

```json
{
  "room_name": "Master Bedroom",
  "room_id": "room_001",
  "tasks": [
    {
      "task_id": "task_001",
      "task_name": "Remove existing carpet",
      "task_type": "removal",
      "quantity": 150.0,
      "unit": "sqft",
      "notes": "200 sqft total - 50 sqft already removed"
    },
    {
      "task_id": "task_002",
      "task_name": "Install luxury vinyl plank flooring",
      "task_type": "installation",
      "quantity": 200.0,
      "unit": "sqft",
      "notes": "Full room area"
    }
  ]
}
```

## 6. Implementation Code Updates

### Update model_interface.py to use structured prompts:

```python
class PromptOptimizer:
    def optimize_for_model(self, base_prompt: str, model_name: str, data: dict) -> str:
        """Optimize prompt for specific model"""
        
        # Add JSON schema definition
        schema = self.get_output_schema()
        
        # Model-specific optimizations
        if model_name == "gpt4":
            return self.format_for_gpt4(base_prompt, schema, data)
        elif model_name == "claude":
            return self.format_for_claude(base_prompt, schema, data)
        elif model_name == "gemini":
            return self.format_for_gemini(base_prompt, schema, data)
    
    def format_for_gpt4(self, prompt: str, schema: dict, data: dict) -> str:
        return f"""
{prompt}

OUTPUT FORMAT (return ONLY this JSON structure):
```json
{json.dumps(schema, indent=2)}
```

INPUT DATA:
{json.dumps(data, indent=2)}
"""
    
    def format_for_claude(self, prompt: str, schema: dict, data: dict) -> str:
        return f"""
{prompt}

<output_format>
Return a JSON object matching this schema:
{json.dumps(schema, indent=2)}
</output_format>

<input_data>
{json.dumps(data, indent=2)}
</input_data>

<instructions>
Analyze the input data and return ONLY a valid JSON object matching the schema above.
</instructions>
"""
```

## 7. Validation and Testing

### Add Pre/Post Processing Validation:

```python
class OutputValidator:
    def validate_llm_response(self, response: str, expected_schema: dict) -> dict:
        """Validate and clean LLM response"""
        
        # Clean response
        cleaned = self.extract_json(response)
        
        # Validate against schema
        errors = self.validate_schema(cleaned, expected_schema)
        
        # Fix common issues
        if "**" in str(cleaned):
            cleaned = self.fix_placeholder_room_names(cleaned)
        
        # Ensure numeric types
        cleaned = self.ensure_numeric_types(cleaned)
        
        return {
            "valid": len(errors) == 0,
            "data": cleaned,
            "errors": errors
        }
```

## 8. Prompt Testing Framework

### Create test cases for each model:

```python
test_cases = [
    {
        "name": "basic_remove_replace",
        "input": {...},
        "expected_output_structure": {...},
        "validation_rules": [...]
    },
    {
        "name": "high_ceiling_premium",
        "input": {...},
        "expected_output_structure": {...},
        "validation_rules": [...]
    }
]

async def test_prompt_consistency():
    for model in ["gpt4", "claude", "gemini"]:
        for test_case in test_cases:
            result = await run_model(model, test_case["input"])
            assert validate_structure(result, test_case["expected_output_structure"])
```

## 9. Quick Implementation Checklist

- [ ] Replace phase1_prompt.txt with structured version
- [ ] Add JSON schema to all prompts
- [ ] Remove validation checkboxes
- [ ] Add model-specific prompt optimizations
- [ ] Implement output validation layer
- [ ] Test with sample data across all three models
- [ ] Monitor and log parsing success rates
- [ ] Create fallback patterns for failed parsing

## 10. Recommended Prompt Structure Template

```text
[ROLE DEFINITION]
You are a [specific role] specialized in [domain].

[OUTPUT REQUIREMENT]
You MUST return ONLY a valid JSON object with no additional text.

[SCHEMA DEFINITION]
```json
{
  // Complete schema here
}
```

[BUSINESS RULES]
1. Specific rule with formula
2. Another rule with example

[DATA EXTRACTION INSTRUCTIONS]
- Field name: How to extract/calculate
- Another field: Extraction method

[VALIDATION REQUIREMENTS]
- What makes output valid
- What to check before returning

[EXAMPLE]
Input: [sample input]
Output: [expected JSON output]

[INPUT DATA MARKER]
// Actual data injected here
```

## Conclusion

These improvements will significantly increase the consistency and reliability of structured outputs across all three AI models. The key is being explicit about format requirements, providing clear schemas, and using model-specific optimizations while maintaining a universal structure that all models can understand.