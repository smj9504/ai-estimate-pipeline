"""
Prompt Optimizer for Multi-Model AI System
Optimizes prompts for consistent JSON output across GPT-4, Claude, and Gemini
"""
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime


class PromptOptimizer:
    """
    Optimizes prompts for specific AI models to ensure consistent JSON output
    """
    
    def __init__(self):
        self.output_schemas = {
            'phase1': self._get_phase1_schema(),
            'phase2': self._get_phase2_schema()
        }
    
    def optimize_prompt(self, 
                       base_prompt: str, 
                       model_name: str, 
                       phase: int,
                       input_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Optimize prompt for specific model and return model-specific parameters
        
        Args:
            base_prompt: Original prompt text
            model_name: One of 'gpt4', 'claude', 'gemini'
            phase: Phase number (1, 2, etc.)
            input_data: The data to be processed
        
        Returns:
            Tuple of (optimized_prompt, model_parameters)
        """
        # Get the appropriate schema
        schema = self.output_schemas.get(f'phase{phase}', {})
        
        # Clean the base prompt
        cleaned_prompt = self._clean_base_prompt(base_prompt)
        
        # Model-specific optimization
        if model_name == 'gpt4':
            return self._optimize_for_gpt4(cleaned_prompt, schema, input_data)
        elif model_name == 'claude':
            return self._optimize_for_claude(cleaned_prompt, schema, input_data)
        elif model_name == 'gemini':
            return self._optimize_for_gemini(cleaned_prompt, schema, input_data)
        else:
            # Default fallback
            return self._optimize_default(cleaned_prompt, schema, input_data)
    
    def _clean_base_prompt(self, prompt: str) -> str:
        """Remove problematic elements from base prompt"""
        # Remove checkbox validation items
        prompt = re.sub(r'✅[^\n]*\n', '', prompt)
        
        # Remove placeholder text
        prompt = prompt.replace('[PASTE FULL JSON DATA HERE]', '')
        prompt = prompt.replace('[JSON DATA BELOW]', '')
        
        # Clean up demo_scope field names
        prompt = prompt.replace("demo_scope(already demo'd)", 'demo_scope')
        prompt = prompt.replace("demo_scope(already demo\\'d)", 'demo_scope')
        
        return prompt.strip()
    
    def _optimize_for_gpt4(self, 
                          base_prompt: str, 
                          schema: Dict[str, Any],
                          input_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Optimize for GPT-4 with JSON mode"""
        
        optimized_prompt = f"""You are a JSON-only construction estimation assistant.

{base_prompt}

CRITICAL REQUIREMENT: Return ONLY a valid JSON object with no additional text or explanation.

REQUIRED JSON STRUCTURE:
```json
{json.dumps(schema, indent=2)}
```

VALIDATION RULES:
1. All room names must be actual strings from input data (never "**" or placeholders)
2. All quantities must be numeric values (not strings)
3. Remove & Replace logic: removal_qty = total_area - demo_scope_amount
4. Installation always uses full area regardless of demo_scope

INPUT DATA:
{json.dumps(input_data, indent=2)}

OUTPUT: JSON object matching the schema above."""
        
        # GPT-4 specific parameters
        model_params = {
            'response_format': {'type': 'json_object'},
            'temperature': 0.3,  # Lower temperature for more consistent output
            'max_tokens': 4000
        }
        
        return optimized_prompt, model_params
    
    def _optimize_for_claude(self, 
                           base_prompt: str,
                           schema: Dict[str, Any],
                           input_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Optimize for Claude with XML tags for structure"""
        
        optimized_prompt = f"""<role>
Senior Reconstruction Estimating Specialist generating JSON-formatted estimates
</role>

<instructions>
{base_prompt}
</instructions>

<output_requirement>
You MUST return ONLY a valid JSON object enclosed in <json_output> tags.
Do not include any text outside the JSON structure.
</output_requirement>

<json_schema>
{json.dumps(schema, indent=2)}
</json_schema>

<validation_rules>
- Room names: Extract exact names from input, never use "**" or generic placeholders
- Quantities: Must be numeric (float or int), not strings
- Remove & Replace: removal = total_area - demo_scope, installation = total_area
- High ceiling: Apply 15% premium when height > 9 feet
</validation_rules>

<input_data>
{json.dumps(input_data, indent=2)}
</input_data>

<json_output>
// Return your JSON response here
</json_output>"""
        
        model_params = {
            'temperature': 0.3,
            'max_tokens': 4000
        }
        
        return optimized_prompt, model_params
    
    def _optimize_for_gemini(self,
                           base_prompt: str,
                           schema: Dict[str, Any],
                           input_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Optimize for Gemini with structured sections"""
        
        optimized_prompt = f"""## ROLE
Construction Estimation Specialist producing JSON-formatted outputs

## TASK
{base_prompt}

## OUTPUT FORMAT
Return ONLY a valid JSON object with this exact structure:

```json
{json.dumps(schema, indent=2)}
```

## PROCESSING RULES
1. **Room Name Extraction**: Use exact room names from input data
2. **Numeric Values**: Ensure all quantities are numbers (not strings)
3. **Remove & Replace Calculation**:
   - Removal quantity = measurement - demo_scope
   - Installation quantity = full measurement
4. **High Ceiling Premium**: Add 15% for rooms with height > 9 feet

## INPUT DATA
```json
{json.dumps(input_data, indent=2)}
```

## REQUIRED OUTPUT
Generate the JSON object following the schema above. Do not include any explanatory text."""
        
        model_params = {
            'temperature': 0.3,
            'candidate_count': 1,
            'max_output_tokens': 4000
        }
        
        return optimized_prompt, model_params
    
    def _optimize_default(self,
                        base_prompt: str,
                        schema: Dict[str, Any],
                        input_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Default optimization for unknown models"""
        
        optimized_prompt = f"""{base_prompt}

OUTPUT REQUIREMENT: Return ONLY a valid JSON object.

EXPECTED JSON STRUCTURE:
{json.dumps(schema, indent=2)}

INPUT DATA:
{json.dumps(input_data, indent=2)}"""
        
        return optimized_prompt, {}
    
    def _get_phase1_schema(self) -> Dict[str, Any]:
        """Get Phase 1 output schema"""
        return {
            "estimate_type": "phase1_reconstruction",
            "processing_timestamp": "ISO-8601 timestamp",
            "rooms": [
                {
                    "room_name": "string - exact room name",
                    "room_id": "string - unique identifier",
                    "tasks": [
                        {
                            "task_id": "string",
                            "task_name": "string",
                            "task_type": "removal|installation|protection|detach|reset",
                            "quantity": "number",
                            "unit": "sqft|lf|item|hour",
                            "notes": "string"
                        }
                    ],
                    "room_totals": {
                        "total_tasks": "number",
                        "total_removal_tasks": "number",
                        "total_installation_tasks": "number"
                    }
                }
            ],
            "summary": {
                "total_rooms": "number",
                "total_tasks": "number",
                "validation_status": {
                    "remove_replace_logic_applied": "boolean",
                    "measurements_used": "boolean",
                    "special_tasks_included": "boolean"
                }
            }
        }
    
    def _get_phase2_schema(self) -> Dict[str, Any]:
        """Get Phase 2 output schema"""
        return {
            "phase": 2,
            "phase_name": "Quantity Survey",
            "rooms": [
                {
                    "room_name": "string",
                    "specifications": {
                        "drywall": "object",
                        "paint": "object",
                        "flooring": "object"
                    },
                    "quantities": {
                        "task_name": {
                            "quantity": "number",
                            "unit": "string",
                            "waste_factor": "number",
                            "total_with_waste": "number"
                        }
                    },
                    "protection_requirements": ["list of items"]
                }
            ],
            "technical_validation": {
                "quantities_match_phase1": "boolean",
                "waste_factors_applied": "boolean",
                "protection_included": "boolean"
            }
        }
    
    def validate_json_response(self, 
                              response: str, 
                              phase: int) -> Dict[str, Any]:
        """
        Validate and extract JSON from model response
        
        Args:
            response: Raw model response
            phase: Phase number for schema validation
        
        Returns:
            Dict with 'valid', 'data', and 'errors' keys
        """
        result = {
            'valid': False,
            'data': None,
            'errors': []
        }
        
        # Try to extract JSON from response
        json_data = self._extract_json(response)
        
        if not json_data:
            result['errors'].append("No valid JSON found in response")
            return result
        
        # Validate against schema
        validation_errors = self._validate_against_schema(
            json_data, 
            self.output_schemas.get(f'phase{phase}', {})
        )
        
        if validation_errors:
            result['errors'].extend(validation_errors)
        
        # Fix common issues
        json_data = self._fix_common_issues(json_data)
        
        result['valid'] = len(result['errors']) == 0
        result['data'] = json_data
        
        return result
    
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from various response formats"""
        
        # Clean response
        response = response.strip()
        
        # Check for direct JSON
        if response.startswith('{') or response.startswith('['):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        
        # Extract from XML tags (Claude)
        xml_pattern = r'<json_output>(.*?)</json_output>'
        xml_match = re.search(xml_pattern, response, re.DOTALL)
        if xml_match:
            try:
                return json.loads(xml_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Extract from code blocks
        code_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'(\{[\s\S]*\})'  # Last resort: largest JSON-like structure
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_against_schema(self, 
                                data: Dict[str, Any],
                                schema: Dict[str, Any]) -> List[str]:
        """Basic schema validation"""
        errors = []
        
        # Check for required top-level fields
        if 'rooms' not in data:
            errors.append("Missing required field: 'rooms'")
        
        # Validate room data
        if 'rooms' in data and isinstance(data['rooms'], list):
            for i, room in enumerate(data['rooms']):
                if not isinstance(room, dict):
                    errors.append(f"Room {i} is not a dictionary")
                    continue
                
                # Check room name
                room_name = room.get('room_name', '')
                if not room_name or room_name == '**':
                    errors.append(f"Room {i} has invalid name: '{room_name}'")
                
                # Check tasks
                if 'tasks' not in room:
                    errors.append(f"Room {i} missing 'tasks' field")
                elif not isinstance(room['tasks'], list):
                    errors.append(f"Room {i} 'tasks' is not a list")
        
        return errors
    
    def _fix_common_issues(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common issues in JSON response"""
        
        # Fix room names with "**"
        if 'rooms' in data:
            for room in data['rooms']:
                if room.get('room_name') == '**':
                    # Try to extract from other fields or use a placeholder
                    room['room_name'] = room.get('room_id', 'Unknown Room')
                
                # Ensure numeric types for quantities
                if 'tasks' in room:
                    for task in room['tasks']:
                        if 'quantity' in task and isinstance(task['quantity'], str):
                            try:
                                task['quantity'] = float(task['quantity'])
                            except (ValueError, TypeError):
                                task['quantity'] = 0.0
        
        # Add timestamp if missing
        if 'processing_timestamp' not in data:
            data['processing_timestamp'] = datetime.now().isoformat()
        
        return data


# Usage example
if __name__ == "__main__":
    optimizer = PromptOptimizer()
    
    # Test with sample data
    sample_input = {
        "jobsite": {"address": "123 Main St"},
        "floors": [
            {
                "name": "Main Level",
                "rooms": [
                    {
                        "name": "Living Room",
                        "measurements": {
                            "floor_area_sqft": 200,
                            "height": 10
                        },
                        "work_scope": {
                            "Flooring": "Remove & Replace"
                        },
                        "demo_scope": {
                            "Flooring": 50
                        }
                    }
                ]
            }
        ]
    }
    
    # Optimize for each model
    for model in ['gpt4', 'claude', 'gemini']:
        prompt, params = optimizer.optimize_prompt(
            base_prompt="Generate reconstruction estimate",
            model_name=model,
            phase=1,
            input_data=sample_input
        )
        print(f"\n{model.upper()} Optimization:")
        print(f"Parameters: {params}")
        print(f"Prompt length: {len(prompt)} characters")