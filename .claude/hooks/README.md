# Claude Code Hooks Documentation

## Overview
Hooks are automated scripts that run at specific points during Claude Code operations. They help automate repetitive tasks, enforce standards, and provide better visibility into the development process.

## Hook Types

### 1. **Standard Hooks**
Triggered by common development actions:
- `pre-commit`: Before committing code
- `post-commit`: After successful commit
- `pre-edit`: Before editing a file
- `post-edit`: After editing a file
- `pre-run`: Before running the application
- `post-run`: After running the application
- `on-error`: When an error occurs
- `on-test`: When running tests
- `on-build`: When building the project
- `on-analyze`: When analyzing code

### 2. **MCP Hooks** (Model Context Protocol)
Specific to AI model operations:
- `on-model-call`: When calling an AI model
- `on-model-response`: When receiving model response
- `on-merge`: When merging model results
- `on-validation`: When validating results

### 3. **Custom Hooks**
Project-specific triggers:
- `phase-complete`: When a pipeline phase completes
- `api-timeout`: When API call times out
- `cache-hit`: When cache is utilized

### 4. **Local Hooks**
Development environment specific:
- `dev-server`: Start development server
- `quick-test`: Quick validation checks
- `api-check`: Verify API keys
- `cleanup-logs`: Clean old log files

## Configuration Files

### `hooks.json`
Main hooks configuration (committed to git):
```json
{
  "hooks": {
    "pre-commit": {
      "enabled": true,
      "commands": ["..."],
      "description": "..."
    }
  }
}
```

### `hooks.local.json`
Local overrides (not committed):
- Override global settings
- Add development-specific hooks
- Configure debug/performance monitoring

## Variables

Hooks can use variables:
- `{file_path}`: Current file being edited
- `{timestamp}`: Current timestamp
- `{model_name}`: AI model name
- `{error_message}`: Error details
- `{phase_number}`: Pipeline phase number
- `{duration}`: Operation duration

## Usage Examples

### Enable a Hook
```json
"pre-commit": {
  "enabled": true,
  "commands": ["black --check src/"]
}
```

### Disable a Hook Locally
In `hooks.local.json`:
```json
"local_overrides": {
  "pre-commit": {
    "enabled": false
  }
}
```

### Add Custom Hook
```json
"custom_hooks": {
  "my-hook": {
    "enabled": true,
    "trigger": "custom_event",
    "commands": ["echo 'Custom hook triggered'"]
  }
}
```

## Best Practices

1. **Keep hooks fast**: Long-running hooks slow down development
2. **Use local overrides**: Customize for your environment without affecting others
3. **Log appropriately**: Use echo for important events, avoid spam
4. **Handle failures gracefully**: Set `fail_on_error: false` for non-critical hooks
5. **Test hooks locally**: Verify hooks work before committing

## Troubleshooting

### Hook Not Running
- Check if enabled in hooks.json
- Verify not overridden in hooks.local.json
- Check Claude Code logs for errors

### Hook Failing
- Review command syntax
- Check environment variables
- Verify dependencies installed
- Check timeout settings

### Performance Issues
- Disable verbose hooks in production
- Use parallel_execution for independent commands
- Increase timeout_seconds if needed

## Project-Specific Hooks

This project includes specialized hooks for:
- Multi-model AI orchestration
- Phase-based pipeline execution
- Consensus analysis and validation
- Performance monitoring
- API key verification

## Disabling All Hooks

To temporarily disable all hooks:
```json
// In hooks.local.json
{
  "settings": {
    "enabled": false
  }
}
```

## Hook Command Examples

### Format Check
```bash
conda run -n ai-estimate black --check src/
```

### Quick Validation
```bash
python -c "from src.models.model_interface import ModelOrchestrator; print('OK')"
```

### API Status
```bash
python -c "import os; print('API Keys:', 'OK' if all([os.getenv(k) for k in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']]) else 'Missing')"
```