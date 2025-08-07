# Claude Project Settings

This directory contains configuration files for Claude Code integration.

## Files

### `settings.json`
Main project settings including:
- Project metadata and description
- Development environment configuration
- AI model configurations
- Common commands and shortcuts
- IDE preferences

### `settings.local.json`
Local environment settings (not committed to git):
- Local paths and overrides
- Development server configuration
- Performance tuning
- Logging preferences
- Model-specific timeouts

## Usage

These settings help Claude Code understand:
- Project structure and conventions
- Available commands and tools
- Development workflow
- Testing and deployment processes

## API Keys

**Important**: Never store API keys in these files. Use the `.env` file instead:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Commands

Quick commands configured for this project:
- `run`: Start the pipeline processor
- `server`: Start the FastAPI development server
- `test`: Run the test suite
- `lint`: Check code quality
- `format`: Auto-format code

## Customization

Feel free to modify these settings based on your preferences and local environment.