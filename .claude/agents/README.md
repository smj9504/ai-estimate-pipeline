# Project Agents

Specialized AI agents for the AI Estimate Pipeline project.

## Available Agents

### 🔍 Estimate Analyzer
**File**: `estimate-analyzer.md`
**Purpose**: Analyze and validate multi-model estimation results
**Specialties**: 
- Consensus analysis
- Statistical validation
- Outlier detection
- Confidence scoring

### 🎯 Pipeline Orchestrator  
**File**: `pipeline-orchestrator.md`
**Purpose**: Manage the complete estimation pipeline flow
**Specialties**:
- Phase coordination
- Error recovery
- Performance optimization
- Progress tracking

## Usage

These agents can be invoked in Claude Code conversations to provide specialized assistance:

```
@estimate-analyzer analyze the Phase 1 results
@pipeline-orchestrator optimize the execution flow
```

## Agent Development

To create a new agent:
1. Create a new `.md` file in this directory
2. Define the agent's purpose, capabilities, and workflow
3. Include specific context about the project domain
4. Add commands and integration points

## Best Practices

- Keep agents focused on specific domains
- Include project-specific context and terminology
- Define clear workflows and decision trees
- Document integration points with project code
- Update agents as the project evolves

## Project Context

These agents are specifically trained on:
- Multi-model AI consensus methodology
- Construction estimation best practices
- Remove & Replace reconstruction logic
- DMV area market conditions
- Phase-based pipeline architecture