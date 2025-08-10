# WonyBot AI Agents Guide

## Overview

WonyBot now includes an advanced AI Agent system that provides specialized assistants for different tasks. Each agent has unique capabilities and expertise, allowing for more efficient and targeted task completion.

## Available Agents

### 1. Research Assistant
- **Role**: `researcher`
- **Specialization**: Information gathering and analysis
- **Capabilities**:
  - Information research
  - Fact-checking
  - Source verification
  - Topic investigation
  - Knowledge synthesis

### 2. Code Assistant
- **Role**: `coder`
- **Specialization**: Programming and development
- **Capabilities**:
  - Code writing
  - Debugging
  - Code review
  - Refactoring
  - Algorithm design
  - API development

### 3. Analysis Assistant
- **Role**: `analyst`
- **Specialization**: Data analysis and insights
- **Capabilities**:
  - Data analysis
  - Pattern recognition
  - Statistical analysis
  - Trend identification
  - Insight generation
  - Report creation

### 4. Summary Assistant
- **Role**: `summarizer`
- **Specialization**: Summarization and extraction
- **Capabilities**:
  - Text summarization
  - Key point extraction
  - Executive summaries
  - Meeting notes
  - Document digests

### 5. Creative Assistant
- **Role**: `creative`
- **Specialization**: Creative tasks and ideation
- **Capabilities**:
  - Creative writing
  - Brainstorming
  - Idea generation
  - Story creation
  - Design concepts
  - Problem solving

## Using Agents

### 1. List Available Agents

```bash
wony agents list
```

Shows all available agents with their capabilities and current status.

### 2. Check Agent Status

```bash
wony agents status
```

Displays overall system status including active agents and queue information.

### 3. Assign Task to Specific Agent

```bash
# Assign to specific agent
wony agents assign researcher --task "Research the latest AI trends in 2024"

# Auto-select best agent
wony agents assign --task "Write a Python function to sort a list"
```

### 4. View Task History

```bash
wony agents history
```

Shows recent completed tasks with results.

## Agent Chat Mode

Start an interactive chat session with a specific agent or auto-selection:

### Chat with Specific Agent

```bash
# Chat with researcher
wony agent-chat --agent researcher

# Chat with coder
wony agent-chat --agent coder

# Chat with analyst
wony agent-chat --agent analyst
```

### Auto Agent Selection

```bash
wony agent-chat --agent auto
```

The system will automatically select the best agent based on your task.

### Agent Chat Commands

During an agent chat session:
- `exit` or `quit`: End the session
- `switch <agent>`: Switch to a different agent (e.g., `switch coder`)

## Examples

### Research Task

```bash
wony agents assign researcher --task "Research the impact of quantum computing on cryptography"
```

### Coding Task

```bash
wony agents assign coder --task "Write a Python class for managing a task queue with priority"
```

### Analysis Task

```bash
wony agents assign analyst --task "Analyze the performance metrics from last quarter"
```

### Summary Task

```bash
wony agents assign summarizer --task "Summarize the key points from this meeting transcript"
```

### Creative Task

```bash
wony agents assign creative --task "Brainstorm 10 innovative features for a task management app"
```

## Integration with Chat

The agent system is fully integrated with WonyBot's chat functionality:

1. **Memory Integration**: Agent interactions are saved to memory
2. **Session Support**: Continue conversations across sessions
3. **Context Awareness**: Agents can access previous conversation context

## Advanced Features

### Parallel Task Processing

Multiple agents can work on different tasks simultaneously:

```python
# In code
tasks = [
    {'task': 'Research AI trends', 'agent_id': 'researcher'},
    {'task': 'Write code example', 'agent_id': 'coder'},
    {'task': 'Analyze data', 'agent_id': 'analyst'}
]
results = await agent_manager.process_parallel_tasks(tasks)
```

### Complex Task Delegation

For complex tasks that require multiple agents:

```python
# The system can break down complex tasks
result = await agent_manager.delegate_complex_task(
    "Create a comprehensive report on AI implementation in healthcare"
)
```

## Best Practices

1. **Choose the Right Agent**: Select agents based on their specialization for best results
2. **Provide Clear Tasks**: Be specific about what you want the agent to accomplish
3. **Use Context**: Provide relevant context for better results
4. **Review Results**: Always review agent outputs before using them
5. **Combine Agents**: Use multiple agents for complex multi-faceted tasks

## API Usage

### Python Integration

```python
from app.services.chat import ChatService
from app.agents.agent_manager import AgentManager

# Initialize chat service with agents
chat_service = ChatService(enable_agents=True)
agent_manager = chat_service.get_agent_manager()

# Assign task to agent
result = await agent_manager.assign_task(
    task="Research blockchain technology",
    agent_id="researcher"
)

# Auto-select agent
result = await agent_manager.assign_task(
    task="Debug this Python code"
)
```

## Troubleshooting

### Agent Not Available
- Ensure Ollama is running: `ollama serve`
- Check model is installed: `ollama pull gpt-oss:20b`

### Task Queued
- Agents process one task at a time
- Tasks are queued when agent is busy
- Check queue status with `wony agents status`

### Poor Results
- Provide more specific task descriptions
- Include relevant context
- Try a different agent for the task

## Hierarchical Agent System

WonyBot now includes an advanced hierarchical agent system where:

### Architecture

1. **Orchestrator Agent**: Central coordinator that manages and distributes tasks
2. **Worker Agents**: Specialized agents that execute assigned tasks
3. **Consensus System**: Collective decision-making through voting

### Using Hierarchical System

#### Check System Status
```bash
wony hierarchical status
```

#### View Hierarchy Structure
```bash
wony hierarchical hierarchy
```

#### Process Complex Task
```bash
wony hierarchical process --task "Your complex task description"
```

#### View Consensus History
```bash
wony hierarchical consensus
```

### Voting Mechanisms

The system supports multiple voting types:
- **Simple Majority**: >50% approval required
- **Super Majority**: >66% approval required
- **Unanimous**: 100% approval required
- **Weighted**: Votes weighted by agent expertise
- **Consensus**: Continue until consensus reached

### Features

1. **Task Decomposition**: Orchestrator breaks complex tasks into subtasks
2. **Automatic Distribution**: Tasks assigned to best-suited agents
3. **Parallel Execution**: Multiple agents work simultaneously
4. **Collective Decision**: All agents vote on final results
5. **Performance Monitoring**: Track agent performance and optimize distribution

### Example Workflow

```bash
# Process a complex research and analysis task
wony hierarchical process --task "Research AI trends and analyze their impact on business"

# The system will:
# 1. Orchestrator decomposes the task
# 2. Assigns subtasks to specialized agents
# 3. Agents execute their tasks in parallel
# 4. All agents vote on the results
# 5. Final consensus decision is made
```

## Future Enhancements

- Custom agent creation
- Agent collaboration on tasks
- Learning and adaptation
- Specialized domain agents
- Multi-language support
- Advanced consensus mechanisms
- Agent performance optimization
- Distributed agent networks