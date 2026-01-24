---
name: claude-agent-sdk
description: Use this skill when building AI agents with the Claude Agent SDK (anthropic Python package) that can use tools and MCP servers
---

# Claude Agent SDK

## Overview

The Claude Agent SDK allows you to build AI agents that can use tools, including MCP (Model Context Protocol) servers. This enables Claude to interact with external systems like web browsers, databases, and APIs.

## Basic Agent Setup

```python
import anthropic
import json

# Initialize the client
client = anthropic.Anthropic()

# Create a message with tools
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    tools=[...],  # Define your tools here
    messages=[
        {"role": "user", "content": "Your task here"}
    ]
)
```

## Using MCP Servers

MCP servers provide tools that Claude can use. To use an MCP server:

1. Start the MCP server process
2. Connect to it via stdio or HTTP
3. List available tools from the server
4. Pass those tools to Claude

### Example with subprocess-based MCP server:

```python
import subprocess
import json

# Start MCP server
process = subprocess.Popen(
    ["npx", "@anthropic-ai/mcp-playwright"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Send JSON-RPC requests to the server
def send_request(method, params=None):
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {}
    }
    process.stdin.write(json.dumps(request).encode() + b'\n')
    process.stdin.flush()
    response = process.stdout.readline()
    return json.loads(response)

# List available tools
tools_response = send_request("tools/list")
tools = tools_response.get("result", {}).get("tools", [])
```

## Agentic Loop Pattern

For complex tasks, use an agentic loop that continues until the task is complete:

```python
def run_agent(task):
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # Check if done
        if response.stop_reason == "end_turn":
            return extract_final_response(response)

        # Handle tool use
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
```

## Tool Definition Format

Tools are defined as JSON schemas:

```python
tools = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to"
                }
            },
            "required": ["url"]
        }
    }
]
```

## Best Practices

1. **Handle rate limits**: Implement exponential backoff for API calls
2. **Validate tool results**: Check for errors in tool execution
3. **Set reasonable timeouts**: Don't let the agent run indefinitely
4. **Log interactions**: Keep track of all messages for debugging
5. **Use structured output**: Parse agent responses into structured data
