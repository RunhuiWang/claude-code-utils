---
name: playwright-mcp
description: Use this skill when you need to automate web browsers for research, scraping, or testing using the Playwright MCP server
---

# Playwright MCP Server

## Overview

The Playwright MCP server provides browser automation capabilities through the Model Context Protocol. It allows AI agents to navigate websites, interact with page elements, and extract content.

## Installation

```bash
npm install -g @anthropic-ai/mcp-playwright
```

Or run directly with npx:
```bash
npx @anthropic-ai/mcp-playwright
```

## Available Tools

The Playwright MCP server provides these tools:

### Navigation Tools
- `browser_navigate`: Navigate to a URL
- `browser_go_back`: Go back in browser history
- `browser_go_forward`: Go forward in browser history
- `browser_refresh`: Refresh the current page

### Content Tools
- `browser_snapshot`: Get a text snapshot of the current page
- `browser_screenshot`: Take a screenshot of the page
- `browser_get_text`: Get text content from the page

### Interaction Tools
- `browser_click`: Click on an element
- `browser_type`: Type text into an input field
- `browser_select`: Select an option from a dropdown
- `browser_scroll`: Scroll the page

### Tab Management
- `browser_tab_new`: Open a new tab
- `browser_tab_close`: Close the current tab
- `browser_tab_list`: List all open tabs
- `browser_tab_select`: Switch to a specific tab

## Example Usage

### Starting the MCP Server

```python
import subprocess
import json

# Start the Playwright MCP server
mcp_process = subprocess.Popen(
    ["npx", "@anthropic-ai/mcp-playwright"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

def mcp_request(method, params=None):
    """Send a JSON-RPC request to the MCP server."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {}
    }
    mcp_process.stdin.write(json.dumps(request) + '\n')
    mcp_process.stdin.flush()
    response = mcp_process.stdout.readline()
    return json.loads(response)
```

### Navigating to a Page

```python
# Navigate to a website
result = mcp_request("tools/call", {
    "name": "browser_navigate",
    "arguments": {"url": "https://example.com"}
})
```

### Getting Page Content

```python
# Get a text snapshot of the page
result = mcp_request("tools/call", {
    "name": "browser_snapshot",
    "arguments": {}
})
page_content = result["result"]["content"]
```

### Clicking Elements

```python
# Click on a link or button (use CSS selector or text)
result = mcp_request("tools/call", {
    "name": "browser_click",
    "arguments": {"selector": "a.read-more"}
})
```

### Typing in Forms

```python
# Type in a search box
result = mcp_request("tools/call", {
    "name": "browser_type",
    "arguments": {
        "selector": "input[name='q']",
        "text": "AI safety research 2024"
    }
})
```

## Converting MCP Tools for Claude

When using MCP tools with the Claude API, convert them to the Claude tool format:

```python
def convert_mcp_tool_to_claude(mcp_tool):
    """Convert an MCP tool definition to Claude tool format."""
    return {
        "name": mcp_tool["name"],
        "description": mcp_tool.get("description", ""),
        "input_schema": mcp_tool.get("inputSchema", {
            "type": "object",
            "properties": {}
        })
    }

# Get tools from MCP server
mcp_tools = mcp_request("tools/list")["result"]["tools"]
claude_tools = [convert_mcp_tool_to_claude(t) for t in mcp_tools]
```

## Best Practices

1. **Wait for page loads**: After navigation, wait for content to load
2. **Handle dynamic content**: Some pages load content via JavaScript
3. **Use robust selectors**: Prefer IDs and data attributes over fragile CSS paths
4. **Rate limit requests**: Don't hammer websites with rapid requests
5. **Close browser when done**: Clean up resources after finishing
