# Web Research Agent Task

You need to build a web research agent using the Claude Agent SDK that can browse the web and gather information using Playwright MCP (Model Context Protocol).

Your task is to create a Python script at `/root/workspace/research_agent.py` that:

1. Uses the Claude Agent SDK to create an AI agent
2. Configures the agent to use Playwright MCP for browser automation
3. Has the agent research the following topic: "Latest developments in AI safety research in 2024"
4. Saves the research findings to `/root/workspace/research_report.json`

The research report JSON file must contain:
- `topic`: The research topic string
- `sources`: A list of at least 3 source URLs that were visited
- `findings`: A list of at least 5 key findings (each as a string)
- `summary`: A 2-3 paragraph summary of the research

Your script should be executable and when run with `python research_agent.py`, it should:
1. Initialize the Claude Agent SDK client
2. Configure Playwright MCP server for web browsing
3. Send a research task to the agent
4. Parse the agent's response and save the structured report

Make sure the agent actually navigates to real websites and extracts information from them rather than making up data.
