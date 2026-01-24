---
name: web-research-patterns
description: Use this skill when structuring web research tasks and compiling findings into reports
---

# Web Research Patterns

## Overview

Effective web research involves systematic information gathering, source validation, and structured reporting. This skill covers patterns for conducting research using browser automation.

## Research Workflow

### 1. Define Research Scope
- Identify the main topic and subtopics
- List key questions to answer
- Define what constitutes a valid source

### 2. Source Discovery
- Start with authoritative sources (academic, official, established news)
- Use search engines to find relevant pages
- Follow links to find related content
- Track all visited URLs

### 3. Information Extraction
- Extract key facts and findings
- Note the source URL for each finding
- Look for dates to ensure information is current
- Cross-reference findings across multiple sources

### 4. Report Generation
- Organize findings by theme or subtopic
- Include source citations
- Write a coherent summary
- Structure output in a machine-readable format (JSON)

## Structured Output Format

For research reports, use a consistent JSON structure:

```json
{
  "topic": "The research topic",
  "timestamp": "2024-01-15T10:30:00Z",
  "sources": [
    {
      "url": "https://example.com/article1",
      "title": "Article Title",
      "accessed": "2024-01-15"
    }
  ],
  "findings": [
    "Key finding 1 with specific details",
    "Key finding 2 with supporting evidence",
    "Key finding 3 from authoritative source"
  ],
  "summary": "A comprehensive summary paragraph that synthesizes the findings..."
}
```

## Research Strategies

### Breadth-First Research
1. Visit multiple top-level sources
2. Extract high-level information from each
3. Compile a broad overview

```python
sources = [
    "https://news-site.com/topic",
    "https://research-org.org/papers",
    "https://official-blog.com/updates"
]

findings = []
for url in sources:
    content = navigate_and_extract(url)
    findings.extend(extract_key_points(content))
```

### Depth-First Research
1. Start with one authoritative source
2. Follow links to related content
3. Build deep understanding of specific aspects

```python
def research_depth_first(start_url, max_depth=3):
    visited = set()
    findings = []

    def explore(url, depth):
        if depth > max_depth or url in visited:
            return
        visited.add(url)

        content = navigate_and_extract(url)
        findings.extend(extract_key_points(content))

        for link in extract_relevant_links(content):
            explore(link, depth + 1)

    explore(start_url, 0)
    return findings
```

## Source Validation

### Quality Indicators
- Domain authority (educational, government, established organizations)
- Publication date (recent content for current topics)
- Author credentials (named experts, research institutions)
- Citations and references (links to primary sources)

### Red Flags
- No author or publication date
- Sensationalist headlines
- No citations or references
- Domain with low authority

## Example Research Implementation

```python
import json
from datetime import datetime

class WebResearcher:
    def __init__(self, browser_tools):
        self.browser = browser_tools
        self.sources = []
        self.findings = []

    def research_topic(self, topic, seed_urls):
        """Conduct research on a topic."""
        for url in seed_urls:
            self.visit_and_extract(url)

        return self.compile_report(topic)

    def visit_and_extract(self, url):
        """Visit a URL and extract findings."""
        # Navigate to the page
        self.browser.navigate(url)

        # Get page content
        content = self.browser.get_snapshot()

        # Record the source
        self.sources.append({
            "url": url,
            "accessed": datetime.now().isoformat()
        })

        # Extract and store findings
        key_points = self.extract_key_points(content)
        self.findings.extend(key_points)

    def compile_report(self, topic):
        """Generate a structured research report."""
        return {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "sources": self.sources,
            "findings": self.findings,
            "summary": self.generate_summary()
        }

    def save_report(self, filepath):
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.report, f, indent=2)
```

## Best Practices

1. **Verify information**: Cross-check facts across multiple sources
2. **Track sources**: Always record where information came from
3. **Stay focused**: Don't get sidetracked by tangential content
4. **Be efficient**: Extract relevant information without reading everything
5. **Handle errors**: Gracefully handle pages that fail to load
6. **Respect rate limits**: Don't overload websites with requests
