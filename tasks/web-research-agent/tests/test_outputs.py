#!/usr/bin/env python3
"""
Test cases for web research agent task.
Verifies that the research agent script and report meet requirements.
"""

import os
import json
import ast

import pytest

# Paths
WORKSPACE = "/root/workspace"
AGENT_SCRIPT = os.path.join(WORKSPACE, "research_agent.py")
RESEARCH_REPORT = os.path.join(WORKSPACE, "research_report.json")


class TestAgentScriptExists:
    """Test that the research agent script exists and is valid Python."""

    def test_script_file_exists(self):
        """The research_agent.py file must exist."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"

    def test_script_not_empty(self):
        """The script must not be empty."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"
        size = os.path.getsize(AGENT_SCRIPT)
        assert size > 100, f"Script is too small: {size} bytes"

    def test_script_is_valid_python(self):
        """The script must be valid Python syntax."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"

        with open(AGENT_SCRIPT, 'r') as f:
            content = f.read()

        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Invalid Python syntax: {e}")


class TestAgentScriptContent:
    """Test that the script contains required components."""

    def test_uses_anthropic_client(self):
        """Script must import and use anthropic client."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"

        with open(AGENT_SCRIPT, 'r') as f:
            content = f.read()

        assert "anthropic" in content.lower(), "Script must use anthropic SDK"
        assert "Anthropic" in content or "anthropic.Anthropic" in content, \
            "Script must instantiate Anthropic client"

    def test_uses_mcp_or_playwright(self):
        """Script must use MCP or Playwright for browser automation."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"

        with open(AGENT_SCRIPT, 'r') as f:
            content = f.read()

        has_mcp = "mcp" in content.lower() or "MCP" in content
        has_playwright = "playwright" in content.lower()

        assert has_mcp or has_playwright, \
            "Script must use MCP or Playwright for browser automation"

    def test_defines_tools_or_messages(self):
        """Script must define tools or messages for Claude API."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"

        with open(AGENT_SCRIPT, 'r') as f:
            content = f.read()

        has_tools = "tools" in content and ("tool_use" in content or "input_schema" in content)
        has_messages = "messages" in content and "role" in content

        assert has_tools or has_messages, \
            "Script must define tools or messages for Claude API"

    def test_saves_json_output(self):
        """Script must save output to JSON file."""
        assert os.path.exists(AGENT_SCRIPT), f"Script not found: {AGENT_SCRIPT}"

        with open(AGENT_SCRIPT, 'r') as f:
            content = f.read()

        has_json_dump = "json.dump" in content or "json.dumps" in content
        has_file_write = "open(" in content and ("'w'" in content or '"w"' in content)

        assert has_json_dump and has_file_write, \
            "Script must save JSON output to a file"


class TestResearchReportExists:
    """Test that the research report exists and is valid JSON."""

    def test_report_file_exists(self):
        """The research_report.json file must exist."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

    def test_report_not_empty(self):
        """The report must not be empty."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"
        size = os.path.getsize(RESEARCH_REPORT)
        assert size > 50, f"Report is too small: {size} bytes"

    def test_report_is_valid_json(self):
        """The report must be valid JSON."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            content = f.read()

        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON: {e}")


class TestResearchReportContent:
    """Test that the report contains required fields."""

    def test_has_topic_field(self):
        """Report must contain a topic field."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        assert "topic" in report, "Report must contain 'topic' field"
        assert isinstance(report["topic"], str), "Topic must be a string"
        assert len(report["topic"]) > 10, "Topic must be a meaningful string"

    def test_has_sources_field(self):
        """Report must contain a sources field with at least 3 URLs."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        assert "sources" in report, "Report must contain 'sources' field"
        assert isinstance(report["sources"], list), "Sources must be a list"
        assert len(report["sources"]) >= 3, \
            f"Report must have at least 3 sources, found {len(report['sources'])}"

    def test_sources_are_urls(self):
        """Sources must be valid URL strings."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        sources = report.get("sources", [])
        for source in sources:
            if isinstance(source, dict):
                url = source.get("url", "")
            else:
                url = source

            assert isinstance(url, str), f"Source URL must be a string: {url}"
            assert url.startswith("http://") or url.startswith("https://"), \
                f"Source must be a valid URL starting with http:// or https://: {url}"

    def test_has_findings_field(self):
        """Report must contain a findings field with at least 5 findings."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        assert "findings" in report, "Report must contain 'findings' field"
        assert isinstance(report["findings"], list), "Findings must be a list"
        assert len(report["findings"]) >= 5, \
            f"Report must have at least 5 findings, found {len(report['findings'])}"

    def test_findings_are_meaningful(self):
        """Each finding must be a meaningful string."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        findings = report.get("findings", [])
        for i, finding in enumerate(findings):
            assert isinstance(finding, str), f"Finding {i} must be a string"
            assert len(finding) >= 20, \
                f"Finding {i} is too short (min 20 chars): '{finding}'"

    def test_has_summary_field(self):
        """Report must contain a summary field."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        assert "summary" in report, "Report must contain 'summary' field"
        assert isinstance(report["summary"], str), "Summary must be a string"

    def test_summary_length(self):
        """Summary must be 2-3 paragraphs (at least 200 characters)."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        summary = report.get("summary", "")
        assert len(summary) >= 200, \
            f"Summary must be at least 200 characters (2-3 paragraphs), found {len(summary)}"


class TestResearchQuality:
    """Test the quality and relevance of the research."""

    def test_topic_relevance(self):
        """Topic should be about AI safety research."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        topic = report.get("topic", "").lower()
        assert "ai" in topic or "artificial intelligence" in topic, \
            "Topic should mention AI"
        assert "safety" in topic or "research" in topic or "development" in topic, \
            "Topic should mention safety, research, or development"

    def test_findings_relevance(self):
        """At least some findings should mention AI-related terms."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        findings = report.get("findings", [])
        ai_related_findings = 0

        ai_keywords = ["ai", "artificial intelligence", "machine learning", "safety",
                       "alignment", "model", "neural", "llm", "language model",
                       "anthropic", "openai", "deepmind", "research"]

        for finding in findings:
            finding_lower = finding.lower()
            if any(keyword in finding_lower for keyword in ai_keywords):
                ai_related_findings += 1

        assert ai_related_findings >= 3, \
            f"At least 3 findings should be AI-related, found {ai_related_findings}"

    def test_summary_coherence(self):
        """Summary should be coherent and mention key topics."""
        assert os.path.exists(RESEARCH_REPORT), f"Report not found: {RESEARCH_REPORT}"

        with open(RESEARCH_REPORT, 'r') as f:
            report = json.load(f)

        summary = report.get("summary", "").lower()

        # Summary should mention at least some key topics
        key_topics = ["ai", "safety", "research", "model", "development"]
        topics_found = sum(1 for topic in key_topics if topic in summary)

        assert topics_found >= 2, \
            f"Summary should mention at least 2 key topics, found {topics_found}"
