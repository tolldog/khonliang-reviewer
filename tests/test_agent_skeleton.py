"""Smoke tests for the reviewer agent class metadata + construction.

Skill behavior itself is covered by ``test_agent_skills.py``; this module
just pins the class-level identity + construction path.
"""

from __future__ import annotations

from reviewer.agent import ReviewerAgent, create_reviewer_agent


def test_reviewer_agent_class_metadata():
    assert ReviewerAgent.agent_id == "reviewer-primary"
    assert ReviewerAgent.agent_type == "reviewer"


def test_reviewer_agent_honors_custom_agent_id():
    agent = create_reviewer_agent(
        agent_id="reviewer-secondary",
        bus_url="http://localhost:8787",
        config_path="",
    )
    assert agent.agent_id == "reviewer-secondary"
