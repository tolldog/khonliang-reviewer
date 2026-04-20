"""Smoke tests for the reviewer agent skeleton.

Concrete skill behavior lands in later work units; this module just verifies
the agent imports, instantiates, and presents the expected agent_id /
agent_type / (empty) skill surface.
"""

from __future__ import annotations

from reviewer.agent import ReviewerAgent, create_reviewer_agent


def test_reviewer_agent_class_metadata():
    assert ReviewerAgent.agent_id == "reviewer-primary"
    assert ReviewerAgent.agent_type == "reviewer"


def test_reviewer_agent_registers_no_skills_at_scaffold():
    agent = create_reviewer_agent(
        agent_id="reviewer-primary",
        bus_url="http://localhost:8787",
        config_path="",
    )
    assert agent.register_skills() == []


def test_reviewer_agent_honors_custom_agent_id():
    agent = create_reviewer_agent(
        agent_id="reviewer-secondary",
        bus_url="http://localhost:8787",
        config_path="",
    )
    assert agent.agent_id == "reviewer-secondary"
