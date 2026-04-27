"""Tests for the cold-start welcome surface (fr_khonliang-bus-lib_6a82732c).

Pins ReviewerAgent.WELCOME so future edits don't silently empty it.
"""

from __future__ import annotations

from khonliang_bus import Welcome
from reviewer.agent import ReviewerAgent


def test_reviewer_welcome_is_populated():
    w = ReviewerAgent.WELCOME
    assert isinstance(w, Welcome)
    assert w.role
    assert w.mission
    assert w.entry_points
    advertised = {ep.skill for ep in w.entry_points}
    # Sanity: at least the canonical reviewer skills.
    assert "review_diff" in advertised
    assert "review_pr" in advertised
