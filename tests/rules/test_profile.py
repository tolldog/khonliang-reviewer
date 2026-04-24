"""Tests for the profile cache protocol + in-memory implementation."""

from __future__ import annotations

import pytest

from reviewer.rules import (
    InMemoryProfileCache,
    PROFILE_KEY_PREFIX,
    ProfileCache,
    profile_key,
)


# ---------------------------------------------------------------------------
# profile_key
# ---------------------------------------------------------------------------


def test_profile_key_has_stable_prefix():
    assert profile_key("reviewer").startswith(PROFILE_KEY_PREFIX)


def test_profile_key_lowercases():
    assert profile_key("Tolldog/Khonliang-Reviewer") == (
        PROFILE_KEY_PREFIX + "tolldog/khonliang-reviewer"
    )


def test_profile_key_trims_whitespace():
    assert profile_key("  reviewer  ") == PROFILE_KEY_PREFIX + "reviewer"


def test_profile_key_is_deterministic():
    assert profile_key("reviewer") == profile_key("reviewer")


def test_profile_key_distinguishes_owners():
    """Owner namespace should be preserved so forks don't collide."""
    assert profile_key("tolldog/reviewer") != profile_key("other/reviewer")


# ---------------------------------------------------------------------------
# InMemoryProfileCache
# ---------------------------------------------------------------------------


async def test_get_returns_none_when_missing():
    cache = InMemoryProfileCache()
    assert await cache.get_profile("reviewer") is None


async def test_put_then_get_round_trips():
    cache = InMemoryProfileCache()
    profile = {"languages": {"python": 134}, "tests": 24}

    cache.put_profile("reviewer", profile)
    got = await cache.get_profile("reviewer")

    assert got == profile


async def test_get_returns_copy_not_shared_state():
    """Mutating a returned profile must not poison the cache."""
    cache = InMemoryProfileCache()
    cache.put_profile("reviewer", {"languages": {"python": 100}})

    got = await cache.get_profile("reviewer")
    assert got is not None
    got["languages"]["python"] = 999

    again = await cache.get_profile("reviewer")
    assert again == {"languages": {"python": 100}}


async def test_put_stores_copy_not_shared_reference():
    """Caller mutations after put must not bleed into the cache."""
    cache = InMemoryProfileCache()
    profile = {"languages": {"python": 100}}

    cache.put_profile("reviewer", profile)
    profile["languages"]["python"] = 999

    got = await cache.get_profile("reviewer")
    assert got == {"languages": {"python": 100}}


async def test_invalidate_removes_entry():
    cache = InMemoryProfileCache()
    cache.put_profile("reviewer", {"x": 1})
    cache.invalidate("reviewer")
    assert await cache.get_profile("reviewer") is None


async def test_invalidate_missing_is_no_op():
    """Invalidate of a non-existent key should not raise."""
    cache = InMemoryProfileCache()
    cache.invalidate("never-stored")  # must not raise


async def test_get_is_case_insensitive():
    """Callers may pass different casings; lookups should still match."""
    cache = InMemoryProfileCache()
    cache.put_profile("Tolldog/Reviewer", {"python": 100})
    assert await cache.get_profile("tolldog/reviewer") == {"python": 100}


def test_age_seconds_none_when_missing():
    cache = InMemoryProfileCache()
    assert cache.age_seconds("unknown") is None


def test_age_seconds_small_when_just_put():
    cache = InMemoryProfileCache()
    cache.put_profile("reviewer", {})
    age = cache.age_seconds("reviewer")
    assert age is not None
    assert age < 1.0


def test_in_memory_cache_satisfies_protocol():
    """InMemoryProfileCache must be accepted where ProfileCache is expected."""
    cache: ProfileCache = InMemoryProfileCache()  # type-checker contract
    assert isinstance(cache, ProfileCache)  # runtime_checkable protocol


# ---------------------------------------------------------------------------
# Integration: rule table can consume a profile from the cache
# ---------------------------------------------------------------------------


async def test_profile_flows_into_policy_input():
    """Smoke test that the two modules compose — cache output fits PolicyInput.

    End-to-end wiring (profile load + decide()) lands in WU5 when the
    bus skill drives both sides. This just confirms the shapes line up.
    """
    from reviewer.rules import PolicyInput, decide

    cache = InMemoryProfileCache()
    cache.put_profile(
        "reviewer",
        {"languages": {"python": 134}, "tests": 24, "files": 40},
    )
    profile = await cache.get_profile("reviewer")

    decision = decide(
        PolicyInput(kind="pr_diff", diff_line_count=20, profile=profile)
    )
    # small diff + profile present → fallback still applies
    assert decision.backend == "ollama"
    assert decision.model == "qwen2.5-coder:14b"
