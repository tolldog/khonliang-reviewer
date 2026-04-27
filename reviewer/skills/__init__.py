"""Standalone helper modules consumed by the reviewer agent's skills.

Modules here own pure formatting / mapping logic so the agent's
``@handler`` methods stay thin glue between the bus envelope and
the helper. Keeps each piece independently testable without
spinning up a full :class:`ReviewerAgent`.

See ``specs/MS-D/spec.md`` for the milestone context.
"""
