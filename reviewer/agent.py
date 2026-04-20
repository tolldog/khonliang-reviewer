"""Reviewer as a bus agent.

Skeleton only at this work unit. Registers with the bus but exposes no
skills yet. Concrete skills (``review_pr``, ``review_diff``,
``usage_summary``) land in subsequent work units along with the provider
adapters they need.

Usage::

    # Install into the bus
    python -m reviewer.agent install --id reviewer-primary --bus http://localhost:8787 --config config.yaml

    # Start (normally done by the bus on boot)
    python -m reviewer.agent --id reviewer-primary --bus http://localhost:8787 --config config.yaml

    # Uninstall
    python -m reviewer.agent uninstall --id reviewer-primary --bus http://localhost:8787
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from khonliang_bus import BaseAgent


logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Bus-native reviewer agent skeleton.

    No skills are registered in this scaffold. They are added in later work
    units together with the provider adapters and rule-table policy.
    """

    agent_id = "reviewer-primary"
    agent_type = "reviewer"

    def register_skills(self) -> list[Any]:
        return []


def create_reviewer_agent(
    *, agent_id: str, bus_url: str, config_path: str
) -> ReviewerAgent:
    agent = ReviewerAgent(
        agent_id=agent_id,
        bus_url=bus_url,
        config_path=config_path,
    )
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="khonliang-reviewer bus agent")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["install", "uninstall"],
        help="install or uninstall from the bus",
    )
    parser.add_argument("--id", default="reviewer-primary")
    parser.add_argument("--bus", default="http://localhost:8787")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command in ("install", "uninstall"):
        BaseAgent.from_cli(
            [
                args.command,
                "--id", args.id,
                "--bus", args.bus,
                "--config", args.config,
            ]
        )
        return

    agent = create_reviewer_agent(
        agent_id=args.id,
        bus_url=args.bus,
        config_path=args.config,
    )
    asyncio.run(agent.start())


if __name__ == "__main__":
    main()
