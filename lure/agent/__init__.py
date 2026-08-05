"""Agents under test.

An agent is whatever Lure is measuring. Two ship here, behind one interface:

* :class:`~lure.agent.mock_agent.MockVulnerableAgent` -- scripted and fully
  credulous, so the offline core can be validated against known-correct
  answers.
* :class:`~lure.agent.llm_agent.LLMAgent` -- a real model, reading the same
  pages and executing through the same defense and checkout path.

Neither is privileged by the environment, the attacks, the defenses or the
scorer. That is what makes their numbers comparable.
"""

from lure.agent.base import Agent
from lure.agent.llm_agent import LLMAgent
from lure.agent.mock_agent import MockVulnerableAgent

__all__ = ["Agent", "LLMAgent", "MockVulnerableAgent"]
