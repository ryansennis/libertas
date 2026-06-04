"""Mood and emotional state for agents."""

from dataclasses import dataclass


@dataclass
class MoodState:
    """Current emotional/motivational state."""
    happiness: float = 0.7  # 0-1
    stress: float = 0.3  # 0-1
    motivation: float = 0.8  # 0-1
    trust_in_leadership: float = 0.7  # 0-1
    solidarity_with_group: float = 0.8  # 0-1
