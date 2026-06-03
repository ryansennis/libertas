"""Cognitive attributes for LLM agents."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PersonalityTraits:
    """OCEAN personality model with political leanings."""
    openness: float = 0.5  # 0-1, creativity, new ideas
    conscientiousness: float = 0.5  # 0-1, organization, discipline
    extraversion: float = 0.5  # 0-1, social engagement
    agreeableness: float = 0.5  # 0-1, cooperation vs competition
    neuroticism: float = 0.5  # 0-1, emotional stability

    # Political leanings
    economic_left_right: float = 0.0  # -1 to 1 (collectivist to individualist)
    authority_libertarian: float = 0.0  # -1 to 1 (hierarchical to anarchist)


@dataclass
class Background:
    """Worker's background and history."""
    education_level: int = 2  # 0-5
    years_experience: int = 0
    specializations: List[str] = field(default_factory=list)
    origin_pod: Optional[str] = None


@dataclass
class MoodState:
    """Current emotional/motivational state."""
    happiness: float = 0.7  # 0-1
    stress: float = 0.3  # 0-1
    motivation: float = 0.8  # 0-1
    trust_in_leadership: float = 0.7  # 0-1
    solidarity_with_group: float = 0.8  # 0-1


__all__ = [
    "PersonalityTraits",
    "Background",
    "MoodState",
]
