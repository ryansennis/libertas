"""Memory systems for agents."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SemanticMemory:
    """Long-term factual knowledge learned through experience."""

    # Market knowledge
    price_patterns: Dict[str, List[float]] = field(default_factory=dict)  # resource -> historical prices
    market_insights: List[str] = field(default_factory=list)  # e.g., "wheat prices spike in winter"

    # Social knowledge
    worker_behaviors: Dict[str, Dict] = field(default_factory=dict)  # worker_id -> behavior patterns
    voting_coalitions: List[List[str]] = field(default_factory=list)  # groups that vote together
    trusted_workers: Dict[str, float] = field(default_factory=dict)  # worker_id -> trust score (0-1)

    # Production knowledge
    recipe_efficiency: Dict[str, float] = field(default_factory=dict)  # recipe -> learned efficiency
    skill_mastery: Dict[str, int] = field(default_factory=dict)  # skill -> mastery level (0-10)

    # Governance knowledge
    motion_outcomes: Dict[str, bool] = field(default_factory=dict)  # motion_type -> typical pass/fail
    constitution_rules: List[str] = field(default_factory=list)  # learned constitutional patterns
