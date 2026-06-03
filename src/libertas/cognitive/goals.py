"""Goal system for agents."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Goal:
    """A specific objective an agent is pursuing."""
    goal_id: str
    goal_type: str  # "economic", "social", "governance", "learning"
    description: str
    target_metric: Optional[str] = None  # e.g., "currency", "trust_score"
    target_value: Optional[float] = None
    deadline_step: Optional[int] = None
    priority: float = 0.5  # 0-1
    progress: float = 0.0  # 0-1
    status: str = "active"  # "active", "achieved", "abandoned"
    created_step: int = 0

    def evaluate_progress(self, current_value: float) -> float:
        """Calculate progress toward goal."""
        if self.target_value and current_value is not None:
            return min(1.0, current_value / self.target_value)
        return self.progress


@dataclass
class GoalSystem:
    """Manages agent's goals and tracks progress."""
    active_goals: List[Goal] = field(default_factory=list)
    achieved_goals: List[Goal] = field(default_factory=list)
    abandoned_goals: List[Goal] = field(default_factory=list)

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal to pursue."""
        self.active_goals.append(goal)

    def update_progress(self, goal_id: str, progress: float) -> None:
        """Update progress on a goal."""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.progress = progress
                if progress >= 1.0:
                    goal.status = "achieved"
                    self.active_goals.remove(goal)
                    self.achieved_goals.append(goal)
                break

    def abandon_goal(self, goal_id: str, reason: str) -> None:
        """Give up on a goal."""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.status = f"abandoned: {reason}"
                self.active_goals.remove(goal)
                self.abandoned_goals.append(goal)
                break

    def get_active_goals_by_priority(self) -> List[Goal]:
        """Return active goals sorted by priority."""
        return sorted(self.active_goals, key=lambda g: g.priority, reverse=True)
