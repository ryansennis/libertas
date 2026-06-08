"""Goal system for agents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class GoalStatus(Enum):
    """Status of a goal in its lifecycle."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


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
    status: GoalStatus = GoalStatus.NOT_STARTED
    created_step: int = 0
    abandon_reason: Optional[str] = None  # Why the goal was abandoned

    def evaluate_progress(self, current_value: float) -> float:
        """Calculate progress toward goal."""
        if self.target_value and current_value is not None:
            return min(1.0, current_value / self.target_value)
        return self.progress


@dataclass
class GoalSystem:
    """Manages agent's goals and tracks progress."""
    active_goals: List[Goal] = field(default_factory=list)
    completed_goals: List[Goal] = field(default_factory=list)
    abandoned_goals: List[Goal] = field(default_factory=list)

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal to pursue."""
        self.active_goals.append(goal)
        # Automatically set to IN_PROGRESS if not already set
        if goal.status == GoalStatus.NOT_STARTED:
            goal.status = GoalStatus.IN_PROGRESS

    def update_progress(self, goal_id: str, progress: float) -> None:
        """Update progress on a goal."""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.progress = progress

                # Update status to IN_PROGRESS if it was NOT_STARTED
                if goal.status == GoalStatus.NOT_STARTED:
                    goal.status = GoalStatus.IN_PROGRESS

                # Complete goal if progress reaches 100%
                if progress >= 1.0:
                    goal.status = GoalStatus.COMPLETED
                    self.active_goals.remove(goal)
                    self.completed_goals.append(goal)
                break

    def abandon_goal(self, goal_id: str, reason: str) -> None:
        """Give up on a goal."""
        for goal in self.active_goals:
            if goal.goal_id == goal_id:
                goal.status = GoalStatus.ABANDONED
                goal.abandon_reason = reason
                self.active_goals.remove(goal)
                self.abandoned_goals.append(goal)
                break

    def revive_goal(self, goal_id: str) -> bool:
        """
        Revive an abandoned goal and return it to active goals.

        Args:
            goal_id: ID of the abandoned goal to revive

        Returns:
            True if goal was found and revived, False otherwise
        """
        for goal in self.abandoned_goals:
            if goal.goal_id == goal_id:
                goal.status = GoalStatus.IN_PROGRESS
                goal.abandon_reason = None
                self.abandoned_goals.remove(goal)
                self.active_goals.append(goal)
                return True
        return False

    def get_active_goals_by_priority(self) -> List[Goal]:
        """Return active goals sorted by priority."""
        return sorted(self.active_goals, key=lambda g: g.priority, reverse=True)
