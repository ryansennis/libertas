"""Tests for Goal and GoalSystem classes."""

import pytest
from libertas.cognitive import Goal, GoalSystem


class TestGoal:
    """Test Goal dataclass."""

    def test_goal_initialization(self):
        """Test basic goal initialization."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Earn 1000 currency",
            target_metric="currency",
            target_value=1000.0,
            priority=0.8
        )

        assert goal.goal_id == "goal_1"
        assert goal.goal_type == "economic"
        assert goal.description == "Earn 1000 currency"
        assert goal.target_metric == "currency"
        assert goal.target_value == 1000.0
        assert goal.priority == 0.8
        assert goal.progress == 0.0
        assert goal.status == "active"

    def test_goal_defaults(self):
        """Test goal default values."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="social",
            description="Build relationships"
        )

        assert goal.target_metric is None
        assert goal.target_value is None
        assert goal.deadline_step is None
        assert goal.priority == 0.5
        assert goal.progress == 0.0
        assert goal.status == "active"
        assert goal.created_step == 0

    def test_evaluate_progress_with_target(self):
        """Test progress evaluation with target value."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Earn 1000 currency",
            target_value=1000.0
        )

        assert goal.evaluate_progress(0.0) == 0.0
        assert goal.evaluate_progress(500.0) == 0.5
        assert goal.evaluate_progress(1000.0) == 1.0
        assert goal.evaluate_progress(1500.0) == 1.0  # Capped at 1.0

    def test_evaluate_progress_without_target(self):
        """Test progress evaluation without target value."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="learning",
            description="Learn farming",
            progress=0.3
        )

        assert goal.evaluate_progress(0.0) == 0.3
        assert goal.evaluate_progress(100.0) == 0.3

    def test_goal_types(self):
        """Test different goal types."""
        goal_types = ["economic", "social", "governance", "learning"]

        for goal_type in goal_types:
            goal = Goal(
                goal_id=f"goal_{goal_type}",
                goal_type=goal_type,
                description=f"A {goal_type} goal"
            )
            assert goal.goal_type == goal_type

    def test_goal_status_values(self):
        """Test different goal status values."""
        statuses = ["active", "achieved", "abandoned"]

        for status in statuses:
            goal = Goal(
                goal_id="goal_1",
                goal_type="economic",
                description="Test goal",
                status=status
            )
            assert goal.status == status


class TestGoalSystem:
    """Test GoalSystem class."""

    def test_goal_system_initialization(self):
        """Test GoalSystem initializes with empty lists."""
        system = GoalSystem()

        assert system.active_goals == []
        assert system.achieved_goals == []
        assert system.abandoned_goals == []

    def test_add_goal(self):
        """Test adding a goal to the system."""
        system = GoalSystem()
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Earn currency"
        )

        system.add_goal(goal)

        assert len(system.active_goals) == 1
        assert system.active_goals[0] == goal

    def test_add_multiple_goals(self):
        """Test adding multiple goals."""
        system = GoalSystem()
        goals = [
            Goal(goal_id="goal_1", goal_type="economic", description="Economic goal"),
            Goal(goal_id="goal_2", goal_type="social", description="Social goal"),
            Goal(goal_id="goal_3", goal_type="governance", description="Governance goal")
        ]

        for goal in goals:
            system.add_goal(goal)

        assert len(system.active_goals) == 3

    def test_update_progress_partial(self):
        """Test updating progress without achieving goal."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="economic", description="Test goal")
        system.add_goal(goal)

        system.update_progress("goal_1", 0.5)

        assert system.active_goals[0].progress == 0.5
        assert system.active_goals[0].status == "active"
        assert len(system.active_goals) == 1
        assert len(system.achieved_goals) == 0

    def test_update_progress_achieve_goal(self):
        """Test achieving a goal through progress update."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="economic", description="Test goal")
        system.add_goal(goal)

        system.update_progress("goal_1", 1.0)

        assert len(system.active_goals) == 0
        assert len(system.achieved_goals) == 1
        assert system.achieved_goals[0].status == "achieved"
        assert system.achieved_goals[0].progress == 1.0

    def test_update_progress_nonexistent_goal(self):
        """Test updating progress for nonexistent goal."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="economic", description="Test goal")
        system.add_goal(goal)

        system.update_progress("nonexistent_goal", 0.5)

        # Should not crash, original goal unchanged
        assert system.active_goals[0].progress == 0.0

    def test_abandon_goal(self):
        """Test abandoning a goal."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="economic", description="Test goal")
        system.add_goal(goal)

        system.abandon_goal("goal_1", "deadline_passed")

        assert len(system.active_goals) == 0
        assert len(system.abandoned_goals) == 1
        assert system.abandoned_goals[0].status == "abandoned: deadline_passed"

    def test_abandon_nonexistent_goal(self):
        """Test abandoning nonexistent goal."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="economic", description="Test goal")
        system.add_goal(goal)

        system.abandon_goal("nonexistent_goal", "reason")

        # Should not crash, original goal unchanged
        assert len(system.active_goals) == 1
        assert len(system.abandoned_goals) == 0

    def test_get_active_goals_by_priority(self):
        """Test getting active goals sorted by priority."""
        system = GoalSystem()
        goals = [
            Goal(goal_id="goal_1", goal_type="economic", description="Low", priority=0.3),
            Goal(goal_id="goal_2", goal_type="social", description="High", priority=0.9),
            Goal(goal_id="goal_3", goal_type="governance", description="Med", priority=0.6)
        ]

        for goal in goals:
            system.add_goal(goal)

        sorted_goals = system.get_active_goals_by_priority()

        assert len(sorted_goals) == 3
        assert sorted_goals[0].priority == 0.9
        assert sorted_goals[1].priority == 0.6
        assert sorted_goals[2].priority == 0.3
        assert sorted_goals[0].goal_id == "goal_2"

    def test_get_active_goals_empty(self):
        """Test getting active goals when empty."""
        system = GoalSystem()

        sorted_goals = system.get_active_goals_by_priority()

        assert sorted_goals == []

    def test_goal_lifecycle(self):
        """Test full lifecycle of a goal."""
        system = GoalSystem()
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Earn 1000 currency",
            priority=0.8
        )

        # Add goal
        system.add_goal(goal)
        assert len(system.active_goals) == 1
        assert len(system.achieved_goals) == 0

        # Update progress
        system.update_progress("goal_1", 0.5)
        assert system.active_goals[0].progress == 0.5

        # Achieve goal
        system.update_progress("goal_1", 1.0)
        assert len(system.active_goals) == 0
        assert len(system.achieved_goals) == 1

    def test_multiple_goal_management(self):
        """Test managing multiple goals simultaneously."""
        system = GoalSystem()

        # Add 3 goals
        for i in range(3):
            system.add_goal(Goal(
                goal_id=f"goal_{i}",
                goal_type="economic",
                description=f"Goal {i}"
            ))

        assert len(system.active_goals) == 3

        # Achieve one
        system.update_progress("goal_0", 1.0)
        assert len(system.active_goals) == 2
        assert len(system.achieved_goals) == 1

        # Abandon one
        system.abandon_goal("goal_1", "changed_priorities")
        assert len(system.active_goals) == 1
        assert len(system.abandoned_goals) == 1

        # One remains active
        assert system.active_goals[0].goal_id == "goal_2"

    def test_evaluate_progress_with_none_current_value(self):
        """Test evaluate_progress when current_value is None."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Test goal",
            target_value=1000.0,
            progress=0.3
        )

        # When current_value is None, should return existing progress
        assert goal.evaluate_progress(None) == 0.3

    def test_evaluate_progress_with_zero_current_value(self):
        """Test evaluate_progress when current_value is 0 (falsy but not None)."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Test goal",
            target_value=1000.0
        )

        # When current_value is 0, should calculate 0/1000 = 0.0
        assert goal.evaluate_progress(0.0) == 0.0
        assert goal.evaluate_progress(0) == 0.0

    def test_evaluate_progress_zero_target(self):
        """Test evaluate_progress when target_value is 0."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="learning",
            description="Test goal",
            target_value=0.0,  # Falsy target value
            progress=0.6
        )

        # Should return existing progress since target is falsy
        assert goal.evaluate_progress(100.0) == 0.6

    def test_evaluate_progress_no_target_with_current_value(self):
        """Test evaluate_progress when no target but current_value provided."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="learning",
            description="Learn skill",
            target_value=None,
            progress=0.5
        )

        # Without target_value, should return existing progress
        assert goal.evaluate_progress(100.0) == 0.5
        assert goal.evaluate_progress(0.0) == 0.5

    def test_goal_with_deadline_and_created_step(self):
        """Test goal with deadline_step and created_step."""
        goal = Goal(
            goal_id="goal_1",
            goal_type="economic",
            description="Time-limited goal",
            deadline_step=100,
            created_step=10
        )

        assert goal.deadline_step == 100
        assert goal.created_step == 10

    def test_update_progress_over_one(self):
        """Test that progress over 1.0 achieves goal."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="economic", description="Test")
        system.add_goal(goal)

        # Progress >= 1.0 should achieve goal
        system.update_progress("goal_1", 1.5)

        assert len(system.active_goals) == 0
        assert len(system.achieved_goals) == 1
        assert system.achieved_goals[0].progress == 1.5

    def test_abandon_with_detailed_reason(self):
        """Test abandoning goal with detailed reason string."""
        system = GoalSystem()
        goal = Goal(goal_id="goal_1", goal_type="governance", description="Test")
        system.add_goal(goal)

        reason = "deadline_passed_on_step_150"
        system.abandon_goal("goal_1", reason)

        assert len(system.abandoned_goals) == 1
        assert system.abandoned_goals[0].status == f"abandoned: {reason}"

    def test_get_active_goals_with_equal_priority(self):
        """Test sorting goals with equal priorities."""
        system = GoalSystem()
        goals = [
            Goal(goal_id="goal_1", goal_type="economic", description="A", priority=0.5),
            Goal(goal_id="goal_2", goal_type="social", description="B", priority=0.5),
            Goal(goal_id="goal_3", goal_type="governance", description="C", priority=0.5)
        ]

        for goal in goals:
            system.add_goal(goal)

        sorted_goals = system.get_active_goals_by_priority()

        # All should have same priority
        assert len(sorted_goals) == 3
        assert all(g.priority == 0.5 for g in sorted_goals)

    def test_goal_type_variations(self):
        """Test all expected goal types."""
        types = ["economic", "social", "governance", "learning"]

        for goal_type in types:
            goal = Goal(
                goal_id=f"goal_{goal_type}",
                goal_type=goal_type,
                description=f"Test {goal_type}",
                priority=0.7
            )
            assert goal.goal_type == goal_type
            assert goal.priority == 0.7
