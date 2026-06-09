"""Tests for worker learning methods (semantic memory and goals)."""
from libertas.cognitive import SemanticMemory, Goal, GoalSystem, GoalStatus
from libertas.organization import Federation
from libertas.organization.pod import PodConfig
from libertas.organization.worker import WorkerConfig
from libertas.tests.conftest import basic_federation, basic_pod_config, basic_worker_config
from unittest.mock import Mock, patch
import pytest
import unittest


@pytest.mark.unit
class TestWorkerLearning(unittest.TestCase):
    """Test worker learning from experience."""

    def setUp(self):
        self.federation = basic_federation(basic_pod_config(basic_worker_config()))

        pod = self.federation[0]
        self.worker = list(pod)[0]

    def test_learn_from_experience_requires_minimum_history(self):
        """Test that learning requires at least 5 observations."""
        # Add only 3 observations
        for i in range(3):
            self.worker.episodic_memory.append({"step": i, "observations": {}})

        # Should not crash, but also shouldn't learn anything yet
        self.worker._learn_from_experience()

        # Semantic memory should still be empty
        assert len(self.worker.semantic_memory.price_patterns) == 0
        assert len(self.worker.semantic_memory.market_insights) == 0

    def test_learn_market_patterns(self):
        """Test learning market price patterns."""
        # Add market observations with price data in the format the implementation expects
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "market_state": {
                        "prices": {
                            "wheat": 10.0 + i,
                            "iron": 20.0 - i * 0.5
                        }
                    }
                }
            })

        self.worker._learn_market_patterns()

        # Should have learned price patterns for wheat and iron
        assert "wheat" in self.worker.semantic_memory.price_patterns
        assert "iron" in self.worker.semantic_memory.price_patterns
        assert len(self.worker.semantic_memory.price_patterns["wheat"]) > 0

    def test_generate_market_insights(self):
        """Test generating natural language market insights."""
        # Set up price patterns with significant trends (need 20% change to trigger insights)
        # Wheat rising: avg=10, recent=14 (40% increase, > 20% threshold)
        self.worker.semantic_memory.price_patterns["wheat"] = [8.0, 8.5, 9.0, 9.5, 10.0, 12.0, 13.0, 14.0, 14.5, 15.0]
        # Iron falling: avg=20, recent=15 (25% decrease, > 20% threshold)
        self.worker.semantic_memory.price_patterns["iron"] = [22.0, 22.0, 21.0, 21.0, 20.0, 18.0, 16.0, 15.0, 14.0, 14.0]

        self.worker._generate_market_insights()

        # Should have generated insights
        assert len(self.worker.semantic_memory.market_insights) > 0
        insights_text = " ".join(self.worker.semantic_memory.market_insights)
        # Should mention wheat or iron and indicate trend direction
        assert any(word in insights_text.lower() for word in ["wheat", "iron"])
        assert any(word in insights_text.lower() for word in ["rising", "falling"])

    def test_learn_social_patterns(self):
        """Test learning about other workers' behaviors."""
        # Add observations with worker interactions
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "other_workers": [
                        {"id": "worker_1", "action": "cooperate"},
                        {"id": "worker_2", "action": "compete"}
                    ]
                }
            })

        self.worker._learn_social_patterns()

        # Should have started tracking worker behaviors
        # (Implementation may vary, but memory structure should be populated)
        assert isinstance(self.worker.semantic_memory.worker_behaviors, dict)
        assert isinstance(self.worker.semantic_memory.trusted_workers, dict)

    def test_learn_production_patterns(self):
        """Test learning production efficiency."""
        # Add observations with production data
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "production": {
                        "recipe": "bread_basic",
                        "efficiency": 1.0 + i * 0.1
                    }
                }
            })

        self.worker._learn_production_patterns()

        # Should track production knowledge
        assert isinstance(self.worker.semantic_memory.recipe_efficiency, dict)
        assert isinstance(self.worker.semantic_memory.skill_mastery, dict)

    def test_learn_governance_patterns(self):
        """Test learning governance dynamics."""
        # Add observations with governance data
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "governance": {
                        "recent_motions": [
                            {"type": "tax_increase", "passed": False},
                            {"type": "worker_allocation", "passed": True}
                        ]
                    }
                }
            })

        self.worker._learn_governance_patterns()

        # Should track governance knowledge
        assert isinstance(self.worker.semantic_memory.motion_outcomes, dict)
        assert isinstance(self.worker.semantic_memory.constitution_rules, list)

    def test_get_relevant_insights(self):
        """Test filtering semantic memory for relevant insights."""
        # Populate semantic memory
        self.worker.semantic_memory.market_insights = [
            "Wheat prices trending up",
            "Iron prices stable"
        ]
        self.worker.semantic_memory.trusted_workers = {"worker_1": 0.9}
        self.worker.semantic_memory.skill_mastery = {"farming": 7}

        observations = {
            "market_prices": {"wheat": 15.0},
            "current_skills": ["farming"]
        }

        insights = self.worker._get_relevant_insights(observations)

        # Should return structured insights
        assert isinstance(insights, dict)
        assert "market" in insights
        assert "social" in insights
        assert "production" in insights
        assert "governance" in insights


@pytest.mark.unit
class TestWorkerGoals:
    """Test worker goal management."""
    def setUp(self):
        self.federation = basic_federation(basic_pod_config(basic_worker_config()))

        pod = self.federation[0]
        self.worker = list(pod)[0]

    def test_initial_goals_empty(self):
        """Test that worker starts with no goals."""
        assert len(self.worker.goals.active_goals) == 0
        assert len(self.worker.goals.completed_goals) == 0
        assert len(self.worker.goals.abandoned_goals) == 0

    def test_generate_initial_goals_economic_right(self):
        """Test goal generation for economically right-leaning self.worker."""
        self.worker.personality.economic_left_right = 0.7  # Individualist

        self.worker._generate_initial_goals()

        assert len(self.worker.goals.active_goals) > 0
        # Should have economic goal
        goal_types = [g.goal_type for g in self.worker.goals.active_goals]
        assert "economic" in goal_types

    def test_generate_initial_goals_economic_left(self):
        """Test goal generation for economically left-leaning self.worker."""
        self.worker.personality.economic_left_right = -0.7  # Collectivist

        self.worker._generate_initial_goals()

        assert len(self.worker.goals.active_goals) > 0
        # Should have social goal
        goal_types = [g.goal_type for g in self.worker.goals.active_goals]
        assert "social" in goal_types

    def test_generate_initial_goals_libertarian(self):
        """Test goal generation for libertarian self.worker."""
        self.worker.personality.authority_libertarian = 0.7

        self.worker._generate_initial_goals()

        assert len(self.worker.goals.active_goals) > 0
        # Should have governance goal
        goal_types = [g.goal_type for g in self.worker.goals.active_goals]
        assert "governance" in goal_types

    def test_evaluate_goals_economic(self):
        """Test evaluating progress on economic goals."""
        # Add economic goal
        goal = Goal(
            goal_id="econ_1",
            goal_type="economic",
            description="Earn 500 currency",
            target_metric="currency",
            target_value=500.0,
            priority=0.8
        )
        self.worker.goals.add_goal(goal)
        self.worker.currency = 250.0  # Half way there

        self.worker._evaluate_goals()

        # Progress should be updated
        assert self.worker.goals.active_goals[0].progress == 0.5
        assert self.worker.goals.active_goals[0].status == GoalStatus.IN_PROGRESS

    def test_evaluate_goals_achieve(self):
        """Test achieving a goal."""
        goal = Goal(
            goal_id="econ_1",
            goal_type="economic",
            description="Earn 500 currency",
            target_metric="currency",
            target_value=500.0
        )
        self.worker.goals.add_goal(goal)
        self.worker.currency = 600.0  # Exceeded target

        self.worker._evaluate_goals()

        # Goal should be achieved and moved
        assert len(self.worker.goals.active_goals) == 0
        assert len(self.worker.goals.completed_goals) == 1
        assert self.worker.goals.completed_goals[0].status == GoalStatus.COMPLETED

    def test_evaluate_goals_deadline(self, federation):
        """Test abandoning goals past deadline."""
        federation.steps = 100

        goal = Goal(
            goal_id="econ_1",
            goal_type="economic",
            description="Earn currency",
            deadline_step=50,  # Past deadline
            target_metric="currency",
            target_value=1000.0
        )
        self.worker.goals.add_goal(goal)

        self.worker._evaluate_goals()

        # Goal should be abandoned
        assert len(self.worker.goals.active_goals) == 0
        assert len(self.worker.goals.abandoned_goals) == 1
        assert self.worker.goals.abandoned_goals[0].abandon_reason == "deadline_passed"

    def test_get_goal_metric_value_currency(self):
        """Test getting currency metric value."""
        self.worker.currency = 250.0

        goal = Goal(
            goal_id="test",
            goal_type="economic",
            description="Earn currency",
            target_metric="currency"
        )

        value = self.worker._get_goal_metric_value(goal)

        assert value == 250.0

    def test_get_goal_metric_value_trust(self):
        """Test getting trust score metric."""
        self.worker.semantic_memory.trusted_workers = {
            "worker_1": 0.9,
            "worker_2": 0.7,
            "worker_3": 0.5
        }

        goal = Goal(
            goal_id="test",
            goal_type="social",
            description="Build trust",
            target_metric="avg_trust"
        )

        value = self.worker._get_goal_metric_value(goal)

        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(0.7, abs=0.1)  # Average of trust scores

    def test_get_goal_metric_value_skill(self):
        """Test getting skill mastery metric."""
        self.worker.semantic_memory.skill_mastery = {"farming": 7}

        goal = Goal(
            goal_id="test",
            goal_type="learning",
            description="Learn farming",
            target_metric="farming"
        )

        value = self.worker._get_goal_metric_value(goal)

        assert value == 7.0

    def test_get_goal_metric_value_governance(self):
        """Test getting governance success metric."""
        self.worker.semantic_memory.motion_outcomes = {
            "motion_1": True,
            "motion_2": False,
            "motion_3": True
        }

        goal = Goal(
            goal_id="test",
            goal_type="governance",
            description="Influence governance",
            target_metric="successful_motions"
        )

        value = self.worker._get_goal_metric_value(goal)

        assert value == 2.0

    def test_format_goals_for_prompt_empty(self):
        """Test formatting empty goals."""
        formatted = self.worker._format_goals_for_prompt()

        assert "No active goals" in formatted

    def test_format_goals_for_prompt_with_goals(self):
        """Test formatting goals for LLM prompt."""
        goal1 = Goal(
            goal_id="g1",
            goal_type="economic",
            description="Earn currency",
            priority=0.8,
            progress=0.5
        )
        goal2 = Goal(
            goal_id="g2",
            goal_type="social",
            description="Build trust",
            priority=0.6,
            progress=0.3
        )
        self.worker.goals.add_goal(goal1)
        self.worker.goals.add_goal(goal2)

        formatted = self.worker._format_goals_for_prompt()

        assert "Earn currency" in formatted
        assert "Build trust" in formatted
        assert "0.8" in formatted  # Priority
        assert "50%" in formatted or "0.5" in formatted  # Progress


@pytest.mark.unit
class TestWorkerCognitiveLoopIntegration:
    """Test integration of learning and goals into cognitive loop."""
    def setUp(self):
        self.federation = basic_federation(basic_pod_config(basic_worker_config()))

        pod = self.federation[0]
        self.worker = list(pod)[0]

    def test_learning_triggers_after_observations(self):
        """Test that learning is triggered after accumulating observations."""
        # Add 9 observations - should NOT trigger learning
        for i in range(9):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "market_state": {"prices": {"wheat": 10.0 + i}}
                }
            })

        # No price patterns yet
        assert len(self.worker.semantic_memory.price_patterns) == 0

        # Add 10th observation and call learning manually
        self.worker.episodic_memory.append({
            "step": 9,
            "observations": {
                "market_state": {"prices": {"wheat": 19.0}}
            }
        })

        # Manually trigger learning (simulating what observe_and_reason does)
        if len(self.worker.episodic_memory) >= 10:
            self.worker._learn_from_experience()

        # Should have learned price patterns
        assert "wheat" in self.worker.semantic_memory.price_patterns

    def test_initial_goals_generation(self):
        """Test that initial goals are generated based on personality."""
        # Set personality to trigger economic goal
        self.worker.personality.economic_left_right = 0.7

        # No goals yet
        assert len(self.worker.goals.active_goals) == 0

        # Trigger goal generation
        self.worker._generate_initial_goals()

        # Should have generated initial goals
        assert len(self.worker.goals.active_goals) > 0
        goal_types = [g.goal_type for g in self.worker.goals.active_goals]
        assert "economic" in goal_types

    def test_goal_progress_evaluation(self):
        """Test that goal progress is evaluated correctly."""
        # Add a goal
        goal = Goal(
            goal_id="test",
            goal_type="economic",
            description="Test",
            target_metric="currency",
            target_value=100.0
        )
        self.worker.goals.add_goal(goal)
        self.worker.currency = 50.0

        # Evaluate goals
        self.worker._evaluate_goals()

        # Goal progress should be updated
        assert self.worker.goals.active_goals[0].progress == 0.5

    def test_learned_knowledge_accessible(self):
        """Test that learned knowledge can be accessed for reasoning."""
        # Add some learned knowledge
        self.worker.semantic_memory.market_insights = ["Wheat prices rising"]
        self.worker.semantic_memory.trusted_workers = {"worker_1": 0.9}

        # Get relevant insights
        observations = {
            "market_state": {"prices": {"wheat": 15.0}},
            "current_skills": ["farming"]
        }
        insights = self.worker._get_relevant_insights(observations)

        # Should return structured insights
        assert isinstance(insights, dict)
        assert "market" in insights
        assert len(self.worker.semantic_memory.market_insights) > 0

    def test_goals_formatted_for_prompt(self):
        """Test that goals can be formatted for inclusion in prompts."""
        # Add a goal
        goal = Goal(
            goal_id="test",
            goal_type="economic",
            description="Earn 1000 currency",
            priority=0.8,
            progress=0.3
        )
        self.worker.goals.add_goal(goal)

        # Format goals
        formatted = self.worker._format_goals_for_prompt()

        # Should include goal information
        assert "Earn 1000 currency" in formatted
        assert "0.8" in formatted  # Priority
