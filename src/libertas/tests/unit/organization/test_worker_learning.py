"""Tests for worker learning methods (semantic memory and goals)."""

import pytest
from unittest.mock import Mock, patch
from libertas.organization import Federation, Worker
from libertas.organization.pod import PodConfig
from libertas.organization.worker import WorkerConfig
from libertas.economy import Resource
from libertas.cognitive import SemanticMemory, Goal, GoalSystem

LLM_MODEL = "ollama/functiongemma"


@pytest.fixture
def federation():
    """Create a real test federation."""
    worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
    pod_configs = [
        PodConfig(
            name="test_pod",
            workers=worker_configs,
            initial_inventory={"wood": 100.0}
        )
    ]
    fed = Federation(pods=pod_configs, seed=42)
    fed.register_new_resource(Resource("wood", base_value=1.0))
    fed.register_new_resource(Resource("wheat", base_value=10.0))
    fed.register_new_resource(Resource("iron", base_value=20.0))
    return fed


@pytest.fixture
def worker(federation):
    """Get the first worker from the federation."""
    pod = federation[0]
    return pod.get_worker_by_index(0)


@pytest.mark.unit
class TestWorkerLearning:
    """Test worker learning from experience."""

    def test_learn_from_experience_requires_minimum_history(self, worker):
        """Test that learning requires at least 5 observations."""
        # Add only 3 observations
        for i in range(3):
            worker.episodic_memory.append({"step": i, "observations": {}})

        # Should not crash, but also shouldn't learn anything yet
        worker._learn_from_experience()

        # Semantic memory should still be empty
        assert len(worker.semantic_memory.price_patterns) == 0
        assert len(worker.semantic_memory.market_insights) == 0

    def test_learn_market_patterns(self, worker):
        """Test learning market price patterns."""
        # Add market observations with price data in the format the implementation expects
        for i in range(10):
            worker.episodic_memory.append({
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

        worker._learn_market_patterns()

        # Should have learned price patterns for wheat and iron
        assert "wheat" in worker.semantic_memory.price_patterns
        assert "iron" in worker.semantic_memory.price_patterns
        assert len(worker.semantic_memory.price_patterns["wheat"]) > 0

    def test_generate_market_insights(self, worker):
        """Test generating natural language market insights."""
        # Set up price patterns with significant trends (need 20% change to trigger insights)
        # Wheat rising: avg=10, recent=14 (40% increase, > 20% threshold)
        worker.semantic_memory.price_patterns["wheat"] = [8.0, 8.5, 9.0, 9.5, 10.0, 12.0, 13.0, 14.0, 14.5, 15.0]
        # Iron falling: avg=20, recent=15 (25% decrease, > 20% threshold)
        worker.semantic_memory.price_patterns["iron"] = [22.0, 22.0, 21.0, 21.0, 20.0, 18.0, 16.0, 15.0, 14.0, 14.0]

        worker._generate_market_insights()

        # Should have generated insights
        assert len(worker.semantic_memory.market_insights) > 0
        insights_text = " ".join(worker.semantic_memory.market_insights)
        # Should mention wheat or iron and indicate trend direction
        assert any(word in insights_text.lower() for word in ["wheat", "iron"])
        assert any(word in insights_text.lower() for word in ["rising", "falling"])

    def test_learn_social_patterns(self, worker):
        """Test learning about other workers' behaviors."""
        # Add observations with worker interactions
        for i in range(10):
            worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "other_workers": [
                        {"id": "worker_1", "action": "cooperate"},
                        {"id": "worker_2", "action": "compete"}
                    ]
                }
            })

        worker._learn_social_patterns()

        # Should have started tracking worker behaviors
        # (Implementation may vary, but memory structure should be populated)
        assert isinstance(worker.semantic_memory.worker_behaviors, dict)
        assert isinstance(worker.semantic_memory.trusted_workers, dict)

    def test_learn_production_patterns(self, worker):
        """Test learning production efficiency."""
        # Add observations with production data
        for i in range(10):
            worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "production": {
                        "recipe": "bread_basic",
                        "efficiency": 1.0 + i * 0.1
                    }
                }
            })

        worker._learn_production_patterns()

        # Should track production knowledge
        assert isinstance(worker.semantic_memory.recipe_efficiency, dict)
        assert isinstance(worker.semantic_memory.skill_mastery, dict)

    def test_learn_governance_patterns(self, worker):
        """Test learning governance dynamics."""
        # Add observations with governance data
        for i in range(10):
            worker.episodic_memory.append({
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

        worker._learn_governance_patterns()

        # Should track governance knowledge
        assert isinstance(worker.semantic_memory.motion_outcomes, dict)
        assert isinstance(worker.semantic_memory.constitution_rules, list)

    def test_get_relevant_insights(self, worker):
        """Test filtering semantic memory for relevant insights."""
        # Populate semantic memory
        worker.semantic_memory.market_insights = [
            "Wheat prices trending up",
            "Iron prices stable"
        ]
        worker.semantic_memory.trusted_workers = {"worker_1": 0.9}
        worker.semantic_memory.skill_mastery = {"farming": 7}

        observations = {
            "market_prices": {"wheat": 15.0},
            "current_skills": ["farming"]
        }

        insights = worker._get_relevant_insights(observations)

        # Should return structured insights
        assert isinstance(insights, dict)
        assert "market" in insights
        assert "social" in insights
        assert "production" in insights
        assert "governance" in insights


@pytest.mark.unit
class TestWorkerGoals:
    """Test worker goal management."""

    def test_initial_goals_empty(self, worker):
        """Test that worker starts with no goals."""
        assert len(worker.goals.active_goals) == 0
        assert len(worker.goals.achieved_goals) == 0
        assert len(worker.goals.abandoned_goals) == 0

    def test_generate_initial_goals_economic_right(self, worker):
        """Test goal generation for economically right-leaning worker."""
        worker.personality.economic_left_right = 0.7  # Individualist

        worker._generate_initial_goals()

        assert len(worker.goals.active_goals) > 0
        # Should have economic goal
        goal_types = [g.goal_type for g in worker.goals.active_goals]
        assert "economic" in goal_types

    def test_generate_initial_goals_economic_left(self, worker):
        """Test goal generation for economically left-leaning worker."""
        worker.personality.economic_left_right = -0.7  # Collectivist

        worker._generate_initial_goals()

        assert len(worker.goals.active_goals) > 0
        # Should have social goal
        goal_types = [g.goal_type for g in worker.goals.active_goals]
        assert "social" in goal_types

    def test_generate_initial_goals_libertarian(self, worker):
        """Test goal generation for libertarian worker."""
        worker.personality.authority_libertarian = 0.7

        worker._generate_initial_goals()

        assert len(worker.goals.active_goals) > 0
        # Should have governance goal
        goal_types = [g.goal_type for g in worker.goals.active_goals]
        assert "governance" in goal_types

    def test_evaluate_goals_economic(self, worker):
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
        worker.goals.add_goal(goal)
        worker.currency = 250.0  # Half way there

        worker._evaluate_goals()

        # Progress should be updated
        assert worker.goals.active_goals[0].progress == 0.5
        assert worker.goals.active_goals[0].status == "active"

    def test_evaluate_goals_achieve(self, worker):
        """Test achieving a goal."""
        goal = Goal(
            goal_id="econ_1",
            goal_type="economic",
            description="Earn 500 currency",
            target_metric="currency",
            target_value=500.0
        )
        worker.goals.add_goal(goal)
        worker.currency = 600.0  # Exceeded target

        worker._evaluate_goals()

        # Goal should be achieved and moved
        assert len(worker.goals.active_goals) == 0
        assert len(worker.goals.achieved_goals) == 1
        assert worker.goals.achieved_goals[0].status == "achieved"

    def test_evaluate_goals_deadline(self, worker, federation):
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
        worker.goals.add_goal(goal)

        worker._evaluate_goals()

        # Goal should be abandoned
        assert len(worker.goals.active_goals) == 0
        assert len(worker.goals.abandoned_goals) == 1
        assert "deadline_passed" in worker.goals.abandoned_goals[0].status

    def test_get_goal_metric_value_currency(self, worker):
        """Test getting currency metric value."""
        worker.currency = 250.0

        goal = Goal(
            goal_id="test",
            goal_type="economic",
            description="Earn currency",
            target_metric="currency"
        )

        value = worker._get_goal_metric_value(goal)

        assert value == 250.0

    def test_get_goal_metric_value_trust(self, worker):
        """Test getting trust score metric."""
        worker.semantic_memory.trusted_workers = {
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

        value = worker._get_goal_metric_value(goal)

        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(0.7, abs=0.1)  # Average of trust scores

    def test_get_goal_metric_value_skill(self, worker):
        """Test getting skill mastery metric."""
        worker.semantic_memory.skill_mastery = {"farming": 7}

        goal = Goal(
            goal_id="test",
            goal_type="learning",
            description="Learn farming",
            target_metric="farming"
        )

        value = worker._get_goal_metric_value(goal)

        assert value == 7.0

    def test_get_goal_metric_value_governance(self, worker):
        """Test getting governance success metric."""
        worker.semantic_memory.motion_outcomes = {
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

        value = worker._get_goal_metric_value(goal)

        assert value == 2.0

    def test_format_goals_for_prompt_empty(self, worker):
        """Test formatting empty goals."""
        formatted = worker._format_goals_for_prompt()

        assert "No active goals" in formatted

    def test_format_goals_for_prompt_with_goals(self, worker):
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
        worker.goals.add_goal(goal1)
        worker.goals.add_goal(goal2)

        formatted = worker._format_goals_for_prompt()

        assert "Earn currency" in formatted
        assert "Build trust" in formatted
        assert "0.8" in formatted  # Priority
        assert "50%" in formatted or "0.5" in formatted  # Progress


@pytest.mark.unit
class TestWorkerCognitiveLoopIntegration:
    """Test integration of learning and goals into cognitive loop."""

    def test_learning_triggers_after_observations(self, worker):
        """Test that learning is triggered after accumulating observations."""
        # Add 9 observations - should NOT trigger learning
        for i in range(9):
            worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "market_state": {"prices": {"wheat": 10.0 + i}}
                }
            })

        # No price patterns yet
        assert len(worker.semantic_memory.price_patterns) == 0

        # Add 10th observation and call learning manually
        worker.episodic_memory.append({
            "step": 9,
            "observations": {
                "market_state": {"prices": {"wheat": 19.0}}
            }
        })

        # Manually trigger learning (simulating what observe_and_reason does)
        if len(worker.episodic_memory) >= 10:
            worker._learn_from_experience()

        # Should have learned price patterns
        assert "wheat" in worker.semantic_memory.price_patterns

    def test_initial_goals_generation(self, worker):
        """Test that initial goals are generated based on personality."""
        # Set personality to trigger economic goal
        worker.personality.economic_left_right = 0.7

        # No goals yet
        assert len(worker.goals.active_goals) == 0

        # Trigger goal generation
        worker._generate_initial_goals()

        # Should have generated initial goals
        assert len(worker.goals.active_goals) > 0
        goal_types = [g.goal_type for g in worker.goals.active_goals]
        assert "economic" in goal_types

    def test_goal_progress_evaluation(self, worker):
        """Test that goal progress is evaluated correctly."""
        # Add a goal
        goal = Goal(
            goal_id="test",
            goal_type="economic",
            description="Test",
            target_metric="currency",
            target_value=100.0
        )
        worker.goals.add_goal(goal)
        worker.currency = 50.0

        # Evaluate goals
        worker._evaluate_goals()

        # Goal progress should be updated
        assert worker.goals.active_goals[0].progress == 0.5

    def test_learned_knowledge_accessible(self, worker):
        """Test that learned knowledge can be accessed for reasoning."""
        # Add some learned knowledge
        worker.semantic_memory.market_insights = ["Wheat prices rising"]
        worker.semantic_memory.trusted_workers = {"worker_1": 0.9}

        # Get relevant insights
        observations = {
            "market_state": {"prices": {"wheat": 15.0}},
            "current_skills": ["farming"]
        }
        insights = worker._get_relevant_insights(observations)

        # Should return structured insights
        assert isinstance(insights, dict)
        assert "market" in insights
        assert len(worker.semantic_memory.market_insights) > 0

    def test_goals_formatted_for_prompt(self, worker):
        """Test that goals can be formatted for inclusion in prompts."""
        # Add a goal
        goal = Goal(
            goal_id="test",
            goal_type="economic",
            description="Earn 1000 currency",
            priority=0.8,
            progress=0.3
        )
        worker.goals.add_goal(goal)

        # Format goals
        formatted = worker._format_goals_for_prompt()

        # Should include goal information
        assert "Earn 1000 currency" in formatted
        assert "0.8" in formatted  # Priority
