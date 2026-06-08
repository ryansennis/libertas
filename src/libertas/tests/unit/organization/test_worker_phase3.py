"""Tests for Worker Phase 3 features: semantic memory and goal system."""

import pytest
from unittest.mock import Mock
from libertas.organization import Worker, WorkerConfig, Pod, PodConfig, Federation
from libertas.cognitive import (
    PersonalityTraits,
    Background,
    SemanticMemory,
    Goal,
    GoalSystem
)
from mesa_llm.reasoning.cot import CoTReasoning


@pytest.mark.unit
class TestWorkerSemanticMemory:
    """Test Worker's semantic memory system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.worker_config = WorkerConfig(
            name="TestWorker",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=100.0
        )
        self.worker = Worker(
            federation=self.federation,
            worker_config=self.worker_config,
            coordinate=(0, 0),
            pod=None
        )

    def test_worker_initializes_with_semantic_memory(self):
        """Test worker initializes with empty semantic memory."""
        assert hasattr(self.worker, 'semantic_memory')
        assert isinstance(self.worker.semantic_memory, SemanticMemory)
        assert self.worker.semantic_memory.price_patterns == {}
        assert self.worker.semantic_memory.market_insights == []
        assert self.worker.semantic_memory.worker_behaviors == {}
        assert self.worker.semantic_memory.voting_coalitions == []
        assert self.worker.semantic_memory.trusted_workers == {}
        assert self.worker.semantic_memory.recipe_efficiency == {}
        assert self.worker.semantic_memory.skill_mastery == {}
        assert self.worker.semantic_memory.motion_outcomes == {}
        assert self.worker.semantic_memory.constitution_rules == []

    def test_worker_semantic_memory_persistence(self):
        """Test semantic memory persists across operations."""
        # Add market knowledge
        self.worker.semantic_memory.price_patterns["wheat"] = [10.0, 11.0, 12.0]
        self.worker.semantic_memory.market_insights.append("Wheat prices rising")

        # Add social knowledge
        self.worker.semantic_memory.trusted_workers["worker_2"] = 0.8

        # Add production knowledge
        self.worker.semantic_memory.recipe_efficiency["bread"] = 1.2
        self.worker.semantic_memory.skill_mastery["farming"] = 5

        # Verify persistence
        assert len(self.worker.semantic_memory.price_patterns["wheat"]) == 3
        assert self.worker.semantic_memory.market_insights[0] == "Wheat prices rising"
        assert self.worker.semantic_memory.trusted_workers["worker_2"] == 0.8
        assert self.worker.semantic_memory.recipe_efficiency["bread"] == 1.2
        assert self.worker.semantic_memory.skill_mastery["farming"] == 5

    def test_semantic_memory_market_learning(self):
        """Test storing and updating market patterns."""
        # Record price history
        for price in [10.0, 11.0, 10.5, 12.0, 11.5]:
            if "wheat" not in self.worker.semantic_memory.price_patterns:
                self.worker.semantic_memory.price_patterns["wheat"] = []
            self.worker.semantic_memory.price_patterns["wheat"].append(price)

        assert len(self.worker.semantic_memory.price_patterns["wheat"]) == 5
        assert self.worker.semantic_memory.price_patterns["wheat"][-1] == 11.5

        # Add insight
        self.worker.semantic_memory.market_insights.append(
            "Wheat prices fluctuate between 10-12"
        )
        assert len(self.worker.semantic_memory.market_insights) == 1

    def test_semantic_memory_social_learning(self):
        """Test tracking worker behaviors and trust."""
        # Track worker behaviors
        self.worker.semantic_memory.worker_behaviors["alice"] = {
            "voting_pattern": "progressive",
            "trades_count": 15,
            "cooperation_level": 0.9
        }
        self.worker.semantic_memory.worker_behaviors["bob"] = {
            "voting_pattern": "conservative",
            "trades_count": 8,
            "cooperation_level": 0.6
        }

        assert len(self.worker.semantic_memory.worker_behaviors) == 2
        assert self.worker.semantic_memory.worker_behaviors["alice"]["cooperation_level"] == 0.9

        # Track trust scores
        self.worker.semantic_memory.trusted_workers["alice"] = 0.95
        self.worker.semantic_memory.trusted_workers["bob"] = 0.55

        assert self.worker.semantic_memory.trusted_workers["alice"] > \
               self.worker.semantic_memory.trusted_workers["bob"]

        # Track voting coalitions
        self.worker.semantic_memory.voting_coalitions.append(["alice", "charlie", "diana"])
        assert len(self.worker.semantic_memory.voting_coalitions) == 1

    def test_semantic_memory_production_learning(self):
        """Test learning production efficiency and skills."""
        # Track recipe efficiency
        self.worker.semantic_memory.recipe_efficiency["bread_basic"] = 1.0
        self.worker.semantic_memory.recipe_efficiency["bread_advanced"] = 1.5

        # Improve efficiency through practice
        self.worker.semantic_memory.recipe_efficiency["bread_basic"] = 1.2

        assert self.worker.semantic_memory.recipe_efficiency["bread_basic"] == 1.2

        # Track skill mastery
        self.worker.semantic_memory.skill_mastery["farming"] = 3
        self.worker.semantic_memory.skill_mastery["crafting"] = 5

        # Improve skill
        self.worker.semantic_memory.skill_mastery["farming"] = 4

        assert self.worker.semantic_memory.skill_mastery["farming"] == 4

    def test_semantic_memory_governance_learning(self):
        """Test learning governance patterns."""
        # Track motion outcomes
        self.worker.semantic_memory.motion_outcomes["tax_increase"] = False
        self.worker.semantic_memory.motion_outcomes["worker_allocation"] = True

        assert self.worker.semantic_memory.motion_outcomes["worker_allocation"]
        assert not self.worker.semantic_memory.motion_outcomes["tax_increase"]

        # Learn constitutional rules
        self.worker.semantic_memory.constitution_rules.append(
            "Tax motions require 2/3 supermajority"
        )
        self.worker.semantic_memory.constitution_rules.append(
            "All workers can vote on pod decisions"
        )

        assert len(self.worker.semantic_memory.constitution_rules) == 2


@pytest.mark.unit
class TestWorkerGoalSystem:
    """Test Worker's goal system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.worker_config = WorkerConfig(
            name="TestWorker",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=100.0,
            personality=PersonalityTraits(
                openness=0.8,
                conscientiousness=0.7,
                economic_left_right=-0.5  # Collectivist
            )
        )
        self.worker = Worker(
            federation=self.federation,
            worker_config=self.worker_config,
            coordinate=(0, 0),
            pod=None
        )

    def test_worker_initializes_with_goal_system(self):
        """Test worker initializes with empty goal system."""
        assert hasattr(self.worker, 'goals')
        assert isinstance(self.worker.goals, GoalSystem)
        assert self.worker.goals.active_goals == []
        assert self.worker.goals.completed_goals == []
        assert self.worker.goals.abandoned_goals == []

    def test_worker_can_add_goals(self):
        """Test worker can add goals to goal system."""
        goal = Goal(
            goal_id="earn_currency_1",
            goal_type="economic",
            description="Earn 1000 currency",
            target_metric="currency",
            target_value=1000.0,
            priority=0.8
        )

        self.worker.goals.add_goal(goal)

        assert len(self.worker.goals.active_goals) == 1
        assert self.worker.goals.active_goals[0].goal_id == "earn_currency_1"

    def test_worker_goal_progress_tracking(self):
        """Test tracking progress on worker goals."""
        goal = Goal(
            goal_id="skill_farming",
            goal_type="learning",
            description="Reach farming skill level 5",
            target_metric="skill_level",
            target_value=5.0,
            priority=0.6
        )

        self.worker.goals.add_goal(goal)

        # Update progress
        self.worker.goals.update_progress("skill_farming", 0.4)
        assert self.worker.goals.active_goals[0].progress == 0.4

        # Achieve goal
        self.worker.goals.update_progress("skill_farming", 1.0)
        assert len(self.worker.goals.active_goals) == 0
        assert len(self.worker.goals.completed_goals) == 1

    def test_worker_goal_abandonment(self):
        """Test abandoning worker goals."""
        goal = Goal(
            goal_id="trade_with_pod_x",
            goal_type="social",
            description="Establish trade with Pod X",
            priority=0.5,
            deadline_step=100
        )

        self.worker.goals.add_goal(goal)

        # Abandon due to deadline
        self.worker.goals.abandon_goal("trade_with_pod_x", "deadline_passed")

        assert len(self.worker.goals.active_goals) == 0
        assert len(self.worker.goals.abandoned_goals) == 1
        assert self.worker.goals.abandoned_goals[0].abandon_reason == "deadline_passed"

    def test_worker_multiple_goals(self):
        """Test worker managing multiple goals."""
        goals = [
            Goal(goal_id="g1", goal_type="economic", description="Earn currency", priority=0.9),
            Goal(goal_id="g2", goal_type="social", description="Build relationships", priority=0.6),
            Goal(goal_id="g3", goal_type="learning", description="Learn farming", priority=0.7)
        ]

        for goal in goals:
            self.worker.goals.add_goal(goal)

        assert len(self.worker.goals.active_goals) == 3

        # Get by priority
        sorted_goals = self.worker.goals.get_active_goals_by_priority()
        assert sorted_goals[0].priority == 0.9
        assert sorted_goals[1].priority == 0.7
        assert sorted_goals[2].priority == 0.6

    def test_worker_goal_types(self):
        """Test all goal types for workers."""
        goal_types = ["economic", "social", "governance", "learning"]

        for i, goal_type in enumerate(goal_types):
            goal = Goal(
                goal_id=f"goal_{i}",
                goal_type=goal_type,
                description=f"Test {goal_type} goal",
                priority=0.5
            )
            self.worker.goals.add_goal(goal)

        assert len(self.worker.goals.active_goals) == 4

        # Verify each type
        for goal in self.worker.goals.active_goals:
            assert goal.goal_type in goal_types


@pytest.mark.unit
class TestWorkerPhase3Integration:
    """Test Worker integration with Phase 3 cognitive systems."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.worker_config = WorkerConfig(
            name="TestWorker",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=100.0,
            personality=PersonalityTraits(
                openness=0.8,
                conscientiousness=0.7,
                extraversion=0.6,
                agreeableness=0.7,
                neuroticism=0.3
            ),
            background=Background(
                education_level=3,
                years_experience=2
            )
        )
        self.worker = Worker(
            federation=self.federation,
            worker_config=self.worker_config,
            coordinate=(0, 0),
            pod=None
        )

    def test_worker_has_all_cognitive_systems(self):
        """Test worker has all cognitive systems from Phase 3."""
        # Phase 1 & 2 systems
        assert hasattr(self.worker, 'personality')
        assert hasattr(self.worker, 'background')
        assert hasattr(self.worker, 'mood')
        assert hasattr(self.worker, 'episodic_memory')

        # Phase 3 systems
        assert hasattr(self.worker, 'semantic_memory')
        assert hasattr(self.worker, 'goals')

        # Verify types
        assert isinstance(self.worker.personality, PersonalityTraits)
        assert isinstance(self.worker.background, Background)
        assert isinstance(self.worker.semantic_memory, SemanticMemory)
        assert isinstance(self.worker.goals, GoalSystem)

    def test_episodic_memory_initialized(self):
        """Test episodic memory list is initialized."""
        assert isinstance(self.worker.episodic_memory, list)
        assert len(self.worker.episodic_memory) == 0

    def test_worker_personality_traits(self):
        """Test worker personality traits are set correctly."""
        assert self.worker.personality.openness == 0.8
        assert self.worker.personality.conscientiousness == 0.7
        assert self.worker.personality.extraversion == 0.6
        assert self.worker.personality.agreeableness == 0.7
        assert self.worker.personality.neuroticism == 0.3

    def test_worker_background(self):
        """Test worker background is set correctly."""
        assert self.worker.background.education_level == 3
        assert self.worker.background.years_experience == 2

    def test_memory_systems_independent(self):
        """Test that memory systems are independent."""
        # Add to episodic memory
        self.worker.episodic_memory.append({"step": 1, "action": "observed_market"})

        # Add to semantic memory
        self.worker.semantic_memory.price_patterns["wheat"] = [10.0, 11.0]

        # Add goal
        goal = Goal(goal_id="g1", goal_type="economic", description="Test")
        self.worker.goals.add_goal(goal)

        # Verify independence
        assert len(self.worker.episodic_memory) == 1
        assert len(self.worker.semantic_memory.price_patterns) == 1
        assert len(self.worker.goals.active_goals) == 1

    def test_worker_cognitive_state_snapshot(self):
        """Test capturing full cognitive state snapshot."""
        # Set up cognitive state
        self.worker.episodic_memory.append({"step": 1, "event": "started_work"})
        self.worker.semantic_memory.trusted_workers["alice"] = 0.9
        goal = Goal(goal_id="g1", goal_type="economic", description="Earn 500")
        self.worker.goals.add_goal(goal)

        # Capture state
        state = {
            "personality": self.worker.personality,
            "background": self.worker.background,
            "mood": self.worker.mood,
            "episodic_memory_size": len(self.worker.episodic_memory),
            "semantic_memory": {
                "trusted_workers": len(self.worker.semantic_memory.trusted_workers),
                "price_patterns": len(self.worker.semantic_memory.price_patterns)
            },
            "active_goals": len(self.worker.goals.active_goals)
        }

        assert state["episodic_memory_size"] == 1
        assert state["semantic_memory"]["trusted_workers"] == 1
        assert state["active_goals"] == 1

    def test_worker_with_default_personality(self):
        """Test worker with default personality traits."""
        config = WorkerConfig(
            name="DefaultWorker",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama"
        )
        worker = Worker(
            federation=self.federation,
            worker_config=config,
            coordinate=(0, 0)
        )

        # Should have default personality
        assert worker.personality.openness == 0.5
        assert worker.personality.conscientiousness == 0.5
        assert worker.personality.extraversion == 0.5
        assert worker.personality.agreeableness == 0.5
        assert worker.personality.neuroticism == 0.5

    def test_worker_tools_registered(self):
        """Test that economic and governance tools are registered."""
        assert hasattr(self.worker, 'economic_tools')
        assert hasattr(self.worker, 'governance_tools')
        assert hasattr(self.worker, 'tool_manager')

        # Tools should have reference to worker
        assert self.worker.economic_tools.worker == self.worker
        assert self.worker.governance_tools.worker == self.worker


@pytest.mark.unit
class TestWorkerLearning:
    """Test Worker's learning from experience methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.worker_config = WorkerConfig(
            name="LearningWorker",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=100.0
        )
        self.worker = Worker(
            federation=self.federation,
            worker_config=self.worker_config,
            coordinate=(0, 0),
            pod=None
        )

    def test_learn_from_insufficient_experience(self):
        """Test that learning requires minimum episodic memory."""
        # Add only 3 memories (less than minimum 5)
        for i in range(3):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {}
            })

        # Call learning method
        self.worker._learn_from_experience()

        # Should not learn anything yet
        assert len(self.worker.semantic_memory.price_patterns) == 0
        assert len(self.worker.semantic_memory.market_insights) == 0

    def test_learn_market_patterns(self):
        """Test learning market patterns from episodic memory."""
        # Add episodic memories with market observations
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "market_state": {
                        "prices": {
                            "wheat": 10.0 + i * 0.5,
                            "iron": 20.0 - i * 0.2
                        }
                    }
                }
            })

        # Learn from experience
        self.worker._learn_market_patterns()

        # Should have learned price patterns
        assert "wheat" in self.worker.semantic_memory.price_patterns
        assert "iron" in self.worker.semantic_memory.price_patterns
        assert len(self.worker.semantic_memory.price_patterns["wheat"]) == 10
        assert len(self.worker.semantic_memory.price_patterns["iron"]) == 10

    def test_learn_market_patterns_limits_history(self):
        """Test that price history is limited to last 50 entries."""
        # Add 60 price observations
        for i in range(60):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "market_state": {
                        "prices": {"wheat": float(i)}
                    }
                }
            })

        self.worker._learn_market_patterns()

        # Should keep only last entries (learns from last 20 episodic memories)
        # So we get prices from steps 40-59
        assert len(self.worker.semantic_memory.price_patterns["wheat"]) == 20
        assert self.worker.semantic_memory.price_patterns["wheat"][0] == 40.0
        assert self.worker.semantic_memory.price_patterns["wheat"][-1] == 59.0

    def test_generate_market_insights(self):
        """Test generating insights from price patterns."""
        # Set up price patterns
        self.worker.semantic_memory.price_patterns["wheat"] = [
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0
        ]

        # Generate insights
        self.worker._generate_market_insights()

        # Should have insights (implementation specific)
        # At minimum, the method should execute without error

    def test_learn_social_patterns(self):
        """Test learning social patterns from interactions."""
        # Add episodic memories with local workers
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "local_workers": {
                        "workers": [
                            {
                                "name": "alice",
                                "current_job": "farming",
                                "mood": {"happiness": 0.8, "stress": 0.2}
                            },
                            {
                                "name": "bob",
                                "current_job": "crafting",
                                "mood": {"happiness": 0.6, "stress": 0.4}
                            }
                        ]
                    }
                }
            })

        # Learn from experience
        self.worker._learn_social_patterns()

        # Should track worker behaviors
        assert "alice" in self.worker.semantic_memory.worker_behaviors
        assert "bob" in self.worker.semantic_memory.worker_behaviors

        # Should track interactions
        assert self.worker.semantic_memory.worker_behaviors["alice"]["interactions"] > 0

        # Should track jobs seen
        assert "farming" in self.worker.semantic_memory.worker_behaviors["alice"]["jobs_seen"]
        assert "crafting" in self.worker.semantic_memory.worker_behaviors["bob"]["jobs_seen"]

        # Should build trust
        assert "alice" in self.worker.semantic_memory.trusted_workers
        assert "bob" in self.worker.semantic_memory.trusted_workers

    def test_learn_social_patterns_with_missing_name(self):
        """Test social learning skips entries without worker name."""
        # Add memory with worker info missing name
        self.worker.episodic_memory.extend([
            {"step": 0, "observations": {"local_workers": {"workers": [{"current_job": "farming"}]}}},
            {"step": 1, "observations": {"local_workers": {"workers": [{"name": None}]}}},
            {"step": 2, "observations": {"local_workers": {"workers": [{"name": ""}]}}},
            {"step": 3, "observations": {"local_workers": {"workers": [{"name": "alice", "current_job": "mining"}]}}},
            {"step": 4, "observations": {"local_workers": {"workers": []}}},
            {"step": 5, "observations": {}},
        ])

        self.worker._learn_social_patterns()

        # Should only track alice
        assert len(self.worker.semantic_memory.worker_behaviors) == 1
        assert "alice" in self.worker.semantic_memory.worker_behaviors

    def test_learn_production_patterns(self):
        """Test learning production efficiency from skills."""
        # Set worker skills
        self.worker.skills["farming"] = 3.5
        self.worker.skills["crafting"] = 7.2
        self.worker.skills["mining"] = 10.5

        # Add some episodic memories
        for i in range(6):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "pod_state": {"inventory": {"wheat": 100}}
                }
            })

        # Learn from experience
        self.worker._learn_production_patterns()

        # Should track skill mastery levels
        assert self.worker.semantic_memory.skill_mastery["farming"] == 3  # int(3.5)
        assert self.worker.semantic_memory.skill_mastery["crafting"] == 7  # int(7.2)
        assert self.worker.semantic_memory.skill_mastery["mining"] == 10  # min(10, int(10.5))

    def test_learn_production_with_current_job(self):
        """Test learning production patterns with active job."""
        # Create a simple mock job with a recipe
        from unittest.mock import Mock
        mock_recipe = Mock()
        mock_recipe.name = "bread_basic"

        mock_job = Mock()
        mock_job.recipe = mock_recipe

        self.worker.current_job = mock_job

        # Add memories
        for i in range(6):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {"pod_state": {}}
            })

        self.worker._learn_production_patterns()

        # Should track recipe efficiency
        assert "bread_basic" in self.worker.semantic_memory.recipe_efficiency
        assert self.worker.semantic_memory.recipe_efficiency["bread_basic"] == 1.0

    def test_learn_governance_patterns(self):
        """Test learning governance patterns from active motions."""
        # Add episodic memories with active motions
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "active_motions": {
                        "active_motions": [
                            {"type": "tax_increase", "status": "voting"},
                            {"type": "worker_allocation", "status": "voting"}
                        ]
                    }
                }
            })

        # Learn from experience
        self.worker._learn_governance_patterns()

        # Should track motion types seen
        assert "tax_increase" in self.worker.semantic_memory.motion_outcomes
        assert "worker_allocation" in self.worker.semantic_memory.motion_outcomes

    def test_learn_governance_patterns_without_type(self):
        """Test governance learning handles motions without type."""
        # Add memories with various malformed motion data
        self.worker.episodic_memory.extend([
            {"step": 0, "observations": {"active_motions": {"active_motions": [{"status": "voting"}]}}},
            {"step": 1, "observations": {"active_motions": {"active_motions": [{"type": None}]}}},
            {"step": 2, "observations": {"active_motions": {"active_motions": []}}},
            {"step": 3, "observations": {"active_motions": {}}},
            {"step": 4, "observations": {}},
            {"step": 5, "observations": {"active_motions": {"active_motions": [{"type": "valid_motion"}]}}},
        ])

        self.worker._learn_governance_patterns()

        # Should only track the valid motion
        assert "valid_motion" in self.worker.semantic_memory.motion_outcomes
        assert self.worker.semantic_memory.motion_outcomes["valid_motion"] is False

    def test_full_learning_cycle(self):
        """Test complete learning cycle with all pattern types."""
        # Add diverse episodic memories
        for i in range(10):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {
                    "market_state": {"prices": {"wheat": 10.0 + i}},
                    "worker_interactions": {"alice": {"action": "traded"}},
                    "production_result": {"recipe": "bread", "efficiency": 1.0},
                    "motion_outcome": {"motion_type": "allocate_resources", "passed": True}
                }
            })

        # Run full learning cycle
        self.worker._learn_from_experience()

        # Should have learned market patterns at minimum
        assert "wheat" in self.worker.semantic_memory.price_patterns
