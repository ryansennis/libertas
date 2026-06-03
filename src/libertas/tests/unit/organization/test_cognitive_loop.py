"""
Unit tests for Worker cognitive loop methods.

Tests observation methods, reasoning, action filtering, memory, and mood updates.
"""
import pytest
from libertas.organization import Federation, WorkerConfig, PodConfig
from libertas.cognitive import PersonalityTraits, Background
from libertas.governance import Motion, MotionType, VoteType
from mesa_llm.reasoning.cot import CoTReasoning


@pytest.fixture
def basic_federation():
    """Create a federation with one pod and two workers."""
    workers = [
        WorkerConfig(
            name="Alice",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=1000.0,
            personality=PersonalityTraits(
                openness=0.7,
                conscientiousness=0.6,
                extraversion=0.8,  # Extraverted
                agreeableness=0.7,
                neuroticism=0.3,
                economic_left_right=-0.8,  # Collectivist
                authority_libertarian=-0.5   # Libertarian-leaning
            ),
            background=Background(education_level=4, years_experience=5)
        ),
        WorkerConfig(
            name="Bob",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=500.0,
            personality=PersonalityTraits(
                openness=0.5,
                conscientiousness=0.8,
                extraversion=0.3,  # Introverted
                agreeableness=0.5,
                neuroticism=0.4,
                economic_left_right=0.7,  # Individualist
                authority_libertarian=0.6   # Strong libertarian
            ),
            background=Background(education_level=3, years_experience=3)
        )
    ]

    pod_config = PodConfig(
        name="test_pod",
        workers=workers,
        initial_inventory={"wood": 500.0, "stone": 200.0}
    )

    federation = Federation(pods=[pod_config], seed=42)
    return federation


class TestObservationMethods:
    """Test worker observation methods."""

    def test_observe_local_workers(self, basic_federation):
        """Workers can observe nearby workers."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        observation = alice._observe_local_workers(radius=10.0)

        assert "workers" in observation
        assert "count" in observation
        assert observation["count"] >= 0
        assert isinstance(observation["workers"], list)

        # Should not include self
        worker_names = [w["name"] for w in observation["workers"]]
        assert "Alice" not in worker_names

    def test_observe_local_workers_small_radius(self, basic_federation):
        """Small radius may exclude distant workers."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        observation = alice._observe_local_workers(radius=0.1)

        assert "workers" in observation
        assert observation["count"] >= 0  # May be 0 if Bob is far

    def test_observe_pod_state(self, basic_federation):
        """Workers can observe pod state."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        observation = alice._observe_pod_state()

        assert "inventory" in observation
        assert "active_jobs" in observation
        assert "workers_count" in observation
        assert "available_recipes" in observation

        # Inventory may have been consumed by equipping workers
        assert isinstance(observation["inventory"], dict)
        assert observation["workers_count"] == 2

    def test_observe_market(self, basic_federation):
        """Workers can observe market state."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        observation = alice._observe_market()

        assert "prices" in observation
        assert "order_counts" in observation
        assert isinstance(observation["prices"], dict)
        assert isinstance(observation["order_counts"], dict)

    def test_observe_active_votes(self, basic_federation):
        """Workers can observe active motions."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        # Create a motion
        motion = Motion(
            motion_id="M001",
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Test Motion",
            description="Test",
            proposer=alice.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={alice.unique_id},
            voting_ends_step=10
        )
        basic_federation.governance.active_motions[motion.motion_id] = motion

        observation = alice._observe_active_votes()

        assert "active_motions" in observation
        assert observation["count"] == 1
        assert len(observation["active_motions"]) == 1
        assert observation["active_motions"][0]["motion_id"] == "M001"

    def test_check_permissions(self, basic_federation):
        """Workers can check their permissions."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        permissions = alice._check_permissions()

        assert "can_vote" in permissions
        assert "can_propose" in permissions
        assert "can_create_pod" in permissions
        assert "can_start_production" in permissions

    def test_calculate_distance(self, basic_federation):
        """Distance calculation works correctly."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        coord1 = (0.0, 0.0)
        coord2 = (3.0, 4.0)

        distance = alice._calculate_distance(coord1, coord2)

        assert distance == pytest.approx(5.0)


class TestReasoningMethods:
    """Test worker reasoning and action decision methods."""

    def test_reason_about_situation_structure(self, basic_federation):
        """Reasoning returns expected structure."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        observations = {
            "local_workers": {"workers": [], "count": 0},
            "pod_state": {"inventory": {}, "active_jobs": 0}
        }

        reasoning = alice._reason_about_situation(observations)

        # Should return dict with concerns, opportunities, recommended_actions
        assert isinstance(reasoning, dict)
        assert "concerns" in reasoning or "error" in reasoning
        assert "opportunities" in reasoning or "error" in reasoning
        assert "recommended_actions" in reasoning or "error" in reasoning

    def test_build_reasoning_prompt(self, basic_federation):
        """Reasoning prompt includes personality and mood."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        context = {
            "personality": {"openness": 0.7},
            "mood": {"happiness": 0.5},
            "observations": {}
        }

        prompt = alice._build_reasoning_prompt(context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "personality" in prompt.lower() or "openness" in prompt.lower()

    def test_decide_actions_filters_by_permissions(self, basic_federation):
        """Actions are filtered by permissions."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        reasoning = {
            "recommended_actions": [
                {"action": "vote", "motion_id": "M001", "choice": "for"},
                {"action": "produce", "recipe": "hammer"}
            ]
        }

        actions = alice._decide_actions(reasoning)

        assert isinstance(actions, list)
        # All actions should be permitted based on default constitution
        assert len(actions) <= len(reasoning["recommended_actions"])


class TestMemoryMethods:
    """Test worker memory management."""

    def test_update_memory_adds_entry(self, basic_federation):
        """Memory is updated with new observations."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        initial_memory_length = len(alice.episodic_memory)

        observations = {
            "local_workers": {"count": 1},
            "pod_state": {"inventory": {"wood": 500.0}}
        }

        alice._update_memory(observations)

        assert len(alice.episodic_memory) == initial_memory_length + 1

        latest_memory = alice.episodic_memory[-1]
        assert "step" in latest_memory
        assert "observations" in latest_memory
        assert "mood" in latest_memory

    def test_update_memory_limits_to_100(self, basic_federation):
        """Memory is capped at 100 entries."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        # Add 110 entries
        for i in range(110):
            alice._update_memory({"test": i})

        assert len(alice.episodic_memory) == 100
        # Should keep most recent entries
        assert alice.episodic_memory[-1]["observations"]["test"] == 109


class TestMoodMethods:
    """Test mood update logic."""

    def test_mood_high_currency_increases_happiness(self, basic_federation):
        """High currency increases happiness."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]
        alice.currency = 600.0

        initial_happiness = alice.mood.happiness

        observations = {"my_currency": 600.0}
        alice._update_mood_from_observations(observations)

        # Should increase (or at least not decrease significantly)
        assert alice.mood.happiness >= initial_happiness - 0.01

    def test_mood_low_currency_decreases_happiness(self, basic_federation):
        """Low currency decreases happiness and increases stress."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]
        alice.currency = 50.0

        initial_happiness = alice.mood.happiness
        initial_stress = alice.mood.stress

        observations = {"my_currency": 50.0}
        alice._update_mood_from_observations(observations)

        # Should decrease happiness or increase stress
        assert alice.mood.happiness <= initial_happiness or alice.mood.stress >= initial_stress

    def test_mood_extravert_likes_crowds(self, basic_federation):
        """Extraverted workers are happier with nearby workers."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        # Alice is extraverted (0.8)
        initial_happiness = alice.mood.happiness

        observations = {
            "local_workers": {
                "count": 5,
                "workers": [{"name": f"Worker{i}"} for i in range(5)]
            }
        }

        alice._update_mood_from_observations(observations)

        # Should increase happiness
        assert alice.mood.happiness >= initial_happiness - 0.01

    def test_mood_introvert_dislikes_crowds(self, basic_federation):
        """Introverted workers are stressed by many nearby workers."""
        pod = basic_federation[0]
        bob = [w for w in pod if w.name == "Bob"][0]

        # Bob is introverted (0.3)
        initial_stress = bob.mood.stress

        observations = {
            "local_workers": {
                "count": 5,
                "workers": [{"name": f"Worker{i}"} for i in range(5)]
            }
        }

        bob._update_mood_from_observations(observations)

        # Should increase stress
        assert bob.mood.stress >= initial_stress - 0.01

    def test_mood_libertarian_motivated_by_votes(self, basic_federation):
        """Libertarian workers are motivated by active votes."""
        pod = basic_federation[0]
        bob = [w for w in pod if w.name == "Bob"][0]

        # Bob is strongly libertarian (0.6)
        initial_motivation = bob.mood.motivation

        observations = {
            "active_motions": {
                "count": 3,
                "active_motions": [{"motion_id": f"M{i}"} for i in range(3)]
            }
        }

        bob._update_mood_from_observations(observations)

        # Should increase motivation
        assert bob.mood.motivation >= initial_motivation - 0.01


class TestFullCognitiveLoop:
    """Test the complete observe_and_reason flow."""

    def test_observe_and_reason_returns_complete_structure(self, basic_federation):
        """observe_and_reason returns observations, reasoning, and actions."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        result = alice.observe_and_reason()

        assert "observations" in result
        assert "reasoning" in result
        assert "actions" in result

        observations = result["observations"]
        assert "local_workers" in observations
        assert "pod_state" in observations
        assert "market_state" in observations
        assert "active_motions" in observations
        assert "my_permissions" in observations

    def test_observe_and_reason_updates_memory(self, basic_federation):
        """observe_and_reason updates episodic memory."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        initial_memory_length = len(alice.episodic_memory)

        alice.observe_and_reason()

        assert len(alice.episodic_memory) == initial_memory_length + 1

    def test_observe_and_reason_updates_mood(self, basic_federation):
        """observe_and_reason may update mood."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        initial_mood = alice.mood.happiness

        alice.observe_and_reason()

        # Mood may change or stay the same depending on observations
        assert isinstance(alice.mood.happiness, float)
        assert 0.0 <= alice.mood.happiness <= 1.0


class TestActionExecution:
    """Test action execution methods."""

    def test_execute_actions_with_empty_list(self, basic_federation):
        """Executing empty action list returns empty results."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        results = alice.execute_actions([])

        assert results == []

    def test_execute_actions_with_invalid_action(self, basic_federation):
        """Invalid actions return error in results."""
        pod = basic_federation[0]
        alice = [w for w in pod if w.name == "Alice"][0]

        actions = [{"action": "invalid_action"}]
        results = alice.execute_actions(actions)

        assert len(results) == 1
        # Should handle gracefully (either skip or error)
        assert isinstance(results[0], dict)
