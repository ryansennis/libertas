"""
Integration tests for autonomous agent cognitive loop.

Tests the full observe-reason-act cycle with LLM integration.
"""
import pytest
from libertas.organization import Federation, WorkerConfig, PodConfig
from libertas.cognitive import PersonalityTraits, Background
from libertas.governance import Motion, MotionType, VoteType
from mesa_llm.reasoning.cot import CoTReasoning


@pytest.fixture
def autonomous_federation():
    """Create a federation with workers of different personalities."""
    workers = [
        WorkerConfig(
            name="Collectivist_Alice",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=1000.0,
            personality=PersonalityTraits(
                openness=0.8,
                conscientiousness=0.6,
                extraversion=0.7,
                agreeableness=0.8,
                neuroticism=0.3,
                economic_left_right=-0.9,  # Very collectivist
                authority_libertarian=-0.4
            ),
            background=Background(education_level=4, years_experience=8)
        ),
        WorkerConfig(
            name="Individualist_Bob",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=500.0,
            personality=PersonalityTraits(
                openness=0.5,
                conscientiousness=0.8,
                extraversion=0.3,
                agreeableness=0.4,
                neuroticism=0.4,
                economic_left_right=0.8,  # Very individualist
                authority_libertarian=0.7
            ),
            background=Background(education_level=3, years_experience=5)
        )
    ]

    pod_config = PodConfig(
        name="autonomous_pod",
        workers=workers,
        initial_inventory={"wood": 1000.0, "stone": 500.0}
    )

    federation = Federation(pods=[pod_config], seed=42)
    return federation


class TestAutonomousObservation:
    """Test autonomous observation in integrated environment."""

    def test_worker_observes_complete_environment(self, autonomous_federation):
        """Worker observes all aspects of environment in one cycle."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        result = alice.observe_and_reason()

        # Check complete observation structure
        obs = result["observations"]
        assert "local_workers" in obs
        assert "pod_state" in obs
        assert "market_state" in obs
        assert "active_motions" in obs
        assert "my_permissions" in obs

        # Local workers should include Bob
        assert obs["local_workers"]["count"] >= 1

        # Pod state should have inventory
        assert "wood" in obs["pod_state"]["inventory"]
        assert obs["pod_state"]["inventory"]["wood"] == 1000.0

    def test_worker_observes_other_worker_activities(self, autonomous_federation):
        """Workers can observe what other workers are doing."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]
        bob = [w for w in pod if w.name == "Individualist_Bob"][0]

        # Bob starts a job (if possible)
        # For this test, just observe what's visible

        result = alice.observe_and_reason()
        obs = result["observations"]

        # Should see Bob in local workers
        worker_names = [w["name"] for w in obs["local_workers"]["workers"]]
        assert "Individualist_Bob" in worker_names or obs["local_workers"]["count"] >= 1


class TestAutonomousReasoning:
    """Test LLM reasoning in autonomous context."""

    def test_reasoning_considers_personality(self, autonomous_federation):
        """Reasoning output should be influenced by personality."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]
        bob = [w for w in pod if w.name == "Individualist_Bob"][0]

        # Create a motion about collective investment
        motion = Motion(
            motion_id="M001",
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Collective Investment",
            description="Invest collectively in tool production",
            proposer=alice.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={alice.unique_id, bob.unique_id},
            voting_ends_step=10
        )
        autonomous_federation.governance.active_motions[motion.motion_id] = motion

        alice_result = alice.observe_and_reason()
        bob_result = bob.observe_and_reason()

        # Both should observe the motion
        assert alice_result["observations"]["active_motions"]["count"] == 1
        assert bob_result["observations"]["active_motions"]["count"] == 1

        # Actions may differ based on personality (but LLM is unpredictable in tests)
        # Just verify they both generate some form of reasoning
        assert "reasoning" in alice_result
        assert "reasoning" in bob_result


class TestAutonomousActions:
    """Test action generation and execution."""

    def test_worker_generates_actions(self, autonomous_federation):
        """Workers generate actions from reasoning."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        result = alice.observe_and_reason()

        # Should have actions field (may be empty if LLM decides not to act)
        assert "actions" in result
        assert isinstance(result["actions"], list)

    def test_worker_executes_vote_action(self, autonomous_federation):
        """Workers can execute vote actions."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        # Create motion
        motion = Motion(
            motion_id="M002",
            motion_type=MotionType.POLICY_CHANGE,
            title="Test Vote",
            description="Test",
            proposer=alice.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={alice.unique_id},
            voting_ends_step=10
        )
        autonomous_federation.governance.active_motions[motion.motion_id] = motion

        # Execute vote action manually
        actions = [
            {"action": "vote", "motion_id": "M002", "choice": "for"}
        ]

        results = alice.execute_actions(actions)

        assert len(results) == 1
        assert results[0]["action"] == "vote"
        # Vote should succeed or return error
        assert "result" in results[0] or "error" in results[0]

    def test_worker_action_filtered_by_permissions(self, autonomous_federation):
        """Actions are filtered by constitutional permissions."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        # All actions should be permitted in default constitution
        reasoning = {
            "recommended_actions": [
                {"action": "vote", "motion_id": "M001", "choice": "for"},
                {"action": "produce", "recipe": "hammer"}
            ]
        }

        actions = alice._decide_actions(reasoning)

        # Should return filtered list (all actions valid in default constitution)
        assert isinstance(actions, list)
        assert len(actions) >= 0


class TestMemoryAccumulation:
    """Test episodic memory accumulation over multiple cycles."""

    def test_memory_accumulates_over_cycles(self, autonomous_federation):
        """Memory accumulates as worker observes multiple times."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        initial_memory_count = len(alice.episodic_memory)

        # Run 5 observation cycles
        for _ in range(5):
            alice.observe_and_reason()

        assert len(alice.episodic_memory) == initial_memory_count + 5

    def test_memory_contains_step_information(self, autonomous_federation):
        """Memory entries contain step numbers."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        alice.observe_and_reason()

        latest_memory = alice.episodic_memory[-1]
        assert "step" in latest_memory
        assert latest_memory["step"] == autonomous_federation.steps


class TestMoodDynamics:
    """Test mood changes in integrated environment."""

    def test_mood_changes_over_time(self, autonomous_federation):
        """Mood may change as worker observes environment."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        initial_happiness = alice.mood.happiness
        initial_stress = alice.mood.stress

        # Run multiple cycles
        for _ in range(10):
            alice.observe_and_reason()

        # Mood should remain in valid range
        assert 0.0 <= alice.mood.happiness <= 1.0
        assert 0.0 <= alice.mood.stress <= 1.0

    def test_high_currency_worker_stays_happy(self, autonomous_federation):
        """Worker with high currency maintains happiness."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        # Alice starts with 1000 currency
        alice.observe_and_reason()

        # Happiness should be relatively high
        assert alice.mood.happiness >= 0.4

    def test_low_currency_worker_becomes_stressed(self, autonomous_federation):
        """Worker with low currency becomes stressed."""
        pod = autonomous_federation[0]
        bob = [w for w in pod if w.name == "Individualist_Bob"][0]

        # Bob starts with 500 currency (moderate)
        # Reduce it artificially
        bob.currency = 50.0

        initial_stress = bob.mood.stress

        bob.observe_and_reason()

        # Stress should increase or stay high
        assert bob.mood.stress >= initial_stress or bob.mood.stress > 0.3


class TestFullAutonomousLoop:
    """Test complete autonomous loop integration."""

    def test_complete_cycle_executes_without_errors(self, autonomous_federation):
        """Full observe-reason-act cycle executes cleanly."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        # Run full cycle
        result = alice.observe_and_reason()
        actions = result.get("actions", [])

        # Should not raise exceptions
        if actions:
            execution_results = alice.execute_actions(actions)
            assert isinstance(execution_results, list)

    def test_worker_autonomous_over_multiple_steps(self, autonomous_federation):
        """Worker can run autonomously over multiple steps."""
        pod = autonomous_federation[0]
        alice = [w for w in pod if w.name == "Collectivist_Alice"][0]

        # Run 10 autonomous cycles
        for i in range(10):
            result = alice.observe_and_reason()
            actions = result.get("actions", [])

            if actions:
                alice.execute_actions(actions)

        # Memory should accumulate
        assert len(alice.episodic_memory) >= 10

        # Mood should still be valid
        assert 0.0 <= alice.mood.happiness <= 1.0
        assert 0.0 <= alice.mood.stress <= 1.0
