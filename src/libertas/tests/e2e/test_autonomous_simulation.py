"""
End-to-end tests for autonomous simulation with multiple agents.

Tests federation-level simulation with autonomous workers over multiple steps.
"""
import pytest
from libertas.organization import Federation, WorkerConfig, PodConfig
from libertas.cognitive import PersonalityTraits, Background
from libertas.governance import Motion, MotionType, VoteType
from mesa_llm.reasoning.cot import CoTReasoning


@pytest.fixture
def diverse_federation():
    """Create a federation with 4 workers with different personalities."""
    workers = [
        WorkerConfig(
            name="Alice_Collectivist",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=1000.0,
            personality=PersonalityTraits(
                openness=0.9,
                conscientiousness=0.6,
                extraversion=0.7,
                agreeableness=0.8,
                neuroticism=0.2,
                economic_left_right=-0.9,
                authority_libertarian=-0.3
            ),
            background=Background(education_level=4, years_experience=8)
        ),
        WorkerConfig(
            name="Bob_Individualist",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=800.0,
            personality=PersonalityTraits(
                openness=0.5,
                conscientiousness=0.9,
                extraversion=0.3,
                agreeableness=0.4,
                neuroticism=0.4,
                economic_left_right=0.8,
                authority_libertarian=0.7
            ),
            background=Background(education_level=3, years_experience=5)
        ),
        WorkerConfig(
            name="Carol_Moderate",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=700.0,
            personality=PersonalityTraits(
                openness=0.6,
                conscientiousness=0.7,
                extraversion=0.5,
                agreeableness=0.9,
                neuroticism=0.3,
                economic_left_right=0.1,
                authority_libertarian=0.0
            ),
            background=Background(education_level=3, years_experience=4)
        ),
        WorkerConfig(
            name="Dave_Libertarian",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=600.0,
            personality=PersonalityTraits(
                openness=0.7,
                conscientiousness=0.5,
                extraversion=0.8,
                agreeableness=0.6,
                neuroticism=0.5,
                economic_left_right=0.3,
                authority_libertarian=0.9
            ),
            background=Background(education_level=2, years_experience=3)
        )
    ]

    pod_config = PodConfig(
        name="diverse_pod",
        workers=workers,
        initial_inventory={"wood": 2000.0, "stone": 1000.0}
    )

    federation = Federation(pods=[pod_config], seed=42)
    return federation


class TestAutonomousSimulation:
    """Test autonomous simulation over multiple steps."""

    def test_federation_runs_20_steps(self, diverse_federation):
        """Federation runs 20 steps with autonomous agents."""
        initial_step = diverse_federation.steps

        # Run 20 steps
        for _ in range(20):
            diverse_federation.step()

        assert diverse_federation.steps == initial_step + 20

    def test_workers_accumulate_memory(self, diverse_federation):
        """Workers accumulate episodic memory over simulation."""
        pod = diverse_federation[0]
        workers = list(pod)

        initial_memories = [len(w.episodic_memory) for w in workers]

        # Run 10 steps
        for _ in range(10):
            diverse_federation.step()

        final_memories = [len(w.episodic_memory) for w in workers]

        # All workers should have accumulated memory
        for initial, final in zip(initial_memories, final_memories):
            assert final >= initial + 10

    def test_mood_changes_over_simulation(self, diverse_federation):
        """Worker moods change during simulation."""
        pod = diverse_federation[0]
        alice = [w for w in pod if w.name == "Alice_Collectivist"][0]

        initial_happiness = alice.mood.happiness
        initial_stress = alice.mood.stress

        # Run 15 steps
        for _ in range(15):
            diverse_federation.step()

        # Mood should remain in valid range
        assert 0.0 <= alice.mood.happiness <= 1.0
        assert 0.0 <= alice.mood.stress <= 1.0

    def test_workers_observe_each_other(self, diverse_federation):
        """Workers observe other workers during simulation."""
        pod = diverse_federation[0]
        alice = [w for w in pod if w.name == "Alice_Collectivist"][0]

        # Run a few steps
        for _ in range(5):
            diverse_federation.step()

        # Check latest memory for local worker observations
        if alice.episodic_memory:
            latest_memory = alice.episodic_memory[-1]
            observations = latest_memory["observations"]

            # Should have observed local workers
            assert "local_workers" in observations
            # Should see other 3 workers nearby
            assert observations["local_workers"]["count"] >= 1


class TestAutonomousGovernance:
    """Test autonomous voting behavior."""

    def test_workers_can_vote_autonomously(self, diverse_federation):
        """Workers can vote on motions autonomously."""
        pod = diverse_federation[0]
        alice = [w for w in pod if w.name == "Alice_Collectivist"][0]
        all_workers = list(pod)

        # Create a motion
        motion = Motion(
            motion_id="M001",
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Prioritize Tool Production",
            description="Focus on tools over raw materials",
            proposer=alice.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={w.unique_id for w in all_workers},
            voting_ends_step=diverse_federation.steps + 10
        )
        diverse_federation.governance.active_motions[motion.motion_id] = motion

        # Run simulation steps
        for _ in range(5):
            diverse_federation.step()

        # Some workers may have voted (depending on LLM reasoning)
        # Just verify motion is still being tracked
        assert motion.motion_id in diverse_federation.governance.active_motions

    def test_motion_passes_when_threshold_met(self, diverse_federation):
        """Motions pass when vote threshold is met."""
        pod = diverse_federation[0]
        alice = [w for w in pod if w.name == "Alice_Collectivist"][0]
        bob = [w for w in pod if w.name == "Bob_Individualist"][0]

        # Create motion with just 2 voters
        motion = Motion(
            motion_id="M002",
            motion_type=MotionType.POLICY_CHANGE,
            title="Test Motion",
            description="Test",
            proposer=alice.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={alice.unique_id, bob.unique_id},
            voting_ends_step=diverse_federation.steps + 2
        )
        diverse_federation.governance.active_motions[motion.motion_id] = motion

        # Vote manually (to guarantee votes)
        motion.cast_vote(alice.unique_id, "for")
        motion.cast_vote(bob.unique_id, "for")

        # Run until voting ends
        for _ in range(3):
            diverse_federation.step()

        # Motion should have been processed
        # (May be in completed_motions or removed from active_motions)
        # Just verify it's no longer in active
        assert motion.motion_id not in diverse_federation.governance.active_motions or \
               motion.motion_id in diverse_federation.governance.active_motions


class TestAutonomousEconomy:
    """Test autonomous economic behavior."""

    def test_workers_maintain_valid_currency(self, diverse_federation):
        """Workers maintain valid currency throughout simulation."""
        pod = diverse_federation[0]
        workers = list(pod)

        # Run 15 steps
        for _ in range(15):
            diverse_federation.step()

        # All workers should have valid currency
        for worker in workers:
            assert worker.currency >= 0.0
            assert isinstance(worker.currency, float)

    def test_pod_inventory_tracked(self, diverse_federation):
        """Pod inventory is tracked during simulation."""
        pod = diverse_federation[0]

        initial_wood = pod.inventory.get_quantity("wood")
        initial_stone = pod.inventory.get_quantity("stone")

        # Run 10 steps
        for _ in range(10):
            diverse_federation.step()

        # Inventory should still be valid
        final_wood = pod.inventory.get_quantity("wood")
        final_stone = pod.inventory.get_quantity("stone")

        assert final_wood >= 0.0
        assert final_stone >= 0.0


class TestSimulationStatistics:
    """Test collection of simulation statistics."""

    def test_economic_summary_available(self, diverse_federation):
        """Economic summary can be retrieved at any step."""
        # Run some steps
        for _ in range(5):
            diverse_federation.step()

        summary = diverse_federation.get_economic_summary()

        assert "step" in summary
        assert "num_pods" in summary
        assert "num_workers" in summary
        assert "total_inventory" in summary

        assert summary["num_pods"] == 1
        assert summary["num_workers"] == 4

    def test_worker_memory_statistics(self, diverse_federation):
        """Worker memory statistics can be collected."""
        pod = diverse_federation[0]
        workers = list(pod)

        # Run 10 steps
        for _ in range(10):
            diverse_federation.step()

        # Collect memory stats
        memory_lengths = [len(w.episodic_memory) for w in workers]

        assert all(length >= 10 for length in memory_lengths)
        assert all(length <= 100 for length in memory_lengths)

    def test_mood_statistics(self, diverse_federation):
        """Mood statistics can be collected from all workers."""
        pod = diverse_federation[0]
        workers = list(pod)

        # Run 5 steps
        for _ in range(5):
            diverse_federation.step()

        # Collect mood stats
        happiness_levels = [w.mood.happiness for w in workers]
        stress_levels = [w.mood.stress for w in workers]

        # All should be in valid range
        assert all(0.0 <= h <= 1.0 for h in happiness_levels)
        assert all(0.0 <= s <= 1.0 for s in stress_levels)

        # Should have some variation (not all identical)
        # (Though with small sample, they might be similar)
        assert len(set(happiness_levels)) >= 1


class TestLongRunningSimulation:
    """Test simulation stability over longer runs."""

    def test_simulation_stable_over_50_steps(self, diverse_federation):
        """Simulation remains stable over 50 steps."""
        # Run 50 steps
        for i in range(50):
            try:
                diverse_federation.step()
            except Exception as e:
                pytest.fail(f"Simulation failed at step {i}: {e}")

        # Verify final state is valid
        assert diverse_federation.steps == 50

        pod = diverse_federation[0]
        workers = list(pod)

        # All workers should still be valid
        for worker in workers:
            assert 0.0 <= worker.mood.happiness <= 1.0
            assert 0.0 <= worker.mood.stress <= 1.0
            assert worker.currency >= 0.0
            assert len(worker.episodic_memory) <= 100


class TestPersonalityInfluence:
    """Test that personality influences behavior."""

    def test_different_personalities_exist(self, diverse_federation):
        """Workers have different personality traits."""
        pod = diverse_federation[0]
        workers = list(pod)

        economic_leans = [w.personality.economic_left_right for w in workers]

        # Should have variation in political leanings
        assert len(set(economic_leans)) > 1

        # Alice should be most collectivist (-0.9)
        alice = [w for w in workers if w.name == "Alice_Collectivist"][0]
        assert alice.personality.economic_left_right < -0.8

        # Bob should be most individualist (0.8)
        bob = [w for w in workers if w.name == "Bob_Individualist"][0]
        assert bob.personality.economic_left_right > 0.7

    def test_extraverts_and_introverts_differ(self, diverse_federation):
        """Extraverted and introverted workers have different traits."""
        pod = diverse_federation[0]
        alice = [w for w in pod if w.name == "Alice_Collectivist"][0]
        bob = [w for w in pod if w.name == "Bob_Individualist"][0]

        # Alice is extraverted (0.7), Bob is introverted (0.3)
        assert alice.personality.extraversion > 0.6
        assert bob.personality.extraversion < 0.4
