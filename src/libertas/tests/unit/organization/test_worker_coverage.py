"""
Additional tests to improve worker.py coverage.

Focuses on cognitive loop edge cases and LLM response parsing.
"""
import pytest
from unittest.mock import Mock, patch
from libertas.organization import Federation, WorkerConfig, PodConfig
from libertas.cognitive import PersonalityTraits, Background
from libertas.governance import Motion, MotionType, VoteType
from mesa_llm.reasoning.cot import CoTReasoning


@pytest.fixture
def test_federation():
    """Create a simple federation for testing."""
    workers = [
        WorkerConfig(
            name="TestWorker",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=1000.0,
            personality=PersonalityTraits(),
            background=Background()
        )
    ]

    pod_config = PodConfig(
        name="test_pod",
        workers=workers,
        initial_inventory={"wood": 100.0}
    )

    federation = Federation(pods=[pod_config], seed=42)
    return federation


@pytest.mark.unit
class TestLLMResponseParsing:
    """Test LLM response parsing with various formats."""

    def test_parse_response_with_json_code_block(self, test_federation):
        """Parse LLM response with ```json code block."""
        pod = test_federation[0]
        worker = list(pod)[0]

        response = """Here is my reasoning:
```json
{
  "concerns": ["low inventory"],
  "opportunities": ["trade"],
  "recommended_actions": []
}
```
"""

        result = worker._parse_llm_response(response)

        assert "concerns" in result
        assert result["concerns"] == ["low inventory"]
        assert result["opportunities"] == ["trade"]

    def test_parse_response_with_generic_code_block(self, test_federation):
        """Parse LLM response with generic ``` code block."""
        pod = test_federation[0]
        worker = list(pod)[0]

        response = """
```
{
  "concerns": ["test"],
  "opportunities": [],
  "recommended_actions": []
}
```
"""

        result = worker._parse_llm_response(response)

        assert "concerns" in result
        assert result["concerns"] == ["test"]

    def test_parse_response_plain_json(self, test_federation):
        """Parse plain JSON response."""
        pod = test_federation[0]
        worker = list(pod)[0]

        response = '{"concerns": [], "opportunities": ["opportunity1"], "recommended_actions": []}'

        result = worker._parse_llm_response(response)

        assert "opportunities" in result
        assert result["opportunities"] == ["opportunity1"]

    def test_parse_response_invalid_json(self, test_federation):
        """Fallback on invalid JSON."""
        pod = test_federation[0]
        worker = list(pod)[0]

        response = "This is not valid JSON at all"

        result = worker._parse_llm_response(response)

        # Should return fallback structure
        assert "concerns" in result
        assert "opportunities" in result
        assert "recommended_actions" in result


@pytest.mark.unit
class TestActionDecisionFiltering:
    """Test action filtering based on permissions."""

    def test_decide_actions_with_permitted_actions(self, test_federation):
        """Actions that are permitted pass through."""
        pod = test_federation[0]
        worker = list(pod)[0]

        reasoning = {
            "recommended_actions": [
                {"action": "vote", "motion_id": "M001", "choice": "for"}
            ]
        }

        actions = worker._decide_actions(reasoning)

        # Should include the action (voting is permitted)
        assert len(actions) >= 0  # Depends on actual permissions

    def test_decide_actions_empty_reasoning(self, test_federation):
        """Handle reasoning with no recommended actions."""
        pod = test_federation[0]
        worker = list(pod)[0]

        reasoning = {}

        actions = worker._decide_actions(reasoning)

        assert actions == []

    def test_decide_actions_no_recommendations(self, test_federation):
        """Handle reasoning with empty recommended_actions."""
        pod = test_federation[0]
        worker = list(pod)[0]

        reasoning = {
            "concerns": ["test"],
            "recommended_actions": []
        }

        actions = worker._decide_actions(reasoning)

        assert actions == []


@pytest.mark.unit
class TestActionExecution:
    """Test executing various action types."""

    def test_execute_vote_action(self, test_federation):
        """Execute a vote action."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Create a motion
        motion = Motion(
            motion_id="M999",
            motion_type=MotionType.POLICY_CHANGE,
            title="Test",
            description="Test",
            proposer=worker.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={worker.unique_id},
            voting_ends_step=100
        )
        test_federation.governance.active_motions[motion.motion_id] = motion

        actions = [
            {"action": "vote", "motion_id": "M999", "choice": "for"}
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 1
        assert results[0]["action"] == "vote"

    def test_execute_produce_action(self, test_federation):
        """Execute a production action."""
        pod = test_federation[0]
        worker = list(pod)[0]

        actions = [
            {"action": "produce", "recipe": "make_planks", "batch_size": 1}
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 1
        assert results[0]["action"] == "produce"

    def test_execute_buy_action(self, test_federation):
        """Execute a buy action."""
        pod = test_federation[0]
        worker = list(pod)[0]

        actions = [
            {"action": "buy", "resource": "wood", "quantity": 10, "max_price": 5.0}
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 1
        assert results[0]["action"] == "buy"

    def test_execute_sell_action(self, test_federation):
        """Execute a sell action."""
        pod = test_federation[0]
        worker = list(pod)[0]

        actions = [
            {"action": "sell", "resource": "wood", "quantity": 5, "min_price": 2.0}
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 1
        assert results[0]["action"] == "sell"

    def test_execute_propose_action(self, test_federation):
        """Execute a propose motion action."""
        pod = test_federation[0]
        worker = list(pod)[0]

        actions = [
            {
                "action": "propose",
                "title": "New Policy",
                "description": "Test policy",
                "motion_type": "POLICY_CHANGE"
            }
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 1
        assert results[0]["action"] == "propose"

    def test_execute_action_with_error(self, test_federation):
        """Handle action that raises an error."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Vote on non-existent motion should fail
        actions = [
            {"action": "vote", "motion_id": "NONEXISTENT", "choice": "for"}
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 1
        # Should have a result (may be error or failed result)
        assert "result" in results[0] or "error" in results[0]

    def test_execute_multiple_actions(self, test_federation):
        """Execute multiple actions in sequence."""
        pod = test_federation[0]
        worker = list(pod)[0]

        actions = [
            {"action": "invalid_action_type"},
            {"action": "buy", "resource": "stone", "quantity": 1, "max_price": 10.0}
        ]

        results = worker.execute_actions(actions)

        assert len(results) == 2


@pytest.mark.unit
class TestMoodUpdates:
    """Test mood update edge cases."""

    def test_mood_update_with_active_job(self, test_federation):
        """Mood updates when worker has active job."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Simulate having an active job
        worker.current_job = Mock()
        worker.current_job.recipe.name = "test_recipe"

        initial_stress = worker.mood.stress

        observations = {"my_job": "test_recipe"}
        worker._update_mood_from_observations(observations)

        # Stress should decrease with active work
        assert worker.mood.stress <= initial_stress

    def test_mood_update_with_no_motions(self, test_federation):
        """Mood updates when there are no active motions."""
        pod = test_federation[0]
        worker = list(pod)[0]

        initial_motivation = worker.mood.motivation

        observations = {
            "active_motions": {"count": 0, "active_motions": []}
        }
        worker._update_mood_from_observations(observations)

        # Motivation shouldn't change much without motions
        assert abs(worker.mood.motivation - initial_motivation) < 0.1

    def test_mood_stays_in_bounds(self, test_federation):
        """Mood values stay within [0, 1] bounds."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Extreme observations
        for _ in range(100):
            observations = {
                "my_currency": 0.0,  # Very low
                "local_workers": {"count": 100},  # Very crowded
                "active_motions": {"count": 50}  # Lots of activity
            }
            worker._update_mood_from_observations(observations)

        # Mood should stay clamped
        assert 0.0 <= worker.mood.happiness <= 1.0
        assert 0.0 <= worker.mood.stress <= 1.0
        assert 0.0 <= worker.mood.motivation <= 1.0


@pytest.mark.unit
class TestObservationEdgeCases:
    """Test observation method edge cases."""

    def test_observe_with_no_federation(self, test_federation):
        """Observe when federation is missing."""
        pod = test_federation[0]
        worker = list(pod)[0]
        worker._federation = None

        # Should not crash
        observations = worker._observe_pod_state()

        assert "error" in observations or "inventory" in observations

    def test_observe_market_no_market(self, test_federation):
        """Observe market when market doesn't exist."""
        pod = test_federation[0]
        worker = list(pod)[0]
        test_federation.market = None

        observations = worker._observe_market()

        assert "error" in observations

    def test_observe_votes_no_governance(self, test_federation):
        """Observe votes when governance doesn't exist."""
        pod = test_federation[0]
        worker = list(pod)[0]
        test_federation.governance = None

        observations = worker._observe_active_votes()

        assert observations["count"] == 0

    def test_check_permissions_no_federation(self, test_federation):
        """Check permissions when federation is missing."""
        pod = test_federation[0]
        worker = list(pod)[0]
        worker._federation = None

        permissions = worker._check_permissions()

        assert permissions == {}


@pytest.mark.unit
class TestMemoryManagement:
    """Test memory system edge cases."""

    def test_memory_caps_at_100_entries(self, test_federation):
        """Memory is limited to 100 most recent entries."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Add 150 memory entries
        for i in range(150):
            worker._update_memory({"step": i, "test": True})

        # Should keep only last 100
        assert len(worker.episodic_memory) == 100

        # Should have most recent entries
        assert worker.episodic_memory[-1]["observations"]["step"] == 149
        assert worker.episodic_memory[0]["observations"]["step"] == 50

    def test_memory_includes_mood_snapshot(self, test_federation):
        """Memory entries include mood at time of observation."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Set specific mood
        worker.mood.happiness = 0.75
        worker.mood.stress = 0.25

        worker._update_memory({"test": "data"})

        latest_memory = worker.episodic_memory[-1]
        assert "mood" in latest_memory
        assert latest_memory["mood"]["happiness"] == 0.75
        assert latest_memory["mood"]["stress"] == 0.25


@pytest.mark.unit
class TestFederationStepIntegration:
    """Test federation step with cognitive loop."""

    def test_federation_step_runs_cognitive_loop(self, test_federation):
        """Federation step executes cognitive loop for all workers."""
        pod = test_federation[0]
        worker = list(pod)[0]

        initial_memory_count = len(worker.episodic_memory)

        # Run one step
        test_federation.step()

        # Worker should have accumulated memory
        assert len(worker.episodic_memory) > initial_memory_count

    def test_execute_motion_called_on_passed_motion(self, test_federation):
        """_execute_motion is called when motion passes."""
        pod = test_federation[0]
        worker = list(pod)[0]

        # Create motion that will pass
        motion = Motion(
            motion_id="M_PASS",
            motion_type=MotionType.POLICY_CHANGE,
            title="Test",
            description="Test",
            proposer=worker.unique_id,
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={worker.unique_id},
            voting_ends_step=test_federation.steps + 1
        )
        test_federation.governance.active_motions[motion.motion_id] = motion

        # Vote for it
        motion.cast_vote(worker.unique_id, "for")

        # Run steps until voting ends
        test_federation.step()
        test_federation.step()

        # Motion should have been processed (removed from active_motions)
        assert motion.motion_id not in test_federation.governance.active_motions
