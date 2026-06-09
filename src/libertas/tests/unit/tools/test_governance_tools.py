# tests/unit/tools/test_governance_tools.py
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.tools.governance_tools import GovernanceTools, get_governance_tool_definitions
from libertas.organization.worker import Worker, WorkerConfig
from libertas.organization.pod import Pod, PodConfig
from libertas.organization.federation import Federation
from libertas.governance import Constitution, MotionType, VoteType

LLM_MODEL = "ollama/tinyllama"


@pytest.mark.unit
class TestGovernanceTools:
    """Test GovernanceTools class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create worker configs
        worker_config1 = WorkerConfig(
            name="Alice",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_currency=1000.0
        )
        worker_config2 = WorkerConfig(
            name="Bob",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_currency=1000.0
        )

        # Create pod config with workers
        pod_config = PodConfig(
            name="TestPod",
            workers=[worker_config1, worker_config2]
        )

        # Create federation
        self.federation = Federation(pods=[pod_config])
        self.federation.steps = 100

        # Get workers
        pod = list(self.federation)[0]
        workers = list(pod)
        self.alice = workers[0]
        self.bob = workers[1]

        # Create governance tools instances
        self.alice_tools = self.alice.governance_tools
        self.bob_tools = self.bob.governance_tools

    def test_read_constitution_pod(self):
        """Test reading pod constitution."""
        result = self.alice_tools.read_constitution("pod")
        result_dict = json.loads(result)

        assert "level" in result_dict
        assert result_dict["level"] == "pod"
        assert "title" in result_dict
        assert "full_text" in result_dict
        assert "version" in result_dict
        assert "TestPod" in result_dict["title"]
        assert "Article" in result_dict["full_text"]

    def test_read_constitution_federation(self):
        """Test reading federation constitution."""
        result = self.alice_tools.read_constitution("federation")
        result_dict = json.loads(result)

        assert result_dict["level"] == "federation"
        assert "Federation" in result_dict["title"]
        assert "Article" in result_dict["full_text"]

    def test_read_constitution_invalid_level(self):
        """Test reading constitution with invalid level."""
        result = self.alice_tools.read_constitution("invalid")
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "Invalid level" in result_dict["error"]

    def test_read_constitution_no_pod(self):
        """Test reading pod constitution when worker has no pod."""
        # Create worker without pod
        worker_config = WorkerConfig(
            name="NoPodWorker",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        federation = Federation(pods=[])
        worker = Worker(federation, worker_config, coordinate=(0, 0), pod=None)
        tools = GovernanceTools(worker)

        result = tools.read_constitution("pod")
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "not assigned to a pod" in result_dict["error"]

    def test_check_my_permissions(self):
        """Test checking worker permissions."""
        result = self.alice_tools.check_my_permissions()
        result_dict = json.loads(result)

        assert "federation" in result_dict
        assert "pod" in result_dict

        # Check federation permissions
        assert "can_vote" in result_dict["federation"]
        assert "can_propose_motion" in result_dict["federation"]
        assert "can_create_pod" in result_dict["federation"]
        assert "can_propose_amendment" in result_dict["federation"]

        # Check pod permissions
        assert "can_vote" in result_dict["pod"]
        assert "can_propose_motion" in result_dict["pod"]
        assert "can_start_production" in result_dict["pod"]

    def test_check_my_permissions_no_pod(self):
        """Test checking permissions when worker has no pod."""
        # Create worker without pod
        worker_config = WorkerConfig(
            name="NoPodWorker",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        federation = Federation(pods=[])
        worker = Worker(federation, worker_config, coordinate=(0, 0), pod=None)
        tools = GovernanceTools(worker)

        result = tools.check_my_permissions()
        result_dict = json.loads(result)

        # Should have federation permissions but not pod permissions
        assert "federation" in result_dict
        assert "pod" not in result_dict

    def test_propose_motion_pod(self):
        """Test proposing a motion at pod level."""
        result = self.alice_tools.propose_motion(
            title="Test Motion",
            description="This is a test motion",
            motion_type="CUSTOM",
            scope="pod",
            voting_duration=50
        )
        result_dict = json.loads(result)

        assert result_dict["success"] is True
        assert "motion_id" in result_dict
        assert "voting_ends_step" in result_dict
        assert "Test Motion" in result_dict["message"]

        # Verify motion was created
        assert len(self.federation.governance.active_motions) == 1

    def test_propose_motion_federation(self):
        """Test proposing a motion at federation level."""
        result = self.alice_tools.propose_motion(
            title="Federation Motion",
            description="Federation-level proposal",
            motion_type="CUSTOM",
            scope="federation"
        )
        result_dict = json.loads(result)

        assert result_dict["success"] is True
        motion = self.federation.governance.active_motions[result_dict["motion_id"]]
        assert motion.scope == "federation"

    def test_propose_motion_invalid_scope(self):
        """Test proposing motion with invalid scope."""
        result = self.alice_tools.propose_motion(
            title="Bad Motion",
            description="Invalid scope",
            scope="invalid"
        )
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "Invalid scope" in result_dict["error"]

    def test_propose_motion_no_pod(self):
        """Test proposing pod motion when worker has no pod."""
        # Create worker without pod
        worker_config = WorkerConfig(
            name="NoPodWorker",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        federation = Federation(pods=[])
        worker = Worker(federation, worker_config, coordinate=(0, 0), pod=None)
        tools = GovernanceTools(worker)

        result = tools.propose_motion(
            title="Test",
            description="Test",
            scope="pod"
        )
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "not assigned to a pod" in result_dict["error"]

    def test_propose_motion_invalid_type(self):
        """Test proposing motion with invalid motion type."""
        result = self.alice_tools.propose_motion(
            title="Test",
            description="Test",
            motion_type="INVALID_TYPE",
            scope="pod"
        )
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "Invalid motion type" in result_dict["error"]

    def test_vote_on_motion(self):
        """Test voting on a motion."""
        # First create a motion
        propose_result = self.alice_tools.propose_motion(
            title="Test Vote",
            description="Testing voting",
            scope="pod"
        )
        motion_id = json.loads(propose_result)["motion_id"]

        # Bob votes on it
        result = self.bob_tools.vote_on_motion(motion_id, "for")
        result_dict = json.loads(result)

        assert result_dict["success"] is True
        assert "Vote recorded" in result_dict["message"] or "for" in result_dict["message"]

        # Verify vote was recorded
        motion = self.federation.governance.active_motions[motion_id]
        assert self.bob.unique_id in motion.votes_for

    def test_vote_on_motion_against(self):
        """Test voting against a motion."""
        propose_result = self.alice_tools.propose_motion(
            title="Test",
            description="Test",
            scope="pod"
        )
        motion_id = json.loads(propose_result)["motion_id"]

        result = self.bob_tools.vote_on_motion(motion_id, "against")
        result_dict = json.loads(result)

        assert result_dict["success"] is True
        motion = self.federation.governance.active_motions[motion_id]
        assert self.bob.unique_id in motion.votes_against

    def test_vote_on_motion_abstain(self):
        """Test abstaining from a motion."""
        propose_result = self.alice_tools.propose_motion(
            title="Test",
            description="Test",
            scope="pod"
        )
        motion_id = json.loads(propose_result)["motion_id"]

        result = self.bob_tools.vote_on_motion(motion_id, "abstain")
        result_dict = json.loads(result)

        assert result_dict["success"] is True
        motion = self.federation.governance.active_motions[motion_id]
        assert self.bob.unique_id in motion.abstentions

    def test_vote_on_nonexistent_motion(self):
        """Test voting on a motion that doesn't exist."""
        result = self.bob_tools.vote_on_motion("M999", "for")
        result_dict = json.loads(result)

        assert result_dict["success"] is False
        assert "not found" in result_dict["message"]

    def test_list_active_motions_all(self):
        """Test listing all active motions."""
        # Create some motions
        self.alice_tools.propose_motion("Motion 1", "First", scope="pod")
        self.alice_tools.propose_motion("Motion 2", "Second", scope="pod")

        result = self.bob_tools.list_active_motions("all")
        result_dict = json.loads(result)

        assert "active_motions" in result_dict
        assert "count" in result_dict
        assert result_dict["count"] == 2

    def test_list_active_motions_pod_scope(self):
        """Test listing motions filtered by pod scope."""
        self.alice_tools.propose_motion("Pod Motion", "Test", scope="pod")

        result = self.bob_tools.list_active_motions("pod")
        result_dict = json.loads(result)

        assert result_dict["count"] == 1
        assert result_dict["active_motions"][0]["scope"] == "pod"

    def test_list_active_motions_empty(self):
        """Test listing motions when none exist."""
        result = self.bob_tools.list_active_motions()
        result_dict = json.loads(result)

        assert result_dict["count"] == 0
        assert result_dict["active_motions"] == []

    def test_list_active_motions_shows_my_vote(self):
        """Test that list shows if worker has already voted."""
        # Create motion and vote
        propose_result = self.alice_tools.propose_motion("Test", "Test", scope="pod")
        motion_id = json.loads(propose_result)["motion_id"]
        self.bob_tools.vote_on_motion(motion_id, "for")

        # List motions
        result = self.bob_tools.list_active_motions()
        result_dict = json.loads(result)

        assert result_dict["active_motions"][0]["my_vote"] == "for"

    def test_get_motion_details(self):
        """Test getting detailed information about a motion."""
        propose_result = self.alice_tools.propose_motion(
            title="Detailed Motion",
            description="Testing details",
            motion_type="PRODUCTION_PRIORITY",
            scope="pod"
        )
        motion_id = json.loads(propose_result)["motion_id"]

        result = self.bob_tools.get_motion_details(motion_id)
        result_dict = json.loads(result)

        assert result_dict["motion_id"] == motion_id
        assert result_dict["title"] == "Detailed Motion"
        assert result_dict["description"] == "Testing details"
        assert result_dict["type"] == "PRODUCTION_PRIORITY"
        assert result_dict["scope"] == "pod"
        assert "votes_for" in result_dict
        assert "votes_against" in result_dict
        assert "participation_rate" in result_dict

    def test_get_motion_details_nonexistent(self):
        """Test getting details for a motion that doesn't exist."""
        result = self.bob_tools.get_motion_details("M999")
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "not found" in result_dict["error"]

    def test_get_motion_details_shows_my_vote(self):
        """Test that motion details show if worker has voted."""
        propose_result = self.alice_tools.propose_motion("Test", "Test", scope="pod")
        motion_id = json.loads(propose_result)["motion_id"]
        self.bob_tools.vote_on_motion(motion_id, "against")

        result = self.bob_tools.get_motion_details(motion_id)
        result_dict = json.loads(result)

        assert result_dict["my_vote"] == "against"

    def test_get_my_vote_helper_for(self):
        """Test _get_my_vote helper when voted for."""
        propose_result = self.alice_tools.propose_motion("Test", "Test", scope="pod")
        motion_id = json.loads(propose_result)["motion_id"]
        motion = self.federation.governance.active_motions[motion_id]
        self.bob_tools.vote_on_motion(motion_id, "for")

        my_vote = self.bob_tools._get_my_vote(motion)
        assert my_vote == "for"

    def test_get_my_vote_helper_against(self):
        """Test _get_my_vote helper when voted against."""
        propose_result = self.alice_tools.propose_motion("Test", "Test", scope="pod")
        motion_id = json.loads(propose_result)["motion_id"]
        motion = self.federation.governance.active_motions[motion_id]
        self.bob_tools.vote_on_motion(motion_id, "against")

        my_vote = self.bob_tools._get_my_vote(motion)
        assert my_vote == "against"

    def test_get_my_vote_helper_abstain(self):
        """Test _get_my_vote helper when abstained."""
        propose_result = self.alice_tools.propose_motion("Test", "Test", scope="pod")
        motion_id = json.loads(propose_result)["motion_id"]
        motion = self.federation.governance.active_motions[motion_id]
        self.bob_tools.vote_on_motion(motion_id, "abstain")

        my_vote = self.bob_tools._get_my_vote(motion)
        assert my_vote == "abstain"

    def test_get_my_vote_helper_not_voted(self):
        """Test _get_my_vote helper when not yet voted."""
        propose_result = self.alice_tools.propose_motion("Test", "Test", scope="pod")
        motion_id = json.loads(propose_result)["motion_id"]
        motion = self.federation.governance.active_motions[motion_id]

        my_vote = self.bob_tools._get_my_vote(motion)
        assert my_vote is None


@pytest.mark.unit
class TestGovernanceToolDefinitions:
    """Test governance tool definitions."""

    def test_get_governance_tool_definitions(self):
        """Test that tool definitions are properly structured."""
        tool_defs = get_governance_tool_definitions()

        assert isinstance(tool_defs, list)
        assert len(tool_defs) == 6  # 6 tools defined

        tool_names = [tool["function"]["name"] for tool in tool_defs]
        expected_tools = [
            "read_constitution",
            "check_my_permissions",
            "list_active_motions",
            "propose_motion",
            "vote_on_motion",
            "get_motion_details"
        ]

        for tool_name in expected_tools:
            assert tool_name in tool_names

    def test_tool_definitions_structure(self):
        """Test that each tool definition has required structure."""
        tool_defs = get_governance_tool_definitions()

        for tool_def in tool_defs:
            assert "type" in tool_def
            assert tool_def["type"] == "function"
            assert "function" in tool_def
            assert "name" in tool_def["function"]
            assert "description" in tool_def["function"]
            assert "parameters" in tool_def["function"]

    def test_read_constitution_definition(self):
        """Test read_constitution tool definition."""
        tool_defs = get_governance_tool_definitions()
        read_const = next(t for t in tool_defs if t["function"]["name"] == "read_constitution")

        assert "level" in read_const["function"]["parameters"]["properties"]
        level_param = read_const["function"]["parameters"]["properties"]["level"]
        assert level_param["type"] == "string"
        assert "federation" in level_param["enum"]
        assert "pod" in level_param["enum"]

    def test_propose_motion_definition(self):
        """Test propose_motion tool definition."""
        tool_defs = get_governance_tool_definitions()
        propose = next(t for t in tool_defs if t["function"]["name"] == "propose_motion")

        params = propose["function"]["parameters"]["properties"]
        assert "title" in params
        assert "description" in params
        assert "motion_type" in params
        assert "scope" in params
        assert "voting_duration" in params

        required = propose["function"]["parameters"]["required"]
        assert "title" in required
        assert "description" in required

    def test_vote_on_motion_definition(self):
        """Test vote_on_motion tool definition."""
        tool_defs = get_governance_tool_definitions()
        vote = next(t for t in tool_defs if t["function"]["name"] == "vote_on_motion")

        params = vote["function"]["parameters"]["properties"]
        assert "motion_id" in params
        assert "choice" in params

        choice_param = params["choice"]
        assert "for" in choice_param["enum"]
        assert "against" in choice_param["enum"]
        assert "abstain" in choice_param["enum"]


@pytest.mark.unit
class TestGovernanceToolsEdgeCases:
    """Test edge cases and coverage gaps."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create workers in different pods
        worker_config1 = WorkerConfig(
            name="Alice",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        worker_config2 = WorkerConfig(
            name="Bob",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )

        pod_config1 = PodConfig(name="PodA", workers=[worker_config1])
        pod_config2 = PodConfig(name="PodB", workers=[worker_config2])

        self.federation = Federation(pods=[pod_config1, pod_config2])
        self.federation.steps = 100

        pod_a = list(self.federation)[0]
        pod_b = list(self.federation)[1]
        self.alice = list(pod_a)[0]
        self.bob = list(pod_b)[0]

    def test_list_active_motions_filters_by_scope(self):
        """Test that list_active_motions filters by scope correctly (line 104)."""
        # Create federation motion
        self.alice.governance_tools.propose_motion(
            "Federation Motion",
            "For all",
            scope="federation"
        )

        # Create pod-specific motion
        self.alice.governance_tools.propose_motion(
            "Pod A Motion",
            "For Pod A only",
            scope="pod"
        )

        # Bob lists federation motions only
        result = self.bob.governance_tools.list_active_motions("federation")
        result_dict = json.loads(result)

        # Should only see federation motion
        assert result_dict["count"] == 1
        assert result_dict["active_motions"][0]["scope"] == "federation"

    def test_list_active_motions_filters_eligible_voters(self):
        """Test that list_active_motions filters by eligible voters (line 108)."""
        # Alice creates a pod-specific motion (only Pod A can vote)
        self.alice.governance_tools.propose_motion(
            "Pod A Only",
            "Only Pod A members",
            scope="pod"
        )

        # Bob (from Pod B) should not see it
        result = self.bob.governance_tools.list_active_motions("all")
        result_dict = json.loads(result)

        # Bob should not see Alice's pod-specific motion
        assert result_dict["count"] == 0

    def test_list_active_motions_with_multiple_scopes(self):
        """Test listing motions with federation and pod scopes (covers lines 104, 108)."""
        # Create federation motion (all can see)
        motion1_result = self.alice.governance_tools.propose_motion(
            "Federation Wide",
            "Everyone can vote",
            scope="federation"
        )

        # Create pod-specific motion (only Pod A)
        motion2_result = self.alice.governance_tools.propose_motion(
            "Pod A Only",
            "Pod A specific",
            scope="pod"
        )

        # Alice (Pod A) should see both when listing "all"
        result = self.alice.governance_tools.list_active_motions("all")
        result_dict = json.loads(result)
        assert result_dict["count"] == 2

        # Bob (Pod B) should only see federation motion
        result = self.bob.governance_tools.list_active_motions("all")
        result_dict = json.loads(result)
        assert result_dict["count"] == 1
        assert result_dict["active_motions"][0]["scope"] == "federation"

    def test_propose_motion_permission_denied(self):
        """Test PermissionError handling in propose_motion (lines 197-198)."""
        from unittest.mock import patch

        # Mock governance.propose_motion to raise PermissionError
        with patch.object(self.federation.governance, 'propose_motion') as mock_propose:
            mock_propose.side_effect = PermissionError("Test permission denied")

            result = self.alice.governance_tools.propose_motion(
                "Should Fail",
                "Test error handling",
                scope="pod"
            )
            result_dict = json.loads(result)

            assert result_dict["success"] is False
            assert "error" in result_dict
            assert "permission denied" in result_dict["error"].lower()


if __name__ == "__main__":
    unittest.main()
