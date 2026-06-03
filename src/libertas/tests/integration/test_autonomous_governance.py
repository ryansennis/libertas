"""Integration tests for autonomous agent governance behavior."""

import pytest
from mesa_llm.reasoning.cot import CoTReasoning
from libertas.organization import Federation, PodConfig, WorkerConfig
from libertas.governance import MotionType


@pytest.mark.integration
def test_agent_reads_constitution_via_llm():
    """Agent uses LLM tool to read constitution."""
    worker_config = WorkerConfig(
        name="Alice",
        reasoning=CoTReasoning,
        llm_model="ollama/mistral",
        initial_currency=1000.0
    )

    pod_config = PodConfig(name="TestPod", workers=[worker_config])
    federation = Federation(pods=[pod_config])

    worker = list(list(federation)[0])[0]

    # Worker should be able to call governance tools
    result = worker.governance_tools.read_constitution("pod")

    assert "full_text" in result
    assert "Article" in result
    assert "TestPod" in result


@pytest.mark.integration
def test_agent_checks_permissions():
    """Agent checks what actions are permitted."""
    worker_config = WorkerConfig(
        name="Bob",
        reasoning=CoTReasoning,
        llm_model="ollama/mistral",
        initial_currency=1000.0
    )

    pod_config = PodConfig(name="TestPod", workers=[worker_config])
    federation = Federation(pods=[pod_config])

    worker = list(list(federation)[0])[0]

    result = worker.governance_tools.check_my_permissions()

    assert "federation" in result
    assert "pod" in result
    assert "can_vote" in result


@pytest.mark.integration
def test_agent_proposes_motion():
    """Agent proposes a motion using LLM tool."""
    worker_config = WorkerConfig(
        name="Charlie",
        reasoning=CoTReasoning,
        llm_model="ollama/mistral",
        initial_currency=1000.0
    )

    pod_config = PodConfig(name="TechCoop", workers=[worker_config])
    federation = Federation(pods=[pod_config])

    worker = list(list(federation)[0])[0]

    result = worker.governance_tools.propose_motion(
        title="Focus on tool production",
        description="Prioritize manufacturing tools over other goods",
        motion_type="PRODUCTION_PRIORITY",
        scope="pod"
    )

    assert "success" in result
    assert "motion_id" in result
    assert len(federation.governance.active_motions) == 1


@pytest.mark.integration
def test_agent_votes_on_motion():
    """Agent votes on an active motion."""
    worker_configs = [
        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/mistral"),
        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/mistral")
    ]

    pod_config = PodConfig(name="DemoCoop", workers=worker_configs)
    federation = Federation(pods=[pod_config])

    pod = list(federation)[0]
    alice, bob = list(pod)

    # Alice proposes
    result = alice.governance_tools.propose_motion(
        title="Test Motion",
        description="Testing voting",
        scope="pod"
    )
    motion_id = eval(result)["motion_id"]

    # Bob votes
    vote_result = bob.governance_tools.vote_on_motion(motion_id, "for")

    assert "success" in vote_result
    motion = federation.governance.active_motions[motion_id]
    assert bob.unique_id in motion.votes_for


@pytest.mark.integration
def test_agent_lists_active_motions():
    """Agent lists all motions they can vote on."""
    worker_configs = [
        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/mistral"),
        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/mistral")
    ]

    pod_config = PodConfig(name="VoteCoop", workers=worker_configs)
    federation = Federation(pods=[pod_config])

    pod = list(federation)[0]
    alice, bob = list(pod)

    # Create multiple motions
    alice.governance_tools.propose_motion("Motion 1", "First proposal", scope="pod")
    alice.governance_tools.propose_motion("Motion 2", "Second proposal", scope="pod")

    # Bob lists them
    result = bob.governance_tools.list_active_motions()

    assert "active_motions" in result
    assert "count" in result
    motions = eval(result)
    assert motions["count"] == 2


@pytest.mark.integration
def test_agent_get_motion_details():
    """Agent gets detailed information about a motion."""
    worker_config = WorkerConfig(
        name="Alice",
        reasoning=CoTReasoning,
        llm_model="ollama/mistral"
    )

    pod_config = PodConfig(name="DetailCoop", workers=[worker_config])
    federation = Federation(pods=[pod_config])

    alice = list(list(federation)[0])[0]

    # Propose motion
    result = alice.governance_tools.propose_motion(
        title="Important Decision",
        description="Very important governance decision",
        scope="pod"
    )
    motion_id = eval(result)["motion_id"]

    # Get details
    details = alice.governance_tools.get_motion_details(motion_id)

    assert "motion_id" in details
    assert "title" in details
    assert "Important Decision" in details
