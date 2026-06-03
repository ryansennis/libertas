"""Integration tests for governance system with Federation and Pod."""

import pytest
from mesa_llm.reasoning.cot import CoTReasoning
from libertas.organization import Federation, PodConfig, WorkerConfig
from libertas.governance import (
    Motion,
    MotionType,
    VoteType,
    Constitution,
    ConstitutionLevel,
    Article,
)


@pytest.mark.integration
class TestGovernanceIntegration:
    """Test governance system integrated with Federation and Pod."""

    def test_federation_has_default_constitution(self):
        """Test that Federation initializes with default constitution."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="TestPod",
                    workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"), WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama")]
                )
            ]
        )

        assert federation.constitution is not None
        assert federation.constitution.level == ConstitutionLevel.FEDERATION
        assert len(federation.constitution.articles) > 0
        assert federation.governance is not None

    def test_pod_has_default_constitution(self):
        """Test that Pod initializes with default constitution."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="TestPod",
                    workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama")]
                )
            ]
        )

        pod = list(federation)[0]
        assert pod.constitution is not None
        assert pod.constitution.level == ConstitutionLevel.POD
        assert pod.name in pod.constitution.title

    def test_federation_with_custom_constitution(self):
        """Test creating Federation with custom constitution."""
        custom_constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Custom Federation Constitution",
            preamble="This is a custom constitution",
            articles=[
                Article("A1", "Custom rule", ["custom_permission"], {})
            ]
        )

        federation = Federation(
            pods=[PodConfig(name="Pod1", workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama")])],
            constitution=custom_constitution
        )

        assert federation.constitution.title == "Custom Federation Constitution"
        assert len(federation.constitution.articles) == 1

    def test_pod_with_custom_constitution(self):
        """Test creating Pod with custom constitution."""
        custom_pod_constitution = Constitution(
            level=ConstitutionLevel.POD,
            title="Custom Pod Constitution",
            preamble="Custom pod rules",
            articles=[
                Article("A1", "Pod-specific rule", ["pod_permission"], {})
            ]
        )

        federation = Federation(
            pods=[
                PodConfig(
                    name="CustomPod",
                    workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama")],
                    constitution=custom_pod_constitution
                )
            ]
        )

        pod = list(federation)[0]
        assert pod.constitution.title == "Custom Pod Constitution"

    def test_read_federation_constitution(self):
        """Test reading full text of federation constitution."""
        federation = Federation(
            pods=[PodConfig(name="Pod1", workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama")])]
        )

        constitution_text = federation.constitution.get_full_text()

        assert "Federation" in constitution_text
        assert "Article" in constitution_text
        assert "voting rights" in constitution_text.lower() or "vote" in constitution_text.lower()

    def test_read_pod_constitution(self):
        """Test reading full text of pod constitution."""
        federation = Federation(
            pods=[PodConfig(name="DemocracyPod", workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama")])]
        )

        pod = list(federation)[0]
        constitution_text = pod.constitution.get_full_text()

        assert "DemocracyPod" in constitution_text
        assert "Article" in constitution_text

    def test_propose_motion_at_federation_level(self):
        """Test proposing a motion at federation level."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="Pod1",
                    workers=[
                        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Charlie", reasoning=CoTReasoning, llm_model="ollama/tinyllama")
                    ]
                )
            ]
        )

        # Get all workers
        pod = list(federation)[0]
        workers = list(pod)
        worker_ids = {w.unique_id for w in workers}

        # Propose a motion
        motion = federation.governance.propose_motion(
            proposer=workers[0],
            motion_type=MotionType.POLICY_CHANGE,
            title="Increase profit sharing to 80%",
            description="Change profit distribution policy to allocate 80% to workers",
            scope="federation",
            eligible_voters=worker_ids,
            vote_type=VoteType.SIMPLE_MAJORITY,
            voting_duration=10,
            current_step=0
        )

        assert motion.motion_id == "M0"
        assert motion.title == "Increase profit sharing to 80%"
        assert motion.proposer == workers[0].unique_id
        assert len(motion.eligible_voters) == 3

    def test_workers_vote_on_motion(self):
        """Test workers voting on a motion."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="Pod1",
                    workers=[
                        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Charlie", reasoning=CoTReasoning, llm_model="ollama/tinyllama")
                    ]
                )
            ]
        )

        pod = list(federation)[0]
        workers = list(pod)
        worker_ids = {w.unique_id for w in workers}

        # Propose motion
        motion = federation.governance.propose_motion(
            proposer=workers[0],
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Focus on tools",
            description="Prioritize tool production",
            scope="pod",
            eligible_voters=worker_ids,
            voting_duration=10,
            current_step=0
        )

        # Workers vote
        success1, msg1 = federation.governance.cast_vote(motion.motion_id, workers[0].unique_id, "for")
        success2, msg2 = federation.governance.cast_vote(motion.motion_id, workers[1].unique_id, "for")
        success3, msg3 = federation.governance.cast_vote(motion.motion_id, workers[2].unique_id, "against")

        assert success1 is True
        assert success2 is True
        assert success3 is True
        assert len(motion.votes_for) == 2
        assert len(motion.votes_against) == 1

    def test_motion_passes_with_majority(self):
        """Test that motion passes with simple majority."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="Pod1",
                    workers=[
                        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Charlie", reasoning=CoTReasoning, llm_model="ollama/tinyllama")
                    ]
                )
            ]
        )

        pod = list(federation)[0]
        workers = list(pod)
        worker_ids = {w.unique_id for w in workers}

        # Propose motion with 10 step duration
        motion = federation.governance.propose_motion(
            proposer=workers[0],
            motion_type=MotionType.CUSTOM,
            title="Test Motion",
            description="Testing voting",
            scope="pod",
            eligible_voters=worker_ids,
            voting_duration=10,
            current_step=0
        )

        # 2 vote for, 1 votes against (majority wins)
        federation.governance.cast_vote(motion.motion_id, workers[0].unique_id, "for")
        federation.governance.cast_vote(motion.motion_id, workers[1].unique_id, "for")
        federation.governance.cast_vote(motion.motion_id, workers[2].unique_id, "against")

        # Process votes at step 10 (voting ends)
        completed = federation.governance.process_votes(current_step=10)

        assert len(completed) == 1
        completed_motion, passed = completed[0]
        assert passed is True
        assert completed_motion.status == "passed"

    def test_motion_fails_without_majority(self):
        """Test that motion fails without majority."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="Pod1",
                    workers=[
                        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Charlie", reasoning=CoTReasoning, llm_model="ollama/tinyllama")
                    ]
                )
            ]
        )

        pod = list(federation)[0]
        workers = list(pod)
        worker_ids = {w.unique_id for w in workers}

        motion = federation.governance.propose_motion(
            proposer=workers[0],
            motion_type=MotionType.CUSTOM,
            title="Test Motion",
            description="Testing voting",
            scope="pod",
            eligible_voters=worker_ids,
            voting_duration=10,
            current_step=0
        )

        # 1 votes for, 2 vote against (fails)
        federation.governance.cast_vote(motion.motion_id, workers[0].unique_id, "for")
        federation.governance.cast_vote(motion.motion_id, workers[1].unique_id, "against")
        federation.governance.cast_vote(motion.motion_id, workers[2].unique_id, "against")

        completed = federation.governance.process_votes(current_step=10)

        assert len(completed) == 1
        completed_motion, passed = completed[0]
        assert passed is False
        assert completed_motion.status == "failed"

    def test_check_worker_permissions_from_constitution(self):
        """Test checking worker permissions based on constitution."""
        # Create constitution with currency requirement
        custom_constitution = Constitution(
            level=ConstitutionLevel.FEDERATION,
            title="Test Constitution",
            preamble="Test",
            articles=[
                Article(
                    "A1",
                    "Workers with 1000+ currency can create pods",
                    ["create_pod"],
                    {"min_currency": 1000}
                )
            ]
        )

        federation = Federation(
            pods=[PodConfig(name="Pod1", workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama")])],
            constitution=custom_constitution
        )

        pod = list(federation)[0]
        worker = list(pod)[0]

        # Worker starts with 100 currency (default)
        allowed, reason = federation.constitution.check_permission(worker, "create_pod")
        assert allowed is False
        assert "Insufficient currency" in reason

        # Give worker enough currency
        worker.currency = 1500
        allowed, reason = federation.constitution.check_permission(worker, "create_pod")
        assert allowed is True

    def test_multiple_active_motions(self):
        """Test managing multiple active motions simultaneously."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="Pod1",
                    workers=[
                        WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"),
                        WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama")
                    ]
                )
            ]
        )

        pod = list(federation)[0]
        workers = list(pod)
        worker_ids = {w.unique_id for w in workers}

        # Create multiple motions
        motion1 = federation.governance.propose_motion(
            proposer=workers[0],
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Motion 1",
            description="First motion",
            scope="pod",
            eligible_voters=worker_ids,
            voting_duration=10,
            current_step=0
        )

        motion2 = federation.governance.propose_motion(
            proposer=workers[1],
            motion_type=MotionType.RESOURCE_ALLOCATION,
            title="Motion 2",
            description="Second motion",
            scope="pod",
            eligible_voters=worker_ids,
            voting_duration=20,
            current_step=0
        )

        assert len(federation.governance.active_motions) == 2
        assert motion1.motion_id != motion2.motion_id

        # Vote on both
        federation.governance.cast_vote(motion1.motion_id, workers[0].unique_id, "for")
        federation.governance.cast_vote(motion1.motion_id, workers[1].unique_id, "for")
        federation.governance.cast_vote(motion2.motion_id, workers[0].unique_id, "for")
        federation.governance.cast_vote(motion2.motion_id, workers[1].unique_id, "against")

        # At step 10, only motion1 completes
        completed = federation.governance.process_votes(current_step=10)
        assert len(completed) == 1
        assert completed[0][0].motion_id == motion1.motion_id
        assert len(federation.governance.active_motions) == 1

        # At step 20, motion2 completes
        completed = federation.governance.process_votes(current_step=20)
        assert len(completed) == 1
        assert completed[0][0].motion_id == motion2.motion_id
        assert len(federation.governance.active_motions) == 0

    def test_governance_statistics(self):
        """Test governance statistics tracking."""
        federation = Federation(
            pods=[
                PodConfig(
                    name="Pod1",
                    workers=[WorkerConfig(name="Alice", reasoning=CoTReasoning, llm_model="ollama/tinyllama"), WorkerConfig(name="Bob", reasoning=CoTReasoning, llm_model="ollama/tinyllama")]
                )
            ]
        )

        pod = list(federation)[0]
        workers = list(pod)
        worker_ids = {w.unique_id for w in workers}

        # Initially empty
        stats = federation.governance.get_statistics()
        assert stats["total_motions"] == 0

        # Create and complete a motion
        motion = federation.governance.propose_motion(
            proposer=workers[0],
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters=worker_ids,
            voting_duration=10,
            current_step=0
        )

        federation.governance.cast_vote(motion.motion_id, workers[0].unique_id, "for")
        federation.governance.cast_vote(motion.motion_id, workers[1].unique_id, "for")
        federation.governance.process_votes(current_step=10)

        stats = federation.governance.get_statistics()
        assert stats["total_motions"] == 1
        assert stats["passed"] == 1
        assert stats["failed"] == 0
        assert stats["average_participation_rate"] == 1.0  # Both workers voted
