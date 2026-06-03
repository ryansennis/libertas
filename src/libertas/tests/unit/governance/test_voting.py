"""Unit tests for voting system."""

import pytest
from libertas.governance import (
    Motion,
    MotionType,
    VoteType,
    GovernanceEngine,
    Constitution,
    ConstitutionLevel,
    Article,
)


@pytest.mark.unit
class TestMotion:
    """Test Motion class."""

    def test_motion_creation(self):
        """Test creating a motion."""
        motion = Motion(
            motion_id="M1",
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Prioritize tool production",
            description="Focus on producing more tools this quarter",
            proposer="worker_001",
            scope="pod_alpha",
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters={"worker_001", "worker_002", "worker_003"}
        )

        assert motion.motion_id == "M1"
        assert motion.motion_type == MotionType.PRODUCTION_PRIORITY
        assert motion.title == "Prioritize tool production"
        assert motion.proposer == "worker_001"
        assert motion.scope == "pod_alpha"
        assert motion.status == "active"
        assert len(motion.eligible_voters) == 3
        assert len(motion.votes_for) == 0
        assert len(motion.votes_against) == 0

    def test_cast_vote_for(self):
        """Test casting a vote for."""
        motion = Motion(
            motion_id="M2",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001", "worker_002"}
        )

        success, message = motion.cast_vote("worker_001", "for")

        assert success is True
        assert "recorded" in message.lower()
        assert "worker_001" in motion.votes_for
        assert "worker_001" not in motion.votes_against
        assert "worker_001" not in motion.abstentions

    def test_cast_vote_against(self):
        """Test casting a vote against."""
        motion = Motion(
            motion_id="M3",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001", "worker_002"}
        )

        success, message = motion.cast_vote("worker_002", "against")

        assert success is True
        assert "worker_002" in motion.votes_against
        assert "worker_002" not in motion.votes_for

    def test_cast_vote_abstain(self):
        """Test abstaining from vote."""
        motion = Motion(
            motion_id="M4",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001", "worker_002"}
        )

        success, message = motion.cast_vote("worker_001", "abstain")

        assert success is True
        assert "worker_001" in motion.abstentions
        assert "worker_001" not in motion.votes_for
        assert "worker_001" not in motion.votes_against

    def test_cast_vote_not_eligible(self):
        """Test voting when not eligible."""
        motion = Motion(
            motion_id="M5",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001"}
        )

        success, message = motion.cast_vote("worker_999", "for")

        assert success is False
        assert "not eligible" in message.lower()
        assert "worker_999" not in motion.votes_for

    def test_cast_vote_motion_not_active(self):
        """Test voting on closed motion."""
        motion = Motion(
            motion_id="M6",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001"},
            status="passed"
        )

        success, message = motion.cast_vote("worker_001", "for")

        assert success is False
        assert "voting is closed" in message.lower()

    def test_cast_vote_invalid_choice(self):
        """Test voting with invalid choice."""
        motion = Motion(
            motion_id="M7",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001"}
        )

        success, message = motion.cast_vote("worker_001", "maybe")

        assert success is False
        assert "invalid" in message.lower()

    def test_change_vote(self):
        """Test changing vote from for to against."""
        motion = Motion(
            motion_id="M8",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"worker_001"}
        )

        # First vote for
        motion.cast_vote("worker_001", "for")
        assert "worker_001" in motion.votes_for

        # Change to against
        motion.cast_vote("worker_001", "against")
        assert "worker_001" in motion.votes_against
        assert "worker_001" not in motion.votes_for

    def test_tally_votes_simple_majority_passes(self):
        """Test tallying with simple majority that passes."""
        motion = Motion(
            motion_id="M9",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            eligible_voters={"worker_001", "worker_002", "worker_003"}
        )

        motion.cast_vote("worker_001", "for")
        motion.cast_vote("worker_002", "for")
        motion.cast_vote("worker_003", "against")

        passed, results = motion.tally_votes()

        assert passed is True
        assert results["votes_for"] == 2
        assert results["votes_against"] == 1
        assert results["support_ratio"] > 0.5

    def test_tally_votes_simple_majority_fails(self):
        """Test tallying with simple majority that fails."""
        motion = Motion(
            motion_id="M10",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            eligible_voters={"worker_001", "worker_002", "worker_003"}
        )

        motion.cast_vote("worker_001", "for")
        motion.cast_vote("worker_002", "against")
        motion.cast_vote("worker_003", "against")

        passed, results = motion.tally_votes()

        assert passed is False
        assert results["votes_for"] == 1
        assert results["votes_against"] == 2

    def test_tally_votes_supermajority(self):
        """Test tallying with supermajority requirement."""
        motion = Motion(
            motion_id="M11",
            motion_type=MotionType.CONSTITUTIONAL_AMENDMENT,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="federation",
            vote_type=VoteType.SUPERMAJORITY,
            required_threshold=0.67,
            eligible_voters={"w1", "w2", "w3", "w4", "w5", "w6"}
        )

        # 4 for, 2 against = 66.67% (fails 67%)
        motion.cast_vote("w1", "for")
        motion.cast_vote("w2", "for")
        motion.cast_vote("w3", "for")
        motion.cast_vote("w4", "for")
        motion.cast_vote("w5", "against")
        motion.cast_vote("w6", "against")

        passed, results = motion.tally_votes()

        assert passed is False
        assert results["support_ratio"] < 0.67

        # Add one more for vote: 5 for, 2 against = 71.43% (passes 67%)
        motion.cast_vote("w6", "for")  # Change vote
        passed, results = motion.tally_votes()

        assert passed is True

    def test_tally_votes_unanimous(self):
        """Test tallying with unanimous requirement."""
        motion = Motion(
            motion_id="M12",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            vote_type=VoteType.UNANIMOUS,
            eligible_voters={"w1", "w2", "w3"}
        )

        # All vote for
        motion.cast_vote("w1", "for")
        motion.cast_vote("w2", "for")
        motion.cast_vote("w3", "for")

        passed, results = motion.tally_votes()

        assert passed is True
        assert results["support_ratio"] == 1.0

        # One changes to against
        motion.cast_vote("w3", "against")
        passed, results = motion.tally_votes()

        assert passed is False

    def test_tally_votes_no_votes(self):
        """Test tallying when no votes cast."""
        motion = Motion(
            motion_id="M13",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"w1", "w2"}
        )

        passed, results = motion.tally_votes()

        assert passed is False
        assert "no votes cast" in results["reason"]

    def test_tally_votes_abstentions_not_counted(self):
        """Test that abstentions don't count toward vote total."""
        motion = Motion(
            motion_id="M14",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            vote_type=VoteType.SIMPLE_MAJORITY,
            eligible_voters={"w1", "w2", "w3", "w4"}
        )

        motion.cast_vote("w1", "for")
        motion.cast_vote("w2", "for")
        motion.cast_vote("w3", "against")
        motion.cast_vote("w4", "abstain")

        passed, results = motion.tally_votes()

        # 2 for, 1 against, 1 abstain → support = 2/3 = 66.67%
        assert passed is True
        assert results["votes_for"] == 2
        assert results["votes_against"] == 1
        assert results["abstentions"] == 1
        assert results["total_votes"] == 3  # Abstentions not counted

    def test_get_participation_rate(self):
        """Test calculating participation rate."""
        motion = Motion(
            motion_id="M15",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters={"w1", "w2", "w3", "w4", "w5"}
        )

        # Initially 0%
        assert motion.get_participation_rate() == 0.0

        # 2 vote (40%)
        motion.cast_vote("w1", "for")
        motion.cast_vote("w2", "against")
        assert motion.get_participation_rate() == 0.4

        # 1 abstains (60%)
        motion.cast_vote("w3", "abstain")
        assert motion.get_participation_rate() == 0.6

        # All vote (100%)
        motion.cast_vote("w4", "for")
        motion.cast_vote("w5", "for")
        assert motion.get_participation_rate() == 1.0

    def test_get_participation_rate_no_eligible(self):
        """Test participation rate with no eligible voters."""
        motion = Motion(
            motion_id="M16",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod",
            eligible_voters=set()
        )

        assert motion.get_participation_rate() == 0.0

    def test_to_dict(self):
        """Test serializing motion to dict."""
        motion = Motion(
            motion_id="M17",
            motion_type=MotionType.POD_CREATION,
            title="Create new pod",
            description="Form research pod",
            proposer="worker_001",
            scope="federation",
            vote_type=VoteType.SUPERMAJORITY,
            required_threshold=0.6,
            eligible_voters={"w1", "w2"},
            created_step=100,
            voting_ends_step=200
        )

        motion.cast_vote("w1", "for")
        data = motion.to_dict()

        assert data["motion_id"] == "M17"
        assert data["motion_type"] == "POD_CREATION"
        assert data["title"] == "Create new pod"
        assert data["proposer"] == "worker_001"
        assert data["vote_type"] == "SUPERMAJORITY"
        assert data["required_threshold"] == 0.6
        assert "w1" in data["eligible_voters"]
        assert "w1" in data["votes_for"]
        assert data["created_step"] == 100
        assert data["voting_ends_step"] == 200

    def test_to_json(self):
        """Test serializing motion to JSON."""
        motion = Motion(
            motion_id="M18",
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            proposer="worker_001",
            scope="pod"
        )

        json_str = motion.to_json()

        assert "M18" in json_str
        assert "CUSTOM" in json_str
        assert "worker_001" in json_str


@pytest.mark.unit
class TestGovernanceEngine:
    """Test GovernanceEngine class."""

    def test_engine_initialization(self):
        """Test creating governance engine."""
        engine = GovernanceEngine()

        assert len(engine.active_motions) == 0
        assert len(engine.motion_history) == 0
        assert engine.motion_counter == 0

    def test_propose_motion(self):
        """Test proposing a new motion."""
        engine = GovernanceEngine()

        # Mock constitution that allows proposals
        class MockConstitution:
            def check_permission(self, agent, action):
                return (True, "Allowed")

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.PRODUCTION_PRIORITY,
            title="Focus on tools",
            description="Prioritize tool production",
            scope="pod_alpha",
            eligible_voters={"worker_001", "worker_002", "worker_003"},
            vote_type=VoteType.SIMPLE_MAJORITY,
            required_threshold=0.5,
            voting_duration=50,
            current_step=100
        )

        assert motion.motion_id == "M0"
        assert motion.title == "Focus on tools"
        assert motion.proposer == "worker_001"
        assert motion.created_step == 100
        assert motion.voting_ends_step == 150  # 100 + 50
        assert motion.status == "active"
        assert "M0" in engine.active_motions
        assert engine.motion_counter == 1

    def test_propose_motion_no_duration(self):
        """Test proposing motion with no voting duration."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters={"worker_001"},
            voting_duration=None,
            current_step=50
        )

        assert motion.voting_ends_step is None

    def test_propose_multiple_motions(self):
        """Test proposing multiple motions."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 1",
            description="First",
            scope="pod",
            eligible_voters={"worker_001"}
        )

        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 2",
            description="Second",
            scope="pod",
            eligible_voters={"worker_001"}
        )

        assert motion1.motion_id == "M0"
        assert motion2.motion_id == "M1"
        assert len(engine.active_motions) == 2
        assert engine.motion_counter == 2

    def test_cast_vote(self):
        """Test casting vote through engine."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters={"worker_001", "worker_002"}
        )

        success, message = engine.cast_vote("M0", "worker_002", "for")

        assert success is True
        assert "worker_002" in motion.votes_for

    def test_cast_vote_motion_not_found(self):
        """Test voting on non-existent motion."""
        engine = GovernanceEngine()

        success, message = engine.cast_vote("M999", "worker_001", "for")

        assert success is False
        assert "not found" in message.lower()

    def test_process_votes_motion_ends(self):
        """Test processing votes when voting period ends."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters={"worker_001", "worker_002", "worker_003"},
            voting_duration=50,
            current_step=100
        )

        # Cast votes
        engine.cast_vote("M0", "worker_001", "for")
        engine.cast_vote("M0", "worker_002", "for")
        engine.cast_vote("M0", "worker_003", "against")

        # Process votes at step 149 (not ended yet)
        completed = engine.process_votes(149)
        assert len(completed) == 0
        assert "M0" in engine.active_motions

        # Process votes at step 150 (ended)
        completed = engine.process_votes(150)
        assert len(completed) == 1
        assert "M0" not in engine.active_motions
        assert len(engine.motion_history) == 1

        completed_motion, passed = completed[0]
        assert completed_motion.motion_id == "M0"
        assert passed is True
        assert completed_motion.status == "passed"

    def test_process_votes_motion_fails(self):
        """Test processing votes for motion that fails."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters={"w1", "w2", "w3"},
            voting_duration=10,
            current_step=0
        )

        # Majority votes against
        engine.cast_vote("M0", "w1", "against")
        engine.cast_vote("M0", "w2", "against")
        engine.cast_vote("M0", "w3", "for")

        completed = engine.process_votes(10)
        completed_motion, passed = completed[0]

        assert passed is False
        assert completed_motion.status == "failed"

    def test_process_votes_multiple_motions(self):
        """Test processing multiple motions ending at different times."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        # Motion ending at step 10
        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 1",
            description="Test",
            scope="pod",
            eligible_voters={"w1"},
            voting_duration=10,
            current_step=0
        )

        # Motion ending at step 20
        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 2",
            description="Test",
            scope="pod",
            eligible_voters={"w1"},
            voting_duration=20,
            current_step=0
        )

        engine.cast_vote("M0", "w1", "for")
        engine.cast_vote("M1", "w1", "for")

        # At step 10, only first motion completes
        completed = engine.process_votes(10)
        assert len(completed) == 1
        assert completed[0][0].motion_id == "M0"
        assert len(engine.active_motions) == 1
        assert "M1" in engine.active_motions

        # At step 20, second motion completes
        completed = engine.process_votes(20)
        assert len(completed) == 1
        assert completed[0][0].motion_id == "M1"
        assert len(engine.active_motions) == 0
        assert len(engine.motion_history) == 2

    def test_close_motion(self):
        """Test manually closing a motion."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters={"w1", "w2"}
        )

        engine.cast_vote("M0", "w1", "for")
        engine.cast_vote("M0", "w2", "for")

        result = engine.close_motion("M0", current_step=50)

        assert result is not None
        motion, passed = result
        assert passed is True
        assert motion.status == "passed"
        assert motion.voting_ends_step == 50
        assert "M0" not in engine.active_motions
        assert len(engine.motion_history) == 1

    def test_close_motion_not_found(self):
        """Test closing non-existent motion."""
        engine = GovernanceEngine()

        result = engine.close_motion("M999", current_step=100)

        assert result is None

    def test_cancel_motion(self):
        """Test cancelling a motion."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        motion = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Test",
            description="Test",
            scope="pod",
            eligible_voters={"w1"}
        )

        result = engine.cancel_motion("M0")

        assert result is True
        assert motion.status == "cancelled"
        assert "M0" not in engine.active_motions
        assert len(engine.motion_history) == 1

    def test_cancel_motion_not_found(self):
        """Test cancelling non-existent motion."""
        engine = GovernanceEngine()

        result = engine.cancel_motion("M999")

        assert result is False

    def test_get_active_motions_for_voter(self):
        """Test getting motions a voter can vote on."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        # Motion 1: worker_001 and worker_002 eligible
        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 1",
            description="Test",
            scope="pod",
            eligible_voters={"worker_001", "worker_002"}
        )

        # Motion 2: only worker_002 eligible
        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 2",
            description="Test",
            scope="pod",
            eligible_voters={"worker_002"}
        )

        # worker_001 sees only Motion 1
        motions = engine.get_active_motions_for_voter("worker_001")
        assert len(motions) == 1
        assert motions[0].motion_id == "M0"

        # worker_002 sees both
        motions = engine.get_active_motions_for_voter("worker_002")
        assert len(motions) == 2

        # worker_003 sees none
        motions = engine.get_active_motions_for_voter("worker_003")
        assert len(motions) == 0

    def test_get_motion(self):
        """Test getting motion by ID."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        # Create active motion
        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Active",
            description="Test",
            scope="pod",
            eligible_voters={"w1"}
        )

        # Create and close motion
        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Historical",
            description="Test",
            scope="pod",
            eligible_voters={"w1"},
            voting_duration=10,
            current_step=0
        )
        engine.cast_vote("M1", "w1", "for")
        engine.process_votes(10)

        # Get active motion
        found = engine.get_motion("M0")
        assert found is not None
        assert found.title == "Active"

        # Get historical motion
        found = engine.get_motion("M1")
        assert found is not None
        assert found.title == "Historical"

        # Get non-existent motion
        found = engine.get_motion("M999")
        assert found is None

    def test_get_motions_by_scope(self):
        """Test getting motions by scope."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        # Federation motion
        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Fed motion",
            description="Test",
            scope="federation",
            eligible_voters={"w1"}
        )

        # Pod A motion
        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Pod A motion",
            description="Test",
            scope="pod_alpha",
            eligible_voters={"w1"}
        )

        # Pod B motion (closed)
        motion3 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Pod B motion",
            description="Test",
            scope="pod_beta",
            eligible_voters={"w1"},
            voting_duration=10,
            current_step=0
        )
        engine.cast_vote("M2", "w1", "for")
        engine.process_votes(10)

        # Get federation motions
        motions = engine.get_motions_by_scope("federation")
        assert len(motions) == 1
        assert motions[0].title == "Fed motion"

        # Get pod_alpha motions
        motions = engine.get_motions_by_scope("pod_alpha")
        assert len(motions) == 1

        # Get pod_beta motions (includes historical)
        motions = engine.get_motions_by_scope("pod_beta")
        assert len(motions) == 1

    def test_get_statistics(self):
        """Test getting governance statistics."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        # Initially empty
        stats = engine.get_statistics()
        assert stats["total_motions"] == 0
        assert stats["active_motions"] == 0
        assert stats["passed"] == 0

        # Create and pass a motion
        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 1",
            description="Test",
            scope="pod",
            eligible_voters={"w1", "w2"},
            voting_duration=10,
            current_step=0
        )
        engine.cast_vote("M0", "w1", "for")
        engine.cast_vote("M0", "w2", "for")
        engine.process_votes(10)

        # Create and fail a motion
        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 2",
            description="Test",
            scope="pod",
            eligible_voters={"w1"},
            voting_duration=10,
            current_step=10
        )
        engine.cast_vote("M1", "w1", "against")
        engine.process_votes(20)

        # Create and cancel a motion
        motion3 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 3",
            description="Test",
            scope="pod",
            eligible_voters={"w1"}
        )
        engine.cancel_motion("M2")

        # Create active motion
        motion4 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="Motion 4",
            description="Test",
            scope="pod",
            eligible_voters={"w1"}
        )

        stats = engine.get_statistics()

        assert stats["total_motions"] == 4
        assert stats["active_motions"] == 1
        assert stats["historical_motions"] == 3
        assert stats["passed"] == 1
        assert stats["failed"] == 1
        assert stats["cancelled"] == 1
        assert stats["average_participation_rate"] > 0

    def test_get_statistics_participation_rate(self):
        """Test average participation rate calculation."""
        engine = GovernanceEngine()

        class MockProposer:
            unique_id = "worker_001"

        # Motion with 100% participation
        motion1 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="M1",
            description="Test",
            scope="pod",
            eligible_voters={"w1", "w2"},
            voting_duration=10,
            current_step=0
        )
        engine.cast_vote("M0", "w1", "for")
        engine.cast_vote("M0", "w2", "for")
        engine.process_votes(10)

        # Motion with 50% participation
        motion2 = engine.propose_motion(
            proposer=MockProposer(),
            motion_type=MotionType.CUSTOM,
            title="M2",
            description="Test",
            scope="pod",
            eligible_voters={"w1", "w2"},
            voting_duration=10,
            current_step=10
        )
        engine.cast_vote("M1", "w1", "for")
        engine.process_votes(20)

        stats = engine.get_statistics()

        # Average = (1.0 + 0.5) / 2 = 0.75
        assert stats["average_participation_rate"] == 0.75
