"""Voting and governance engine for democratic decision-making."""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto
import json
import uuid


class VoteType(Enum):
    """Type of voting mechanism."""
    SIMPLE_MAJORITY = auto()  # >50%
    SUPERMAJORITY = auto()    # Configurable threshold (e.g., 2/3)
    UNANIMOUS = auto()         # 100%
    RANKED_CHOICE = auto()     # Future: ranked preferences
    APPROVAL = auto()          # Future: approve multiple options


class MotionType(Enum):
    """Type of motion being voted on."""
    CONSTITUTIONAL_AMENDMENT = auto()
    POD_CREATION = auto()
    PRODUCTION_PRIORITY = auto()
    RESOURCE_ALLOCATION = auto()
    LEADERSHIP_ELECTION = auto()
    CONTRACT_APPROVAL = auto()
    BUDGET_APPROVAL = auto()
    POLICY_CHANGE = auto()
    CUSTOM = auto()


@dataclass
class Motion:
    """A proposal to be voted on by members of an organization.

    Motions are the primary mechanism for democratic decision-making.
    They can propose constitutional amendments, new policies, resource
    allocations, or any other decision requiring collective approval.
    """

    motion_id: str
    """Unique identifier"""

    motion_type: MotionType
    """Type of motion"""

    title: str
    """Short title of the motion"""

    description: str
    """Full description for voters to read"""

    proposer: str
    """Worker ID of proposer"""

    scope: str
    """Scope of motion (e.g., 'federation', 'pod_001')"""

    vote_type: VoteType = VoteType.SIMPLE_MAJORITY
    """Voting mechanism to use"""

    required_threshold: float = 0.5
    """Vote threshold for passage (0-1)"""

    eligible_voters: Set[str] = field(default_factory=set)
    """Set of worker IDs eligible to vote"""

    votes_for: Set[str] = field(default_factory=set)
    """Worker IDs who voted for"""

    votes_against: Set[str] = field(default_factory=set)
    """Worker IDs who voted against"""

    abstentions: Set[str] = field(default_factory=set)
    """Worker IDs who abstained"""

    status: str = "active"
    """Status: active, passed, failed, cancelled"""

    created_step: int = 0
    """Simulation step when created"""

    voting_ends_step: Optional[int] = None
    """Simulation step when voting closes (None = stays open)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata for specific motion types"""

    def cast_vote(self, voter_id: str, choice: str) -> Tuple[bool, str]:
        """Record a vote from an eligible voter.

        Args:
            voter_id: ID of the voting worker
            choice: 'for', 'against', or 'abstain'

        Returns:
            Tuple of (success, message)
        """
        if self.status != "active":
            return False, f"Motion is {self.status}, voting is closed"

        if voter_id not in self.eligible_voters:
            return False, "Not eligible to vote on this motion"

        # Remove from other vote sets if already voted
        self.votes_for.discard(voter_id)
        self.votes_against.discard(voter_id)
        self.abstentions.discard(voter_id)

        # Record new vote
        if choice == "for":
            self.votes_for.add(voter_id)
        elif choice == "against":
            self.votes_against.add(voter_id)
        elif choice == "abstain":
            self.abstentions.add(voter_id)
        else:
            return False, f"Invalid choice: {choice}"

        return True, f"Vote recorded: {choice}"

    def tally_votes(self) -> Tuple[bool, Dict[str, Any]]:
        """Tally votes and determine if motion passes.

        Returns:
            Tuple of (passed, results_dict)
        """
        total_votes = len(self.votes_for) + len(self.votes_against)
        total_eligible = len(self.eligible_voters)

        if total_votes == 0:
            return False, {
                "reason": "no votes cast",
                "votes_for": 0,
                "votes_against": 0,
                "abstentions": len(self.abstentions),
                "total_eligible": total_eligible,
            }

        support_ratio = len(self.votes_for) / total_votes

        # Check based on vote type
        if self.vote_type == VoteType.SIMPLE_MAJORITY:
            passed = support_ratio > 0.5
        elif self.vote_type == VoteType.SUPERMAJORITY:
            passed = support_ratio >= self.required_threshold
        elif self.vote_type == VoteType.UNANIMOUS:
            passed = support_ratio == 1.0
        else:
            passed = False

        results = {
            "votes_for": len(self.votes_for),
            "votes_against": len(self.votes_against),
            "abstentions": len(self.abstentions),
            "total_votes": total_votes,
            "total_eligible": total_eligible,
            "support_ratio": support_ratio,
            "required_threshold": self.required_threshold,
            "passed": passed,
        }

        return passed, results

    def get_participation_rate(self) -> float:
        """Calculate voter participation rate."""
        if not self.eligible_voters:
            return 0.0

        total_participated = len(self.votes_for) + len(self.votes_against) + len(self.abstentions)
        return total_participated / len(self.eligible_voters)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize motion to dictionary."""
        return {
            "motion_id": self.motion_id,
            "motion_type": self.motion_type.name,
            "title": self.title,
            "description": self.description,
            "proposer": self.proposer,
            "scope": self.scope,
            "vote_type": self.vote_type.name,
            "required_threshold": self.required_threshold,
            "eligible_voters": list(self.eligible_voters),
            "votes_for": list(self.votes_for),
            "votes_against": list(self.votes_against),
            "abstentions": list(self.abstentions),
            "status": self.status,
            "created_step": self.created_step,
            "voting_ends_step": self.voting_ends_step,
            "metadata": self.metadata.copy(),
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class GovernanceEngine:
    """Manages all active motions and voting processes.

    The GovernanceEngine is the central coordinator for democratic
    decision-making. It tracks active motions, processes votes,
    and executes approved motions.
    """

    def __init__(self):
        self.active_motions: Dict[str, Motion] = {}
        """Currently active motions"""

        self.motion_history: List[Motion] = []
        """History of completed motions"""

        self.motion_counter: int = 0
        """Counter for generating motion IDs"""

    def propose_motion(
        self,
        proposer: str,
        motion_type: MotionType,
        title: str,
        description: str,
        scope: str,
        eligible_voters: Set[str],
        vote_type: VoteType = VoteType.SIMPLE_MAJORITY,
        required_threshold: float = 0.5,
        voting_duration: Optional[int] = None,
        current_step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Motion:
        """Create a new motion for voting.

        Args:
            proposer: Worker ID of proposer
            motion_type: Type of motion
            title: Short title
            description: Full description
            scope: Scope identifier (e.g., 'federation', 'pod_001')
            eligible_voters: Set of worker IDs eligible to vote
            vote_type: Voting mechanism
            required_threshold: Threshold for passage (if applicable)
            voting_duration: Number of steps until voting closes (None = indefinite)
            current_step: Current simulation step
            metadata: Additional data for specific motion types

        Returns:
            Created Motion object
        """
        motion_id = f"M{self.motion_counter}"
        self.motion_counter += 1

        voting_ends_step = None
        if voting_duration is not None:
            voting_ends_step = current_step + voting_duration

        # Extract worker ID from proposer
        proposer_id = proposer if isinstance(proposer, str) else proposer.unique_id

        motion = Motion(
            motion_id=motion_id,
            motion_type=motion_type,
            title=title,
            description=description,
            proposer=proposer_id,
            scope=scope,
            vote_type=vote_type,
            required_threshold=required_threshold,
            eligible_voters=eligible_voters.copy(),
            created_step=current_step,
            voting_ends_step=voting_ends_step,
            metadata=metadata or {},
        )

        self.active_motions[motion_id] = motion
        return motion

    def cast_vote(
        self,
        motion_id: str,
        voter_id: str,
        choice: str
    ) -> Tuple[bool, str]:
        """Cast a vote on an active motion.

        Args:
            motion_id: ID of the motion
            voter_id: ID of the voting worker
            choice: 'for', 'against', or 'abstain'

        Returns:
            Tuple of (success, message)
        """
        motion = self.active_motions.get(motion_id)
        if not motion:
            return False, f"Motion {motion_id} not found or not active"

        return motion.cast_vote(voter_id, choice)

    def process_votes(self, current_step: int) -> List[Tuple[Motion, bool]]:
        """Process and tally votes for motions whose voting period has ended.

        Args:
            current_step: Current simulation step

        Returns:
            List of (motion, passed) tuples for completed motions
        """
        completed = []

        for motion_id, motion in list(self.active_motions.items()):
            # Check if voting period has ended
            if motion.voting_ends_step is not None and current_step >= motion.voting_ends_step:
                passed, results = motion.tally_votes()

                motion.status = "passed" if passed else "failed"
                motion.metadata["final_results"] = results

                self.motion_history.append(motion)
                del self.active_motions[motion_id]

                completed.append((motion, passed))

        return completed

    def close_motion(self, motion_id: str, current_step: int) -> Optional[Tuple[Motion, bool]]:
        """Manually close a motion and tally votes.

        Args:
            motion_id: ID of motion to close
            current_step: Current simulation step

        Returns:
            Tuple of (motion, passed) or None if not found
        """
        motion = self.active_motions.get(motion_id)
        if not motion:
            return None

        passed, results = motion.tally_votes()
        motion.status = "passed" if passed else "failed"
        motion.metadata["final_results"] = results
        motion.voting_ends_step = current_step

        self.motion_history.append(motion)
        del self.active_motions[motion_id]

        return (motion, passed)

    def cancel_motion(self, motion_id: str) -> bool:
        """Cancel an active motion.

        Args:
            motion_id: ID of motion to cancel

        Returns:
            True if motion was cancelled
        """
        motion = self.active_motions.get(motion_id)
        if not motion:
            return False

        motion.status = "cancelled"
        self.motion_history.append(motion)
        del self.active_motions[motion_id]

        return True

    def get_active_motions_for_voter(self, voter_id: str) -> List[Motion]:
        """Get all active motions a voter is eligible to vote on.

        Args:
            voter_id: Worker ID

        Returns:
            List of motions the voter can vote on
        """
        return [
            motion for motion in self.active_motions.values()
            if voter_id in motion.eligible_voters
        ]

    def get_motion(self, motion_id: str) -> Optional[Motion]:
        """Get a motion by ID (active or historical).

        Args:
            motion_id: Motion ID

        Returns:
            Motion object or None
        """
        # Check active motions first
        if motion_id in self.active_motions:
            return self.active_motions[motion_id]

        # Check history
        for motion in self.motion_history:
            if motion.motion_id == motion_id:
                return motion

        return None

    def get_motions_by_scope(self, scope: str) -> List[Motion]:
        """Get all motions (active and historical) for a specific scope.

        Args:
            scope: Scope identifier

        Returns:
            List of motions
        """
        motions = []

        # Active motions
        motions.extend([m for m in self.active_motions.values() if m.scope == scope])

        # Historical motions
        motions.extend([m for m in self.motion_history if m.scope == scope])

        return motions

    def get_statistics(self) -> Dict[str, Any]:
        """Get governance statistics.

        Returns:
            Dictionary of statistics
        """
        total_motions = len(self.active_motions) + len(self.motion_history)
        passed = sum(1 for m in self.motion_history if m.status == "passed")
        failed = sum(1 for m in self.motion_history if m.status == "failed")
        cancelled = sum(1 for m in self.motion_history if m.status == "cancelled")

        avg_participation = 0.0
        if self.motion_history:
            avg_participation = sum(
                m.get_participation_rate() for m in self.motion_history
            ) / len(self.motion_history)

        return {
            "total_motions": total_motions,
            "active_motions": len(self.active_motions),
            "historical_motions": len(self.motion_history),
            "passed": passed,
            "failed": failed,
            "cancelled": cancelled,
            "average_participation_rate": avg_participation,
        }
