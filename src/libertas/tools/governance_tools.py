"""Function calling tools for workers to interact with governance systems."""

from typing import Dict, List, Optional
import json


class GovernanceTools:
    """
    Tool functions for workers to interact with governance systems.

    These are registered with the LLM agent and called via function calling.
    """

    def __init__(self, worker):
        self.worker = worker

    def get_pod(self):
        """Get the pod this worker belongs to."""
        return self.worker.pod

    def get_federation(self):
        """Get the federation this worker belongs to."""
        return self.worker.federation

    # ===== Constitution Tools =====

    def read_constitution(self, level: str = "pod") -> str:
        """
        Read the full constitutional text.

        Args:
            level: "federation" or "pod" (default: "pod")

        Returns:
            JSON string with constitutional text
        """
        if level == "federation":
            constitution = self.get_federation().constitution
        elif level == "pod":
            pod = self.get_pod()
            if not pod:
                return json.dumps({"error": "Worker not assigned to a pod"})
            constitution = pod.constitution
        else:
            return json.dumps({"error": f"Invalid level: {level}"})

        return json.dumps({
            "level": level,
            "title": constitution.title,
            "full_text": constitution.get_full_text(),
            "version": constitution.version
        }, indent=2)

    def check_my_permissions(self) -> str:
        """
        Check what actions are currently permitted under constitutions.

        Returns:
            JSON string with permission details
        """
        federation = self.get_federation()
        pod = self.get_pod()

        permissions = {}

        # Check federation permissions
        fed_const = federation.constitution
        permissions["federation"] = {
            "can_vote": fed_const.check_permission(self.worker, "vote")[0],
            "can_propose_motion": fed_const.check_permission(self.worker, "propose_motion")[0],
            "can_create_pod": fed_const.check_permission(self.worker, "create_pod")[0],
            "can_propose_amendment": fed_const.check_permission(self.worker, "propose_amendment")[0]
        }

        # Check pod permissions
        if pod:
            pod_const = pod.constitution
            permissions["pod"] = {
                "can_vote": pod_const.check_permission(self.worker, "vote")[0],
                "can_propose_motion": pod_const.check_permission(self.worker, "propose_motion")[0],
                "can_start_production": pod_const.check_permission(self.worker, "start_production")[0]
            }

        return json.dumps(permissions, indent=2)

    # ===== Motion Tools =====

    def list_active_motions(self, scope: str = "all") -> str:
        """
        List all motions currently open for voting.

        Args:
            scope: "federation", "pod", or "all" (default: "all")

        Returns:
            JSON string with motion list
        """
        federation = self.get_federation()
        motions = []

        for motion in federation.governance.active_motions.values():
            # Filter by scope
            if scope != "all" and motion.scope != scope:
                continue

            # Only show motions this worker can vote on
            if self.worker.unique_id not in motion.eligible_voters:
                continue

            motions.append({
                "motion_id": motion.motion_id,
                "title": motion.title,
                "description": motion.description,
                "type": motion.motion_type.name,
                "scope": motion.scope,
                "proposer": motion.proposer,
                "vote_type": motion.vote_type.name,
                "voting_ends_step": motion.voting_ends_step,
                "votes_for": len(motion.votes_for),
                "votes_against": len(motion.votes_against),
                "my_vote": self._get_my_vote(motion)
            })

        return json.dumps({
            "active_motions": motions,
            "count": len(motions)
        }, indent=2)

    def _get_my_vote(self, motion) -> Optional[str]:
        """Check if this worker has voted on this motion."""
        worker_id = self.worker.unique_id
        if worker_id in motion.votes_for:
            return "for"
        elif worker_id in motion.votes_against:
            return "against"
        elif worker_id in motion.abstentions:
            return "abstain"
        return None

    def propose_motion(self, title: str, description: str,
                      motion_type: str = "CUSTOM", scope: str = "pod",
                      voting_duration: int = 50) -> str:
        """
        Create a new motion for others to vote on.

        Args:
            title: Short title for the motion
            description: Full description of the proposal
            motion_type: Type of motion (CUSTOM, PRODUCTION_PRIORITY, etc.)
            scope: "federation" or "pod" (default: "pod")
            voting_duration: Number of steps until voting closes (default: 50)

        Returns:
            Status message with motion_id if successful
        """
        from ..governance import MotionType

        federation = self.get_federation()
        pod = self.get_pod()

        # Determine constitution and eligible voters
        if scope == "federation":
            constitution = federation.constitution
            eligible_voters = {w.unique_id for pod in federation for w in pod}
        elif scope == "pod":
            if not pod:
                return json.dumps({"error": "Worker not assigned to a pod"})
            constitution = pod.constitution
            eligible_voters = {w.unique_id for w in pod}
        else:
            return json.dumps({"error": f"Invalid scope: {scope}"})

        # Try to propose motion
        try:
            motion_type_enum = MotionType[motion_type.upper()]
        except KeyError:
            return json.dumps({"error": f"Invalid motion type: {motion_type}"})

        try:
            motion = federation.governance.propose_motion(
                proposer=self.worker,
                motion_type=motion_type_enum,
                title=title,
                description=description,
                scope=scope,
                eligible_voters=eligible_voters,
                voting_duration=voting_duration
            )

            return json.dumps({
                "success": True,
                "motion_id": motion.motion_id,
                "voting_ends_step": motion.voting_ends_step,
                "message": f"Motion '{title}' proposed successfully"
            }, indent=2)

        except PermissionError as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

    def vote_on_motion(self, motion_id: str, choice: str) -> str:
        """
        Cast a vote on an active motion.

        Args:
            motion_id: ID of the motion (e.g., "M0", "M1")
            choice: "for", "against", or "abstain"

        Returns:
            Status message
        """
        federation = self.get_federation()

        success, message = federation.governance.cast_vote(
            motion_id=motion_id,
            voter_id=self.worker.unique_id,
            choice=choice
        )

        return json.dumps({
            "success": success,
            "message": message
        }, indent=2)

    def get_motion_details(self, motion_id: str) -> str:
        """
        Get detailed information about a specific motion.

        Args:
            motion_id: ID of the motion

        Returns:
            JSON string with motion details
        """
        federation = self.get_federation()
        motion = federation.governance.active_motions.get(motion_id)

        if not motion:
            return json.dumps({"error": f"Motion '{motion_id}' not found"})

        return json.dumps({
            "motion_id": motion.motion_id,
            "title": motion.title,
            "description": motion.description,
            "type": motion.motion_type.name,
            "scope": motion.scope,
            "proposer": motion.proposer,
            "vote_type": motion.vote_type.name,
            "required_threshold": motion.required_threshold,
            "voting_ends_step": motion.voting_ends_step,
            "current_step": federation.steps,
            "votes_for": len(motion.votes_for),
            "votes_against": len(motion.votes_against),
            "abstentions": len(motion.abstentions),
            "participation_rate": motion.get_participation_rate(),
            "my_vote": self._get_my_vote(motion)
        }, indent=2)


# Tool definitions for LLM function calling registration
def get_governance_tool_definitions() -> List[Dict]:
    """Get the tool definitions for OpenAI/LLM function calling."""

    return [
        {
            "type": "function",
            "function": {
                "name": "read_constitution",
                "description": "Read the full constitutional text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["federation", "pod"],
                            "description": "Which constitution to read (default: pod)"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_my_permissions",
                "description": "Check what actions are currently permitted under constitutions",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_active_motions",
                "description": "List all motions currently open for voting",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["federation", "pod", "all"],
                            "description": "Filter by scope (default: all)"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "propose_motion",
                "description": "Create a new motion for others to vote on",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short title"},
                        "description": {"type": "string", "description": "Full description"},
                        "motion_type": {"type": "string", "description": "Type of motion (default: CUSTOM)"},
                        "scope": {"type": "string", "enum": ["federation", "pod"], "description": "Scope (default: pod)"},
                        "voting_duration": {"type": "integer", "description": "Steps until voting closes (default: 50)"}
                    },
                    "required": ["title", "description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "vote_on_motion",
                "description": "Cast a vote on an active motion",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "motion_id": {"type": "string", "description": "ID of the motion (e.g., M0)"},
                        "choice": {"type": "string", "enum": ["for", "against", "abstain"], "description": "Your vote"}
                    },
                    "required": ["motion_id", "choice"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_motion_details",
                "description": "Get detailed information about a specific motion",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "motion_id": {"type": "string", "description": "ID of the motion"}
                    },
                    "required": ["motion_id"]
                }
            }
        }
    ]
