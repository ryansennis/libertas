"""Function calling tools for workers to manage their cognitive state."""

from typing import Dict, List, Optional
import json


class CognitiveTools:
    """
    Tool functions that workers can call to manage goals, memory, and mood.

    These are registered with the LLM agent and called via function calling.
    """

    def __init__(self, worker):
        self.worker = worker

    # ===== Goal Management Tools =====

    def view_my_goals(self) -> str:
        """
        View all your goals (active, completed, and abandoned).

        Returns:
            JSON string with goal information
        """
        from libertas.cognitive import GoalStatus

        active_goals = []
        for goal in self.worker.goals.active_goals:
            active_goals.append({
                "goal_id": goal.goal_id,
                "type": goal.goal_type,
                "description": goal.description,
                "status": goal.status.value,
                "priority": goal.priority,
                "progress": goal.progress,
                "target_value": goal.target_value,
                "deadline_step": goal.deadline_step
            })

        completed_goals = []
        for goal in self.worker.goals.completed_goals[-10:]:  # Last 10
            completed_goals.append({
                "goal_id": goal.goal_id,
                "type": goal.goal_type,
                "description": goal.description,
                "status": goal.status.value,
                "progress": goal.progress
            })

        abandoned_goals = []
        for goal in self.worker.goals.abandoned_goals[-10:]:  # Last 10
            abandoned_goals.append({
                "goal_id": goal.goal_id,
                "type": goal.goal_type,
                "description": goal.description,
                "status": goal.status.value,
                "abandon_reason": goal.abandon_reason
            })

        return json.dumps({
            "active_goals": active_goals,
            "completed_goals": completed_goals,
            "abandoned_goals": abandoned_goals,
            "total_active": len(self.worker.goals.active_goals),
            "total_completed": len(self.worker.goals.completed_goals),
            "total_abandoned": len(self.worker.goals.abandoned_goals)
        }, indent=2)

    def create_goal(
        self,
        goal_type: str,
        description: str,
        priority: float = 0.5,
        target_value: Optional[float] = None,
        target_metric: Optional[str] = None
    ) -> str:
        """
        Create a new goal to pursue.

        Args:
            goal_type: Type of goal (economic, social, governance, learning)
            description: Clear description of the goal
            priority: Priority level 0-1 (default 0.5)
            target_value: Optional numeric target for progress tracking
            target_metric: Optional metric name (e.g., "currency", "skill_level")

        Returns:
            JSON string with creation result
        """
        from libertas.cognitive import Goal
        import uuid

        # Validate goal type
        valid_types = ["economic", "social", "governance", "learning"]
        if goal_type not in valid_types:
            return json.dumps({
                "success": False,
                "error": f"Invalid goal_type. Must be one of: {valid_types}"
            }, indent=2)

        # Validate priority
        if not 0 <= priority <= 1:
            return json.dumps({
                "success": False,
                "error": "Priority must be between 0 and 1"
            }, indent=2)

        # Create the goal
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        goal = Goal(
            goal_id=goal_id,
            goal_type=goal_type,
            description=description,
            target_metric=target_metric,
            target_value=target_value,
            priority=priority,
            created_step=self.worker.federation.steps if self.worker.federation else 0
        )

        self.worker.goals.add_goal(goal)

        return json.dumps({
            "success": True,
            "goal_id": goal_id,
            "goal_type": goal_type,
            "description": description,
            "priority": priority,
            "status": goal.status.value
        }, indent=2)

    def update_goal_progress(self, goal_id: str, progress: float) -> str:
        """
        Update progress on an active goal.

        Args:
            goal_id: ID of the goal to update
            progress: Progress value 0-1

        Returns:
            JSON string with update result
        """
        # Find the goal
        goal = None
        for g in self.worker.goals.active_goals:
            if g.goal_id == goal_id:
                goal = g
                break

        if not goal:
            return json.dumps({
                "success": False,
                "error": f"Goal '{goal_id}' not found in active goals"
            }, indent=2)

        # Update progress
        old_progress = goal.progress
        self.worker.goals.update_progress(goal_id, progress)

        # Check if goal was completed
        if progress >= 1.0:
            return json.dumps({
                "success": True,
                "goal_id": goal_id,
                "old_progress": old_progress,
                "new_progress": progress,
                "status": "completed",
                "message": f"Goal '{goal.description}' completed!"
            }, indent=2)

        return json.dumps({
            "success": True,
            "goal_id": goal_id,
            "old_progress": old_progress,
            "new_progress": progress,
            "status": "in_progress"
        }, indent=2)

    def abandon_goal(self, goal_id: str, reason: str) -> str:
        """
        Abandon an active goal.

        Args:
            goal_id: ID of the goal to abandon
            reason: Reason for abandoning the goal

        Returns:
            JSON string with abandon result
        """
        # Find the goal
        goal = None
        for g in self.worker.goals.active_goals:
            if g.goal_id == goal_id:
                goal = g
                break

        if not goal:
            return json.dumps({
                "success": False,
                "error": f"Goal '{goal_id}' not found in active goals"
            }, indent=2)

        # Abandon it
        self.worker.goals.abandon_goal(goal_id, reason)

        return json.dumps({
            "success": True,
            "goal_id": goal_id,
            "description": goal.description,
            "reason": reason,
            "status": "abandoned"
        }, indent=2)

    def revive_goal(self, goal_id: str) -> str:
        """
        Revive a previously abandoned goal.

        Args:
            goal_id: ID of the abandoned goal to revive

        Returns:
            JSON string with revive result
        """
        result = self.worker.goals.revive_goal(goal_id)

        if not result:
            return json.dumps({
                "success": False,
                "error": f"Goal '{goal_id}' not found in abandoned goals"
            }, indent=2)

        # Find the revived goal
        revived_goal = None
        for g in self.worker.goals.active_goals:
            if g.goal_id == goal_id:
                revived_goal = g
                break

        return json.dumps({
            "success": True,
            "goal_id": goal_id,
            "description": revived_goal.description if revived_goal else "Unknown",
            "status": "in_progress",
            "message": "Goal has been revived and is now active"
        }, indent=2)

    # ===== Mood and State Tools =====

    def check_my_mood(self) -> str:
        """
        View your current mood and emotional state.

        Returns:
            JSON string with mood information
        """
        mood = self.worker.mood

        return json.dumps({
            "happiness": mood.happiness,
            "stress": mood.stress,
            "motivation": mood.motivation,
            "trust_in_leadership": mood.trust_in_leadership,
            "solidarity_with_group": mood.solidarity_with_group,
            "interpretation": self._interpret_mood(mood)
        }, indent=2)

    def _interpret_mood(self, mood) -> str:
        """Generate a natural language interpretation of mood."""
        if mood.happiness > 0.7 and mood.stress < 0.3:
            return "Feeling good - happy and relaxed"
        elif mood.stress > 0.7:
            return "Highly stressed - may need a break"
        elif mood.motivation < 0.3:
            return "Low motivation - struggling to find drive"
        elif mood.happiness < 0.3:
            return "Unhappy - things aren't going well"
        else:
            return "Mood is balanced"

    # ===== Memory Tools =====

    def recall_memory(self, topic: str, limit: int = 5) -> str:
        """
        Query semantic memory for information on a topic.

        Args:
            topic: Topic to search for (market, social, production, governance)
            limit: Maximum number of results to return

        Returns:
            JSON string with relevant memories
        """
        topic_lower = topic.lower()
        results = {}

        if "market" in topic_lower or "price" in topic_lower:
            results["price_patterns"] = dict(list(self.worker.semantic_memory.price_patterns.items())[:limit])
            results["market_insights"] = self.worker.semantic_memory.market_insights[-limit:]

        if "social" in topic_lower or "worker" in topic_lower:
            results["worker_behaviors"] = dict(list(self.worker.semantic_memory.worker_behaviors.items())[:limit])
            results["trusted_workers"] = dict(list(self.worker.semantic_memory.trusted_workers.items())[:limit])

        if "production" in topic_lower or "skill" in topic_lower:
            results["recipe_efficiency"] = dict(list(self.worker.semantic_memory.recipe_efficiency.items())[:limit])
            results["skill_mastery"] = dict(list(self.worker.semantic_memory.skill_mastery.items())[:limit])

        if "governance" in topic_lower or "motion" in topic_lower:
            results["motion_outcomes"] = dict(list(self.worker.semantic_memory.motion_outcomes.items())[:limit])
            results["constitution_rules"] = self.worker.semantic_memory.constitution_rules[-limit:]

        if not results:
            return json.dumps({
                "error": "No matching memories found",
                "hint": "Try topics like: market, social, production, governance"
            }, indent=2)

        return json.dumps(results, indent=2)

    def view_recent_experiences(self, limit: int = 10) -> str:
        """
        View recent episodic memories.

        Args:
            limit: Number of recent experiences to return (default 10)

        Returns:
            JSON string with recent memories
        """
        recent = self.worker.episodic_memory[-limit:] if self.worker.episodic_memory else []

        experiences = []
        for memory in recent:
            experiences.append({
                "step": memory.get("step", "unknown"),
                "observations_count": len(memory.get("observations", {})),
                "mood_at_time": memory.get("mood", {})
            })

        return json.dumps({
            "total_memories": len(self.worker.episodic_memory),
            "recent_experiences": experiences,
            "showing": len(experiences)
        }, indent=2)


def get_cognitive_tool_definitions() -> List[Dict]:
    """
    Return OpenAI function calling definitions for cognitive tools.

    Returns:
        List of tool definitions in OpenAI format
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "view_my_goals",
                "description": "View all your goals (active, completed, and abandoned)",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_goal",
                "description": "Create a new goal to pursue",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_type": {
                            "type": "string",
                            "description": "Type of goal",
                            "enum": ["economic", "social", "governance", "learning"]
                        },
                        "description": {"type": "string", "description": "Clear description of the goal"},
                        "priority": {"type": "number", "description": "Priority level 0-1 (default 0.5)"},
                        "target_value": {"type": "number", "description": "Optional numeric target"},
                        "target_metric": {"type": "string", "description": "Optional metric name"}
                    },
                    "required": ["goal_type", "description"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_goal_progress",
                "description": "Update progress on an active goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_id": {"type": "string", "description": "ID of the goal to update"},
                        "progress": {"type": "number", "description": "Progress value 0-1"}
                    },
                    "required": ["goal_id", "progress"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "abandon_goal",
                "description": "Abandon an active goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_id": {"type": "string", "description": "ID of the goal to abandon"},
                        "reason": {"type": "string", "description": "Reason for abandoning"}
                    },
                    "required": ["goal_id", "reason"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "revive_goal",
                "description": "Revive a previously abandoned goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal_id": {"type": "string", "description": "ID of the abandoned goal to revive"}
                    },
                    "required": ["goal_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_my_mood",
                "description": "View your current mood and emotional state",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memory",
                "description": "Query semantic memory for information on a topic (market, social, production, governance)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to search for"},
                        "limit": {"type": "integer", "description": "Maximum results (default 5)"}
                    },
                    "required": ["topic"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "view_recent_experiences",
                "description": "View recent episodic memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of recent experiences (default 10)"}
                    }
                }
            }
        }
    ]
