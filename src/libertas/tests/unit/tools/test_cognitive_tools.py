"""Tests for CognitiveTools class."""

import unittest
import json
import pytest
from unittest.mock import Mock

from libertas.tools.cognitive_tools import CognitiveTools, get_cognitive_tool_definitions
from libertas.organization.worker import Worker, WorkerConfig
from libertas.organization.pod import PodConfig
from libertas.organization.federation import Federation
from libertas.cognitive import Goal, GoalStatus, PersonalityTraits, Background
from mesa_llm.reasoning.cot import CoTReasoning

LLM_MODEL = "ollama/tinyllama"


@pytest.mark.unit
class TestCognitiveTools(unittest.TestCase):
    """Test CognitiveTools class."""

    def setUp(self):
        """Set up test fixtures."""
        worker_config = WorkerConfig(
            name="TestWorker",
            reasoning=CoTReasoning,
            llm_model=LLM_MODEL,
            initial_currency=1000.0,
            personality=PersonalityTraits(openness=0.8),
            background=Background(education_level=4)
        )

        pod_config = PodConfig(
            name="TestPod",
            workers=[worker_config]
        )

        self.federation = Federation(pods=[pod_config], seed=42)
        self.federation.steps = 100  # Set current step

        self.pod = self.federation[0]
        self.worker = list(self.pod)[0]
        self.tools = CognitiveTools(self.worker)

    def test_initialization(self):
        """Test CognitiveTools initializes correctly."""
        self.assertEqual(self.tools.worker, self.worker)

    def test_view_my_goals_empty(self):
        """Test viewing goals when empty."""
        result = self.tools.view_my_goals()
        data = json.loads(result)

        self.assertEqual(len(data["active_goals"]), 0)
        self.assertEqual(len(data["completed_goals"]), 0)
        self.assertEqual(len(data["abandoned_goals"]), 0)
        self.assertEqual(data["total_active"], 0)

    def test_view_my_goals_with_goals(self):
        """Test viewing goals with some goals present."""
        # Add goals
        goal1 = Goal(
            goal_id="g1",
            goal_type="economic",
            description="Earn 1000 currency",
            priority=0.8
        )
        goal2 = Goal(
            goal_id="g2",
            goal_type="social",
            description="Make friends",
            priority=0.5
        )

        self.worker.goals.add_goal(goal1)
        self.worker.goals.add_goal(goal2)

        result = self.tools.view_my_goals()
        data = json.loads(result)

        self.assertEqual(data["total_active"], 2)
        self.assertEqual(len(data["active_goals"]), 2)

        # Check goal structure
        first_goal = data["active_goals"][0]
        self.assertIn("goal_id", first_goal)
        self.assertIn("type", first_goal)
        self.assertIn("description", first_goal)
        self.assertIn("status", first_goal)
        self.assertIn("priority", first_goal)

    def test_create_goal_valid(self):
        """Test creating a valid goal."""
        result = self.tools.create_goal(
            goal_type="economic",
            description="Earn 500 currency",
            priority=0.7,
            target_value=500.0,
            target_metric="currency"
        )
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertIn("goal_id", data)
        self.assertEqual(data["goal_type"], "economic")
        self.assertEqual(data["priority"], 0.7)

        # Verify goal was added
        self.assertEqual(len(self.worker.goals.active_goals), 1)

    def test_create_goal_invalid_type(self):
        """Test creating goal with invalid type."""
        result = self.tools.create_goal(
            goal_type="invalid_type",
            description="Test"
        )
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertIn("Invalid goal_type", data["error"])

    def test_create_goal_invalid_priority(self):
        """Test creating goal with invalid priority."""
        result = self.tools.create_goal(
            goal_type="economic",
            description="Test",
            priority=1.5  # Invalid, should be 0-1
        )
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertIn("Priority must be", data["error"])

    def test_update_goal_progress(self):
        """Test updating goal progress."""
        # Create a goal
        goal = Goal(
            goal_id="test_goal",
            goal_type="economic",
            description="Test",
            priority=0.5
        )
        self.worker.goals.add_goal(goal)

        # Update progress
        result = self.tools.update_goal_progress("test_goal", 0.5)
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertEqual(data["old_progress"], 0.0)
        self.assertEqual(data["new_progress"], 0.5)
        self.assertEqual(data["status"], "in_progress")

    def test_update_goal_progress_complete(self):
        """Test updating goal progress to completion."""
        goal = Goal(goal_id="test_goal", goal_type="economic", description="Test")
        self.worker.goals.add_goal(goal)

        result = self.tools.update_goal_progress("test_goal", 1.0)
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "completed")
        self.assertIn("completed", data["message"])

        # Verify goal moved to completed
        self.assertEqual(len(self.worker.goals.active_goals), 0)
        self.assertEqual(len(self.worker.goals.completed_goals), 1)

    def test_update_goal_progress_not_found(self):
        """Test updating progress for non-existent goal."""
        result = self.tools.update_goal_progress("nonexistent", 0.5)
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("error", data)
        self.assertIn("not found", data["error"])

    def test_abandon_goal(self):
        """Test abandoning a goal."""
        goal = Goal(goal_id="test_goal", goal_type="economic", description="Test")
        self.worker.goals.add_goal(goal)

        result = self.tools.abandon_goal("test_goal", "Changed priorities")
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertEqual(data["goal_id"], "test_goal")
        self.assertEqual(data["reason"], "Changed priorities")
        self.assertEqual(data["status"], "abandoned")

        # Verify goal moved to abandoned
        self.assertEqual(len(self.worker.goals.active_goals), 0)
        self.assertEqual(len(self.worker.goals.abandoned_goals), 1)

    def test_abandon_goal_not_found(self):
        """Test abandoning non-existent goal."""
        result = self.tools.abandon_goal("nonexistent", "reason")
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_revive_goal(self):
        """Test reviving an abandoned goal."""
        goal = Goal(goal_id="test_goal", goal_type="economic", description="Test")
        self.worker.goals.add_goal(goal)
        self.worker.goals.abandon_goal("test_goal", "reason")

        result = self.tools.revive_goal("test_goal")
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertEqual(data["goal_id"], "test_goal")
        self.assertEqual(data["status"], "in_progress")
        self.assertIn("revived", data["message"])

        # Verify goal moved back to active
        self.assertEqual(len(self.worker.goals.active_goals), 1)
        self.assertEqual(len(self.worker.goals.abandoned_goals), 0)

    def test_revive_goal_not_found(self):
        """Test reviving non-existent goal."""
        result = self.tools.revive_goal("nonexistent")
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_check_my_mood(self):
        """Test checking mood."""
        result = self.tools.check_my_mood()
        data = json.loads(result)

        self.assertIn("happiness", data)
        self.assertIn("stress", data)
        self.assertIn("motivation", data)
        self.assertIn("trust_in_leadership", data)
        self.assertIn("solidarity_with_group", data)
        self.assertIn("interpretation", data)

        # Check values are in valid range
        self.assertGreaterEqual(data["happiness"], 0.0)
        self.assertLessEqual(data["happiness"], 1.0)

    def test_interpret_mood_happy(self):
        """Test mood interpretation for happy state."""
        self.worker.mood.happiness = 0.9
        self.worker.mood.stress = 0.2

        interpretation = self.tools._interpret_mood(self.worker.mood)

        self.assertIn("happy", interpretation.lower())

    def test_interpret_mood_stressed(self):
        """Test mood interpretation for stressed state."""
        self.worker.mood.stress = 0.9

        interpretation = self.tools._interpret_mood(self.worker.mood)

        self.assertIn("stress", interpretation.lower())

    def test_recall_memory_market(self):
        """Test recalling market memories."""
        # Add market memories
        self.worker.semantic_memory.price_patterns["wheat"] = [10.0, 11.0, 12.0]
        self.worker.semantic_memory.market_insights.append("Wheat prices rising")

        result = self.tools.recall_memory("market", limit=5)
        data = json.loads(result)

        self.assertIn("price_patterns", data)
        self.assertIn("market_insights", data)
        self.assertIn("wheat", data["price_patterns"])

    def test_recall_memory_social(self):
        """Test recalling social memories."""
        self.worker.semantic_memory.trusted_workers["Alice"] = 0.9

        result = self.tools.recall_memory("social", limit=5)
        data = json.loads(result)

        self.assertIn("trusted_workers", data)
        self.assertIn("Alice", data["trusted_workers"])

    def test_recall_memory_production(self):
        """Test recalling production memories."""
        self.worker.semantic_memory.skill_mastery["farming"] = 7

        result = self.tools.recall_memory("production", limit=5)
        data = json.loads(result)

        self.assertIn("skill_mastery", data)
        self.assertIn("farming", data["skill_mastery"])

    def test_recall_memory_governance(self):
        """Test recalling governance memories."""
        self.worker.semantic_memory.motion_outcomes["tax_increase"] = False
        self.worker.semantic_memory.constitution_rules.append("Rule 1")

        result = self.tools.recall_memory("governance", limit=5)
        data = json.loads(result)

        self.assertIn("motion_outcomes", data)
        self.assertIn("constitution_rules", data)

    def test_recall_memory_no_match(self):
        """Test recalling with no matching topic."""
        result = self.tools.recall_memory("unknown_topic", limit=5)
        data = json.loads(result)

        self.assertIn("error", data)
        self.assertIn("No matching memories", data["error"])

    def test_view_recent_experiences_empty(self):
        """Test viewing experiences when empty."""
        result = self.tools.view_recent_experiences(limit=10)
        data = json.loads(result)

        self.assertEqual(data["total_memories"], 0)
        self.assertEqual(len(data["recent_experiences"]), 0)

    def test_view_recent_experiences_with_data(self):
        """Test viewing experiences with data."""
        # Add episodic memories
        for i in range(5):
            self.worker.episodic_memory.append({
                "step": i,
                "observations": {"test": "data"},
                "mood": {"happiness": 0.7}
            })

        result = self.tools.view_recent_experiences(limit=10)
        data = json.loads(result)

        self.assertEqual(data["total_memories"], 5)
        self.assertEqual(len(data["recent_experiences"]), 5)

        exp = data["recent_experiences"][0]
        self.assertIn("step", exp)
        self.assertIn("observations_count", exp)
        self.assertIn("mood_at_time", exp)


@pytest.mark.unit
class TestCognitiveToolDefinitions(unittest.TestCase):
    """Test cognitive tool definitions."""

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        defs = get_cognitive_tool_definitions()

        self.assertIsInstance(defs, list)
        self.assertGreater(len(defs), 0)

        # Check structure
        for tool_def in defs:
            self.assertIn("type", tool_def)
            self.assertEqual(tool_def["type"], "function")
            self.assertIn("function", tool_def)
            self.assertIn("name", tool_def["function"])
            self.assertIn("description", tool_def["function"])

    def test_tool_names(self):
        """Test that all expected tools are defined."""
        defs = get_cognitive_tool_definitions()
        tool_names = [d["function"]["name"] for d in defs]

        expected_tools = [
            "view_my_goals",
            "create_goal",
            "update_goal_progress",
            "abandon_goal",
            "revive_goal",
            "check_my_mood",
            "recall_memory",
            "view_recent_experiences"
        ]

        for expected in expected_tools:
            self.assertIn(expected, tool_names)

    def test_create_goal_definition(self):
        """Test create_goal has proper parameter definitions."""
        defs = get_cognitive_tool_definitions()
        tool = next(d for d in defs if d["function"]["name"] == "create_goal")

        params = tool["function"]["parameters"]
        self.assertIn("goal_type", params["properties"])
        self.assertIn("description", params["properties"])
        self.assertIn("goal_type", params["required"])
        self.assertIn("description", params["required"])

        # Check enum values
        self.assertIn("enum", params["properties"]["goal_type"])
        self.assertIn("economic", params["properties"]["goal_type"]["enum"])

    def test_update_goal_progress_definition(self):
        """Test update_goal_progress has required parameters."""
        defs = get_cognitive_tool_definitions()
        tool = next(d for d in defs if d["function"]["name"] == "update_goal_progress")

        params = tool["function"]["parameters"]
        self.assertIn("goal_id", params["required"])
        self.assertIn("progress", params["required"])


if __name__ == "__main__":
    unittest.main()
