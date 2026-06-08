"""Tests for SemanticMemory dataclass and learning methods."""

import pytest
from libertas.cognitive import SemanticMemory


@pytest.mark.unit
class TestSemanticMemory:
    """Test SemanticMemory dataclass."""

    def test_initialization_empty(self):
        """Test that SemanticMemory initializes with empty structures."""
        memory = SemanticMemory()

        assert memory.price_patterns == {}
        assert memory.market_insights == []
        assert memory.worker_behaviors == {}
        assert memory.voting_coalitions == []
        assert memory.trusted_workers == {}
        assert memory.recipe_efficiency == {}
        assert memory.skill_mastery == {}
        assert memory.motion_outcomes == {}
        assert memory.constitution_rules == []

    def test_price_patterns_storage(self):
        """Test storing price patterns."""
        memory = SemanticMemory()

        memory.price_patterns["wheat"] = [10.0, 11.0, 12.0, 11.5, 13.0]
        memory.price_patterns["iron"] = [20.0, 22.0, 21.0]

        assert len(memory.price_patterns) == 2
        assert memory.price_patterns["wheat"] == [10.0, 11.0, 12.0, 11.5, 13.0]
        assert len(memory.price_patterns["wheat"]) == 5

    def test_market_insights_storage(self):
        """Test storing market insights."""
        memory = SemanticMemory()

        memory.market_insights.append("Wheat prices trending upward")
        memory.market_insights.append("Iron prices volatile, avg 21.0")

        assert len(memory.market_insights) == 2
        assert "trending upward" in memory.market_insights[0]

    def test_worker_behaviors_tracking(self):
        """Test tracking worker behaviors."""
        memory = SemanticMemory()

        memory.worker_behaviors["worker_1"] = {
            "voting_pattern": "conservative",
            "interaction_count": 5,
            "cooperation_level": 0.8
        }

        assert "worker_1" in memory.worker_behaviors
        assert memory.worker_behaviors["worker_1"]["voting_pattern"] == "conservative"
        assert memory.worker_behaviors["worker_1"]["cooperation_level"] == 0.8

    def test_voting_coalitions_storage(self):
        """Test storing voting coalitions."""
        memory = SemanticMemory()

        memory.voting_coalitions.append(["worker_1", "worker_2", "worker_3"])
        memory.voting_coalitions.append(["worker_4", "worker_5"])

        assert len(memory.voting_coalitions) == 2
        assert "worker_1" in memory.voting_coalitions[0]
        assert len(memory.voting_coalitions[0]) == 3

    def test_trusted_workers_scores(self):
        """Test trust score tracking."""
        memory = SemanticMemory()

        memory.trusted_workers["worker_1"] = 0.9
        memory.trusted_workers["worker_2"] = 0.3
        memory.trusted_workers["worker_3"] = 0.7

        assert len(memory.trusted_workers) == 3
        assert memory.trusted_workers["worker_1"] == 0.9
        assert 0.0 <= memory.trusted_workers["worker_2"] <= 1.0

    def test_recipe_efficiency_tracking(self):
        """Test recipe efficiency storage."""
        memory = SemanticMemory()

        memory.recipe_efficiency["bread_basic"] = 1.2
        memory.recipe_efficiency["tool_simple"] = 0.8

        assert len(memory.recipe_efficiency) == 2
        assert memory.recipe_efficiency["bread_basic"] == 1.2

    def test_skill_mastery_levels(self):
        """Test skill mastery level tracking."""
        memory = SemanticMemory()

        memory.skill_mastery["farming"] = 7
        memory.skill_mastery["crafting"] = 3
        memory.skill_mastery["trading"] = 10

        assert len(memory.skill_mastery) == 3
        assert memory.skill_mastery["farming"] == 7
        assert 0 <= memory.skill_mastery["crafting"] <= 10

    def test_motion_outcomes_tracking(self):
        """Test governance motion outcome tracking."""
        memory = SemanticMemory()

        memory.motion_outcomes["tax_increase"] = False
        memory.motion_outcomes["worker_allocation"] = True
        memory.motion_outcomes["constitution_amendment"] = False

        assert len(memory.motion_outcomes) == 3
        assert memory.motion_outcomes["worker_allocation"] is True
        assert memory.motion_outcomes["tax_increase"] is False

    def test_constitution_rules_learning(self):
        """Test learning constitutional rules."""
        memory = SemanticMemory()

        memory.constitution_rules.append("Tax motions require 2/3 majority")
        memory.constitution_rules.append("Worker allocation decided by simple majority")

        assert len(memory.constitution_rules) == 2
        assert "2/3 majority" in memory.constitution_rules[0]

    def test_memory_persistence(self):
        """Test that memory persists across operations."""
        memory = SemanticMemory()

        # Add market data
        memory.price_patterns["wheat"] = [10.0, 11.0]
        memory.market_insights.append("Wheat trending up")

        # Add social data
        memory.trusted_workers["worker_1"] = 0.8

        # Add production data
        memory.skill_mastery["farming"] = 5

        # Verify all data persists
        assert len(memory.price_patterns) == 1
        assert len(memory.market_insights) == 1
        assert len(memory.trusted_workers) == 1
        assert len(memory.skill_mastery) == 1

    def test_memory_update_existing(self):
        """Test updating existing memory entries."""
        memory = SemanticMemory()

        # Initial values
        memory.trusted_workers["worker_1"] = 0.5
        memory.skill_mastery["farming"] = 3

        # Update values
        memory.trusted_workers["worker_1"] = 0.8
        memory.skill_mastery["farming"] = 5

        assert memory.trusted_workers["worker_1"] == 0.8
        assert memory.skill_mastery["farming"] == 5

    def test_empty_memory_queries(self):
        """Test querying empty memory structures."""
        memory = SemanticMemory()

        assert memory.price_patterns.get("wheat") is None
        assert "worker_1" not in memory.trusted_workers
        assert len(memory.market_insights) == 0
        assert len(memory.voting_coalitions) == 0
