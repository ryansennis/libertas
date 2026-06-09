"""Tests for Tool class."""

import pytest
from libertas.resources import Tool


@pytest.mark.unit
class TestTool:
    """Test Tool class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = Tool(
            name="hammer",
            base_value=50.0,
            weight=2.0,
            durability=100,
            max_durability=100,
            required_skill="crafting",
            repair_cost=10.0
        )

    def test_tool_creation(self):
        """Test creating a tool."""
        assert self.tool.name == "hammer"
        assert self.tool.durability == 100
        assert self.tool.required_skill == "crafting"

    def test_use_tool(self):
        """Test using a tool degrades durability."""
        result = self.tool.use()

        assert result
        assert self.tool.durability == 99

    def test_use_until_broken(self):
        """Test using tool until it breaks."""
        for _ in range(100):
            result = self.tool.use()

        assert not (result)  # Last use returns False
        assert self.tool.durability == 0
        assert self.tool.is_broken()

    def test_is_broken(self):
        """Test checking if tool is broken."""
        assert not (self.tool.is_broken())

        self.tool.durability = 0
        assert self.tool.is_broken()

    def test_repair(self):
        """Test repairing a tool."""
        self.tool.durability = 50
        cost = self.tool.repair()

        assert self.tool.durability == 100
        assert cost == 5.0  # (50/100) * 10.0

    def test_repair_partial(self):
        """Test partial repair."""
        self.tool.durability = 50
        cost = self.tool.repair(amount=25)

        assert self.tool.durability == 75
        assert cost == 2.5  # (25/100) * 10.0

    def test_repair_beyond_max(self):
        """Test repair doesn't exceed max durability."""
        self.tool.durability = 95
        self.tool.repair(amount=20)

        assert self.tool.durability == 100

    def test_get_value(self):
        """Test tool value scales with durability."""
        value = self.tool.get_buy_price(market_multiplier=1.0)
        assert value == 50.0  # Full durability = full value

        self.tool.durability = 50
        value = self.tool.get_buy_price(market_multiplier=1.0)
        assert value == 25.0  # Half durability = half value

    def test_get_value_broken(self):
        """Test broken tool has zero value."""
        self.tool.durability = 0
        value = self.tool.get_buy_price()
        assert value == 0.0

    def test_serialization(self):
        """Test tool serialization."""
        data = self.tool.to_dict()

        assert data['type'] == 'tool'
        assert data['durability'] == 100
        assert data['required_skill'] == 'crafting'
        assert data['name'] == 'hammer'
        assert data['base_value'] == 50.0

    def test_deserialization(self):
        """Test tool deserialization."""
        data = self.tool.to_dict()
        restored = Tool.from_dict(data)

        assert restored.name == 'hammer'
        assert restored.durability == 100
        assert restored.required_skill == 'crafting'
        assert restored.repair_cost == 10.0

    def test_infinite_durability_tool(self):
        """Test tool with None durability (infinite)."""
        eternal_tool = Tool(
            name="eternal_hammer", base_value=100.0,
            durability=None
        )

        # Should always return True and never break
        for _ in range(1000):
            result = eternal_tool.use()
            assert result

        assert not (eternal_tool.is_broken())

    def test_enables_recipes(self):
        """Test tool with enabled recipes."""
        tool = Tool(
            name="saw", base_value=40.0,
            enables_recipes=["cut_wood", "build_frame"]
        )

        assert len(tool.enables_recipes) == 2
        assert "cut_wood" in tool.enables_recipes


