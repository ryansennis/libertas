"""Tests for Tool class."""

import unittest
import pytest
from libertas.resources import Tool


@pytest.mark.unit
class TestTool(unittest.TestCase):
    """Test Tool class."""

    def setUp(self):
        """Set up test fixtures."""
        self.info = ResourceInfo(name="hammer", base_value=50.0, weight=2.0)
        self.tool = Tool(
            info=self.info,
            durability=100,
            max_durability=100,
            required_skill="crafting",
            repair_cost=10.0
        )

    def test_tool_creation(self):
        """Test creating a tool."""
        self.assertEqual(self.tool.name, "hammer")
        self.assertEqual(self.tool.durability, 100)
        self.assertEqual(self.tool.required_skill, "crafting")

    def test_use_tool(self):
        """Test using a tool degrades durability."""
        result = self.tool.use()

        self.assertTrue(result)
        self.assertEqual(self.tool.durability, 99)

    def test_use_until_broken(self):
        """Test using tool until it breaks."""
        for _ in range(100):
            result = self.tool.use()

        self.assertFalse(result)  # Last use returns False
        self.assertEqual(self.tool.durability, 0)
        self.assertTrue(self.tool.is_broken())

    def test_is_broken(self):
        """Test checking if tool is broken."""
        self.assertFalse(self.tool.is_broken())

        self.tool.durability = 0
        self.assertTrue(self.tool.is_broken())

    def test_repair(self):
        """Test repairing a tool."""
        self.tool.durability = 50
        cost = self.tool.repair()

        self.assertEqual(self.tool.durability, 100)
        self.assertEqual(cost, 5.0)  # (50/100) * 10.0

    def test_repair_partial(self):
        """Test partial repair."""
        self.tool.durability = 50
        cost = self.tool.repair(amount=25)

        self.assertEqual(self.tool.durability, 75)
        self.assertEqual(cost, 2.5)  # (25/100) * 10.0

    def test_repair_beyond_max(self):
        """Test repair doesn't exceed max durability."""
        self.tool.durability = 95
        self.tool.repair(amount=20)

        self.assertEqual(self.tool.durability, 100)

    def test_get_value(self):
        """Test tool value scales with durability."""
        value = self.tool.get_value(market_multiplier=1.0)
        self.assertEqual(value, 50.0)  # Full durability = full value

        self.tool.durability = 50
        value = self.tool.get_value(market_multiplier=1.0)
        self.assertEqual(value, 25.0)  # Half durability = half value

    def test_get_value_broken(self):
        """Test broken tool has zero value."""
        self.tool.durability = 0
        value = self.tool.get_value()
        self.assertEqual(value, 0.0)

    def test_serialization(self):
        """Test tool serialization."""
        data = self.tool.to_dict()

        self.assertEqual(data['type'], 'tool')
        self.assertEqual(data['durability'], 100)
        self.assertEqual(data['required_skill'], 'crafting')
        self.assertEqual(data['info']['name'], 'hammer')

    def test_deserialization(self):
        """Test tool deserialization."""
        data = self.tool.to_dict()
        restored = Tool.from_dict(data)

        self.assertEqual(restored.name, 'hammer')
        self.assertEqual(restored.durability, 100)
        self.assertEqual(restored.required_skill, 'crafting')
        self.assertEqual(restored.repair_cost, 10.0)

    def test_infinite_durability_tool(self):
        """Test tool with None durability (infinite)."""
        eternal_tool = Tool(
            name="eternal_hammer", base_value=100.0,
            durability=None
        )

        # Should always return True and never break
        for _ in range(1000):
            result = eternal_tool.use()
            self.assertTrue(result)

        self.assertFalse(eternal_tool.is_broken())

    def test_enables_recipes(self):
        """Test tool with enabled recipes."""
        tool = Tool(
            name="saw", base_value=40.0,
            enables_recipes=["cut_wood", "build_frame"]
        )

        self.assertEqual(len(tool.enables_recipes), 2)
        self.assertIn("cut_wood", tool.enables_recipes)


if __name__ == "__main__":
    unittest.main()
