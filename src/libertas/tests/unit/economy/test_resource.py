# tests/unit/economy/test_resource.py
import unittest
import sys
import os
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.economy import (
    Resource, ResourceTag, Inventory, ResourceRegistry
)


@pytest.mark.unit
class TestResource(unittest.TestCase):
    """Test Resource class functionality."""
    
    def test_resource_creation(self):
        """Test basic resource creation."""
        resource = Resource(
            name="test_resource",
            base_value=10.0,
            weight=2.0,
            tags=[ResourceTag.RAW]
        )
        
        self.assertEqual(resource.name, "test_resource")
        self.assertEqual(resource.base_value, 10.0)
        self.assertEqual(resource.weight, 2.0)
        self.assertEqual(resource.tags, [ResourceTag.RAW])
        self.assertIsNotNone(resource.resource_id)
    
    def test_tool_resource(self):
        """Test tool resource creation."""
        tool = Resource(
            name="hammer",
            base_value=15.0,
            is_tool=True,
            durability=100,
            required_skill="crafting",
            enables_recipes=["forge", "assemble"]
        )
        
        self.assertTrue(tool.is_tool)
        self.assertEqual(tool.durability, 100)
        self.assertEqual(tool.required_skill, "crafting")
        self.assertEqual(tool.enables_recipes, ["forge", "assemble"])
    
    def test_get_value(self):
        """Test resource value calculation with market multipliers."""
        resource = Resource(name="test", base_value=10.0)
        
        # Default multiplier
        self.assertEqual(resource.get_value(), 10.0)
        
        # With market multiplier
        self.assertEqual(resource.get_value(1.5), 15.0)
        
        # With quality property
        resource.properties = {"quality": 0.8}
        self.assertEqual(resource.get_value(), 13.0)  # 10 * (0.5 + 0.8) = 13
    
    def test_use_tool(self):
        """Test tool usage and degradation."""
        tool = Resource(name="test_tool", is_tool=True, durability=10)
        
        # Use tool multiple times
        for i in range(9):
            self.assertTrue(tool.use_tool())
            self.assertEqual(tool.durability, 9 - i)
        
        # Last use before breaking
        self.assertTrue(tool.use_tool())
        self.assertEqual(tool.durability, 0)
        
        # Tool should be broken now
        self.assertTrue(tool.is_broken())
        self.assertFalse(tool.use_tool())
    
    def test_infinite_durability_tool(self):
        """Test tools with infinite durability."""
        tool = Resource(name="eternal_tool", is_tool=True, durability=None)
        
        for _ in range(100):
            self.assertTrue(tool.use_tool())
        self.assertFalse(tool.is_broken())
    
    def test_serialization(self):
        """Test resource serialization to/from dict."""
        original = Resource(
            name="test",
            base_value=20.0,
            properties={"strength": 0.9},
            is_tool=True,
            durability=50
        )
        
        data = original.to_dict()
        restored = Resource.from_dict(data)
        
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.base_value, original.base_value)
        self.assertEqual(restored.properties, original.properties)
        self.assertEqual(restored.is_tool, original.is_tool)
        self.assertEqual(restored.durability, original.durability)
    
    def test_invent_resource(self):
        """Test runtime resource invention."""
        resource = Resource.invent(
            name="mithril",
            inventor_id="worker_001",
            step=100,
            base_value=50.0,
            tags=[ResourceTag.RAW],
            properties={"rarity": 0.9}
        )
        
        self.assertEqual(resource.name, "mithril")
        self.assertEqual(resource.invented_by, "worker_001")
        self.assertEqual(resource.invention_step, 100)
        self.assertEqual(resource.base_value, 50.0)
        self.assertTrue("rarity" in resource.properties)


class TestInventory(unittest.TestCase):
    """Test Inventory class functionality."""
    
    def setUp(self):
        self.resource = Resource(name="wood", base_value=1.0)
        self.tool = Resource(name="hammer", is_tool=True, durability=100)
    
    def test_add_bulk_resource(self):
        """Test adding bulk resources to inventory."""
        inventory = Inventory()
        
        inventory.add(self.resource, 10)
        self.assertEqual(inventory.get_quantity("wood"), 10)
        
        inventory.add(self.resource, 5)
        self.assertEqual(inventory.get_quantity("wood"), 15)
    
    def test_remove_bulk_resource(self):
        """Test removing bulk resources from inventory."""
        inventory = Inventory()
        inventory.add(self.resource, 10)
        
        self.assertTrue(inventory.remove("wood", 3))
        self.assertEqual(inventory.get_quantity("wood"), 7)
        
        self.assertFalse(inventory.remove("wood", 10))
        self.assertEqual(inventory.get_quantity("wood"), 7)
    
    def test_add_tool(self):
        """Test adding tools to inventory."""
        inventory = Inventory()
        
        inventory.add(self.tool, 1)
        self.assertEqual(len(inventory.instances["hammer"]), 1)
        
        inventory.add(self.tool, 2)
        self.assertEqual(len(inventory.instances["hammer"]), 3)
    
    def test_get_tool(self):
        """Test retrieving tools from inventory."""
        inventory = Inventory()
        inventory.add(self.tool, 3)
        
        tool = inventory.get_tool("hammer")
        self.assertIsNotNone(tool)
        if tool is not None:
            self.assertEqual(tool.name, "hammer")
        self.assertEqual(len(inventory.instances["hammer"]), 2)
    
    def test_use_tool(self):
        """Test using tools from inventory."""
        inventory = Inventory()
        inventory.add(self.tool, 1)
        
        self.assertTrue(inventory.use_tool("hammer"))
        # Tool durability should be 99 now
        tool = inventory.get_tool("hammer")
        if tool is not None:
            self.assertEqual(tool.durability, 99)
    
    def test_capacity_limit(self):
        """Test inventory capacity limits."""
        inventory = Inventory(capacity=10)
        
        inventory.add(self.resource, 5)
        self.assertTrue(inventory.add(self.resource, 5))
        
        # Should fail due to capacity
        self.assertFalse(inventory.add(self.resource, 1))
    
    def test_has_enough(self):
        """Test checking if inventory has enough resources."""
        inventory = Inventory()
        inventory.add(self.resource, 10)
        
        self.assertTrue(inventory.has_enough({"wood": 5}))
        self.assertFalse(inventory.has_enough({"wood": 15}))
        self.assertFalse(inventory.has_enough({"wood": 5, "stone": 1}))
    
    def test_inventory_serialization(self):
        """Test inventory serialization to/from dict."""
        inventory = Inventory(capacity=100)
        inventory.add(self.resource, 20)
        inventory.add(self.tool, 2)
        
        data = inventory.to_dict()
        restored = Inventory.from_dict(data)
        
        self.assertEqual(restored.capacity, inventory.capacity)
        self.assertEqual(restored.get_quantity("wood"), 20)
        self.assertEqual(len(restored.instances.get("hammer", [])), 2)


class TestResourceRegistry(unittest.TestCase):
    """Test ResourceRegistry class functionality."""
    
    def setUp(self):
        self.registry = ResourceRegistry()
    
    def test_register_and_get(self):
        """Test registering and retrieving resources."""
        resource = Resource(name="coal", base_value=2.0)
        self.registry.register(resource)
        
        retrieved = self.registry.get("coal")
        if retrieved is not None:
            self.assertEqual(retrieved.name, "coal")
            self.assertEqual(retrieved.base_value, 2.0)
    
    def test_invent_resource(self):
        """Test inventing new resources at runtime."""
        resource = self.registry.invent(
            name="diamond",
            inventor_id="worker_001",
            step=50,
            base_value=100.0
        )
        
        registry_name = self.registry.get("diamond")
        if registry_name is not None:
            self.assertEqual(registry_name.name, "diamond")
        self.assertEqual(len(self.registry.invention_history), 1)
        self.assertEqual(self.registry.invention_history[0]["inventor"], "worker_001")
    
    def test_list_resources(self):
        """Test listing all registered resources."""
        self.registry.register(Resource(name="wood"))
        self.registry.register(Resource(name="stone"))
        self.registry.register(Resource(name="metal"))
        
        resources = self.registry.list_resources()
        self.assertEqual(len(resources), 3)
        self.assertIn("wood", resources)
        self.assertIn("stone", resources)
    
    def test_is_known(self):
        """Test checking if a resource is known."""
        self.registry.register(Resource(name="known_resource"))
        
        self.assertTrue(self.registry.is_known("known_resource"))
        self.assertFalse(self.registry.is_known("unknown_resource"))


class TestResourceEdgeCases(unittest.TestCase):
    """Test edge cases for resource system."""
    
    def test_negative_quantity(self):
        """Test handling negative quantities."""
        inventory = Inventory()
        resource = Resource(name="test")
        
        inventory.add(resource, -5)
        self.assertEqual(inventory.get_quantity("test"), 0)
    
    def test_remove_non_existent(self):
        """Test removing non-existent resources."""
        inventory = Inventory()
        self.assertFalse(inventory.remove("nonexistent", 10))
    
    def test_get_tool_from_empty(self):
        """Test getting tool from empty inventory."""
        inventory = Inventory()
        self.assertIsNone(inventory.get_tool("hammer"))
    
    def test_broken_tool_removal(self):
        """Test that broken tools are automatically removed."""
        inventory = Inventory()
        tool = Resource(name="fragile", is_tool=True, durability=1)
        inventory.add(tool, 1)
        
        # Use once - should break
        inventory.use_tool("fragile")
        
        # Tool should be gone
        self.assertIsNone(inventory.get_tool("fragile"))
    
    def test_duplicate_registration(self):
        """Test registering duplicate resources."""
        registry = ResourceRegistry()
        registry.register(Resource(name="unique"))
        registry.register(Resource(name="unique"))  # Should not duplicate

        self.assertEqual(len(registry.list_resources()), 1)

    def test_resource_with_rarity_property(self):
        """Test resource value calculation with rarity property."""
        resource = Resource(
            name="rare_gem",
            base_value=50.0,
            properties={"rarity": 2.0}  # 2x multiplier
        )

        # Value should be base_value * (1 + rarity)
        # 50 * (1 + 2.0) = 150
        self.assertEqual(resource.get_value(), 150.0)

    def test_resource_with_quality_and_rarity(self):
        """Test resource with both quality and rarity properties."""
        resource = Resource(
            name="premium_item",
            base_value=100.0,
            properties={"quality": 0.8, "rarity": 1.5}
        )

        # Value = base * (0.5 + quality) * (1 + rarity)
        # 100 * (0.5 + 0.8) * (1 + 1.5) = 100 * 1.3 * 2.5 = 325
        self.assertEqual(resource.get_value(), 325.0)

    def test_use_tool_on_non_tool(self):
        """Test that use_tool returns False for regular resources."""
        resource = Resource(name="wood", base_value=10.0, is_tool=False)

        self.assertFalse(resource.use_tool())

    def test_invention_history_property(self):
        """Test getting invention history from registry (line 171)."""
        registry = ResourceRegistry()

        # Invent some resources
        registry.invent("diamond", "worker1", 10, base_value=100.0)
        registry.invent("ruby", "worker2", 20, base_value=80.0)

        # Get history via property
        history = registry.invention_history

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["name"], "diamond")
        self.assertEqual(history[1]["name"], "ruby")


if __name__ == '__main__':
    unittest.main()