# tests/unit/economy/test_inventory.py
"""Unit tests for Inventory class."""

import pytest
from libertas.economy import Inventory, Resource


@pytest.mark.unit
class TestInventory:
    """Test basic inventory operations."""

    def test_inventory_creation(self):
        """Test creating an empty inventory."""
        inv = Inventory()
        assert inv.quantities == {}
        assert inv.instances == {}
        assert inv.capacity is None

    def test_inventory_with_capacity(self):
        """Test creating inventory with capacity limit."""
        inv = Inventory(capacity=100.0)
        assert inv.capacity == 100.0

    def test_add_bulk_resource(self):
        """Test adding bulk (non-tool) resources."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, 50.0)
        assert result is True
        assert inv.get_quantity("wood") == 50.0

    def test_add_bulk_resource_multiple_times(self):
        """Test adding same resource multiple times accumulates."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)

        inv.add(wood, 30.0)
        inv.add(wood, 20.0)
        assert inv.get_quantity("wood") == 50.0

    def test_add_negative_quantity(self):
        """Test adding negative quantity fails."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, -10.0)
        assert result is False
        assert inv.get_quantity("wood") == 0.0

    def test_add_tool_resource(self):
        """Test adding tool resources creates instances."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0,
            required_skill="crafting"
        )

        result = inv.add(hammer, 2.0)
        assert result is True
        assert "hammer" in inv.instances
        assert len(inv.instances["hammer"]) == 2
        assert all(tool.durability == 100.0 for tool in inv.instances["hammer"])

    def test_remove_bulk_resource(self):
        """Test removing bulk resources."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)
        inv.add(wood, 50.0)

        result = inv.remove("wood", 20.0)
        assert result is True
        assert inv.get_quantity("wood") == 30.0

    def test_remove_all_of_resource(self):
        """Test removing all of a resource deletes the key."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)
        inv.add(wood, 50.0)

        result = inv.remove("wood", 50.0)
        assert result is True
        assert "wood" not in inv.quantities
        assert inv.get_quantity("wood") == 0.0

    def test_remove_more_than_available(self):
        """Test removing more than available fails."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)
        inv.add(wood, 50.0)

        result = inv.remove("wood", 100.0)
        assert result is False
        assert inv.get_quantity("wood") == 50.0  # Unchanged

    def test_remove_nonexistent_resource(self):
        """Test removing nonexistent resource fails."""
        inv = Inventory()
        result = inv.remove("wood", 10.0)
        assert result is False

    def test_get_quantity_nonexistent(self):
        """Test getting quantity of nonexistent resource returns 0."""
        inv = Inventory()
        assert inv.get_quantity("wood") == 0.0

    def test_has_enough(self):
        """Test checking if inventory meets requirements."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)
        metal = Resource(name="metal", base_value=20.0)

        inv.add(wood, 50.0)
        inv.add(metal, 30.0)

        assert inv.has_enough({"wood": 40.0, "metal": 20.0}) is True
        assert inv.has_enough({"wood": 60.0}) is False
        assert inv.has_enough({"wood": 50.0, "metal": 40.0}) is False

    def test_has_enough_empty_requirements(self):
        """Test checking empty requirements passes."""
        inv = Inventory()
        assert inv.has_enough({}) is True


@pytest.mark.unit
class TestInventoryCapacity:
    """Test inventory capacity limits."""

    def test_add_within_capacity(self):
        """Test adding resources within capacity limit."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, 50.0)
        assert result is True
        assert inv.get_quantity("wood") == 50.0

    def test_add_exceeds_capacity(self):
        """Test adding resources that exceed capacity fails."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, 150.0)
        assert result is False
        assert inv.get_quantity("wood") == 0.0

    def test_add_exactly_at_capacity(self):
        """Test adding resources exactly at capacity succeeds."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, 100.0)
        assert result is True
        assert inv.get_quantity("wood") == 100.0

    def test_multiple_resources_exceed_capacity(self):
        """Test multiple resources exceeding capacity."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)
        metal = Resource(name="metal", base_value=20.0)

        inv.add(wood, 60.0)
        result = inv.add(metal, 50.0)  # Would total 110
        assert result is False
        assert inv.get_quantity("metal") == 0.0
        assert inv.get_quantity("wood") == 60.0


@pytest.mark.unit
class TestInventoryTools:
    """Test tool instance management."""

    def test_get_tool(self):
        """Test retrieving a tool from inventory."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0,
            required_skill="crafting"
        )
        inv.add(hammer, 2.0)

        tool = inv.get_tool("hammer")
        assert tool is not None
        assert tool.name == "hammer"
        assert tool.durability == 100.0
        assert len(inv.instances["hammer"]) == 1  # One removed

    def test_get_tool_removes_from_instances(self):
        """Test that get_tool removes tool from instances."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0
        )
        inv.add(hammer, 1.0)

        initial_count = len(inv.instances["hammer"])
        inv.get_tool("hammer")
        final_count = len(inv.instances.get("hammer", []))

        assert final_count == initial_count - 1

    def test_get_tool_nonexistent(self):
        """Test getting nonexistent tool returns None."""
        inv = Inventory()
        tool = inv.get_tool("hammer")
        assert tool is None

    def test_get_tool_empty_list(self):
        """Test getting tool from empty list returns None."""
        inv = Inventory()
        inv.instances["hammer"] = []  # Empty list

        tool = inv.get_tool("hammer")
        assert tool is None

    def test_get_tool_skips_broken(self):
        """Test that get_tool skips broken tools."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=1.0  # Will break after one use
        )
        inv.add(hammer, 2.0)

        # Break the first tool
        inv.instances["hammer"][0].durability = 0.0

        # Should skip broken tool and get second one
        tool = inv.get_tool("hammer")
        assert tool is not None
        assert tool.durability > 0.0

    def test_return_tool(self):
        """Test returning a tool to inventory."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0
        )
        inv.add(hammer, 1.0)

        tool = inv.get_tool("hammer")
        tool.durability = 80.0  # Degrade it
        inv.return_tool(tool)

        assert len(inv.instances["hammer"]) == 1
        assert inv.instances["hammer"][0].durability == 80.0

    def test_return_tool_new_tool_type(self):
        """Test returning a tool of new type creates list."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0
        )

        inv.return_tool(hammer)
        assert "hammer" in inv.instances
        assert len(inv.instances["hammer"]) == 1

    def test_use_tool(self):
        """Test using a tool degrades durability."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0
        )
        inv.add(hammer, 1.0)

        result = inv.use_tool("hammer")
        assert result is True

        # Tool should be returned with reduced durability
        assert len(inv.instances["hammer"]) == 1
        assert inv.instances["hammer"][0].durability < 100.0

    def test_use_tool_breaks(self):
        """Test that broken tools are not returned."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=1.0  # Will break
        )
        inv.add(hammer, 1.0)

        result = inv.use_tool("hammer")
        assert result is True

        # Broken tool should not be returned
        assert len(inv.instances.get("hammer", [])) == 0

    def test_use_tool_nonexistent(self):
        """Test using nonexistent tool fails."""
        inv = Inventory()
        result = inv.use_tool("hammer")
        assert result is False


@pytest.mark.unit
class TestInventorySerialization:
    """Test inventory serialization."""

    def test_to_dict_empty(self):
        """Test serializing empty inventory."""
        inv = Inventory()
        data = inv.to_dict()

        assert data['quantities'] == {}
        assert data['instances'] == {}
        assert data['capacity'] is None

    def test_to_dict_with_quantities(self):
        """Test serializing inventory with bulk resources."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)
        metal = Resource(name="metal", base_value=20.0)

        inv.add(wood, 50.0)
        inv.add(metal, 30.0)

        data = inv.to_dict()
        assert data['quantities']['wood'] == 50.0
        assert data['quantities']['metal'] == 30.0
        assert data['capacity'] == 100.0

    def test_to_dict_with_tools(self):
        """Test serializing inventory with tools."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0,
            required_skill="crafting"
        )
        inv.add(hammer, 2.0)

        data = inv.to_dict()
        assert "hammer" in data['instances']
        assert len(data['instances']['hammer']) == 2
        assert all(tool['durability'] == 100.0 for tool in data['instances']['hammer'])

    def test_from_dict_empty(self):
        """Test deserializing empty inventory."""
        data = {
            'quantities': {},
            'instances': {},
            'capacity': None
        }
        inv = Inventory.from_dict(data)

        assert inv.quantities == {}
        assert inv.instances == {}
        assert inv.capacity is None

    def test_from_dict_with_quantities(self):
        """Test deserializing inventory with bulk resources."""
        data = {
            'quantities': {'wood': 50.0, 'metal': 30.0},
            'instances': {},
            'capacity': 100.0
        }
        inv = Inventory.from_dict(data)

        assert inv.get_quantity('wood') == 50.0
        assert inv.get_quantity('metal') == 30.0
        assert inv.capacity == 100.0

    def test_from_dict_with_tools(self):
        """Test deserializing inventory with tools."""
        data = {
            'quantities': {},
            'instances': {
                'hammer': [
                    {
                        'name': 'hammer',
                        'base_value': 50.0,
                        'is_tool': True,
                        'durability': 80.0,
                        'required_skill': 'crafting'
                    },
                    {
                        'name': 'hammer',
                        'base_value': 50.0,
                        'is_tool': True,
                        'durability': 100.0,
                        'required_skill': 'crafting'
                    }
                ]
            },
            'capacity': None
        }
        inv = Inventory.from_dict(data)

        assert "hammer" in inv.instances
        assert len(inv.instances['hammer']) == 2
        assert inv.instances['hammer'][0].durability == 80.0
        assert inv.instances['hammer'][1].durability == 100.0

    def test_round_trip_serialization(self):
        """Test that serialization round-trips correctly."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0,
            required_skill="crafting"
        )

        inv.add(wood, 50.0)
        inv.add(hammer, 2.0)

        data = inv.to_dict()
        restored = Inventory.from_dict(data)

        assert restored.get_quantity('wood') == 50.0
        assert len(restored.instances['hammer']) == 2
        assert restored.capacity == 100.0


@pytest.mark.unit
class TestInventoryEdgeCases:
    """Test inventory edge cases."""

    def test_add_zero_quantity(self):
        """Test adding zero quantity."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, 0.0)
        assert result is True
        assert inv.get_quantity("wood") == 0.0

    def test_remove_zero_quantity(self):
        """Test removing zero quantity."""
        inv = Inventory()
        wood = Resource(name="wood", base_value=10.0)
        inv.add(wood, 50.0)

        result = inv.remove("wood", 0.0)
        assert result is True
        assert inv.get_quantity("wood") == 50.0

    def test_fractional_tool_quantity(self):
        """Test adding fractional tool quantity (should add floor)."""
        inv = Inventory()
        hammer = Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0
        )

        inv.add(hammer, 2.5)  # Should add 2 tools
        assert len(inv.instances["hammer"]) == 2

    def test_capacity_with_no_resources(self):
        """Test capacity check with empty inventory."""
        inv = Inventory(capacity=100.0)
        wood = Resource(name="wood", base_value=10.0)

        result = inv.add(wood, 50.0)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
