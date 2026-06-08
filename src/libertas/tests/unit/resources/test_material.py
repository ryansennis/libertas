"""Tests for Material class."""

import unittest
import pytest
from libertas.resources import ResourceInfo, Material


@pytest.mark.unit
class TestMaterial(unittest.TestCase):
    """Test Material class."""

    def setUp(self):
        """Set up test fixtures."""
        self.info = ResourceInfo(name="wood", base_value=10.0, weight=1.0)
        self.material = Material(info=self.info, quantity=5.0)

    def test_material_creation(self):
        """Test creating a material."""
        self.assertEqual(self.material.info.name, "wood")
        self.assertEqual(self.material.quantity, 5.0)
        self.assertEqual(self.material.info.base_value, 10.0)

    def test_get_value(self):
        """Test calculating material value."""
        value = self.material.get_value(market_multiplier=1.0)
        self.assertEqual(value, 50.0)  # 10.0 * 5.0

    def test_get_value_with_multiplier(self):
        """Test value with market multiplier."""
        value = self.material.get_value(market_multiplier=2.0)
        self.assertEqual(value, 100.0)  # 10.0 * 2.0 * 5.0

    def test_split(self):
        """Test splitting material quantity."""
        split = self.material.split(2.0)

        self.assertIsNotNone(split)
        self.assertEqual(split.quantity, 2.0)
        self.assertEqual(self.material.quantity, 3.0)
        self.assertEqual(split.info.name, "wood")

    def test_split_too_much(self):
        """Test splitting more than available fails."""
        split = self.material.split(10.0)
        self.assertIsNone(split)
        self.assertEqual(self.material.quantity, 5.0)

    def test_split_zero(self):
        """Test splitting zero fails."""
        split = self.material.split(0.0)
        self.assertIsNone(split)

    def test_split_negative(self):
        """Test splitting negative fails."""
        split = self.material.split(-1.0)
        self.assertIsNone(split)

    def test_merge(self):
        """Test merging materials."""
        other = Material(info=self.info, quantity=3.0)
        result = self.material.merge(other)

        self.assertTrue(result)
        self.assertEqual(self.material.quantity, 8.0)

    def test_merge_different_type(self):
        """Test merging different material types fails."""
        other_info = ResourceInfo(name="stone", base_value=15.0)
        other = Material(info=other_info, quantity=3.0)

        result = self.material.merge(other)
        self.assertFalse(result)
        self.assertEqual(self.material.quantity, 5.0)

    def test_serialization(self):
        """Test material serialization."""
        data = self.material.to_dict()

        self.assertEqual(data['type'], 'material')
        self.assertEqual(data['quantity'], 5.0)
        self.assertEqual(data['info']['name'], 'wood')

    def test_deserialization(self):
        """Test material deserialization."""
        data = self.material.to_dict()
        restored = Material.from_dict(data)

        self.assertEqual(restored.info.name, 'wood')
        self.assertEqual(restored.quantity, 5.0)
        self.assertEqual(restored.info.base_value, 10.0)

    def test_with_properties(self):
        """Test material with quality properties."""
        info = ResourceInfo(
            name="gold",
            base_value=100.0,
            properties={"quality": 0.8, "rarity": 0.5}
        )
        material = Material(info=info, quantity=1.0)

        # Value should be affected by quality and rarity
        value = material.get_value(market_multiplier=1.0)
        # base(100) * quality_factor(0.5 + 0.8 = 1.3) * rarity_factor(1 + 0.5 = 1.5) * quantity(1)
        expected = round(100.0 * 1.3 * 1.5, 2)
        self.assertEqual(value, expected)


if __name__ == "__main__":
    unittest.main()
