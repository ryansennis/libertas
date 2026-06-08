"""Tests for Material class."""

import unittest
import pytest
from libertas.resources import Material


@pytest.mark.unit
class TestMaterial(unittest.TestCase):
    """Test Material class."""

    def setUp(self):
        """Set up test fixtures."""
        self.material = Material(name="wood", base_value=10.0, weight=1.0, quantity=5.0)

    def test_material_creation(self):
        """Test creating a material."""
        self.assertEqual(self.material.name, "wood")
        self.assertEqual(self.material.quantity, 5.0)
        self.assertEqual(self.material.base_value, 10.0)

    def test_get_buy_price(self):
        """Test calculating material buy price."""
        buy_price = self.material.get_buy_price(market_multiplier=1.0)
        self.assertEqual(buy_price, 50.0)  # 10.0 * 5.0

    def test_get_buy_price_with_multiplier(self):
        """Test buy price with market multiplier."""
        buy_price = self.material.get_buy_price(market_multiplier=2.0)
        self.assertEqual(buy_price, 100.0)  # 10.0 * 2.0 * 5.0

    def test_get_sell_price(self):
        """Test calculating material sell price (with spread)."""
        buy_price = self.material.get_buy_price(market_multiplier=1.0)  # 50.0
        sell_price = self.material.get_sell_price(market_multiplier=1.0)
        self.assertEqual(sell_price, 37.5)  # 50.0 * 0.75 spread

    def test_sell_price_floor(self):
        """Test sell price has floor at production cost."""
        # Material with high production cost
        expensive = Material(name="steel", base_value=5.0, production_cost=10.0, quantity=1.0)
        sell_price = expensive.get_sell_price()
        # Sell price should be at least production cost
        self.assertEqual(sell_price, 10.0)  # Floored at production_cost

    def test_split(self):
        """Test splitting material quantity."""
        split = self.material.split(2.0)

        self.assertIsNotNone(split)
        self.assertEqual(split.quantity, 2.0)
        self.assertEqual(self.material.quantity, 3.0)
        self.assertEqual(split.name, "wood")

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
        other = Material(name="wood", base_value=10.0, quantity=3.0)
        result = self.material.merge(other)

        self.assertTrue(result)
        self.assertEqual(self.material.quantity, 8.0)

    def test_merge_different_type(self):
        """Test merging different material types fails."""
        other = Material(name="stone", base_value=15.0, quantity=3.0)

        result = self.material.merge(other)
        self.assertFalse(result)
        self.assertEqual(self.material.quantity, 5.0)

    def test_serialization(self):
        """Test material serialization."""
        data = self.material.to_dict()

        self.assertEqual(data['type'], 'material')
        self.assertEqual(data['quantity'], 5.0)
        self.assertEqual(data['name'], 'wood')

    def test_deserialization(self):
        """Test material deserialization."""
        data = self.material.to_dict()
        restored = Material.from_dict(data)

        self.assertEqual(restored.name, 'wood')
        self.assertEqual(restored.quantity, 5.0)
        self.assertEqual(restored.base_value, 10.0)

    def test_with_properties(self):
        """Test material with quality properties."""
        material = Material(
            name="gold",
            base_value=100.0,
            properties={"quality": 0.8, "rarity": 0.5},
            quantity=1.0
        )

        # Buy price should be affected by quality and rarity
        buy_price = material.get_buy_price(market_multiplier=1.0)
        # base(100) * quality_factor(0.5 + 0.8 = 1.3) * rarity_factor(1 + 0.5 = 1.5) * quantity(1)
        expected = round(100.0 * 1.3 * 1.5, 2)
        self.assertEqual(buy_price, expected)


if __name__ == "__main__":
    unittest.main()
