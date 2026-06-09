"""Material class - fungible/bulk resources like wood, stone, wheat."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .resource import Resource, ResourceTag


@dataclass
class Material(Resource):
    """
    Fungible/bulk material that can be stacked (wood, stone, wheat, metal).
    Used in production recipes. Stored in pod inventory.
    Multiple units of the same material can be merged together.

    Economic model:
    - Raw materials (wood, stone): production_cost = 0 (found in nature), base_value = labor cost
    - Processed materials (planks, metal ingots): production_cost = sum(inputs), base_value = cost + margin
    """
    quantity: float = 1.0

    def get_buy_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to buy from market (what you pay).
        Based on base_value (production cost + profit margin).
        Market multiplier adjusts for supply/demand.
        """
        base = self.base_value * market_multiplier
        # Apply property-based modifiers
        for prop_name, prop_value in self.properties.items():
            if prop_name == "quality":
                base *= (0.5 + prop_value)  # quality 0-1 scale
            elif prop_name == "rarity":
                base *= (1 + prop_value)
        return round(base * self.quantity, 2)

    def get_sell_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to sell to market (what you receive).
        Typically 70-80% of buy price (market spread).
        Floor at production cost (can't sell below cost of inputs).
        """
        buy_price = self.get_buy_price(market_multiplier)
        spread = 0.75  # Market takes 25% spread
        sell_price = buy_price * spread

        # Floor: never sell below production cost per unit
        cost_floor = self.production_cost * self.quantity
        return round(max(sell_price, cost_floor), 2)

    def split(self, amount: float) -> Optional['Material']:
        """Split off some quantity into a new Material instance."""
        if amount <= 0 or amount > self.quantity:
            return None
        self.quantity -= amount
        # Create new material with same properties
        return Material(
            name=self.name,
            resource_id=self.resource_id,
            base_value=self.base_value,
            weight=self.weight,
            tags=self.tags.copy(),
            properties=self.properties.copy(),
            invented_by=self.invented_by,
            invention_step=self.invention_step,
            quantity=amount
        )

    def merge(self, other: 'Material') -> bool:
        """Merge with another Material of same type."""
        if other.name != self.name:
            return False
        self.quantity += other.quantity
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = self._base_to_dict()
        result['type'] = 'material'
        result['quantity'] = self.quantity
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Material':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            resource_id=data.get('resource_id'),
            base_value=data.get('base_value', 1.0),
            production_cost=data.get('production_cost', 0.0),
            weight=data.get('weight', 1.0),
            tags=[ResourceTag(t) for t in data.get('tags', [])],
            properties=data.get('properties', {}).copy(),
            invented_by=data.get('invented_by'),
            invention_step=data.get('invention_step'),
            quantity=data.get('quantity', 1.0)
        )
