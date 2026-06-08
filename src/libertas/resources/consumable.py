"""Consumable class - single-use items for worker needs."""

from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum
from .resource import Resource, ResourceTag


class ConsumableType(Enum):
    """Types of consumables - things that can be consumed/used up."""
    FOOD = "food"              # Edible items that satisfy hunger
    WATER = "water"            # Drinkable liquids that satisfy thirst
    ALCOHOL = "alcohol"        # Alcoholic beverages (satisfy recreation, potentially hunger/thirst)
    ENTERTAINMENT = "entertainment"  # Consumable entertainment (movie tickets, event access)


@dataclass
class Consumable(Resource):
    """
    Single-use consumable item (food, water, entertainment).
    Used to satisfy worker needs. Stackable like materials.
    """
    need_type: ConsumableType = ConsumableType.FOOD
    satisfaction_value: float = 0.4  # How much need is satisfied (0-1)
    quantity: float = 1.0  # Can be stacked

    def consume(self) -> float:
        """Consume one unit, return satisfaction value."""
        if self.quantity >= 1.0:
            self.quantity -= 1.0
            return self.satisfaction_value
        return 0.0

    def get_buy_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to buy consumables from market.
        Higher satisfaction value = premium price.
        """
        satisfaction_premium = 1.0 + (self.satisfaction_value * 0.5)
        return round(
            self.base_value * self.quantity * market_multiplier * satisfaction_premium,
            2
        )

    def get_sell_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to sell consumables to market.
        Perishability and demand volatility = higher spread.
        Floor at production cost (raw ingredients).
        """
        buy_price = self.get_buy_price(market_multiplier)
        spread = 0.80  # Consumables have lower spread (20%) due to fast turnover
        sell_price = buy_price * spread

        # Floor: production cost per unit
        cost_floor = self.production_cost * self.quantity
        return round(max(sell_price, cost_floor), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = self._base_to_dict()
        result['type'] = 'consumable'
        result['need_type'] = self.need_type
        result['satisfaction_value'] = self.satisfaction_value
        result['quantity'] = self.quantity
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Consumable':
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
            need_type=data.get('need_type', 'hunger'),
            satisfaction_value=data.get('satisfaction_value', 0.4),
            quantity=data.get('quantity', 1.0)
        )
