"""Equipment class - pod-level machinery."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .resource import Resource, ResourceTag


@dataclass
class Equipment(Resource):
    """
    Pod-level equipment/machinery (lathe, CNC machine, forge, loom).
    Too large/expensive for workers to carry. Shared by pod.
    Requires periodic maintenance.
    """
    durability: int = 1000
    max_durability: int = 1000
    required_skill: Optional[str] = None
    enables_recipes: List[str] = field(default_factory=list)
    maintenance_cost: float = 50.0  # Per maintenance cycle
    maintenance_interval: int = 100  # Steps between maintenance
    last_maintenance: int = 0
    capacity: int = 1  # How many workers can use simultaneously

    def use(self) -> bool:
        """Use equipment, degrade durability. Returns True if still usable."""
        self.durability = max(0, self.durability - 1)
        return self.durability > 0

    def needs_maintenance(self, current_step: int) -> bool:
        """Check if equipment needs maintenance."""
        return (current_step - self.last_maintenance) >= self.maintenance_interval

    def perform_maintenance(self, current_step: int) -> float:
        """Perform maintenance, return cost."""
        self.durability = min(self.max_durability, self.durability + 100)
        self.last_maintenance = current_step
        return self.maintenance_cost

    def get_buy_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to buy equipment from market.
        Depreciates with durability. Capacity adds premium.
        """
        durability_factor = self.durability / self.max_durability
        capacity_factor = 1.0 + (self.capacity - 1) * 0.5  # More capacity = higher value
        return round(self.base_value * durability_factor * capacity_factor * market_multiplier, 2)

    def get_sell_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to sell equipment to market.
        Heavy depreciation due to maintenance needs and size.
        Floor at scrap value (30% of production cost - more salvageable than tools).
        """
        buy_price = self.get_buy_price(market_multiplier)
        spread = 0.70  # Equipment has higher spread (30%) due to size/complexity
        sell_price = buy_price * spread

        # Floor: scrap value
        scrap_value = self.production_cost * 0.3
        return round(max(sell_price, scrap_value), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = self._base_to_dict()
        result['type'] = 'equipment'
        result['durability'] = self.durability
        result['max_durability'] = self.max_durability
        result['required_skill'] = self.required_skill
        result['enables_recipes'] = self.enables_recipes.copy()
        result['maintenance_cost'] = self.maintenance_cost
        result['maintenance_interval'] = self.maintenance_interval
        result['last_maintenance'] = self.last_maintenance
        result['capacity'] = self.capacity
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Equipment':
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
            durability=data.get('durability', 1000),
            max_durability=data.get('max_durability', 1000),
            required_skill=data.get('required_skill'),
            enables_recipes=data.get('enables_recipes', []).copy(),
            maintenance_cost=data.get('maintenance_cost', 50.0),
            maintenance_interval=data.get('maintenance_interval', 100),
            last_maintenance=data.get('last_maintenance', 0),
            capacity=data.get('capacity', 1)
        )
