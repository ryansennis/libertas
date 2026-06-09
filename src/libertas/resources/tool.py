"""Tool class - worker-held tools with durability."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .resource import Resource, ResourceTag


@dataclass
class Tool(Resource):
    """
    Worker-held tool with durability (hammer, saw, wrench).
    Portable, used in recipes, degrades with use.
    Each instance is unique and tracked separately.
    """
    durability: int = 100
    max_durability: int = 100
    required_skill: Optional[str] = None
    enables_recipes: List[str] = field(default_factory=list)
    repair_cost: float = 10.0

    def use(self) -> bool:
        """Use tool, degrade durability. Returns True if still usable."""
        if self.durability is None:
            return True  # Infinite durability
        self.durability = max(0, self.durability - 1)
        return self.durability > 0

    def is_broken(self) -> bool:
        """Check if tool is broken."""
        if self.durability is None:
            return False  # Infinite durability never breaks
        return self.durability <= 0

    def repair(self, amount: Optional[int] = None) -> float:
        """Repair tool, return cost."""
        if amount is None:
            amount = self.max_durability - self.durability

        cost = (amount / self.max_durability) * self.repair_cost
        self.durability = min(self.max_durability, self.durability + amount)
        return cost

    def get_buy_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to buy tool from market.
        New tools sell at base_value (includes material + labor costs).
        Used tools depreciate linearly with durability.
        """
        if self.durability is None:
            # Infinite durability
            durability_factor = 1.0
        else:
            durability_factor = self.durability / self.max_durability
        return round(self.base_value * durability_factor * market_multiplier, 2)

    def get_sell_price(self, market_multiplier: float = 1.0) -> float:
        """
        Price to sell tool to market.
        Market spread (75%) + durability depreciation.
        Floor at scrap value (20% of production cost for materials).
        """
        buy_price = self.get_buy_price(market_multiplier)
        spread = 0.75  # Market takes 25% spread
        sell_price = buy_price * spread

        # Floor: scrap value (can salvage materials)
        scrap_value = self.production_cost * 0.2
        return round(max(sell_price, scrap_value), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = self._base_to_dict()
        result['type'] = 'tool'
        result['durability'] = self.durability
        result['max_durability'] = self.max_durability
        result['required_skill'] = self.required_skill
        result['enables_recipes'] = self.enables_recipes.copy()
        result['repair_cost'] = self.repair_cost
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
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
            durability=data.get('durability', 100),
            max_durability=data.get('max_durability', 100),
            required_skill=data.get('required_skill'),
            enables_recipes=data.get('enables_recipes', []).copy(),
            repair_cost=data.get('repair_cost', 10.0)
        )
