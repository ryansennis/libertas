"""Tool class - worker-held tools with durability."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .resource import ResourceInfo


@dataclass
class Tool:
    """
    Worker-held tool with durability (hammer, saw, wrench).
    Portable, used in recipes, degrades with use.
    Each instance is unique and tracked separately.
    """
    info: ResourceInfo  # Composition
    durability: int = 100
    max_durability: int = 100
    required_skill: Optional[str] = None
    enables_recipes: List[str] = field(default_factory=list)
    repair_cost: float = 10.0

    def use(self) -> bool:
        """Use tool, degrade durability. Returns True if still usable."""
        self.durability = max(0, self.durability - 1)
        return self.durability > 0

    def is_broken(self) -> bool:
        """Check if tool is broken."""
        return self.durability <= 0

    def repair(self, amount: Optional[int] = None) -> float:
        """Repair tool, return cost."""
        if amount is None:
            amount = self.max_durability - self.durability

        cost = (amount / self.max_durability) * self.repair_cost
        self.durability = min(self.max_durability, self.durability + amount)
        return cost

    def get_value(self, market_multiplier: float = 1.0) -> float:
        """Value scales with durability."""
        durability_factor = self.durability / self.max_durability
        return round(self.info.base_value * durability_factor * market_multiplier, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'tool',
            'info': self.info.to_dict(),
            'durability': self.durability,
            'max_durability': self.max_durability,
            'required_skill': self.required_skill,
            'enables_recipes': self.enables_recipes.copy(),
            'repair_cost': self.repair_cost
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tool':
        """Deserialize from dictionary."""
        return cls(
            info=ResourceInfo.from_dict(data['info']),
            durability=data.get('durability', 100),
            max_durability=data.get('max_durability', 100),
            required_skill=data.get('required_skill'),
            enables_recipes=data.get('enables_recipes', []).copy(),
            repair_cost=data.get('repair_cost', 10.0)
        )
