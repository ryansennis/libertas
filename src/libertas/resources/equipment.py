"""Equipment class - pod-level machinery and large tools."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .resource import ResourceInfo


@dataclass
class Equipment:
    """
    Pod-level equipment/machinery (lathe, CNC machine, forge, loom).
    Too large/expensive for workers to carry. Shared by pod.
    Requires maintenance and can be used by multiple workers.
    """
    info: ResourceInfo  # Composition
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

    def is_broken(self) -> bool:
        """Check if equipment is broken."""
        return self.durability <= 0

    def needs_maintenance(self, current_step: int) -> bool:
        """Check if equipment needs maintenance."""
        return (current_step - self.last_maintenance) >= self.maintenance_interval

    def perform_maintenance(self, current_step: int) -> float:
        """Perform maintenance, return cost."""
        self.durability = min(self.max_durability, self.durability + 100)
        self.last_maintenance = current_step
        return self.maintenance_cost

    def get_value(self, market_multiplier: float = 1.0) -> float:
        """Value scales with durability and capacity."""
        durability_factor = self.durability / self.max_durability
        capacity_bonus = 1.0 + (self.capacity * 0.2)
        return round(self.info.base_value * durability_factor * capacity_bonus * market_multiplier, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'equipment',
            'info': self.info.to_dict(),
            'durability': self.durability,
            'max_durability': self.max_durability,
            'required_skill': self.required_skill,
            'enables_recipes': self.enables_recipes.copy(),
            'maintenance_cost': self.maintenance_cost,
            'maintenance_interval': self.maintenance_interval,
            'last_maintenance': self.last_maintenance,
            'capacity': self.capacity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Equipment':
        """Deserialize from dictionary."""
        return cls(
            info=ResourceInfo.from_dict(data['info']),
            durability=data.get('durability', 1000),
            max_durability=data.get('max_durability', 1000),
            required_skill=data.get('required_skill'),
            enables_recipes=data.get('enables_recipes', []).copy(),
            maintenance_cost=data.get('maintenance_cost', 50.0),
            maintenance_interval=data.get('maintenance_interval', 100),
            last_maintenance=data.get('last_maintenance', 0),
            capacity=data.get('capacity', 1)
        )
