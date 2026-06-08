"""Consumable class - single-use items for worker needs."""

from dataclasses import dataclass
from typing import Dict, Any
from .resource import ResourceInfo


@dataclass
class Consumable:
    """
    Single-use consumable item (food, water, entertainment).
    Used to satisfy worker needs. Can be stacked like materials.
    """
    info: ResourceInfo  # Composition
    need_type: str = "hunger"  # "hunger", "thirst", "recreation"
    satisfaction_value: float = 0.4  # How much need is satisfied (0-1)
    quantity: float = 1.0  # Can be stacked

    def consume(self) -> float:
        """Consume one unit, return satisfaction value."""
        if self.quantity >= 1.0:
            self.quantity -= 1.0
            return self.satisfaction_value
        return 0.0

    def get_value(self, market_multiplier: float = 1.0) -> float:
        """Value scales with quantity and satisfaction."""
        return round(self.info.base_value * self.quantity * market_multiplier * (1 + self.satisfaction_value), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'consumable',
            'info': self.info.to_dict(),
            'need_type': self.need_type,
            'satisfaction_value': self.satisfaction_value,
            'quantity': self.quantity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Consumable':
        """Deserialize from dictionary."""
        return cls(
            info=ResourceInfo.from_dict(data['info']),
            need_type=data.get('need_type', 'hunger'),
            satisfaction_value=data.get('satisfaction_value', 0.4),
            quantity=data.get('quantity', 1.0)
        )
