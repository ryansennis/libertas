"""Material class - fungible/bulk resources like wood, stone, wheat."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .resource import ResourceInfo


@dataclass
class Material:
    """
    Fungible/bulk material that can be stacked (wood, stone, wheat, metal).
    Used in production recipes. Stored in pod inventory.
    Multiple units of the same material can be merged together.
    """
    info: ResourceInfo  # Composition, not inheritance
    quantity: float = 1.0

    def get_value(self, market_multiplier: float = 1.0) -> float:
        """Value scales with quantity."""
        base = self.info.base_value * market_multiplier
        # Apply property-based modifiers
        for prop_name, prop_value in self.info.properties.items():
            if prop_name == "quality":
                base *= (0.5 + prop_value)  # quality 0-1 scale
            elif prop_name == "rarity":
                base *= (1 + prop_value)
        return round(base * self.quantity, 2)

    def split(self, amount: float) -> Optional['Material']:
        """Split off some quantity into a new Material instance."""
        if amount <= 0 or amount > self.quantity:
            return None
        self.quantity -= amount
        # Create new material with same info (shared reference is fine for immutable-ish data)
        return Material(info=self.info, quantity=amount)

    def merge(self, other: 'Material') -> bool:
        """Merge with another Material of same type."""
        if other.info.name != self.info.name:
            return False
        self.quantity += other.quantity
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'type': 'material',
            'info': self.info.to_dict(),
            'quantity': self.quantity
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Material':
        """Deserialize from dictionary."""
        return cls(
            info=ResourceInfo.from_dict(data['info']),
            quantity=data.get('quantity', 1.0)
        )
