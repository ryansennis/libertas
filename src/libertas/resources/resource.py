"""Base Resource class with proper inheritance hierarchy."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from uuid import uuid4


class ResourceTag(Enum):
    """Tags for categorizing resources."""
    RAW = "raw"
    INTERMEDIATE = "intermediate"
    FINISHED = "finished"
    TOOL = "tool"
    CONSUMABLE = "consumable"


@dataclass
class Resource(ABC):
    """
    Abstract base class for all resources.
    Provides common fields and polymorphic interface.

    Economic model:
    - production_cost: Cost of inputs to produce this resource (materials, labor, tool depreciation)
    - base_value: Base market selling price (typically production_cost + profit margin)
    - Buy/sell spread: Markets charge more to sell than they pay to buy
    """
    name: str
    resource_id: str = field(default_factory=lambda: str(uuid4()))
    base_value: float = 1.0  # Base selling price
    production_cost: float = 0.0  # Cost to produce (sum of inputs)
    weight: float = 1.0
    tags: List[ResourceTag] = field(default_factory=list)
    properties: Dict[str, float] = field(default_factory=dict)
    invented_by: Optional[str] = None
    invention_step: Optional[int] = None

    @abstractmethod
    def get_buy_price(self, market_multiplier: float = 1.0) -> float:
        """
        Calculate buy price (what you pay to acquire from market).
        This is the higher price - markets charge more to sell to you.
        """
        pass

    @abstractmethod
    def get_sell_price(self, market_multiplier: float = 1.0) -> float:
        """
        Calculate sell price (what you receive when selling to market).
        This is the lower price - markets pay less when buying from you.
        Typical spread: sell_price = buy_price * 0.7-0.9
        """
        pass

    def get_profit_margin(self) -> float:
        """Calculate profit margin: (base_value - production_cost) / production_cost"""
        if self.production_cost == 0:
            return 0.0
        return (self.base_value - self.production_cost) / self.production_cost

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resource':
        """Deserialize from dictionary."""
        pass

    def _base_to_dict(self) -> Dict[str, Any]:
        """Helper to serialize common base fields."""
        return {
            'name': self.name,
            'resource_id': self.resource_id,
            'base_value': self.base_value,
            'production_cost': self.production_cost,
            'weight': self.weight,
            'tags': [t.value for t in self.tags],
            'properties': self.properties.copy(),
            'invented_by': self.invented_by,
            'invention_step': self.invention_step
        }
