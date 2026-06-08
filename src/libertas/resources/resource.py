"""Resource information component - shared metadata for all resource types."""

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
class ResourceInfo:
    """
    Shared metadata component for all resource types.
    Not a base class - embedded via composition in Material, Tool, Equipment, Consumable.
    """
    name: str
    resource_id: str = field(default_factory=lambda: str(uuid4()))
    base_value: float = 1.0
    weight: float = 1.0
    tags: List[ResourceTag] = field(default_factory=list)
    properties: Dict[str, float] = field(default_factory=dict)
    invented_by: Optional[str] = None
    invention_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize common fields."""
        return {
            'name': self.name,
            'resource_id': self.resource_id,
            'base_value': self.base_value,
            'weight': self.weight,
            'tags': [t.value for t in self.tags],
            'properties': self.properties.copy(),
            'invented_by': self.invented_by,
            'invention_step': self.invention_step
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceInfo':
        """Deserialize from dict."""
        return cls(
            name=data['name'],
            resource_id=data.get('resource_id', str(uuid4())),
            base_value=data.get('base_value', 1.0),
            weight=data.get('weight', 1.0),
            tags=[ResourceTag(t) for t in data.get('tags', [])],
            properties=data.get('properties', {}).copy(),
            invented_by=data.get('invented_by'),
            invention_step=data.get('invention_step')
        )
