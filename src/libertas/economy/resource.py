from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
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
class Resource:
    """
    A resource in the economy. Can be raw materials, components, finished goods, or tools.
    
    Resources can be invented at runtime by workers, making the economy open-ended.
    """
    
    name: str
    resource_id: str = field(default_factory=lambda: str(uuid4()))
    
    base_value: float = 1.0  # Base market value in currency units
    weight: float = 1.0  # Storage weight (affects transport costs)
    tags: List[ResourceTag] = field(default_factory=list)
    
    # Dynamic properties discovered/defined at invention time
    properties: Dict[str, float] = field(default_factory=dict)
    
    invented_by: Optional[str] = None  # worker_id or pod_id who invented it
    invention_step: Optional[int] = None  # simulation step when invented
    
    is_tool: bool = False
    durability: Optional[int] = None  # max uses before breaking
    required_skill: Optional[str] = None  # skill needed to use this tool
    enables_recipes: Optional[List[str]] = None  # recipe names this tool enables
    
    def get_value(self, market_multiplier: float = 1.0) -> float:
        """Calculate current value based on base value and market conditions."""
        value = self.base_value * market_multiplier
        
        # Apply property-based modifiers
        for prop_name, prop_value in self.properties.items():
            if prop_name == "quality":
                value *= (0.5 + prop_value)  # quality 0-1 scale
            elif prop_name == "rarity":
                value *= (1 + prop_value)
        
        return round(value, 2)
    
    def use_tool(self) -> bool:
        """Use the tool, degrading durability. Returns True if still usable."""
        if not self.is_tool:
            return False
        
        if self.durability is None:
            return True
        
        self.durability -= 1
        return self.durability >= 0
    
    def is_broken(self) -> bool:
        """Check if tool is broken."""
        return self.is_tool is not None and self.durability is not None and self.durability <= 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            'resource_id': self.resource_id,
            'name': self.name,
            'base_value': self.base_value,
            'weight': self.weight,
            'tags': [tag.value for tag in self.tags],
            'properties': self.properties,
            'invented_by': self.invented_by,
            'invention_step': self.invention_step,
            'is_tool': self.is_tool,
            'durability': self.durability,
            'required_skill': self.required_skill,
            'enables_recipes': self.enables_recipes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resource':
        """Create resource from dictionary."""
        return cls(
            name=data['name'],
            resource_id=data.get('resource_id', str(uuid4())),
            base_value=data.get('base_value', 1.0),
            weight=data.get('weight', 1.0),
            tags=[ResourceTag(t) for t in data.get('tags', [])],
            properties=data.get('properties', {}),
            invented_by=data.get('invented_by'),
            invention_step=data.get('invention_step'),
            is_tool=data.get('is_tool', False),
            durability=data.get('durability'),
            required_skill=data.get('required_skill'),
            enables_recipes=data.get('enables_recipes', [])
        )
    
    @classmethod
    def invent(
        cls,
        name: str,
        inventor_id: str,
        step: int,
        base_value: float = 1.0,
        is_tool: bool = False,
        tags: Optional[List[ResourceTag]] = None,
        properties: Optional[Dict[str, float]] = None,
        required_skill: Optional[str] = None
    ) -> 'Resource':
        """Invent a new resource type at runtime."""
        return cls(
            name=name,
            base_value=base_value,
            tags=tags or [],
            properties=properties or {},
            invented_by=inventor_id,
            invention_step=step,
            is_tool=is_tool,
            required_skill=required_skill
        )

class ResourceRegistry:
    """Registry of all known resource types in the economy.

    During migration: supports BOTH old Resource and new Material/Tool/Equipment/Consumable.
    """

    def __init__(self):
        self._resources: Dict[str, Resource] = {}
        self._invention_history: List[Dict] = []

        # NEW SYSTEM: parallel tracking (Phase 2)
        try:
            from ..resources import Material, Tool, Equipment, Consumable
            self._materials: Dict[str, 'Material'] = {}
            self._tools: Dict[str, 'Tool'] = {}
            self._equipment: Dict[str, 'Equipment'] = {}
            self._consumables: Dict[str, 'Consumable'] = {}
            self._new_system_available = True
        except ImportError:
            self._new_system_available = False

    def register(self, resource: Resource) -> None:
        """Register an old-style resource type."""
        if resource.name not in self._resources:
            self._resources[resource.name] = resource

    def register_material(self, material: 'Material') -> None:
        """Register a new-style Material."""
        if self._new_system_available and material.info.name not in self._materials:
            self._materials[material.info.name] = material

    def register_tool(self, tool: 'Tool') -> None:
        """Register a new-style Tool."""
        if self._new_system_available and tool.info.name not in self._tools:
            self._tools[tool.info.name] = tool

    def register_equipment(self, equipment: 'Equipment') -> None:
        """Register a new-style Equipment."""
        if self._new_system_available and equipment.info.name not in self._equipment:
            self._equipment[equipment.info.name] = equipment

    def register_consumable(self, consumable: 'Consumable') -> None:
        """Register a new-style Consumable."""
        if self._new_system_available and consumable.info.name not in self._consumables:
            self._consumables[consumable.info.name] = consumable
    
    def get(self, name: str) -> Optional[Resource]:
        """Get a resource by name.

        During migration: returns OLD Resource type for compatibility.
        Use get_material(), get_tool(), etc. for new types.
        """
        return self._resources.get(name)

    def get_material(self, name: str) -> Optional['Material']:
        """Get a Material by name (new system)."""
        if self._new_system_available:
            return self._materials.get(name)
        return None

    def get_tool(self, name: str) -> Optional['Tool']:
        """Get a Tool by name (new system)."""
        if self._new_system_available:
            return self._tools.get(name)
        return None

    
    def invent(self, name: str, inventor_id: str, step: int,
               **kwargs) -> Resource:
        """Invent a new resource type."""
        resource = Resource.invent(name, inventor_id, step, **kwargs)
        self.register(resource)
        self._invention_history.append({
            'name': name,
            'inventor': inventor_id,
            'step': step,
            'properties': resource.properties,
            'is_tool': resource.is_tool
        })
        return resource
    
    def list_resources(self) -> List[str]:
        """List all known resource names."""
        return list(self._resources.keys())
    
    def is_known(self, name: str) -> bool:
        """Check if a resource type is known."""
        return name in self._resources
    
    @property
    def invention_history(self) -> List[Dict]:
        """Get history of all resource inventions."""
        return self._invention_history.copy()