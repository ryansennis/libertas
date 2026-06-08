"""Resource registry for tracking all known resource types."""

from typing import Dict, List, Optional, Union
from .resource import ResourceInfo
from .material import Material
from .tool import Tool
from .equipment import Equipment
from .consumable import Consumable

ResourceType = Union[Material, Tool, Equipment, Consumable]


class ResourceRegistry:
    """
    Registry of all known resource types in the simulation.
    Tracks materials, tools, equipment, and consumables that can be produced or traded.
    """

    def __init__(self):
        # Store by name for quick lookup
        self._materials: Dict[str, Material] = {}
        self._tools: Dict[str, Tool] = {}
        self._equipment: Dict[str, Equipment] = {}
        self._consumables: Dict[str, Consumable] = {}
        self._invention_history: List[Dict] = []

    def register_material(self, material: Material) -> None:
        """Register a material type."""
        if material.info.name not in self._materials:
            self._materials[material.info.name] = material

    def register_tool(self, tool: Tool) -> None:
        """Register a tool type."""
        if tool.info.name not in self._tools:
            self._tools[tool.info.name] = tool

    def register_equipment(self, equipment: Equipment) -> None:
        """Register an equipment type."""
        if equipment.info.name not in self._equipment:
            self._equipment[equipment.info.name] = equipment

    def register_consumable(self, consumable: Consumable) -> None:
        """Register a consumable type."""
        if consumable.info.name not in self._consumables:
            self._consumables[consumable.info.name] = consumable

    def register(self, resource: ResourceType) -> None:
        """Register any resource type (polymorphic)."""
        if isinstance(resource, Material):
            self.register_material(resource)
        elif isinstance(resource, Tool):
            self.register_tool(resource)
        elif isinstance(resource, Equipment):
            self.register_equipment(resource)
        elif isinstance(resource, Consumable):
            self.register_consumable(resource)

    def get(self, name: str) -> Optional[ResourceType]:
        """Get a resource by name (checks all types)."""
        if name in self._materials:
            return self._materials[name]
        if name in self._tools:
            return self._tools[name]
        if name in self._equipment:
            return self._equipment[name]
        if name in self._consumables:
            return self._consumables[name]
        return None

    def get_material(self, name: str) -> Optional[Material]:
        """Get a material by name."""
        return self._materials.get(name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_equipment(self, name: str) -> Optional[Equipment]:
        """Get equipment by name."""
        return self._equipment.get(name)

    def get_consumable(self, name: str) -> Optional[Consumable]:
        """Get a consumable by name."""
        return self._consumables.get(name)

    def invent_material(self, name: str, inventor_id: str, step: int, **kwargs) -> Material:
        """Invent a new material type at runtime."""
        info = ResourceInfo(
            name=name,
            invented_by=inventor_id,
            invention_step=step,
            **{k: v for k, v in kwargs.items() if k in ['base_value', 'weight', 'tags', 'properties']}
        )
        material = Material(info=info, **{k: v for k, v in kwargs.items() if k == 'quantity'})
        self.register_material(material)
        self._record_invention(name, inventor_id, step, 'material', kwargs)
        return material

    def invent_tool(self, name: str, inventor_id: str, step: int, **kwargs) -> Tool:
        """Invent a new tool type at runtime."""
        info = ResourceInfo(
            name=name,
            invented_by=inventor_id,
            invention_step=step,
            **{k: v for k, v in kwargs.items() if k in ['base_value', 'weight', 'tags', 'properties']}
        )
        tool = Tool(info=info, **{k: v for k, v in kwargs.items() if k in ['durability', 'max_durability', 'required_skill', 'enables_recipes', 'repair_cost']})
        self.register_tool(tool)
        self._record_invention(name, inventor_id, step, 'tool', kwargs)
        return tool

    def _record_invention(self, name: str, inventor_id: str, step: int, resource_type: str, properties: dict):
        """Record an invention in history."""
        self._invention_history.append({
            'name': name,
            'inventor': inventor_id,
            'step': step,
            'type': resource_type,
            'properties': properties
        })

    def list_resources(self) -> List[str]:
        """List all known resource names."""
        return (list(self._materials.keys()) +
                list(self._tools.keys()) +
                list(self._equipment.keys()) +
                list(self._consumables.keys()))

    def is_known(self, name: str) -> bool:
        """Check if a resource type is known."""
        return (name in self._materials or
                name in self._tools or
                name in self._equipment or
                name in self._consumables)

    @property
    def invention_history(self) -> List[Dict]:
        """Get history of all resource inventions."""
        return self._invention_history.copy()
