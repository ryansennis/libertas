"""Inventory classes for managing resource collections.

Following the pattern of Federation being a MutableSet[Pod],
inventories are collection-like structures for resources.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, List, Dict, Any, Iterator
from itertools import chain

from .material import Material
from .tool import Tool
from .equipment import Equipment
from .consumable import Consumable

# Union type for all resource types
ResourceType = Union[Material, Tool, Equipment, Consumable]


class BaseInventory(ABC):
    """
    Abstract base for all inventory types.
    Similar to how Federation is a MutableSet[Pod], inventories manage collections of resources.
    """

    def __init__(self, capacity: Optional[float] = None):
        self.capacity = capacity

    @abstractmethod
    def add(self, resource: ResourceType, quantity: float = 1.0) -> bool:
        """Add a resource to inventory."""
        pass

    @abstractmethod
    def remove(self, resource_name: str, quantity: float = 1.0) -> bool:
        """Remove a resource from inventory."""
        pass

    @abstractmethod
    def get_total_weight(self) -> float:
        """Calculate total weight of contents."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[ResourceType]:
        """Iterate over all resources."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get count of distinct resource types."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseInventory':
        """Deserialize from dict."""
        pass


class WorkerInventory(BaseInventory):
    """
    Inventory for a worker - holds tools only.
    Limited capacity enforces realistic tool-carrying limits.
    """

    def __init__(self, capacity: float = 10.0):
        super().__init__(capacity)
        self._tools: Dict[str, Tool] = {}  # resource_id → Tool

    def add(self, resource: ResourceType, quantity: float = 1.0) -> bool:
        """Add a resource (only tools allowed for workers)."""
        if not isinstance(resource, Tool):
            return False  # Workers can only hold tools

        if self.capacity and self.get_total_weight() + resource.info.weight > self.capacity:
            return False

        self._tools[resource.info.resource_id] = resource
        return True

    def remove(self, resource_name: str, quantity: float = 1.0) -> bool:
        """Remove a tool by name."""
        for tool_id, tool in list(self._tools.items()):
            if tool.info.name == resource_name:
                del self._tools[tool_id]
                return True
        return False

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get first available tool by name (removes from inventory)."""
        for tool_id, tool in list(self._tools.items()):
            if tool.info.name == tool_name and not tool.is_broken():
                del self._tools[tool_id]
                return tool
        return None

    def count_tools(self, tool_name: str) -> int:
        """Count how many tools of a given name we have."""
        return sum(1 for t in self._tools.values() if t.info.name == tool_name)

    def get_total_weight(self) -> float:
        return sum(t.info.weight for t in self._tools.values())

    def __iter__(self) -> Iterator[Tool]:
        """Iterate over tools."""
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'type': 'worker',
            'capacity': self.capacity,
            'tools': {tid: tool.to_dict() for tid, tool in self._tools.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkerInventory':
        """Deserialize from dict."""
        inv = cls(capacity=data.get('capacity', 10.0))
        for tid, tool_data in data.get('tools', {}).items():
            tool = Tool.from_dict(tool_data)
            inv._tools[tid] = tool
        return inv


class PodInventory(BaseInventory):
    """
    Inventory for a pod - holds materials, tools, equipment, and consumables.
    Shared by all pod members. No strict capacity (can be unlimited).
    Collection-based like Federation[Pod].
    """

    def __init__(self, capacity: Optional[float] = None):
        super().__init__(capacity)
        self._materials: Dict[str, Material] = {}  # name → Material (fungible)
        self._tools: Dict[str, Tool] = {}  # resource_id → Tool (non-fungible)
        self._equipment: Dict[str, Equipment] = {}  # resource_id → Equipment (non-fungible)
        self._consumables: Dict[str, Consumable] = {}  # name → Consumable (stackable)

    def add(self, resource: ResourceType, quantity: float = 1.0) -> bool:
        """Add a resource, with type-specific handling."""
        if self.capacity:
            added_weight = self._calculate_added_weight(resource, quantity)
            if self.get_total_weight() + added_weight > self.capacity:
                return False

        if isinstance(resource, Material):
            if resource.info.name in self._materials:
                self._materials[resource.info.name].merge(resource)
            else:
                self._materials[resource.info.name] = resource

        elif isinstance(resource, Tool):
            self._tools[resource.info.resource_id] = resource

        elif isinstance(resource, Equipment):
            self._equipment[resource.info.resource_id] = resource

        elif isinstance(resource, Consumable):
            if resource.info.name in self._consumables:
                self._consumables[resource.info.name].quantity += quantity
            else:
                self._consumables[resource.info.name] = resource

        else:
            return False

        return True

    def remove(self, resource_name: str, quantity: float = 1.0) -> bool:
        """Remove a resource by name."""
        # Try materials first
        if resource_name in self._materials:
            material = self._materials[resource_name]
            if material.quantity >= quantity:
                material.quantity -= quantity
                if material.quantity <= 0:
                    del self._materials[resource_name]
                return True

        # Try consumables
        if resource_name in self._consumables:
            consumable = self._consumables[resource_name]
            if consumable.quantity >= quantity:
                consumable.quantity -= quantity
                if consumable.quantity <= 0:
                    del self._consumables[resource_name]
                return True

        # Try tools (quantity ignored for non-fungible)
        for tool_id, tool in list(self._tools.items()):
            if tool.info.name == resource_name:
                del self._tools[tool_id]
                return True

        # Try equipment
        for equip_id, equip in list(self._equipment.items()):
            if equip.info.name == resource_name:
                del self._equipment[equip_id]
                return True

        return False

    def get_material_quantity(self, name: str) -> float:
        """Get quantity of a material."""
        return self._materials[name].quantity if name in self._materials else 0.0

    def get_consumables(self) -> List[Consumable]:
        """Get list of all available consumables."""
        return [c for c in self._consumables.values() if c.quantity > 0]

    def get_consumable(self, name: str) -> Optional[Consumable]:
        """Get a consumable if available (does not remove)."""
        return self._consumables.get(name)

    def get_total_weight(self) -> float:
        """Calculate total weight of all resources."""
        material_weight = sum(m.info.weight * m.quantity for m in self._materials.values())
        tool_weight = sum(t.info.weight for t in self._tools.values())
        equipment_weight = sum(e.info.weight for e in self._equipment.values())
        consumable_weight = sum(c.info.weight * c.quantity for c in self._consumables.values())
        return material_weight + tool_weight + equipment_weight + consumable_weight

    def __iter__(self) -> Iterator[ResourceType]:
        """Iterate over all resources."""
        return chain(
            self._materials.values(),
            self._tools.values(),
            self._equipment.values(),
            self._consumables.values()
        )

    def __len__(self) -> int:
        """Count distinct resource types."""
        return (len(self._materials) + len(self._tools) +
                len(self._equipment) + len(self._consumables))

    def _calculate_added_weight(self, resource: ResourceType, quantity: float) -> float:
        """Calculate weight that would be added."""
        if isinstance(resource, (Material, Consumable)):
            return resource.info.weight * quantity
        else:
            return resource.info.weight

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'type': 'pod',
            'capacity': self.capacity,
            'materials': {name: mat.to_dict() for name, mat in self._materials.items()},
            'tools': {tid: tool.to_dict() for tid, tool in self._tools.items()},
            'equipment': {eid: equip.to_dict() for eid, equip in self._equipment.items()},
            'consumables': {name: cons.to_dict() for name, cons in self._consumables.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PodInventory':
        """Deserialize from dict."""
        inv = cls(capacity=data.get('capacity'))

        for name, mat_data in data.get('materials', {}).items():
            inv._materials[name] = Material.from_dict(mat_data)

        for tid, tool_data in data.get('tools', {}).items():
            inv._tools[tid] = Tool.from_dict(tool_data)

        for eid, equip_data in data.get('equipment', {}).items():
            inv._equipment[eid] = Equipment.from_dict(equip_data)

        for name, cons_data in data.get('consumables', {}).items():
            inv._consumables[name] = Consumable.from_dict(cons_data)

        return inv


class FederationInventory(BaseInventory):
    """
    Federation-level shared resource pool.
    Can hold any resource type - some pods manufacture tools/equipment/consumables.
    Resources available to all pods via distribution/trade mechanisms.
    Similar to Federation being a collection of Pods.
    """

    def __init__(self, capacity: Optional[float] = None):
        super().__init__(capacity)
        self._materials: Dict[str, Material] = {}
        self._tools: Dict[str, Tool] = {}
        self._equipment: Dict[str, Equipment] = {}
        self._consumables: Dict[str, Consumable] = {}

    def add(self, resource: ResourceType, quantity: float = 1.0) -> bool:
        """Add any resource type to federation pool."""
        if isinstance(resource, Material):
            if resource.info.name in self._materials:
                self._materials[resource.info.name].merge(resource)
            else:
                self._materials[resource.info.name] = resource
        elif isinstance(resource, Tool):
            self._tools[resource.info.resource_id] = resource
        elif isinstance(resource, Equipment):
            self._equipment[resource.info.resource_id] = resource
        elif isinstance(resource, Consumable):
            if resource.info.name in self._consumables:
                self._consumables[resource.info.name].quantity += quantity
            else:
                self._consumables[resource.info.name] = resource
        else:
            return False
        return True

    def remove(self, resource_name: str, quantity: float = 1.0) -> bool:
        """Remove resource from federation pool."""
        # Same logic as PodInventory.remove()
        if resource_name in self._materials:
            material = self._materials[resource_name]
            if material.quantity >= quantity:
                material.quantity -= quantity
                if material.quantity <= 0:
                    del self._materials[resource_name]
                return True

        if resource_name in self._consumables:
            consumable = self._consumables[resource_name]
            if consumable.quantity >= quantity:
                consumable.quantity -= quantity
                if consumable.quantity <= 0:
                    del self._consumables[resource_name]
                return True

        for tool_id, tool in list(self._tools.items()):
            if tool.info.name == resource_name:
                del self._tools[tool_id]
                return True

        for equip_id, equip in list(self._equipment.items()):
            if equip.info.name == resource_name:
                del self._equipment[equip_id]
                return True

        return False

    def distribute_to_pod(self, pod_inventory: 'PodInventory', resource_name: str, quantity: float = 1.0) -> bool:
        """
        Distribute resources from federation to a pod.
        Removes from federation, adds to pod.
        """
        resource = None

        if resource_name in self._materials:
            material = self._materials[resource_name]
            if material.quantity >= quantity:
                resource = material.split(quantity)

        elif resource_name in self._consumables:
            consumable = self._consumables[resource_name]
            if consumable.quantity >= quantity:
                consumable.quantity -= quantity
                resource = Consumable(
                    info=consumable.info,
                    quantity=quantity,
                    need_type=consumable.need_type,
                    satisfaction_value=consumable.satisfaction_value
                )

        # For tools/equipment, transfer the instance
        if not resource:
            for tool_id, tool in list(self._tools.items()):
                if tool.info.name == resource_name:
                    resource = self._tools.pop(tool_id)
                    break

        if not resource:
            for equip_id, equip in list(self._equipment.items()):
                if equip.info.name == resource_name:
                    resource = self._equipment.pop(equip_id)
                    break

        if resource:
            return pod_inventory.add(resource, quantity)
        return False

    def get_total_weight(self) -> float:
        """Calculate total weight (usually not enforced for federation)."""
        material_weight = sum(m.info.weight * m.quantity for m in self._materials.values())
        tool_weight = sum(t.info.weight for t in self._tools.values())
        equipment_weight = sum(e.info.weight for e in self._equipment.values())
        consumable_weight = sum(c.info.weight * c.quantity for c in self._consumables.values())
        return material_weight + tool_weight + equipment_weight + consumable_weight

    def __iter__(self) -> Iterator[ResourceType]:
        """Iterate over all resources in federation pool."""
        return chain(
            self._materials.values(),
            self._tools.values(),
            self._equipment.values(),
            self._consumables.values()
        )

    def __len__(self) -> int:
        return (len(self._materials) + len(self._tools) +
                len(self._equipment) + len(self._consumables))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'type': 'federation',
            'capacity': self.capacity,
            'materials': {name: mat.to_dict() for name, mat in self._materials.items()},
            'tools': {tid: tool.to_dict() for tid, tool in self._tools.items()},
            'equipment': {eid: equip.to_dict() for eid, equip in self._equipment.items()},
            'consumables': {name: cons.to_dict() for name, cons in self._consumables.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FederationInventory':
        """Deserialize from dict."""
        inv = cls(capacity=data.get('capacity'))

        for name, mat_data in data.get('materials', {}).items():
            inv._materials[name] = Material.from_dict(mat_data)

        for tid, tool_data in data.get('tools', {}).items():
            inv._tools[tid] = Tool.from_dict(tool_data)

        for eid, equip_data in data.get('equipment', {}).items():
            inv._equipment[eid] = Equipment.from_dict(equip_data)

        for name, cons_data in data.get('consumables', {}).items():
            inv._consumables[name] = Consumable.from_dict(cons_data)

        return inv
