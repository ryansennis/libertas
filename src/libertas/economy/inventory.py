from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from .resource import Resource

# Import new resource types for parallel tracking
try:
    from ..resources import Material, Tool, Equipment, Consumable
    NEW_SYSTEM_AVAILABLE = True
except ImportError:
    NEW_SYSTEM_AVAILABLE = False

@dataclass
class Inventory:
    """Inventory of resources for a worker or pod.

    Currently maintains both old (quantities/instances) and new (materials/tools/equipment/consumables)
    tracking systems during migration period.
    """

    # OLD SYSTEM (will be removed in Phase 6)
    quantities: Dict[str, float] = field(default_factory=dict)
    instances: Dict[str, List[Resource]] = field(default_factory=dict)

    # NEW SYSTEM (parallel tracking)
    materials: Dict[str, 'Material'] = field(default_factory=dict) if NEW_SYSTEM_AVAILABLE else field(default_factory=dict)
    tools: Dict[str, 'Tool'] = field(default_factory=dict) if NEW_SYSTEM_AVAILABLE else field(default_factory=dict)
    equipment: Dict[str, 'Equipment'] = field(default_factory=dict) if NEW_SYSTEM_AVAILABLE else field(default_factory=dict)
    consumables: Dict[str, 'Consumable'] = field(default_factory=dict) if NEW_SYSTEM_AVAILABLE else field(default_factory=dict)

    capacity: Optional[float] = None
    
    def add(self, resource: Resource, quantity: float = 1.0) -> bool:
        """Add a resource to inventory.

        During migration: writes to BOTH old and new systems.
        """
        if quantity < 0:
            return False

        # Check capacity
        if self.capacity is not None:
            current_total = sum(self.quantities.values())
            if current_total + quantity > self.capacity:
                return False

        # OLD SYSTEM: For bulk resources (non-tools), just track quantity
        if not resource.is_tool:
            self.quantities[resource.name] = self.quantities.get(resource.name, 0) + quantity
        else:
            # For tools, track individual instances with their durability
            if resource.name not in self.instances:
                self.instances[resource.name] = []
            for _ in range(int(quantity)):
                # Create a new instance with same properties
                new_tool = Resource(
                    name=resource.name,
                    base_value=resource.base_value,
                    is_tool=True,
                    durability=resource.durability,
                    required_skill=resource.required_skill,
                    enables_recipes=resource.enables_recipes
                )
                self.instances[resource.name].append(new_tool)

        # NEW SYSTEM: Also write to new-style tracking (if available)
        if NEW_SYSTEM_AVAILABLE:
            self._add_to_new_system(resource, quantity)

        return True

    def _add_to_new_system(self, resource: Resource, quantity: float):
        """Helper to add to new-style materials/tools/equipment/consumables dicts."""
        # This is a temporary bridge - in Phase 3 we'll convert resources to new types
        # For now, just mirror the old system's storage
        if not resource.is_tool:
            # Store in materials dict (will be properly converted in Phase 3)
            if resource.name in self.materials:
                self.materials[resource.name].quantity += quantity
            else:
                # Create a Material from old Resource
                from ..resources import ResourceInfo, Material
                info = ResourceInfo(
                    name=resource.name,
                    base_value=resource.base_value,
                    weight=resource.weight,
                    tags=resource.tags,
                    properties=resource.properties,
                    invented_by=resource.invented_by,
                    invention_step=resource.invention_step
                )
                self.materials[resource.name] = Material(info=info, quantity=quantity)
        else:
            # Store in tools dict
            from ..resources import ResourceInfo, Tool
            for _ in range(int(quantity)):
                info = ResourceInfo(
                    name=resource.name,
                    base_value=resource.base_value,
                    weight=resource.weight,
                    tags=resource.tags,
                    properties=resource.properties,
                    invented_by=resource.invented_by,
                    invention_step=resource.invention_step
                )
                tool = Tool(
                    info=info,
                    durability=resource.durability if resource.durability else 100,
                    required_skill=resource.required_skill,
                    enables_recipes=resource.enables_recipes if resource.enables_recipes else []
                )
                self.tools[tool.info.resource_id] = tool
    
    def remove(self, resource_name: str, quantity: float = 1.0) -> bool:
        """Remove a resource from inventory.

        During migration: removes from BOTH old and new systems.
        """
        # OLD SYSTEM
        old_removed = False
        if resource_name in self.quantities:
            if self.quantities[resource_name] >= quantity:
                self.quantities[resource_name] -= quantity
                if self.quantities[resource_name] <= 0:
                    del self.quantities[resource_name]
                old_removed = True

        # NEW SYSTEM: Also remove from new tracking
        new_removed = False
        if NEW_SYSTEM_AVAILABLE:
            if resource_name in self.materials:
                material = self.materials[resource_name]
                if material.quantity >= quantity:
                    material.quantity -= quantity
                    if material.quantity <= 0:
                        del self.materials[resource_name]
                    new_removed = True
            elif resource_name in self.consumables:
                consumable = self.consumables[resource_name]
                if consumable.quantity >= quantity:
                    consumable.quantity -= quantity
                    if consumable.quantity <= 0:
                        del self.consumables[resource_name]
                    new_removed = True

        return old_removed or new_removed
    
    def get_tool(self, tool_name: str) -> Optional[Resource]:
        """Get an available tool instance."""
        if tool_name in self.instances and self.instances[tool_name]:
            tool = self.instances[tool_name][0]
            self.instances[tool_name].pop(0)
            if not tool.is_broken():
                return tool
            else:
                return self.get_tool(tool_name)
        return None
    
    def return_tool(self, tool: Resource) -> None:
        """Return a tool to inventory after use."""
        if tool.name not in self.instances:
            self.instances[tool.name] = []
        self.instances[tool.name].append(tool)
    
    def use_tool(self, tool_name: str) -> bool:
        """Use a tool, degrading it. Returns True if successfully used."""
        tool = self.get_tool(tool_name)
        if tool:
            success = tool.use_tool()
            if not tool.is_broken():
                self.return_tool(tool)
            return success
        return False
    
    def get_quantity(self, resource_name: str) -> float:
        """Get quantity of a bulk resource.

        During migration: reads from NEW system if available, falls back to old.
        """
        # Prefer new system
        if NEW_SYSTEM_AVAILABLE and resource_name in self.materials:
            return self.materials[resource_name].quantity

        # Fallback to old system
        return self.quantities.get(resource_name, 0)
    
    def has_enough(self, requirements: Dict[str, float]) -> bool:
        """Check if inventory meets requirements."""
        for resource_name, needed in requirements.items():
            if self.get_quantity(resource_name) < needed:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize inventory to dictionary.

        During migration: saves BOTH old and new systems.
        """
        result = {
            'quantities': self.quantities.copy(),
            'instances': {
                name: [tool.to_dict() for tool in tools]
                for name, tools in self.instances.items()
            },
            'capacity': self.capacity
        }

        # Also save new system if available
        if NEW_SYSTEM_AVAILABLE:
            result['materials'] = {name: mat.to_dict() for name, mat in self.materials.items()}
            result['tools'] = {tid: tool.to_dict() for tid, tool in self.tools.items()}
            result['equipment'] = {eid: equip.to_dict() for eid, equip in self.equipment.items()}
            result['consumables'] = {name: cons.to_dict() for name, cons in self.consumables.items()}

        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Inventory':
        """Create inventory from dictionary.

        During migration: supports loading from BOTH old and new formats.
        """
        inv = cls(capacity=data.get('capacity'))

        # Restore OLD system
        inv.quantities = data.get('quantities', {}).copy()

        # Restore tool instances
        for name, tool_data in data.get('instances', {}).items():
            inv.instances[name] = [Resource.from_dict(t) for t in tool_data]

        # Restore NEW system if present
        if NEW_SYSTEM_AVAILABLE:
            from ..resources import Material, Tool, Equipment, Consumable

            for name, mat_data in data.get('materials', {}).items():
                inv.materials[name] = Material.from_dict(mat_data)

            for tid, tool_data in data.get('tools', {}).items():
                inv.tools[tid] = Tool.from_dict(tool_data)

            for eid, equip_data in data.get('equipment', {}).items():
                inv.equipment[eid] = Equipment.from_dict(equip_data)

            for name, cons_data in data.get('consumables', {}).items():
                inv.consumables[name] = Consumable.from_dict(cons_data)

        return inv