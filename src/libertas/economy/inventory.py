from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from .resource import Resource

@dataclass
class Inventory:
    """Inventory of resources for a worker or pod."""
    
    quantities: Dict[str, float] = field(default_factory=dict)
    
    instances: Dict[str, List[Resource]] = field(default_factory=dict)
    
    capacity: Optional[float] = None
    
    def add(self, resource: Resource, quantity: float = 1.0) -> bool:
        """Add a resource to inventory."""
        if quantity < 0:
            return False
        
        # Check capacity
        if self.capacity is not None:
            current_total = sum(self.quantities.values())
            if current_total + quantity > self.capacity:
                return False
        
        # For bulk resources (non-tools), just track quantity
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
        
        return True
    
    def remove(self, resource_name: str, quantity: float = 1.0) -> bool:
        """Remove a resource from inventory."""
        if resource_name in self.quantities:
            if self.quantities[resource_name] >= quantity:
                self.quantities[resource_name] -= quantity
                if self.quantities[resource_name] <= 0:
                    del self.quantities[resource_name]
                return True
        return False
    
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
        """Get quantity of a bulk resource."""
        return self.quantities.get(resource_name, 0)
    
    def has_enough(self, requirements: Dict[str, float]) -> bool:
        """Check if inventory meets requirements."""
        for resource_name, needed in requirements.items():
            if self.get_quantity(resource_name) < needed:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize inventory to dictionary."""
        return {
            'quantities': self.quantities.copy(),
            'instances': {
                name: [tool.to_dict() for tool in tools]
                for name, tools in self.instances.items()
            },
            'capacity': self.capacity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Inventory':
        """Create inventory from dictionary."""
        inv = cls(capacity=data.get('capacity'))
        
        # Restore quantities
        inv.quantities = data.get('quantities', {}).copy()
        
        # Restore tool instances
        for name, tool_data in data.get('instances', {}).items():
            inv.instances[name] = [Resource.from_dict(t) for t in tool_data]
        
        return inv