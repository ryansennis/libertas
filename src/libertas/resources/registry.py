"""Unified registry for resources and their production recipes."""

from typing import Dict, List, Optional
from .resource import Resource
from .material import Material
from .tool import Tool
from .equipment import Equipment
from .consumable import Consumable
from .recipes import Recipe


class ResourceRegistry:
    """
    Unified registry for resources and their production recipes.

    Every resource must have at least one recipe that defines how to produce it.
    Production costs are automatically calculated from recipe inputs + labor.

    Economic model:
    - base_labor_rate: Minimum wage (unskilled labor cost per time unit)
    - Skilled labor: base_rate × skill_multiplier (1.0 + skill_level × 0.1)
    - production_cost = sum(input costs) + (duration × base_labor_rate)
    """

    def __init__(self, base_labor_rate: float = 0.5):
        """
        Initialize registry.

        Args:
            base_labor_rate: Minimum wage per time unit (e.g., $0.50/hour for unskilled labor)
        """
        # Economic parameters
        self.base_labor_rate = base_labor_rate

        # Resources by name
        self._materials: Dict[str, Material] = {}
        self._tools: Dict[str, Tool] = {}
        self._equipment: Dict[str, Equipment] = {}
        self._consumables: Dict[str, Consumable] = {}

        # Recipes by output resource name (1 resource : many recipes)
        self._recipes: Dict[str, List[Recipe]] = {}  # resource_name → [recipes]

        # Quick recipe lookup by name
        self._recipes_by_name: Dict[str, Recipe] = {}  # recipe_name → recipe

        # Invention history
        self._invention_history: List[Dict] = []

    # ============================================================================
    # Resource Registration (polymorphic)
    # ============================================================================

    def register_material(self, material: Material) -> None:
        """Register a material type."""
        if material.name not in self._materials:
            self._materials[material.name] = material

    def register_tool(self, tool: Tool) -> None:
        """Register a tool type."""
        if tool.name not in self._tools:
            self._tools[tool.name] = tool

    def register_equipment(self, equipment: Equipment) -> None:
        """Register an equipment type."""
        if equipment.name not in self._equipment:
            self._equipment[equipment.name] = equipment

    def register_consumable(self, consumable: Consumable) -> None:
        """Register a consumable type."""
        if consumable.name not in self._consumables:
            self._consumables[consumable.name] = consumable

    def register(self, resource: Resource) -> None:
        """Register any resource type (polymorphic)."""
        if isinstance(resource, Material):
            self.register_material(resource)
        elif isinstance(resource, Tool):
            self.register_tool(resource)
        elif isinstance(resource, Equipment):
            self.register_equipment(resource)
        elif isinstance(resource, Consumable):
            self.register_consumable(resource)

    # ============================================================================
    # Resource + Recipe Registration (NEW - recommended API)
    # ============================================================================

    def register_resource_with_recipe(
        self,
        resource: Resource,
        recipe: Recipe,
        is_primary: bool = True
    ) -> None:
        """
        Register a resource with its production recipe.
        Automatically calculates production_cost from recipe inputs + labor.

        This is the recommended way to register non-raw resources.

        Args:
            resource: The resource being produced
            recipe: How to produce it
            is_primary: If True (or if resource.production_cost == 0), set production_cost

        Example:
            plank = Material(name="plank", base_value=15.0)
            plank_recipe = Recipe(
                name="process_wood",
                steps=[ProductionStep(
                    name="cut",
                    duration=5,
                    inputs={"wood": 2.0},
                    outputs={"plank": 1.0}
                )]
            )
            registry.register_resource_with_recipe(plank, plank_recipe)
            # plank.production_cost = (2 × wood.base_value) + (5 × base_labor_rate)
        """
        # Calculate production cost: materials + labor
        material_cost = self.calculate_production_cost(recipe.total_inputs)
        labor_cost = recipe.get_base_labor_cost(self.base_labor_rate)
        production_cost = material_cost + labor_cost

        # Update resource's production cost if this is primary recipe
        if is_primary or resource.production_cost == 0:
            resource.production_cost = production_cost

        # Register resource
        self.register(resource)

        # Link recipe to resource
        output_name = recipe.get_primary_output()
        if output_name:
            if output_name not in self._recipes:
                self._recipes[output_name] = []
            self._recipes[output_name].append(recipe)

        # Add to quick lookup
        self._recipes_by_name[recipe.name] = recipe

    def register_resource_with_gathering_recipe(
        self,
        resource: Resource,
        gathering_recipe: Recipe
    ) -> None:
        """
        Register a raw resource with its gathering recipe.
        Raw materials still have production cost (labor to gather/extract).

        This is the recommended way to register raw materials.

        Args:
            resource: Raw material (wood, stone, ore)
            gathering_recipe: How to gather it (should have no material inputs)

        Example:
            wood = Material(name="wood", base_value=5.0)
            wood_gathering = Recipe(
                name="gather_wood",
                steps=[ProductionStep(
                    name="chop_tree",
                    duration=10,
                    inputs={},  # No material inputs
                    outputs={"wood": 1.0}
                )]
            )
            registry.register_resource_with_gathering_recipe(wood, wood_gathering)
            # wood.production_cost = 10 × base_labor_rate = $5.0
        """
        # Calculate cost as labor only (no material inputs)
        labor_cost = gathering_recipe.get_base_labor_cost(self.base_labor_rate)
        resource.production_cost = labor_cost

        # Register
        self.register(resource)

        # Link recipe
        output_name = gathering_recipe.get_primary_output()
        if output_name:
            if output_name not in self._recipes:
                self._recipes[output_name] = []
            self._recipes[output_name].append(gathering_recipe)

        self._recipes_by_name[gathering_recipe.name] = gathering_recipe

    # ============================================================================
    # Resource Lookup
    # ============================================================================

    def get(self, name: str) -> Optional[Resource]:
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

    # ============================================================================
    # Recipe Lookup
    # ============================================================================

    def get_recipes_for(self, resource_name: str) -> List[Recipe]:
        """Get all recipes that produce a resource."""
        return self._recipes.get(resource_name, [])

    def get_recipe_by_name(self, recipe_name: str) -> Optional[Recipe]:
        """Get a recipe by its name (backward compatibility with RecipeRegistry.get())."""
        return self._recipes_by_name.get(recipe_name)

    def get_cheapest_recipe(self, resource_name: str) -> Optional[Recipe]:
        """
        Get the cheapest recipe to produce a resource.

        This determines the market production_cost (most efficient method).
        """
        recipes = self.get_recipes_for(resource_name)
        if not recipes:
            return None

        def recipe_cost(recipe: Recipe) -> float:
            material_cost = self.calculate_production_cost(recipe.total_inputs)
            labor_cost = recipe.get_base_labor_cost(self.base_labor_rate)
            return material_cost + labor_cost

        return min(recipes, key=recipe_cost)

    def list_recipes(self) -> List[str]:
        """List all known recipe names (backward compatibility)."""
        return list(self._recipes_by_name.keys())

    # ============================================================================
    # Cost Calculation
    # ============================================================================

    def calculate_production_cost(self, inputs: Dict[str, float]) -> float:
        """
        Calculate total production cost from recipe inputs.
        Looks up each input resource and sums their buy prices.

        Args:
            inputs: Dict of {resource_name: quantity}

        Returns:
            Total cost to acquire all inputs at current market prices
        """
        total_cost = 0.0
        for resource_name, quantity in inputs.items():
            resource = self.get(resource_name)
            if resource:
                # Use buy price (what it costs to acquire)
                # For materials/consumables, adjust for quantity in resource
                resource_quantity = getattr(resource, 'quantity', 1.0)
                total_cost += resource.get_buy_price() * (quantity / resource_quantity)
        return total_cost

    # ============================================================================
    # Validation
    # ============================================================================

    def validate_economy(self) -> List[str]:
        """
        Validate that all resources have recipes.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        for resource in self.all_resources():
            recipes = self.get_recipes_for(resource.name)
            if not recipes:
                errors.append(f"Resource '{resource.name}' has no recipe (all resources need recipes)")
        return errors

    def all_resources(self) -> List[Resource]:
        """Get all registered resources."""
        from itertools import chain
        return list(chain(
            self._materials.values(),
            self._tools.values(),
            self._equipment.values(),
            self._consumables.values()
        ))

    # ============================================================================
    # Runtime Invention
    # ============================================================================

    def invent_material(self, name: str, inventor_id: str, step: int, **kwargs) -> Material:
        """Invent a new material type at runtime."""
        material = Material(
            name=name,
            invented_by=inventor_id,
            invention_step=step,
            **{k: v for k, v in kwargs.items() if k in ['base_value', 'production_cost', 'weight', 'tags', 'properties', 'quantity']}
        )
        self.register_material(material)
        self._record_invention(name, inventor_id, step, 'material', kwargs)
        return material

    def invent_tool(self, name: str, inventor_id: str, step: int, **kwargs) -> Tool:
        """Invent a new tool type at runtime."""
        tool = Tool(
            name=name,
            invented_by=inventor_id,
            invention_step=step,
            **{k: v for k, v in kwargs.items() if k in ['base_value', 'production_cost', 'weight', 'tags', 'properties', 'durability', 'max_durability', 'required_skill', 'enables_recipes', 'repair_cost']}
        )
        self.register_tool(tool)
        self._record_invention(name, inventor_id, step, 'tool', kwargs)
        return tool

    def invent_recipe(self, name: str, steps: List, inventor_id: str, step: int,
                     description: str = "", category: str = "general") -> Recipe:
        """
        Invent a new recipe at runtime (backward compatibility with RecipeRegistry).

        Note: This doesn't automatically link to a resource. Use register_resource_with_recipe()
        for the full integration.
        """
        recipe = Recipe(
            name=name,
            steps=steps,
            description=description,
            category=category,
            invented_by=inventor_id,
            invention_step=step
        )
        self._recipes_by_name[recipe.name] = recipe

        # Try to link to existing resource
        output_name = recipe.get_primary_output()
        if output_name:
            if output_name not in self._recipes:
                self._recipes[output_name] = []
            self._recipes[output_name].append(recipe)

        self._invention_history.append({
            'name': name,
            'inventor': inventor_id,
            'step': step,
            'type': 'recipe',
            'steps': [s.name for s in steps],
            'total_duration': recipe.total_duration
        })
        return recipe

    def _record_invention(self, name: str, inventor_id: str, step: int, resource_type: str, properties: dict):
        """Record an invention in history."""
        self._invention_history.append({
            'name': name,
            'inventor': inventor_id,
            'step': step,
            'type': resource_type,
            'properties': properties
        })

    # ============================================================================
    # Utility
    # ============================================================================

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
