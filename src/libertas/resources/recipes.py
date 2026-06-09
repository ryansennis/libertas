"""Production recipes and step definitions.

Recipes define how to transform inputs into outputs through sequential steps.
Each resource must have at least one recipe (including raw materials with gathering recipes).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from uuid import uuid4


class StepType(Enum):
    """Types of production steps."""
    PROCESSING = "processing"      # Transform raw materials
    ASSEMBLY = "assembly"          # Combine components
    QUALITY_CHECK = "quality"      # Inspection/validation
    PACKAGING = "packaging"        # Prepare for shipment
    MAINTENANCE = "maintenance"    # Tool/equipment upkeep
    RESEARCH = "research"          # Discovery/invention
    REPAIR = "repair"              # Fix broken tools
    GATHERING = "gathering"        # Collect raw materials (chopping wood, mining)


@dataclass
class ProductionStep:
    """
    A discrete step in a production recipe.

    Each step can be performed by a single worker and takes time.
    Steps can require tools, skills, and specific inputs.
    """

    name: str
    step_type: StepType
    duration: int  # Time steps required to complete

    # Inputs consumed by this step
    inputs: Dict[str, float] = field(default_factory=dict)  # resource_name -> quantity

    # Outputs produced by this step
    outputs: Dict[str, float] = field(default_factory=dict)  # resource_name -> quantity

    # Requirements
    required_tool: Optional[str] = None  # Tool name needed (must be equipped)
    required_skill: Optional[str] = None  # Skill name needed
    required_skill_level: float = 1.0    # Minimum proficiency (0-10)

    # Conditions
    requires_approval: bool = False  # Needs supervisor/council approval
    quality_threshold: Optional[float] = None  # Min quality to pass (0-1)

    # Parallel execution
    can_parallel: bool = False  # Can multiple workers do this step simultaneously?
    max_parallel: int = 1       # Maximum workers if parallel allowed

    def can_perform(self, worker_skills: Dict[str, float],
                    has_tool: bool = False) -> tuple[bool, str]:
        """
        Check if a worker can perform this step.

        Returns:
            (can_perform, reason) tuple
        """
        if self.required_skill:
            skill_level = worker_skills.get(self.required_skill, 0)
            if skill_level < self.required_skill_level:
                return False, f"Need {self.required_skill} level {self.required_skill_level}, have {skill_level}"

        if self.required_tool and not has_tool:
            return False, f"Need tool: {self.required_tool}"

        return True, "OK"

    def get_skill_multiplier(self, worker_skills: Dict[str, float]) -> float:
        """
        Get labor cost multiplier based on required skill level.
        Higher skill requirements = higher labor costs.

        Args:
            worker_skills: Worker's skill levels

        Returns:
            Multiplier (1.0 = base rate, 1.5 = 50% premium for skilled work)
        """
        if not self.required_skill:
            return 1.0  # Unskilled labor

        skill_level = worker_skills.get(self.required_skill, 0)
        # Skill multiplier: 1.0 + (skill_level * 0.1)
        # Level 0 = 1.0x, Level 5 = 1.5x, Level 10 = 2.0x
        return 1.0 + (skill_level * 0.1)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'step_type': self.step_type.value,
            'duration': self.duration,
            'inputs': self.inputs.copy(),
            'outputs': self.outputs.copy(),
            'required_tool': self.required_tool,
            'required_skill': self.required_skill,
            'required_skill_level': self.required_skill_level,
            'requires_approval': self.requires_approval,
            'quality_threshold': self.quality_threshold,
            'can_parallel': self.can_parallel,
            'max_parallel': self.max_parallel
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionStep':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            step_type=StepType(data['step_type']),
            duration=data['duration'],
            inputs=data.get('inputs', {}),
            outputs=data.get('outputs', {}),
            required_tool=data.get('required_tool'),
            required_skill=data.get('required_skill'),
            required_skill_level=data.get('required_skill_level', 1.0),
            requires_approval=data.get('requires_approval', False),
            quality_threshold=data.get('quality_threshold'),
            can_parallel=data.get('can_parallel', False),
            max_parallel=data.get('max_parallel', 1)
        )


@dataclass
class Recipe:
    """
    A production recipe composed of sequential steps.

    Recipes define how to transform inputs into outputs.
    Every resource must have at least one recipe (including raw materials).
    They can be invented at runtime by workers.
    """

    name: str
    steps: List[ProductionStep]
    recipe_id: str = field(default_factory=lambda: str(uuid4()))

    # Recipe-level properties
    description: str = ""
    category: str = "general"  # e.g., "crafting", "processing", "assembly", "gathering"

    # Invention tracking
    invented_by: Optional[str] = None  # worker_id or pod_id
    invention_step: Optional[int] = None

    # Recipe stats (computed)
    total_duration: int = 0
    total_inputs: Dict[str, float] = field(default_factory=dict)
    total_outputs: Dict[str, float] = field(default_factory=dict)
    requires_tools: Set[str] = field(default_factory=set)
    requires_skills: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Compute recipe statistics."""
        self.total_duration = sum(step.duration for step in self.steps)

        # Aggregate inputs and outputs
        for step in self.steps:
            for resource, qty in step.inputs.items():
                self.total_inputs[resource] = self.total_inputs.get(resource, 0) + qty
            for resource, qty in step.outputs.items():
                self.total_outputs[resource] = self.total_outputs.get(resource, 0) + qty

            if step.required_tool:
                self.requires_tools.add(step.required_tool)
            if step.required_skill:
                self.requires_skills.add(step.required_skill)

    def get_primary_output(self) -> Optional[str]:
        """Get the primary output resource name (highest quantity)."""
        if not self.total_outputs:
            return None
        return max(self.total_outputs.items(), key=lambda x: x[1])[0]

    def get_base_labor_cost(self, base_labor_rate: float) -> float:
        """
        Calculate base labor cost (unskilled labor).

        Args:
            base_labor_rate: Minimum wage per time unit

        Returns:
            Total labor cost at base rate
        """
        return self.total_duration * base_labor_rate

    def get_skilled_labor_cost(self, base_labor_rate: float, worker_skills: Dict[str, float]) -> float:
        """
        Calculate actual labor cost based on worker skill levels.

        Args:
            base_labor_rate: Minimum wage per time unit
            worker_skills: Worker's skill levels (skill_name -> level 0-10)

        Returns:
            Total labor cost with skill multipliers applied
        """
        total_cost = 0.0
        for step in self.steps:
            multiplier = step.get_skill_multiplier(worker_skills)
            step_cost = step.duration * base_labor_rate * multiplier
            total_cost += step_cost
        return total_cost

    def get_step(self, index: int) -> Optional[ProductionStep]:
        """Get step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_step_inputs(self, step_index: int) -> Dict[str, float]:
        """Get inputs required for a specific step."""
        step = self.get_step(step_index)
        return step.inputs if step else {}

    def get_step_outputs(self, step_index: int) -> Dict[str, float]:
        """Get outputs produced by a specific step."""
        step = self.get_step(step_index)
        return step.outputs if step else {}

    def get_required_tools_for_step(self, step_index: int) -> Optional[str]:
        """Get required tool for a specific step."""
        step = self.get_step(step_index)
        return step.required_tool if step else None

    def can_start(self, available_inputs: Dict[str, float]) -> tuple[bool, str]:
        """
        Check if recipe can be started with available inputs.

        Returns:
            (can_start, reason) tuple
        """
        for resource, needed in self.total_inputs.items():
            if available_inputs.get(resource, 0) < needed:
                return False, f"Need {needed} {resource}, have {available_inputs.get(resource, 0)}"
        return True, "OK"

    def requires_parallel_coordination(self) -> bool:
        """Check if recipe needs parallel execution coordination."""
        parallel_steps = [s for s in self.steps if s.can_parallel and s.max_parallel > 1]
        return len(parallel_steps) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'recipe_id': self.recipe_id,
            'steps': [step.to_dict() for step in self.steps],
            'description': self.description,
            'category': self.category,
            'invented_by': self.invented_by,
            'invention_step': self.invention_step,
            'total_duration': self.total_duration,
            'total_inputs': self.total_inputs,
            'total_outputs': self.total_outputs,
            'requires_tools': list(self.requires_tools),
            'requires_skills': list(self.requires_skills)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recipe':
        """Create from dictionary."""
        steps = [ProductionStep.from_dict(s) for s in data['steps']]
        recipe = cls(
            name=data['name'],
            steps=steps,
            recipe_id=data.get('recipe_id', str(uuid4())),
            description=data.get('description', ''),
            category=data.get('category', 'general'),
            invented_by=data.get('invented_by'),
            invention_step=data.get('invention_step')
        )
        return recipe
