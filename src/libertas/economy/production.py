"""Production recipes and step definitions for the cooperative economy."""

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
    They can be invented at runtime by workers.
    """
    
    name: str
    steps: List[ProductionStep]
    recipe_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Recipe-level properties
    description: str = ""
    category: str = "general"  # e.g., "crafting", "processing", "assembly"
    
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


@dataclass
class ProductionJob:
    """An active production job tracking progress through recipe steps."""
    
    recipe: Recipe
    job_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Job metadata
    started_by: str = ""  # worker_id or pod_id
    started_at_step: int = 0  # simulation step when started
    batch_size: int = 1
    
    # Progress tracking
    current_step_index: int = 0
    completed_steps: List[int] = field(default_factory=list)
    assigned_worker_id: Optional[str] = None
    step_start_step: Optional[int] = None
    
    # Resource accounting
    inputs_consumed: Dict[str, float] = field(default_factory=dict)
    outputs_produced: Dict[str, float] = field(default_factory=dict)
    
    # Status
    is_active: bool = True
    is_paused: bool = False
    error_message: str = ""
    
    def __post_init__(self):
        """Initialize tracking for inputs."""
        # Pre-calculate total inputs needed for batch
        for resource, qty in self.recipe.total_inputs.items():
            self.inputs_consumed[resource] = 0
        for resource, qty in self.recipe.total_outputs.items():
            self.outputs_produced[resource] = 0
    
    def assign_worker(self, worker_id: str, current_step: int) -> None:
        """Assign a worker to the current step."""
        self.assigned_worker_id = worker_id
        self.current_step_index = current_step
        self.step_start_step = None  # Will be set when work actually begins
    
    def start_current_step(self, simulation_step: int) -> None:
        """Begin work on the current step."""
        self.step_start_step = simulation_step
    
    def get_step_remaining_duration(self, simulation_step: int) -> int:
        """Get remaining duration for current step."""
        step = self.current_step
        if step is None:
            return 0
        
        if self.step_start_step is None:
            return step.duration
        
        elapsed = simulation_step - self.step_start_step
        return max(0, step.duration - elapsed)
    
    @property
    def current_step(self) -> Optional[ProductionStep]:
        """Get the current step."""
        return self.recipe.get_step(self.current_step_index)
    
    def consume_step_inputs(self) -> bool:
        """Mark step inputs as consumed. Returns True if successful."""
        step = self.current_step
        if not step:
            return False
        
        for resource, qty in step.inputs.items():
            self.inputs_consumed[resource] = self.inputs_consumed.get(resource, 0) + (qty * self.batch_size)
        return True
    
    def complete_step(self, simulation_step: int) -> Dict[str, float]:
        """
        Mark current step as complete.
        
        Returns:
            Outputs produced by this step
        """
        if not self.current_step:
            return {}
        
        # Consume inputs if not already consumed
        self.consume_step_inputs()
        
        # Add outputs
        for resource, qty in self.current_step.outputs.items():
            self.outputs_produced[resource] = self.outputs_produced.get(resource, 0) + (qty * self.batch_size)
        
        # Move to next step

        outputs = self.get_step_outputs()
        
        self.current_step_index += 1

        # Record completion
        self.completed_steps.append(self.current_step_index)
        self.assigned_worker_id = None
        self.step_start_step = None
        
        # Check if job is complete
        if self.current_step_index >= len(self.recipe.steps):
            self.is_active = False
        
        return outputs
    
    def get_step_outputs(self) -> Dict[str, float]:
        """Get outputs for the current step."""
        if not self.current_step:
            return {}
        return {
            resource: quantity * self.batch_size
            for resource, quantity in self.current_step.outputs.items()
        }
    
    def get_total_outputs(self) -> Dict[str, float]:
        """Get all outputs produced so far."""
        return self.outputs_produced.copy()
    
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return len(self.completed_steps) >= len(self.recipe.steps)
    
    def get_next_step(self) -> Optional[ProductionStep]:
        """Get the next uncompleted step."""
        if self.current_step_index < len(self.recipe.steps):
            return self.recipe.get_step(self.current_step_index)
        return None
    
    def get_progress(self) -> float:
        """Get completion progress (0-1)."""
        if not self.recipe.steps:
            return 1.0
        return len(self.completed_steps) / len(self.recipe.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize job to dictionary."""
        return {
            'recipe': self.recipe.to_dict(),
            'job_id': self.job_id,
            'started_by': self.started_by,
            'started_at_step': self.started_at_step,
            'batch_size': self.batch_size,
            'current_step_index': self.current_step_index,
            'completed_steps': self.completed_steps,
            'assigned_worker_id': self.assigned_worker_id,
            'step_start_step': self.step_start_step,
            'inputs_consumed': self.inputs_consumed.copy(),
            'outputs_produced': self.outputs_produced.copy(),
            'is_active': self.is_active,
            'is_paused': self.is_paused,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionJob':
        """Create job from dictionary."""
        
        job = cls(
            recipe=Recipe.from_dict(data['recipe']),
            job_id=data['job_id'],
            started_by=data['started_by'],
            started_at_step=data['started_at_step'],
            batch_size=data.get('batch_size', 1)
        )
        job.current_step_index = data['current_step_index']
        job.completed_steps = data['completed_steps']
        job.assigned_worker_id = data.get('assigned_worker_id')
        job.step_start_step = data.get('step_start_step')
        job.inputs_consumed = data.get('inputs_consumed', {})
        job.outputs_produced = data.get('outputs_produced', {})
        job.is_active = data.get('is_active', True)
        job.is_paused = data.get('is_paused', False)
        job.error_message = data.get('error_message', '')
        return job


class RecipeRegistry:
    """Registry of all known production recipes."""
    
    def __init__(self):
        self._recipes: Dict[str, Recipe] = {}
        self._invention_history: List[Dict] = []
    
    def register(self, recipe: Recipe) -> None:
        """Register a recipe."""
        self._recipes[recipe.name] = recipe
    
    def get(self, name: str) -> Optional[Recipe]:
        """Get a recipe by name."""
        return self._recipes.get(name)
    
    def invent(self, name: str, steps: List[ProductionStep],
               inventor_id: str, step: int,
               description: str = "",
               category: str = "general") -> Recipe:
        """Invent a new recipe at runtime."""
        recipe = Recipe(
            name=name,
            steps=steps,
            description=description,
            category=category,
            invented_by=inventor_id,
            invention_step=step
        )
        self.register(recipe)
        self._invention_history.append({
            'name': name,
            'inventor': inventor_id,
            'step': step,
            'steps': [s.name for s in steps],
            'total_duration': recipe.total_duration
        })
        return recipe
    
    def list_recipes(self) -> List[str]:
        """List all known recipe names."""
        return list(self._recipes.keys())
    
    def get_by_category(self, category: str) -> List[Recipe]:
        """Get all recipes in a category."""
        return [r for r in self._recipes.values() if r.category == category]
    
    @property
    def invention_history(self) -> List[Dict]:
        """Get history of all recipe inventions."""
        return self._invention_history.copy()


# Predefined starting recipes
def get_starting_recipes() -> List[Recipe]:
    """Get the initial set of recipes available at simulation start."""
    
    return [
        # Basic processing
        Recipe(
            name="smelt_metal",
            steps=[
                ProductionStep(
                    name="smelt",
                    step_type=StepType.PROCESSING,
                    duration=10,
                    inputs={"raw_metal": 2},
                    outputs={"metal_ingot": 1}
                )
            ],
            description="Smelt raw metal into usable ingots",
            category="processing"
        ),
        
        # Simple crafting requiring tool
        Recipe(
            name="craft_component",
            steps=[
                ProductionStep(
                    name="assemble_component",
                    step_type=StepType.ASSEMBLY,
                    duration=15,
                    inputs={"metal_ingot": 1, "wood": 1},
                    outputs={"component": 1},
                    required_tool="hammer",
                    required_skill="basic_crafting",
                    required_skill_level=1.0
                )
            ],
            description="Craft a basic component from materials",
            category="crafting"
        ),
        
        # Multi-step production
        Recipe(
            name="craft_widget",
            steps=[
                ProductionStep(
                    name="prepare_parts",
                    step_type=StepType.PROCESSING,
                    duration=8,
                    inputs={"component": 2},
                    outputs={"prepared_parts": 2},
                    required_tool="hammer"
                ),
                ProductionStep(
                    name="assemble_widget",
                    step_type=StepType.ASSEMBLY,
                    duration=12,
                    inputs={"prepared_parts": 2},
                    outputs={"widget": 1},
                    required_tool="hammer",
                    required_skill="assembly",
                    required_skill_level=2.0
                ),
                ProductionStep(
                    name="quality_check",
                    step_type=StepType.QUALITY_CHECK,
                    duration=5,
                    inputs={"widget": 1},
                    outputs={"widget": 1},
                    quality_threshold=0.8,
                    requires_approval=True
                )
            ],
            description="Craft a finished widget from components",
            category="manufacturing"
        ),
        
        # Tool making
        Recipe(
            name="craft_hammer",
            steps=[
                ProductionStep(
                    name="forge_head",
                    step_type=StepType.PROCESSING,
                    duration=20,
                    inputs={"metal_ingot": 3},
                    outputs={"hammer_head": 1},
                    required_skill="forging",
                    required_skill_level=2.0
                ),
                ProductionStep(
                    name="attach_handle",
                    step_type=StepType.ASSEMBLY,
                    duration=10,
                    inputs={"hammer_head": 1, "wood": 2},
                    outputs={"hammer": 1},
                    required_tool="hammer"  # Need a hammer to make a hammer!
                )
            ],
            description="Craft a hammer tool",
            category="tools"
        )
    ]