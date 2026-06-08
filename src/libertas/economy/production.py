"""Production job tracking for active work.

Production definitions (Recipe, ProductionStep) have moved to resources module.
This module focuses on tracking active production work through ProductionJob.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from uuid import uuid4

# Import recipe definitions from resources
from ..resources import Recipe, ProductionStep


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