# src/libertas/organization/worker.py
from dataclasses import dataclass
from mesa import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.reasoning import Reasoning
from mesa.discrete_space.cell import Cell
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import json
import importlib

if TYPE_CHECKING:
    from .pod import Pod
    from .federation import Federation
    from ..economy import ProductionJob, Resource

from ..cognitive import PersonalityTraits, Background, MoodState

@dataclass
class WorkerConfig:
    name: str
    reasoning: type[Reasoning]
    llm_model: str
    system_prompt: Optional[str] = None
    vision: Optional[float] = None
    internal_state: Optional[Union[List[str], str]] = None
    step_prompt: Optional[str] = None
    api_base: Optional[str] = None
    initial_skills: Optional[Dict[str, float]] = None
    initial_tools: Optional[List[str]] = None
    initial_currency: float = 100.0
    personality: Optional[PersonalityTraits] = None
    background: Optional[Background] = None  

    @property
    def llm_inputs(self):
        return (
            self.reasoning,
            self.llm_model,
            self.system_prompt,
            self.vision,
            self.internal_state,
            self.step_prompt,
            self.api_base
        )
    
    def to_json(self, filepath: Optional[str] = None) -> Union[str, None]:
        """Convert WorkerConfig to JSON string or save to file."""
        config_dict = {
            'name': self.name,
            'reasoning': f"{self.reasoning.__module__}.{self.reasoning.__name__}",
            'llm_model': self.llm_model,
            'system_prompt': self.system_prompt,
            'vision': self.vision,
            'internal_state': self.internal_state,
            'step_prompt': self.step_prompt,
            'api_base': self.api_base,
            'initial_skills': self.initial_skills,
            'initial_tools': self.initial_tools,
            'initial_currency': self.initial_currency
        }
        
        json_str = json.dumps(config_dict, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        return json_str
    
    @staticmethod
    def from_json(data: Union[str, Dict, object], filepath: Optional[str] = None) -> 'WorkerConfig':
        """Create WorkerConfig from JSON string, dictionary, or file."""
        if filepath:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(data, str):
            config_dict = json.loads(data)
        elif isinstance(data, dict):
            config_dict = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Convert reasoning string back to class reference
        if 'reasoning' in config_dict and isinstance(config_dict['reasoning'], str):
            reasoning_path = config_dict['reasoning'].split('.')
            module_path = '.'.join(reasoning_path[:-1])
            class_name = reasoning_path[-1]
            
            module = importlib.import_module(module_path)
            config_dict['reasoning'] = getattr(module, class_name)
        
        return WorkerConfig(**config_dict)
    
    @staticmethod
    def from_json_file(filepath: str) -> 'WorkerConfig':
        """Convenience method to load from a JSON file."""
        return WorkerConfig.from_json(None, filepath=filepath)


class Worker(LLMAgent):
    """Worker agent that combines LLM capabilities with production work and market trading."""
    
    def __init__(
        self,
        federation: Model,
        worker_config: WorkerConfig,
        coordinate: tuple,
        pod: Optional['Pod'] = None,
    ):
        self._name = worker_config.name
        self._pod = pod
        self._federation = federation

        super().__init__(
            model=federation,
            reasoning=worker_config.reasoning,
            llm_model=worker_config.llm_model,
            system_prompt=worker_config.system_prompt,
            vision=worker_config.vision,
            internal_state=worker_config.internal_state if worker_config.internal_state else [],
            step_prompt=worker_config.step_prompt,
            api_base=worker_config.api_base
        )
        
        # Create a Cell instance for spatial functionality (composition)
        self._cell = Cell(coordinate=coordinate)
        self._random_storage = None
        
        # Production attributes
        self.current_job: Optional['ProductionJob'] = None
        self.assigned_step_index: Optional[int] = None
        self.step_start_step: Optional[int] = None
        
        # Skills (skill_name -> proficiency level 0-10)
        self.skills: Dict[str, float] = worker_config.initial_skills or {}
        
        # Personal tool inventory
        self.tools: Dict[str, List['Resource']] = {}  # tool_name -> list of tool instances
        self.equipped_tool: Optional[str] = None
        
        # Personal currency (for market transactions)
        self.currency: float = worker_config.initial_currency
        
        # Work history
        self.completed_jobs: List[str] = []  # job_ids
        self.total_outputs: Dict[str, float] = {}  # resource_name -> quantity
        self.transaction_history: List[Dict] = []  # Market transaction log
        
        # Initialize tools from config
        if worker_config.initial_tools:
            federation = self._federation
            for tool_name in worker_config.initial_tools:
                tool_template = federation.get_resource(tool_name)
                if tool_template and tool_template.is_tool:
                    self._add_tool(tool_template)
        
        # Cognitive attributes
        self.personality = worker_config.personality or PersonalityTraits()
        self.background = worker_config.background or Background()
        self.mood = MoodState()

        # Initialize economic tools
        from ..tools.economic_tools import EconomicTools
        self.economic_tools = EconomicTools(self)

        # Initialize governance tools
        from ..tools.governance_tools import GovernanceTools
        self.governance_tools = GovernanceTools(self)

        # Register tools with LLMAgent's tool manager
        self._register_economic_tools()
        self._register_governance_tools()
    
    def _register_economic_tools(self):
        """Register economic tools with the LLM agent's tool manager."""
        from ..tools.economic_tools import get_economic_tool_definitions

        # Get tool definitions
        tool_defs = get_economic_tool_definitions()

        # Register each tool with the tool manager
        for tool_def in tool_defs:
            tool_name = tool_def['function']['name']
            tool_func = getattr(self.economic_tools, tool_name, None)
            if tool_func:
                self.tool_manager.register(tool_func)

    def _register_governance_tools(self):
        """Register governance tools with the LLM agent's tool manager."""
        from ..tools.governance_tools import get_governance_tool_definitions

        # Get tool definitions
        tool_defs = get_governance_tool_definitions()

        # Register each tool with the tool manager
        for tool_def in tool_defs:
            tool_name = tool_def['function']['name']
            tool_func = getattr(self.governance_tools, tool_name, None)
            if tool_func:
                self.tool_manager.register(tool_func)
    
    def __hash__(self) -> int:
        """Make Worker hashable based on its name."""
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        """Define equality for Worker objects."""
        if not isinstance(other, Worker):
            return False
        return self.name == other.name
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the worker's name."""
        self._name = value
    
    @property
    def pod(self):
        return self._pod
    
    @pod.setter
    def pod(self, pod: 'Pod'):
        self._pod = pod

    @property
    def federation(self):
        return self._federation

    # Delegate Cell properties
    @property
    def coordinate(self):
        return self._cell.coordinate
    
    @coordinate.setter
    def coordinate(self, value):
        self._cell.coordinate = value
    
    @property
    def position(self):
        return self._cell.position
    
    @position.setter
    def position(self, value):
        self._cell.position = value
        
    @property
    def federation(self):
        return self._federation
    
    @federation.setter
    def federation(self, value: 'Federation'):
        self._federation = value
    
    def _add_tool(self, tool_template: 'Resource') -> None:
        """Add a tool to worker's inventory."""
        from ..economy.resource import Resource
        
        # Create a new instance of the tool
        tool = Resource(
            name=tool_template.name,
            base_value=tool_template.base_value,
            is_tool=True,
            durability=tool_template.durability,
            required_skill=tool_template.required_skill,
            enables_recipes=tool_template.enables_recipes.copy() if tool_template.enables_recipes else []
        )
        
        if tool.name not in self.tools:
            self.tools[tool.name] = []
        self.tools[tool.name].append(tool)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if worker has a specific tool."""
        return tool_name in self.tools and len(self.tools[tool_name]) > 0
    
    def equip_tool(self, tool_name: str) -> bool:
        """Equip a tool from inventory."""
        if self.has_tool(tool_name):
            self.equipped_tool = tool_name
            return True
        return False
    
    def unequip_tool(self) -> None:
        """Unequip current tool."""
        self.equipped_tool = None
    
    def use_equipped_tool(self) -> bool:
        """Use the equipped tool, degrading it. Returns True if still usable."""
        if not self.equipped_tool:
            return False
        
        if self.equipped_tool not in self.tools:
            return False
        
        tool = self.tools[self.equipped_tool][0]
        success = tool.use_tool()
        
        if tool.is_broken():
            # Remove broken tool
            self.tools[self.equipped_tool].pop(0)
            if not self.tools[self.equipped_tool]:
                del self.tools[self.equipped_tool]
                self.equipped_tool = None
            return False
        
        return success
    
    # Skill Management
    def improve_skill(self, skill_name: str, amount: float = 0.1) -> None:
        """Improve a skill through practice."""
        current = self.skills.get(skill_name, 0)
        self.skills[skill_name] = min(10.0, current + amount)
    
    def get_skill_level(self, skill_name: str) -> float:
        """Get skill level."""
        return self.skills.get(skill_name, 0)
    
    # Market Methods
    def add_currency(self, amount: float) -> None:
        """Add currency to worker's personal account."""
        self.currency += amount
        self.transaction_history.append({
            'type': 'credit',
            'amount': amount,
            'new_balance': self.currency,
            'step': self._get_current_step()
        })
    
    def subtract_currency(self, amount: float) -> bool:
        """Subtract currency if sufficient balance."""
        if self.currency >= amount:
            self.currency -= amount
            self.transaction_history.append({
                'type': 'debit',
                'amount': amount,
                'new_balance': self.currency,
                'step': self._get_current_step()
            })
            return True
        return False
    
    def _get_current_step(self) -> int:
        """Get current simulation step."""
        return self._federation.steps if hasattr(self._federation, 'steps') else 0
    
    # Production Methods
    def assign_to_job(self, job: 'ProductionJob', step_index: int, 
                      current_step: int) -> bool:
        """Assign worker to a production job."""
        if self.current_job is not None:
            return False
        
        self.current_job = job
        self.assigned_step_index = step_index
        self.step_start_step = current_step
        job.assign_worker(self.name, step_index)
        return True
    
    def is_available(self) -> bool:
        """Check if worker is available for new work."""
        return self.current_job is None
    
    def work_on_current_step(self, current_step: int) -> Dict[str, float]:
        """Work on assigned production step."""
        if self.current_job is None:
            return {}
        
        remaining = self.current_job.get_step_remaining_duration(current_step)
        
        if remaining <= 0:
            return self.complete_current_step(current_step)
        
        return {}
    
    def complete_current_step(self, current_step: int) -> Dict[str, float]:
        """Complete the current production step."""
        if self.current_job is None:
            return {}
        
        step = self.current_job.current_step
        if not step:
            return {}
        
        # Check if we need a tool for this step
        if step.required_tool:
            if self.equipped_tool != step.required_tool:
                if not self.equip_tool(step.required_tool):
                    return {}
                
                if not self.use_equipped_tool():
                    return {}
        
        # Improve skill from practice
        if step.required_skill:
            self.improve_skill(step.required_skill, 0.05)
        
        # Complete the step
        outputs = self.current_job.complete_step(current_step)
        
        # Record outputs
        for resource_name, quantity in outputs.items():
            self.total_outputs[resource_name] = self.total_outputs.get(resource_name, 0) + quantity
        
        # Check if job is complete
        if self.current_job.is_complete():
            self.completed_jobs.append(self.current_job.job_id)
            self.current_job = None
            self.assigned_step_index = None
            self.step_start_step = None
        else:
            self.assigned_step_index = self.current_job.current_step_index
            self.step_start_step = current_step
        
        return outputs
    
    def cancel_current_job(self) -> None:
        """Cancel the current production job."""
        if self.current_job:
            self.current_job.is_active = False
            self.current_job.error_message = "Cancelled by worker"
            self.current_job = None
            self.assigned_step_index = None
            self.step_start_step = None
    
    def step(self) -> None:
        """Execute a step for the worker."""
        # Work on production if assigned
        if self.current_job:
            current_step = self._get_current_step()
            outputs = self.work_on_current_step(current_step)
            
            if outputs:
                # Optional: Notify LLM about completion
                pass
    
    # Status Methods
    def get_status(self) -> Dict[str, Any]:
        """Get worker's current status."""
        return {
            'name': self.name,
            'skills': self.skills.copy(),
            'equipped_tool': self.equipped_tool,
            'tools': {name: len(tools) for name, tools in self.tools.items()},
            'currency': self.currency,
            'current_job': self.current_job.job_id if self.current_job else None,
            'completed_jobs': len(self.completed_jobs),
            'total_outputs': self.total_outputs.copy()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize worker state."""
        return {
            'name': self.name,
            'skills': self.skills.copy(),
            'tools': {
                name: [tool.to_dict() for tool in tools]
                for name, tools in self.tools.items()
            },
            'equipped_tool': self.equipped_tool,
            'currency': self.currency,
            'completed_jobs': self.completed_jobs.copy(),
            'total_outputs': self.total_outputs.copy(),
            'transaction_history': self.transaction_history.copy()
        }