from __future__ import annotations
from ..economy import ProductionJob
from ..resources import PodInventory
from ..governance import Constitution
from .worker import Worker, WorkerConfig
from dataclasses import dataclass
from mesa.agentset import AgentSet
from mesa.discrete_space.cell import Cell
from random import Random
from typing import List, Optional, Union, Dict, TYPE_CHECKING
import json
import networkx as nx

if TYPE_CHECKING:
    from .federation import Federation

@dataclass
class PodConfig:
    name: str
    workers: List[WorkerConfig]
    random: Optional[Random] = None
    initial_inventory: Optional[Dict[str, float]] = None
    initial_tools: Optional[List[str]] = None
    constitution: Optional[Constitution] = None
    
    def to_json(self, filepath: Optional[str] = None) -> Union[str, None]:
        """Convert PodConfig to JSON string or save to file."""
        config_dict = {
            'name': self.name,
            'workers': [worker_config.to_json() for worker_config in self.workers],
            'random': None,
            'initial_inventory': self.initial_inventory,
            'initial_tools': self.initial_tools
        }
        
        json_str = json.dumps(config_dict, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        return json_str
    
    @staticmethod
    def from_json(data: Union[str, Dict, object], filepath: Optional[str] = None) -> 'PodConfig':
        """Create PodConfig from JSON string, dictionary, or file."""
        if filepath:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif isinstance(data, str):
            config_dict = json.loads(data)
        elif isinstance(data, dict):
            config_dict = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        from .worker import WorkerConfig
        
        workers = []
        for worker_config_dict in config_dict.get('workers', []):
            worker_config = WorkerConfig.from_json(worker_config_dict)
            workers.append(worker_config)

        name: str = ''
        pod_name = config_dict.get('name')
        if pod_name is not None:
            name = pod_name
        
        return PodConfig(
            name=name,
            workers=workers,
            random=None,
            initial_inventory=config_dict.get('initial_inventory'),
            initial_tools=config_dict.get('initial_tools')
        )
    
    @staticmethod
    def from_json_file(filepath: str) -> 'PodConfig':
        """Convenience method to load from a JSON file."""
        return PodConfig.from_json(None, filepath=filepath)


class Pod(AgentSet[Worker]):
    """Pod that manages a group of workers with spatial positioning and production."""

    def __init__(self, federation, pod_config, coordinate):
        self.name = pod_config.name
        self.federation = federation
        self.pod_config = pod_config

        # Initialize governance
        self.constitution = pod_config.constitution or Constitution.create_default_pod_constitution(pod_config.name)

        worker_instances: List[Worker] = []
        for idx, config in enumerate(pod_config.workers):
            temp_coordinate = (idx, 0)
            # Pass both federation and pod reference
            worker = Worker(federation, config, coordinate=temp_coordinate, pod=self)
            worker_instances.append(worker)

        super().__init__(worker_instances, pod_config.random)

        self._cell = Cell(coordinate=coordinate)

        self.inventory = PodInventory(capacity=None)
        self.production_queue: List[ProductionJob] = []
        self.active_jobs: List[ProductionJob] = []
        self.completed_jobs: List[ProductionJob] = []

        if pod_config.initial_inventory:
            for resource_name, quantity in pod_config.initial_inventory.items():
                resource = federation.resource_registry.get(resource_name)
                if resource:
                    self.inventory.add(resource, quantity)

        if pod_config.initial_tools:
            for tool_name in pod_config.initial_tools:
                tool = federation.resource_registry.get(tool_name)
                if tool:
                    self.inventory.add(tool, 1)

        # Create graph AFTER workers are in AgentSet
        self._worker_graph: nx.Graph = self._create_worker_network()

        # No need to register workers - Worker.__init__ already does this via Agent.__init__

    def __hash__(self) -> int:
        """Make Pod hashable based on its name."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Define equality for Pod objects based on name."""
        if not isinstance(other, Pod):
            return False
        return self.name == other.name

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
    
    def start_production(self, recipe_name: str, batch_size: int = 1, 
                        started_by: str = "") -> tuple[bool, str]:
        """
        Start a new production job.
        
        Returns:
            (success, message_or_job_id)
        """
        recipe = self.federation.recipe_registry.get(recipe_name)
        if not recipe:
            return False, f"Recipe '{recipe_name}' not found"
        
        # Check if pod has required inputs
        can_start, reason = recipe.can_start(self._get_available_inputs())
        if not can_start:
            return False, reason
        
        # Create job
        job = ProductionJob(
            recipe=recipe,
            started_by=started_by,
            started_at_step=self.federation.steps,
            batch_size=batch_size
        )
        
        # Consume inputs immediately (or could consume step-by-step)
        for resource_name, quantity in recipe.total_inputs.items():
            required = quantity * batch_size
            if not self.inventory.remove(resource_name, required):
                return False, f"Failed to consume {required} {resource_name}"
        
        self.production_queue.append(job)
        return True, job.job_id
    
    def _get_available_inputs(self) -> Dict[str, float]:
        """Get current inventory levels for recipe checking.

        During migration: use new system if available, fallback to old.
        """
        # Try new system first
        if hasattr(self.inventory, 'materials') and self.inventory.materials:
            return {name: mat.quantity for name, mat in self.inventory.materials.items()}
        # Fallback to old system
        return self.inventory.quantities.copy()
    
    def process_production(self, simulation_step: int):
        """Process active production jobs and assign workers."""
        while self.production_queue:
            job = self.production_queue.pop(0)
            self.active_jobs.append(job)
        
        for job in self.active_jobs[:]:
            if job.is_complete():
                for resource_name, quantity in job.get_total_outputs().items():
                    resource = self.federation.resource_registry.get(resource_name)
                    if resource:
                        self.inventory.add(resource, quantity)
                
                self.completed_jobs.append(job)
                self.active_jobs.remove(job)
                continue
            
            if job.assigned_worker_id:
                worker = self.get_worker_by_name(job.assigned_worker_id)
                if worker and worker.current_job == job:
                    continue
            
            next_step = job.get_next_step()
            if next_step:
                available_workers = [
                    w for w in self 
                    if w.current_job is None and w.is_available()
                ]
                
                for worker in available_workers:
                    has_tool = False
                    if next_step.required_tool:
                        has_tool = worker.has_tool(next_step.required_tool)
                    
                    can_perform, reason = next_step.can_perform(
                        worker.skills, has_tool
                    )
                    
                    if can_perform:
                        worker.assign_to_job(job, job.current_step_index, simulation_step)
                        break
    
    def step(self):
        """Execute a step for all workers and process production."""
        self.process_production(self.federation.steps if hasattr(self.federation, 'steps') else 0)
        
        for worker in self:
            worker.step()
    
    def get_inventory_summary(self) -> Dict[str, float]:
        """Get summary of current inventory.

        During migration: use new system if available, fallback to old.
        """
        # Try new system first
        if hasattr(self.inventory, 'materials') and self.inventory.materials:
            result = {name: mat.quantity for name, mat in self.inventory.materials.items()}
            # Also include consumables if present
            if hasattr(self.inventory, 'consumables') and self.inventory.consumables:
                for name, cons in self.inventory.consumables.items():
                    result[name] = result.get(name, 0) + cons.quantity
            return result
        # Fallback to old system
        return self.inventory.quantities.copy()
    
    def get_tools_summary(self) -> Dict[str, int]:
        """Get summary of tools in inventory.

        During migration: use new system if available, fallback to old.
        """
        # Try new system first
        if hasattr(self.inventory, 'tools') and self.inventory.tools:
            tool_counts = {}
            for tool_id, tool in self.inventory.tools.items():
                name = tool.info.name
                tool_counts[name] = tool_counts.get(name, 0) + 1
            return tool_counts
        # Fallback to old system
        return {
            name: len(tools)
            for name, tools in self.inventory.instances.items()
        }
    
    def transfer_to_pod(self, resource_name: str, quantity: float, 
                       target_pod: 'Pod') -> bool:
        """Transfer resources to another pod."""
        if self.inventory.remove(resource_name, quantity):
            resource = self.federation.resource_registry.get(resource_name)
            if resource:
                target_pod.inventory.add(resource, quantity)
                return True
        return False
    
    def add_worker(self, worker: Worker) -> None:
        """Add a worker to the pod and update the network."""
        if worker not in self:
            # Add to AgentSet
            super().add(worker)
            
            # Rebuild the entire graph from scratch (cleaner than updating)
            self._worker_graph = self._create_worker_network()
            
            # Register with federation
            self.federation.register_agent(worker)
    
    def remove_worker(self, worker: Worker) -> None:
        """Remove a worker from the pod and update the network."""
        if worker in self:
            # Remove from AgentSet
            super().remove(worker)
            
            # Rebuild graph from remaining workers
            self._worker_graph = self._create_worker_network()
    
    def get_worker_by_name(self, name: str) -> Optional[Worker]:
        """Get a worker by its unique_id."""
        for worker in self:
            if worker.name == name:
                return worker
        return None
    
    def get_worker_by_index(self, index: int) -> Optional[Worker]:
        """Get a worker by its index."""
        try:
            return self[index]
        except (IndexError, KeyError):
            return None
    
    def _create_worker_network(self) -> nx.Graph:
        workers_list = list(self)
        if len(workers_list) == 0:
            return nx.Graph()
        
        # Create complete graph using indices first, then relabel
        graph = nx.complete_graph(len(workers_list))
        
        # Use worker.name (string) as node key
        mapping = {i: worker.name for i, worker in enumerate(workers_list)}
        graph = nx.relabel_nodes(graph, mapping)
        
        self._update_worker_coordinates(graph)
        
        return graph
        
    def _update_worker_coordinates(self, graph: nx.Graph):
        """Update coordinates of all workers based on current network layout."""
        if len(self) == 0 or graph.number_of_nodes() == 0:
            return
        
        try:
            pos = nx.spring_layout(graph)
            for worker_id, position in pos.items():
                # Find worker by unique_id
                for worker in self:
                    if worker.unique_id == worker_id:
                        worker.coordinate = tuple(position)
                        break
        except Exception:
            # Fallback: simple circular layout
            import math
            n = len(self)
            for i, worker in enumerate(self):
                angle = 2 * math.pi * i / n
                worker.coordinate = (math.cos(angle), math.sin(angle))
    
    def set_worker_layout(self, layout_type: str = "spring"):
        if len(self) == 0:
            return
        
        if layout_type == "spring":
            pos = nx.spring_layout(self._worker_graph)
        elif layout_type == "circular":
            pos = nx.circular_layout(self._worker_graph)
        elif layout_type == "random":
            pos = nx.random_layout(self._worker_graph)
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self._worker_graph)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
        
        for worker_id, position in pos.items():
            for worker in self:
                if worker.unique_id == worker_id:
                    worker.coordinate = tuple(position)
                    break
    
    def get_worker_neighbors(self, worker: Worker) -> List[Worker]:
        neighbors = []
        worker_name = worker.name
        
        if worker_name in self._worker_graph:
            for neighbor_name in self._worker_graph.neighbors(worker_name):
                for w in self:
                    if w.name == neighbor_name:
                        neighbors.append(w)
                        break
        return neighbors
    
    def get_worker_network(self) -> nx.Graph:
        """Return a copy of the worker network graph."""
        return self._worker_graph.copy()
    
    def get_worker_degrees(self) -> Dict[Worker, int]:
        """Get the degree for each worker."""
        return dict(self._worker_graph.degree())
    
    def is_fully_connected(self) -> bool:
        """Check if the worker network is fully connected (complete graph)."""
        n = len(self)
        if n <= 1:
            return True
        expected_edges = n * (n - 1) // 2
        return self._worker_graph.number_of_edges() == expected_edges
    
    def num_workers(self) -> int:
        """Return the number of workers in this pod."""
        return len(self)
    
    def to_list(self) -> List[Worker]:
        """Convert to list when needed."""
        return list(self)
    
    def to_json(self, filepath: Optional[str] = None) -> Union[str, None]:
        """Convert Pod to JSON configuration."""
        return self.pod_config.to_json(filepath)
    
    @classmethod
    def from_json(
        cls, 
        federation: Federation, 
        data: Union[str, Dict, object],
        coordinate: tuple,
        filepath: Optional[str] = None
    ) -> 'Pod':
        """Create Pod instance from JSON configuration."""
        pod_config = PodConfig.from_json(data, filepath)
        return cls(federation, pod_config, coordinate)