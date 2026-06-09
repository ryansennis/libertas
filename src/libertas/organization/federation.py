# src/libertas/organization/federation.py
from .pod import Pod, PodConfig
from collections.abc import MutableSet, Sequence
from typing import Optional, Union, List, Dict, cast
import mesa
import networkx as nx
import numpy as np

from ..economy import Market
from ..resources import ResourceRegistry, Resource, Recipe, ProductionStep
from ..governance import Constitution, GovernanceEngine

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


class Federation(mesa.Model, MutableSet[Pod]):
    def __init__(
        self,
        pods: List[PodConfig],
        seed: Optional[Union[float, int]] = None,
        rng: Optional[Union[RNGLike, SeedLike]] = None,
        resource_registry: Optional[ResourceRegistry] = None,
        market: Optional[Market] = None,
        initialize_market: bool = True,
        constitution: Optional[Constitution] = None,
        enable_cognitive_loop: bool = True,
        base_labor_rate: float = 0.5,
    ):
        # Call mesa.Model.__init__
        super().__init__(seed=seed, rng=rng)

        # Track simulation step
        self.steps = 0
        self.enable_cognitive_loop = enable_cognitive_loop

        # Initialize unified registry (resources + recipes)
        self.resource_registry = resource_registry or ResourceRegistry(base_labor_rate=base_labor_rate)
        # Alias for backwards compatibility with tests
        self.recipe_registry = self.resource_registry

        # Initialize governance
        self.constitution = constitution or Constitution.create_default_federation_constitution()
        self.governance = GovernanceEngine()

        # NEW SYSTEM: Federation-level shared inventory (parallel tracking)
        try:
            from ..resources import FederationInventory
            self.shared_inventory: FederationInventory = FederationInventory(capacity=None)
        except ImportError:
            self.shared_inventory = None

        # Create pods with temporary coordinates
        pod_instances: List[Pod] = []
        self._pod_map: Dict[str, Pod] = {}  # Map name to pod
        for idx, config in enumerate(pods):
            temp_coordinate = (idx, 0)
            pod = Pod(self, config, coordinate=temp_coordinate)
            pod_instances.append(pod)
            self._pod_map[pod.name] = pod
        
        self._pods: List[Pod] = pod_instances
        
        # Create complete graph for pods (using names as nodes)
        self._create_pod_network()
        
        # Register pods
        for pod in self._pods:
            self.register_agent(pod)

        # Initialize market
        if initialize_market:
            self.market = market or Market(random_seed=seed if isinstance(seed, int) else None)
            
            # Register existing resources with market
            from ..resources import Tool
            for resource_name in self.resource_registry.list_resources():
                resource = self.resource_registry.get(resource_name)
                if resource and not isinstance(resource, Tool):
                    self.market.register_resource(resource_name, resource.base_value)
        else:
            self.market = market
    
    def _create_pod_network(self):
        """Create a network for pods within the federation."""
        if len(self._pods) == 0:
            return
        
        # Use pod.name as node identifiers
        pod_names = [pod.name for pod in self._pods]
        graph: nx.Graph = nx.complete_graph(pod_names)
        
        # Store the graph (nodes are names)
        self._pod_graph = graph
        
        # Assign positions using names
        self._update_pod_coordinates(graph)
        
        # Create cell graph
        cell_graph = nx.Graph()
        for pod_name in pod_names:
            pod = self._pod_map[pod_name]
            cell_graph.add_node(pod._cell)
        
        # Add edges
        for i, pod_name1 in enumerate(pod_names):
            for pod_name2 in pod_names[i+1:]:
                pod1 = self._pod_map[pod_name1]
                pod2 = self._pod_map[pod_name2]
                cell_graph.add_edge(pod1._cell, pod2._cell)
        
        self._grid = mesa.discrete_space.Network(cell_graph)
    
    def _update_pod_coordinates(self, graph: nx.Graph):
        """Update coordinates of all pods based on current network layout."""
        if len(self._pods) == 0:
            return
        
        pos = nx.spring_layout(graph)
        for pod_name, position in pos.items():
            pod = self._pod_map.get(pod_name)
            if pod:
                pod.coordinate = tuple(position)
    
    # Economic Methods
    def register_new_resource(self, resource: Resource) -> bool:
        registry = self.resource_registry
        if registry is not None:
            registry.register(resource)
            return True
        else:
            return False
    
    def register_new_recipe(self, name: str, steps: List,
                           inventor_id: str,
                           description: str = "",
                           category: str = "general") -> Recipe:
        return self.resource_registry.invent_recipe(
            name=name,
            steps=steps,
            inventor_id=inventor_id,
            step=self.steps,
            description=description,
            category=category
        )

    def get_resource(self, name: str) -> Optional[Resource]:
        return self.resource_registry.get(name)

    def get_recipe(self, name: str) -> Optional[Recipe]:
        return self.resource_registry.get_recipe_by_name(name)

    def list_resources(self) -> List[str]:
        return self.resource_registry.list_resources()

    def list_recipes(self) -> List[str]:
        return self.resource_registry.list_recipes()
    
    # MutableSet required methods
    def __contains__(self, pod: object) -> bool:
        return pod in self._pods
    
    def __iter__(self):
        return iter(self._pods)
    
    def __len__(self) -> int:
        return len(self._pods)
    
    def __getitem__(self, key: Union[int, str]) -> Pod:
        """Get pod by index or name."""
        if isinstance(key, str):
            if key in self._pod_map:
                return self._pod_map[key]
            raise KeyError(f"Pod with name '{key}' not found")
        elif isinstance(key, int):
            if 0 <= key < len(self._pods):
                return self._pods[key]
            raise IndexError(f"Pod index {key} out of range")
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def add(self, value: Pod) -> None:
        """Add a pod to the federation."""
        if value not in self._pods:
            self._pods.append(value)
            self._pod_map[value.name] = value
            
            # Rebuild the graph with all pods
            pod_names = [pod.name for pod in self._pods]
            self._pod_graph = nx.complete_graph(pod_names)
            self._update_pod_coordinates(self._pod_graph)
            
            # Rebuild cell graph
            cell_graph = nx.Graph()
            for pod_name in pod_names:
                pod = self._pod_map[pod_name]
                cell_graph.add_node(pod._cell)
            
            for i, pod_name1 in enumerate(pod_names):
                for pod_name2 in pod_names[i+1:]:
                    pod1 = self._pod_map[pod_name1]
                    pod2 = self._pod_map[pod_name2]
                    cell_graph.add_edge(pod1._cell, pod2._cell)
            
            self._grid = mesa.discrete_space.Network(cell_graph)
            self.register_agent(value)
    
    def discard(self, value: Pod) -> None:
        """Remove a pod from the federation if present."""
        if value in self._pods:
            self._pods.remove(value)
            del self._pod_map[value.name]
            
            if len(self._pods) > 0:
                pod_names = [pod.name for pod in self._pods]
                self._pod_graph = nx.complete_graph(pod_names)
                self._update_pod_coordinates(self._pod_graph)
                
                cell_graph = nx.Graph()
                for pod_name in pod_names:
                    pod = self._pod_map[pod_name]
                    cell_graph.add_node(pod._cell)
                
                for i, pod_name1 in enumerate(pod_names):
                    for pod_name2 in pod_names[i+1:]:
                        pod1 = self._pod_map[pod_name1]
                        pod2 = self._pod_map[pod_name2]
                        cell_graph.add_edge(pod1._cell, pod2._cell)
                
                self._grid = mesa.discrete_space.Network(cell_graph)
            else:
                self._grid = None
                self._pod_graph = None
    
    def remove(self, value: Pod) -> None:
        if value not in self._pods:
            raise KeyError(f"Pod {value} not in federation")
        self.discard(value)
    
    # Convenience methods
    def get_neighbors(self, pod: Pod) -> List[Pod]:
        """Get neighboring pods in the network."""
        if not hasattr(self, '_pod_graph') or self._pod_graph is None:
            return []
        
        try:
            neighbors: List[Pod] = []
            for neighbor_name in self._pod_graph.neighbors(pod.name):
                neighbor_pod = self._pod_map.get(neighbor_name)
                if neighbor_pod is not None:
                    neighbors.append(neighbor_pod)
            return neighbors
        except nx.NetworkXError:
            return []
    
    def get_pod_by_name(self, name: str) -> Optional[Pod]:
        """Get a pod by its name."""
        return self._pod_map.get(name)
    
    def get_pod_by_index(self, index: int) -> Optional[Pod]:
        if 0 <= index < len(self._pods):
            return self._pods[index]
        return None
    
    def to_list(self) -> List[Pod]:
        return list(self._pods)
    
    def get_pod_network(self) -> nx.Graph:
        if hasattr(self, '_pod_graph') and self._pod_graph is not None:
            return self._pod_graph.copy()
        return nx.Graph()
    
    def set_pod_layout(self, layout_type: str = "spring"):
        if len(self._pods) == 0 or not hasattr(self, '_pod_graph') or self._pod_graph is None:
            return
        
        graph = self._pod_graph
        
        if layout_type == "spring":
            pos = nx.spring_layout(graph)
        elif layout_type == "circular":
            pos = nx.circular_layout(graph)
        elif layout_type == "random":
            pos = nx.random_layout(graph)
        elif layout_type == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
        
        for pod_name, position in pos.items():
            pod = self._pod_map.get(pod_name)
            if pod:
                pod.coordinate = tuple(position)
    
    def get_economic_summary(self) -> Dict:
        total_inventory = {}
        total_tools = {}
        
        for pod in self._pods:
            for resource, qty in pod.get_inventory_summary().items():
                total_inventory[resource] = total_inventory.get(resource, 0) + qty
            
            for tool, count in pod.get_tools_summary().items():
                total_tools[tool] = total_tools.get(tool, 0) + count
        
        return {
            'step': self.steps,
            'num_pods': len(self._pods),
            'num_workers': sum(pod.num_workers() for pod in self._pods),
            'total_inventory': total_inventory,
            'total_tools': total_tools,
            'known_resources': self.list_resources(),
            'known_recipes': self.list_recipes(),
            'resource_inventions': len(self.resource_registry.invention_history),
            'recipe_inventions': sum(1 for item in self.resource_registry.invention_history if item['type'] == 'recipe')
        }

    def step(self) -> None:
        super().step()

        if hasattr(self, 'market') and self.market:
            transactions = self.market.process_market(
                timestamp=self.steps,
                pod_inventory_getter=self._get_pod_inventory,
                pod_inventory_setter=self._update_pod_inventory
            )

            for tx in transactions:
                buyer_worker = self._find_worker_by_name(tx['buyer_worker'])
                seller_worker = self._find_worker_by_name(tx['seller_worker'])

                if buyer_worker and seller_worker:
                    cost = tx['total_value']
                    if buyer_worker.subtract_currency(cost):
                        seller_worker.add_currency(cost)

        # Autonomous agent cognitive loop (only if enabled)
        if self.enable_cognitive_loop:
            for pod in self._pods:
                for worker in pod:
                    try:
                        cognitive_result = worker.observe_and_reason()
                        actions = cognitive_result.get("actions", [])
                        if actions:
                            worker.execute_actions(actions)
                    except Exception:
                        # Skip cognitive loop if LLM unavailable or worker not properly configured
                        pass

        # Process governance votes
        if hasattr(self, 'governance') and self.governance:
            completed_motions = self.governance.process_votes(self.steps)
            for motion, passed in completed_motions:
                if passed:
                    self._execute_motion(motion)

        for pod in self._pods:
            pod.step()

    def _get_pod_inventory(self, pod_name: str, resource_name: str) -> float:
        pod = self.get_pod_by_name(pod_name)
        if pod:
            return pod.inventory.get_quantity(resource_name)
        return 0.0

    def _update_pod_inventory(self, pod_name: str, resource_name: str, delta: float) -> None:
        pod = self.get_pod_by_name(pod_name)
        if pod:
            resource = self.resource_registry.get(resource_name)
            if resource:
                if delta > 0:
                    pod.inventory.add(resource, delta)
                else:
                    pod.inventory.remove(resource_name, -delta)

    def _find_worker_by_name(self, name: str):
        for pod in self._pods:
            for worker in pod:
                if worker.name == name:
                    return worker
        return None

    def _execute_motion(self, motion):
        """Execute a passed motion."""
        # Placeholder for motion execution logic
        # Will implement specific motion type handlers in future PRs
        pass