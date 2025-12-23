'''
    Pod Model
'''
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import mesa
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import Memory
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.tools.tool_manager import ToolManager

from .worker import Worker

class Pod(LLMAgent):
    '''
        Pod Model
    '''
    def __init__(
        self: 'Pod',
        model: mesa.Model,
        reasoning: type[Reasoning],
        llm_model: str,
        system_prompt: Optional[str] = None,
        vision: Optional[float] = None,
        internal_state: Optional[Union[List[str], str]] = None,
        step_prompt: Optional[str] = None,
        memory: Optional[Memory] = None,
        tool_manager: Optional[ToolManager] = None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
        )

        if isinstance(memory, Memory):
            self.memory = memory

        if isinstance(tool_manager, ToolManager):
            self.tool_manager = tool_manager

        self.resources = {}
        self.task_queue = []

        self.workers: List[Worker] = []
        self.max_workers = 0
        self.min_workers = 0

        self.network_grid: Optional[mesa.space.NetworkGrid] = None

    def setup_pod(
        self: 'Pod',
        resources: Dict[str, Any],
        workers: List[Worker],
        max_workers: int = 12,
        min_workers: int = 3,
    ) -> None:
        '''
        Docstring for setup_pod
        '''
        self.resources = resources

        worker_graph = nx.Graph()

        for worker in workers:
            worker_graph.add_node(worker.unique_id)

        for i, worker1 in enumerate(workers):
            for worker2 in workers[i + 1 :]:
                worker_graph.add_edge(worker1.unique_id, worker2.unique_id)

        self.network_grid = mesa.space.NetworkGrid(worker_graph)

        for worker in workers:
            worker.pod_id = self.unique_id
            self.workers.append(worker)
            self.network_grid.place_agent(worker, worker.unique_id)

        self.max_workers = max_workers
        self.min_workers = min_workers
    
    def step(self) -> None:
        for worker in self.workers:
            worker.step()
