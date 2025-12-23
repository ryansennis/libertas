'''
    Worker Agent
'''
from typing import List, Optional, Union
import mesa
import json

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import Memory
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.tools.tool_manager import ToolManager

class Worker(LLMAgent):
    '''
    Docstring for Worker
    '''
    def __init__(
        self: 'Worker',
        model: mesa.Model,
        reasoning: type[Reasoning],
        llm_model: str,
        system_prompt: Optional[str] = None,
        vision: Optional[float] = None,
        internal_state: Optional[Union[List[str], str]] = None,
        memory: Optional[Memory] = None,
        tool_manager: Optional[ToolManager] = None
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state
        )

        if isinstance(memory, Memory):
            self.memory = memory

        if isinstance(tool_manager, ToolManager):
            self.tool_manager = tool_manager
        
        self.last_llm_output = ""
        self.pod_id = 0

    def _generate_step_prompt(self) -> str:
        observation = self.generate_obs()

        prompt = f"""
            You are Worker {self.unique_id} in Pod {self.pod_id}.

            It is day {observation.step}.

            SELF CONTEXT:
            {observation.self_state}

            LOCAL CONTEXT:
            {observation.local_state}

            POD SITUATION:
            Pod members: {getattr(self.model, 'workers', lambda _: [])(self.pod_id)}
            Pending tasks: {getattr(self.model, 'task_queue', lambda _: [])(self.pod_id)}

            CONSTITUTION RULES:
            {getattr(self.model, 'constitution', lambda _: [])}

            Make a decision. Consider:
            1. What matches your skills and current state?
            2. What helps your pod?
            3. What aligns with the constitution?
            4. What maintains your well-being?
            """

        return prompt

    def step(self) -> None:
        prompt = str(self.system_prompt) + self._generate_step_prompt()

        self.last_llm_output = self.llm.generate(
            prompt=prompt,
            tool_schema=[],
            tool_choice="auto",
            response_format={}
        )


