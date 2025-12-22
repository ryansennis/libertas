'''
    Worker Agent
'''
import mesa

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
        tool_manager = ToolManager,
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
        step_prompt: str | None = None,
        memory: Memory | None = None,
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

        self.memory = memory
        self.tool_manager = tool_manager

    def step(self) -> None:
        return
