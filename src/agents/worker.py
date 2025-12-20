from enum import Enum

from litellm import Reasoning
import mesa

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

worker_tool_manager = ToolManager()

class WorkerState(Enum):
    QUIET = 1
    ACTIVE = 2
    ARRESTED = 3

class Worker(LLMAgent):
    """
    """

    def __init__(
        self,
        model: mesa.Model,
        reasoning: str,
        llm_model: str,
        system_prompt: str,
        vision: float,
        internal_state: list[str],
        step_prompt: str
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

        self.memory = STLTMemory(
            agent=self,
            display=True,
            llm_model=llm_model,
        )

        self.tool_manager = worker_tool_manager
        self.system_prompt = ""

    def step(self) -> None:
        return
