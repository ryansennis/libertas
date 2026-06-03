# tests/unit/organization/test_worker_additional.py
"""Additional unit tests for Worker class to improve coverage."""

import pytest
import unittest
from unittest.mock import Mock

from libertas.organization import Worker, WorkerConfig, Pod, PodConfig, Federation
from libertas.economy import Resource, Recipe, ProductionStep, StepType
from libertas.economy.production import ProductionJob


LLM_MODEL = "ollama/qwen3"


@pytest.mark.unit
class TestWorkerToolBreakage(unittest.TestCase):
    """Test tool breakage scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.federation.steps = 0

        # Register tools
        self.federation.register_new_resource(
            Resource("hammer", "system", base_value=50.0, is_tool=True, durability=2.0)
        )

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_tools=["hammer"]
        )

        pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config]
        )

        pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.worker = list(pod)[0]

    def test_use_equipped_tool_breaks(self):
        """Test that using a tool until it breaks removes it."""
        self.worker.equip_tool("hammer")

        initial_count = len(self.worker.tools.get("hammer", []))
        self.assertGreater(initial_count, 0)

        # Use tool until it breaks
        for _ in range(10):  # More than durability
            result = self.worker.use_equipped_tool()
            if not result:
                break

        # Tool should be removed and unequipped
        self.assertIsNone(self.worker.equipped_tool)
        self.assertEqual(len(self.worker.tools.get("hammer", [])), 0)

    def test_use_equipped_tool_no_tool_equipped(self):
        """Test using tool when none equipped."""
        self.worker.unequip_tool()
        result = self.worker.use_equipped_tool()
        self.assertFalse(result)

    def test_use_equipped_tool_tool_not_in_inventory(self):
        """Test using equipped tool that's no longer in inventory."""
        self.worker.equipped_tool = "nonexistent"
        result = self.worker.use_equipped_tool()
        self.assertFalse(result)


@pytest.mark.unit
class TestWorkerStepMethod(unittest.TestCase):
    """Test Worker.step() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.federation.steps = 10

        self.federation.register_new_resource(
            Resource("wood", "system", base_value=10.0)
        )

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_skills={"crafting": 2.0}
        )

        pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.worker = list(pod)[0]

    def test_step_with_no_job(self):
        """Test step when worker has no job."""
        # Should not raise error
        self.worker.step()
        self.assertIsNone(self.worker.current_job)

    def test_step_with_job(self):
        """Test step when worker has a job."""
        recipe = Recipe(
            name="simple",
            steps=[
                ProductionStep(
                    name="work",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    required_skill="crafting"
                )
            ]
        )
        self.federation.recipe_registry.register(recipe)

        job = ProductionJob(recipe=recipe, batch_size=1, started_by="system")
        self.worker.assign_to_job(job, 0, self.federation.steps)

        # Worker should have job
        self.assertIsNotNone(self.worker.current_job)

        # Step should work
        self.worker.step()


@pytest.mark.unit
class TestWorkerSkillMethods(unittest.TestCase):
    """Test worker skill-related methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_skills={"crafting": 3.0, "smelting": 2.0}
        )

        pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config]
        )

        pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.worker = list(pod)[0]

    def test_get_skill_level_existing(self):
        """Test getting level of existing skill."""
        level = self.worker.get_skill_level("crafting")
        self.assertEqual(level, 3.0)

    def test_get_skill_level_nonexistent(self):
        """Test getting level of non-existent skill returns 0."""
        level = self.worker.get_skill_level("magic")
        self.assertEqual(level, 0.0)

    def test_skills_dict_contains_skill(self):
        """Test that skills dict contains expected skills."""
        self.assertIn("crafting", self.worker.skills)
        self.assertNotIn("magic", self.worker.skills)


@pytest.mark.unit
class TestWorkerJobCompletion(unittest.TestCase):
    """Test worker job completion scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.federation.steps = 0

        self.federation.register_new_resource(
            Resource("wood", "system", base_value=10.0)
        )
        self.federation.register_new_resource(
            Resource("plank", "system", base_value=15.0)
        )

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_skills={"crafting": 3.0}
        )

        pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        self.pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.worker = list(self.pod)[0]

        # Create recipe
        self.recipe = Recipe(
            name="make_plank",
            steps=[
                ProductionStep(
                    name="cut",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    inputs={"wood": 1.0},
                    outputs={"plank": 2.0},
                    required_skill="crafting",
                    required_skill_level=2.0
                )
            ]
        )
        self.federation.recipe_registry.register(self.recipe)

    def test_job_assignment_creates_connection(self):
        """Test that assigning a job creates connection between worker and job."""
        job = ProductionJob(recipe=self.recipe, batch_size=2, started_by="system")
        self.worker.assign_to_job(job, 0, self.federation.steps)

        # Worker should have the job
        self.assertEqual(self.worker.current_job, job)
        self.assertIsNotNone(self.worker.step_start_step)


@pytest.mark.unit
class TestWorkerCurrency(unittest.TestCase):
    """Test worker currency methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_currency=100.0
        )

        pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config]
        )

        pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.worker = list(pod)[0]

    def test_subtract_currency_success(self):
        """Test subtracting currency when sufficient funds."""
        result = self.worker.subtract_currency(50.0)
        self.assertTrue(result)
        self.assertEqual(self.worker.currency, 50.0)

    def test_subtract_currency_insufficient(self):
        """Test subtracting currency when insufficient funds."""
        result = self.worker.subtract_currency(200.0)
        self.assertFalse(result)
        self.assertEqual(self.worker.currency, 100.0)  # Unchanged

    def test_subtract_currency_exact_amount(self):
        """Test subtracting exact currency amount."""
        result = self.worker.subtract_currency(100.0)
        self.assertTrue(result)
        self.assertEqual(self.worker.currency, 0.0)

    def test_add_currency(self):
        """Test adding currency."""
        self.worker.add_currency(50.0)
        self.assertEqual(self.worker.currency, 150.0)

    def test_add_currency_negative(self):
        """Test adding negative currency (should still work)."""
        self.worker.add_currency(-20.0)
        self.assertEqual(self.worker.currency, 80.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
