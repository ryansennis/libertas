# tests/e2e/test_simple_simulation.py
"""End-to-end simulation tests."""

import pytest
from unittest.mock import Mock

from libertas.economy import Resource, ResourceRegistry, Recipe, RecipeRegistry, ProductionStep, StepType
from libertas.organization import WorkerConfig, PodConfig, Federation


@pytest.mark.e2e
class TestSimpleSimulation:
    """Test complete simulation scenarios."""

    def test_basic_federation_setup(self, basic_federation):
        """
        Test that a basic federation can be created and stepped through.

        Uses shared fixture from conftest.py.
        """
        federation = basic_federation

        # Federation should be set up correctly
        assert len(list(federation)) == 1  # 1 pod
        pod = federation[0]
        assert pod.num_workers() == 1

        # Run simulation for 10 steps
        for step in range(10):
            federation.steps = step
            federation.step()

        # Verify simulation ran without errors
        assert federation.steps == 9

    def test_single_pod_production_cycle(self):
        """
        Complete simulation: Pod produces and trades resources.

        Scenario:
        - Pod starts with wood
        - Processes wood into planks
        - Sells planks on market
        - Workers earn currency
        """
        # Setup resources
        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))
        resource_registry.register(Resource(name="plank", base_value=15.0))

        # Setup recipe
        recipe_registry = RecipeRegistry()
        recipe = Recipe(
            name="process_wood",
            steps=[
                ProductionStep(
                    name="cut",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    inputs={"wood": 2.0},
                    outputs={"plank": 4.0},
                    required_skill="crafting"
                )
            ]
        )
        recipe_registry.register(recipe)

        # Setup worker and pod
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            initial_currency=100.0,
            initial_skills={"crafting": 3.0}
        )
        pod_config = PodConfig(
            name="woodworkers",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        # Create federation
        federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry,
            recipe_registry=recipe_registry,
            initialize_market=True
        )

        pod = federation[0]
        worker = list(pod)[0]

        # Record initial state
        initial_wood = pod.inventory.get_quantity("wood")
        initial_plank = pod.inventory.get_quantity("plank")
        initial_currency = worker.currency

        # Start production
        success, job_id = pod.start_production("process_wood", batch_size=2, started_by=worker.name)
        assert success is True

        # Run simulation for enough steps to complete production
        for step in range(20):
            federation.steps = step
            federation.step()  # This calls pod.step() which calls worker.step()

        # Verify production completed
        assert len(pod.completed_jobs) >= 1

        # Verify inventory changed
        final_wood = pod.inventory.get_quantity("wood")
        final_plank = pod.inventory.get_quantity("plank")

        assert final_wood < initial_wood  # Wood was consumed
        assert final_plank > initial_plank  # Planks were produced

        # Verify at least 4 wood consumed and 8 planks produced (2 batches)
        wood_consumed = initial_wood - final_wood
        planks_produced = final_plank - initial_plank

        assert wood_consumed >= 4.0
        assert planks_produced >= 8.0

    def test_worker_learns_from_experience(self):
        """
        Test that workers improve skills through repeated production.

        Scenario:
        - Worker starts with low crafting skill
        - Completes multiple production jobs
        - Skill level increases
        """
        # Setup
        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))
        resource_registry.register(Resource(name="plank", base_value=15.0))

        recipe_registry = RecipeRegistry()
        recipe = Recipe(
            name="process_wood",
            steps=[
                ProductionStep(
                    name="cut",
                    step_type=StepType.PROCESSING,
                    duration=3,
                    inputs={"wood": 1.0},
                    outputs={"plank": 2.0},
                    required_skill="crafting"
                )
            ]
        )
        recipe_registry.register(recipe)

        worker_config = WorkerConfig(
            name="apprentice",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            initial_skills={"crafting": 1.0}  # Start low
        )
        pod_config = PodConfig(
            name="workshop",
            workers=[worker_config],
            initial_inventory={"wood": 50.0}
        )

        federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry,
            recipe_registry=recipe_registry
        )

        pod = federation[0]
        worker = list(pod)[0]

        initial_skill = worker.get_skill_level("crafting")

        # Complete multiple jobs
        jobs_completed = 0
        for batch in range(5):  # Try to complete 5 jobs
            success, _ = pod.start_production("process_wood", batch_size=1, started_by=worker.name)
            if not success:
                break

            # Run until job completes
            for step in range(10):
                federation.steps += 1
                federation.step()

                if worker.current_job is None:
                    jobs_completed += 1
                    break

        # Skill should have improved
        final_skill = worker.get_skill_level("crafting")
        assert final_skill > initial_skill
        assert jobs_completed >= 3  # At least some jobs completed

    def test_federation_statistics(self):
        """Test that federation tracks overall statistics."""
        # Setup basic federation
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model="ollama/tinyllama"
        )
        pod_config = PodConfig(
            name="pod_001",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))

        federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry
        )

        # Get statistics
        stats = federation.get_economic_summary()

        assert stats["num_pods"] == 1
        assert stats["num_workers"] == 1
        assert "wood" in stats["total_inventory"]
        assert stats["total_inventory"]["wood"] == 100.0
        assert len(stats["known_resources"]) >= 1

    def test_simulation_runs_without_errors(self):
        """
        Smoke test: Verify simulation can run for many steps without crashing.
        """
        # Minimal setup
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model="ollama/tinyllama"
        )
        pod_config = PodConfig(
            name="pod_001",
            workers=[worker_config]
        )

        federation = Federation(pods=[pod_config])

        # Run for 100 steps
        for step in range(100):
            federation.steps = step
            try:
                federation.step()
            except Exception as e:
                pytest.fail(f"Simulation failed at step {step}: {e}")

        # If we got here, simulation ran successfully
        assert federation.steps == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
