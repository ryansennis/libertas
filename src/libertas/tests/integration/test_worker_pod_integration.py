# tests/integration/test_worker_pod_integration.py
"""Integration tests for Worker-Pod interactions."""

import pytest
from mesa_llm.reasoning.cot import CoTReasoning

from libertas.resources import Resource, ResourceRegistry, Recipe, ProductionStep, StepType, Material, Tool
from libertas.organization import WorkerConfig, PodConfig, Federation


@pytest.mark.integration
class TestWorkerPodIntegration:
    """Test interactions between workers and pods."""

    @pytest.fixture
    def federation_with_resources(self):
        """Create a federation with resources and recipes."""
        # Create resources
        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))
        resource_registry.register(Resource(name="plank", base_value=15.0))
        resource_registry.register(Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0,
            required_skill="crafting"
        ))

        # Create recipes
        recipe_registry = RecipeRegistry()
        process_recipe = Recipe(
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
        recipe_registry.register(process_recipe)

        # Create worker and pod
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_currency=500.0,
            initial_skills={"crafting": 3.0},
            initial_tools=["hammer"]
        )
        pod_config = PodConfig(
            name="pod_001",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry,
            recipe_registry=recipe_registry,
            initialize_market=True
        )
        federation.steps = 0

        return federation

    def test_worker_accesses_pod_inventory(self, federation_with_resources):
        """Test that workers can access their pod's inventory."""
        pod = federation_with_resources[0]
        worker = list(pod)[0]

        # Worker should have access to pod
        assert worker.pod == pod
        assert worker.pod.name == "pod_001"

        # Worker should see pod inventory
        inventory = worker.pod.get_inventory_summary()
        assert "wood" in inventory
        assert inventory["wood"] == 100.0

    def test_worker_starts_production(self, federation_with_resources):
        """Test worker starting a production job."""
        pod = federation_with_resources[0]
        worker = list(pod)[0]

        # Start production
        success, job_id = pod.start_production("process_wood", batch_size=1, started_by=worker.name)

        assert success is True
        assert len(job_id) > 0  # UUID should be generated

        # Job should be in queue initially
        assert len(pod.production_queue) >= 1 or len(pod.active_jobs) >= 1

        # Process queue to move job to active
        pod.process_production(federation_with_resources.steps)

        # Pod should now have active job
        assert len(pod.active_jobs) == 1
        assert pod.active_jobs[0].recipe.name == "process_wood"

    def test_worker_completes_production_job(self, federation_with_resources):
        """Test worker completing a production job from start to finish."""
        federation = federation_with_resources
        pod = federation[0]
        worker = list(pod)[0]

        # Check initial inventory
        initial_wood = pod.inventory.get_quantity("wood")
        initial_planks = pod.inventory.get_quantity("plank")

        # Start production
        success, job_id = pod.start_production("process_wood", batch_size=1, started_by=worker.name)
        assert success is True

        # Run simulation loop until job completes (or timeout after 20 steps)
        max_steps = 20
        for step_num in range(max_steps):
            federation.steps += 1

            # Process production queue (assigns/manages workers)
            pod.process_production(federation.steps)

            # Worker performs work
            worker.step()

            # Check if job completed
            if worker.current_job is None:
                break

        # Process production one final time to move completed job to completed_jobs
        pod.process_production(federation.steps)

        # Verify job completed within reasonable time
        assert worker.current_job is None, f"Job did not complete after {max_steps} steps"
        assert len(pod.completed_jobs) == 1

        # Verify resource transformation
        final_wood = pod.inventory.get_quantity("wood")
        final_planks = pod.inventory.get_quantity("plank")

        assert final_wood == initial_wood - 1.0  # Consumed 1 wood
        assert final_planks == initial_planks + 2.0  # Produced 2 planks

    def test_worker_uses_tool_from_inventory(self, federation_with_resources):
        """Test worker equipping and using a tool."""
        pod = federation_with_resources[0]
        worker = list(pod)[0]

        # Worker should have hammer tool
        assert worker.has_tool("hammer")

        # Equip hammer
        success = worker.equip_tool("hammer")
        assert success is True
        assert worker.equipped_tool == "hammer"

        # Use hammer (degrades durability)
        initial_tool_count = len(worker.tools["hammer"])
        initial_durability = worker.tools["hammer"][0].durability

        success = worker.use_equipped_tool()
        assert success is True

        # Durability should decrease
        assert worker.tools["hammer"][0].durability < initial_durability

        # Unequip
        worker.unequip_tool()
        assert worker.equipped_tool is None

    def test_worker_skill_improvement(self, federation_with_resources):
        """Test worker skills improve through production work."""
        federation = federation_with_resources
        pod = federation[0]
        worker = list(pod)[0]

        initial_skill = worker.get_skill_level("crafting")

        # Start and complete production
        pod.start_production("process_wood", batch_size=1, started_by=worker.name)

        # Complete the job
        for _ in range(10):  # Run enough steps to complete job
            federation.steps += 1
            pod.process_production(federation.steps)
            worker.step()

            # Break if job completes
            if worker.current_job is None:
                break

        # Skill should have improved
        final_skill = worker.get_skill_level("crafting")
        assert final_skill > initial_skill

    def test_multiple_workers_in_pod(self):
        """Test multiple workers sharing a pod."""
        # Create multiple workers
        worker1_config = WorkerConfig(
            name="worker_001",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_skills={"crafting": 2.0}
        )
        worker2_config = WorkerConfig(
            name="worker_002",
            reasoning=CoTReasoning,
            llm_model="ollama/tinyllama",
            initial_skills={"smelting": 3.0}
        )

        pod_config = PodConfig(
            name="pod_001",
            workers=[worker1_config, worker2_config],
            initial_inventory={"wood": 100.0}
        )

        federation = Federation(pods=[pod_config])
        pod = federation[0]

        # Pod should have both workers
        assert pod.num_workers() == 2
        workers = list(pod)
        assert len(workers) == 2

        # Both workers should reference same pod
        assert workers[0].pod == pod
        assert workers[1].pod == pod

        # Workers should know each other
        neighbors = pod.get_worker_neighbors(workers[0])
        assert workers[1] in neighbors


