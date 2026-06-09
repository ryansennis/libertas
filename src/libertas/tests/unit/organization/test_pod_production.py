# tests/unit/organization/test_pod_production.py
"""Unit tests for Pod production processing."""

import pytest
from unittest.mock import Mock

from libertas.organization import Pod, PodConfig, Worker, WorkerConfig, Federation
from libertas.resources import Recipe, ProductionStep, StepType, Resource


LLM_MODEL = "ollama/qwen3"


@pytest.mark.unit
class TestPodProductionProcessing:
    """Test Pod.process_production method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.federation.steps = 0

        # Register resources
        self.federation.resource_registry.register(
            Material("wood", "system", base_value=10.0)
        )
        self.federation.resource_registry.register(
            Material("plank", "system", base_value=15.0)
        )
        self.federation.resource_registry.register(
            Tool("hammer", "system", base_value=50.0, is_tool=True, durability=100.0)
        )

        # Create workers with skills
        worker_configs = [
            WorkerConfig(
                name="worker1",
                reasoning=Mock,
                llm_model=LLM_MODEL,
                initial_skills={"crafting": 3.0},
                initial_tools=["hammer"]
            ),
            WorkerConfig(
                name="worker2",
                reasoning=Mock,
                llm_model=LLM_MODEL,
                initial_skills={"crafting": 2.0}
            )
        ]

        self.pod_config = PodConfig(
            name="test_pod",
            workers=worker_configs,
            initial_inventory={"wood": 100.0}
        )

        self.pod = Pod(self.federation, self.pod_config, coordinate=(0, 0))

        # Register simple recipe
        self.simple_recipe = Recipe(
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
        self.federation.recipe_registry.register(self.simple_recipe)

    def test_process_production_moves_queue_to_active(self):
        """Test that process_production moves jobs from queue to active."""
        success, job_id = self.pod.start_production("make_plank", batch_size=1)
        assert success
        assert len(self.pod.production_queue) == 1
        assert len(self.pod.active_jobs) == 0

        self.pod.process_production(self.federation.steps)

        assert len(self.pod.production_queue) == 0
        assert len(self.pod.active_jobs) == 1

    def test_process_production_assigns_worker(self):
        """Test that process_production assigns available worker to job."""
        success, job_id = self.pod.start_production("make_plank", batch_size=1)
        assert success

        # Get workers
        workers = list(self.pod)
        assert workers[0].current_job == None
        assert workers[1].current_job == None

        self.pod.process_production(self.federation.steps)

        # One worker should be assigned
        assigned_workers = [w for w in workers if w.current_job is not None]
        assert len(assigned_workers) == 1

    def test_process_production_assigns_skilled_worker(self):
        """Test that process_production prefers workers with required skills."""
        success, job_id = self.pod.start_production("make_plank", batch_size=1)
        assert success

        self.pod.process_production(self.federation.steps)

        # worker1 has crafting:3.0, worker2 has crafting:2.0
        # Both meet requirement (2.0), but first available should get it
        workers = list(self.pod)
        assigned = [w for w in workers if w.current_job is not None]
        assert len(assigned) == 1

    def test_process_production_completes_job(self):
        """Test that completed jobs are moved to completed list."""
        success, job_id = self.pod.start_production("make_plank", batch_size=1)
        assert success

        self.pod.process_production(self.federation.steps)

        # Get the active job and manually mark as complete
        job = self.pod.active_jobs[0]
        # Mark all steps as completed
        for i in range(len(job.recipe.steps)):
            job.completed_steps.append(i)
        # Add outputs that would have been produced
        job.outputs_produced["plank"] = 2.0

        initial_plank = self.pod.inventory.get_quantity("plank")

        self.pod.process_production(self.federation.steps)

        # Job should be moved to completed
        assert len(self.pod.active_jobs) == 0
        assert len(self.pod.completed_jobs) == 1

        # Outputs should be added to inventory
        assert self.pod.inventory.get_quantity("plank") == initial_plank + 2.0

    def test_process_production_no_available_workers(self):
        """Test process_production when no workers are available."""
        success, job_id = self.pod.start_production("make_plank", batch_size=1)
        assert success

        # Make all workers unavailable by assigning them to a fake job
        for worker in self.pod:
            worker.current_job = Mock()  # Simulate busy worker

        self.pod.process_production(self.federation.steps)

        # Job should be in active but not assigned
        assert len(self.pod.active_jobs) == 1
        job = self.pod.active_jobs[0]
        assert job.assigned_worker_id is None

    def test_process_production_requires_tool(self):
        """Test that jobs requiring tools assign workers with tools."""
        # Create recipe requiring hammer
        tool_recipe = Recipe(
            name="hammer_work",
            steps=[
                ProductionStep(
                    name="hammer",
                    step_type=StepType.PROCESSING,
                    duration=3,
                    required_skill="crafting",
                    required_tool="hammer"
                )
            ]
        )
        self.federation.recipe_registry.register(tool_recipe)

        success, job_id = self.pod.start_production("hammer_work", batch_size=1)
        assert success

        self.pod.process_production(self.federation.steps)

        # worker1 has hammer, worker2 doesn't
        workers = list(self.pod)
        assigned = [w for w in workers if w.current_job is not None]

        assert len(assigned) == 1
        assert assigned[0].has_tool("hammer")

    def test_process_production_worker_already_assigned(self):
        """Test that already assigned workers aren't reassigned."""
        success, job_id = self.pod.start_production("make_plank", batch_size=1)
        assert success

        self.pod.process_production(self.federation.steps)

        # Get assigned worker
        workers = list(self.pod)
        assigned_worker = [w for w in workers if w.current_job is not None][0]
        job = assigned_worker.current_job

        # Process again - should not reassign
        self.pod.process_production(self.federation.steps)

        # Same worker should still be assigned
        assert assigned_worker.current_job == job

    def test_process_production_multiple_jobs(self):
        """Test processing multiple jobs simultaneously."""
        # Start two jobs
        success1, job_id1 = self.pod.start_production("make_plank", batch_size=1)
        success2, job_id2 = self.pod.start_production("make_plank", batch_size=1)

        assert success1
        assert success2
        assert len(self.pod.production_queue) == 2

        self.pod.process_production(self.federation.steps)

        # Both jobs should be active
        assert len(self.pod.production_queue) == 0
        assert len(self.pod.active_jobs) == 2

        # Both workers should be assigned
        workers = list(self.pod)
        assigned = [w for w in workers if w.current_job is not None]
        assert len(assigned) == 2


@pytest.mark.unit
class TestPodProductionEdgeCases:
    """Test edge cases in production processing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.federation.steps = 0

        self.federation.resource_registry.register(
            Material("wood", "system", base_value=10.0)
        )

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_skills={"crafting": 2.0}
        )

        self.pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        self.pod = Pod(self.federation, self.pod_config, coordinate=(0, 0))

    def test_process_production_empty_queue(self):
        """Test process_production with no jobs."""
        self.pod.process_production(self.federation.steps)

        assert len(self.pod.production_queue) == 0
        assert len(self.pod.active_jobs) == 0
        assert len(self.pod.completed_jobs) == 0

    def test_process_production_no_workers(self):
        """Test process_production with no workers in pod."""
        empty_config = PodConfig(name="empty_pod", workers=[])
        empty_pod = Pod(self.federation, empty_config, coordinate=(0, 0))

        # Register recipe
        recipe = Recipe(
            name="simple",
            steps=[
                ProductionStep(
                    name="step1",
                    step_type=StepType.PROCESSING,
                    duration=5
                )
            ]
        )
        self.federation.recipe_registry.register(recipe)

        # Manually add a job (bypassing start_production which might check workers)
        from libertas.economy.production import ProductionJob
        job = ProductionJob(recipe=recipe, batch_size=1, started_by="system")
        empty_pod.production_queue.append(job)

        empty_pod.process_production(self.federation.steps)

        # Job should move to active but remain unassigned
        assert len(empty_pod.active_jobs) == 1
        assert empty_pod.active_jobs[0].assigned_worker_id is None

    def test_process_production_worker_not_found(self):
        """Test process_production when assigned worker is removed."""
        recipe = Recipe(
            name="simple",
            steps=[
                ProductionStep(
                    name="step1",
                    step_type=StepType.PROCESSING,
                    duration=5
                )
            ]
        )
        self.federation.recipe_registry.register(recipe)

        success, job_id = self.pod.start_production("simple", batch_size=1)
        assert success

        self.pod.process_production(self.federation.steps)

        # Get job and manually assign to nonexistent worker
        job = self.pod.active_jobs[0]
        job.assigned_worker_id = "nonexistent_worker"

        # Process again - should handle gracefully
        self.pod.process_production(self.federation.steps)

        # Job should still be active
        assert len(self.pod.active_jobs) == 1

    def test_complete_job_adds_outputs_to_inventory(self):
        """Test that completing job adds outputs to inventory."""
        recipe = Recipe(
            name="produce_plank",
            steps=[
                ProductionStep(
                    name="cut",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    outputs={"plank": 3.0}
                )
            ]
        )
        self.federation.recipe_registry.register(recipe)
        self.federation.resource_registry.register(
            Material("plank", "system", base_value=15.0)
        )

        initial_plank = self.pod.inventory.get_quantity("plank")

        success, job_id = self.pod.start_production("produce_plank", batch_size=2)
        assert success

        self.pod.process_production(self.federation.steps)

        # Complete the job manually
        job = self.pod.active_jobs[0]
        for i in range(len(job.recipe.steps)):
            job.completed_steps.append(i)
        # Add outputs that would have been produced
        job.outputs_produced["plank"] = 6.0

        self.pod.process_production(self.federation.steps)

        # Should have 3.0 * 2 = 6.0 planks added
        assert self.pod.inventory.get_quantity("plank") == initial_plank + 6.0
        


@pytest.mark.unit
class TestPodStepMethod:
    """Test Pod.step() method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])
        self.federation.steps = 10

        self.federation.resource_registry.register(
            Material("wood", "system", base_value=10.0)
        )

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )

        self.pod_config = PodConfig(
            name="test_pod",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        self.pod = Pod(self.federation, self.pod_config, coordinate=(0, 0))

    def test_step_calls_process_production(self):
        """Test that step() calls process_production."""
        recipe = Recipe(
            name="simple",
            steps=[
                ProductionStep(
                    name="step1",
                    step_type=StepType.PROCESSING,
                    duration=5
                )
            ]
        )
        self.federation.recipe_registry.register(recipe)

        success, job_id = self.pod.start_production("simple", batch_size=1)
        assert success

        assert len(self.pod.production_queue) == 1

        self.pod.step()

        # Job should be moved from queue to active
        assert len(self.pod.production_queue) == 0
        assert len(self.pod.active_jobs) == 1

    def test_step_with_no_federation_steps(self):
        """Test step() when federation doesn't have steps attribute."""
        # Create federation without steps
        fed = Federation(pods=[])

        pod_config = PodConfig(name="test_pod", workers=[])
        pod = Pod(fed, pod_config, coordinate=(0, 0))

        # Should not raise error
        pod.step()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
