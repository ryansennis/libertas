# tests/test_recipe.py
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.resources import (
    ProductionStep,
    StepType,
    Recipe,
    ResourceRegistry
)
from libertas.economy import ProductionJob


@pytest.mark.unit
class TestProductionStep:
    """Test ProductionStep class."""
    
    def test_step_creation(self):
        """Test basic step creation."""
        step = ProductionStep(
            name="cut_wood",
            step_type=StepType.PROCESSING,
            duration=10,
            inputs={"wood": 2},
            outputs={"planks": 1}
        )
        
        assert step.name == "cut_wood"
        assert step.step_type == StepType.PROCESSING
        assert step.duration == 10
        assert step.inputs == {"wood": 2}
        assert step.outputs == {"planks": 1}
    
    def test_step_with_requirements(self):
        """Test step with tool and skill requirements."""
        step = ProductionStep(
            name="craft",
            step_type=StepType.ASSEMBLY,
            duration=15,
            required_tool="hammer",
            required_skill="crafting",
            required_skill_level=2.0
        )
        
        assert step.required_tool == "hammer"
        assert step.required_skill == "crafting"
        assert step.required_skill_level == 2.0
    
    def test_can_perform(self):
        """Test checking if worker can perform a step."""
        step = ProductionStep(
            name="skilled_step",
            duration=5,
            required_skill="expertise",
            required_skill_level=3.0,
            step_type=StepType.QUALITY_CHECK
        )
        
        # Worker with insufficient skill
        can_perform, reason = step.can_perform({"expertise": 2.0}, False)
        assert not (can_perform)
        assert "Need expertise level 3.0" in reason
        
        # Worker with sufficient skill
        can_perform, reason = step.can_perform({"expertise": 4.0}, False)
        assert can_perform
        
        # Step requiring tool
        step.required_tool = "hammer"
        can_perform, reason = step.can_perform({"expertise": 4.0}, False)
        assert not (can_perform)
        assert "Need tool: hammer" in reason
        
        # Worker with tool
        can_perform, reason = step.can_perform({"expertise": 4.0}, True)
        assert can_perform
    
    def test_step_serialization(self):
        """Test step serialization."""
        original = ProductionStep(
            name="test_step",
            step_type=StepType.QUALITY_CHECK,
            duration=8,
            inputs={"input": 1},
            outputs={"output": 1},
            quality_threshold=0.9
        )
        
        data = original.to_dict()
        restored = ProductionStep.from_dict(data)
        
        assert restored.name == original.name
        assert restored.step_type == original.step_type
        assert restored.duration == original.duration
        assert restored.quality_threshold == original.quality_threshold


@pytest.mark.unit
class TestRecipe:
    """Test Recipe class."""
    
    def setup_method(self):
        self.steps = [
            ProductionStep(name="step1", duration=5, inputs={"a": 1}, outputs={"b": 1},
            step_type=StepType.QUALITY_CHECK),
            ProductionStep(name="step2", duration=3, inputs={"b": 1}, outputs={"c": 1},
            step_type=StepType.QUALITY_CHECK)
        ]
        self.recipe = Recipe(name="test_recipe", steps=self.steps)
    
    def test_recipe_creation(self):
        """Test basic recipe creation."""
        assert self.recipe.name == "test_recipe"
        assert len(self.recipe.steps) == 2
        assert self.recipe.total_duration == 8
    
    def test_aggregated_inputs_outputs(self):
        """Test input/output aggregation."""
        assert self.recipe.total_inputs == {"a": 1, "b": 1}
        assert self.recipe.total_outputs == {"b": 1, "c": 1}
    
    def test_requirements_detection(self):
        """Test detection of required tools and skills."""
        step_with_tool = ProductionStep(
            name="step3", duration=5,
            required_tool="hammer",
            step_type=StepType.QUALITY_CHECK
        )
        step_with_skill = ProductionStep(
            name="step4", duration=5,
            required_skill="crafting",
            step_type=StepType.QUALITY_CHECK
        )
        
        recipe = Recipe(name="complex", steps=[step_with_tool, step_with_skill])
        
        assert "hammer" in recipe.requires_tools
        assert "crafting" in recipe.requires_skills
    
    def test_get_step(self):
        """Test retrieving steps by index."""
        step = self.recipe.get_step(0)
        if step:
            assert step.name == "step1"
        
        assert self.recipe.get_step(10) is None
    
    def test_can_start(self):
        """Test checking if recipe can be started with available inputs."""
        # Sufficient inputs
        can_start, reason = self.recipe.can_start({"a": 5, "b": 5})
        assert can_start
        
        # Insufficient inputs
        can_start, reason = self.recipe.can_start({"a": 0})
        assert not (can_start)
        assert "Need 1 a" in reason
    
    def test_requires_parallel_coordination(self):
        """Test detection of parallel steps."""
        # No parallel steps
        assert not (self.recipe.requires_parallel_coordination())
        
        # Add parallel step
        parallel_step = ProductionStep(
            name="parallel", duration=5,
            can_parallel=True, max_parallel=3,
            step_type=StepType.QUALITY_CHECK
        )
        recipe = Recipe(name="parallel_recipe", steps=[parallel_step])
        assert recipe.requires_parallel_coordination()
    
    def test_recipe_serialization(self):
        """Test recipe serialization."""
        data = self.recipe.to_dict()
        restored = Recipe.from_dict(data)
        
        assert restored.name == self.recipe.name
        assert len(restored.steps) == len(self.recipe.steps)
        assert restored.total_duration == self.recipe.total_duration


@pytest.mark.unit
class TestProductionJob:
    """Test ProductionJob class."""
    
    def setup_method(self):
        steps = [
            ProductionStep(name="step1", duration=5, outputs={"output": 1},
            step_type=StepType.QUALITY_CHECK),
            ProductionStep(name="step2", duration=3, outputs={"output": 1},
            step_type=StepType.QUALITY_CHECK)
        ]
        self.recipe = Recipe(name="test", steps=steps)
    
    def test_job_creation(self):
        """Test job creation."""
        job = ProductionJob(
            recipe=self.recipe,
            started_by="worker_001",
            started_at_step=100,
            batch_size=2
        )
        
        assert job.recipe.name == "test"
        assert job.started_by == "worker_001"
        assert job.batch_size == 2
        assert job.is_active
        assert job.current_step_index == 0
    
    def test_assign_worker(self):
        """Test assigning worker to job."""
        job = ProductionJob(recipe=self.recipe)
        job.assign_worker("worker_001")
        
        assert job.assigned_worker_id == "worker_001"
        assert job.current_step_index == 0
    
    def test_step_completion(self):
        """Test completing production steps."""
        job = ProductionJob(recipe=self.recipe, batch_size=1)
        job.assign_worker("worker_001")
        
        # Complete first step
        assert job.complete_step(10) == {"output": 1}
        assert len(job.completed_steps) == 1
        assert job.current_step_index == 1
        
        # Not complete yet
        assert not (job.is_complete())
        
        # Complete second step
        job.assign_worker("worker_001", 1)
        assert job.complete_step(15) == {"output": 1}
        assert job.is_complete()
        assert not (job.is_active)
    
    def test_get_progress(self):
        """Test progress calculation."""
        job = ProductionJob(recipe=self.recipe)
        
        assert job.get_progress() == 0.0
        
        job.completed_steps = [0]
        assert job.get_progress() == 0.5
        
        job.completed_steps = [0, 1]
        assert job.get_progress() == 1.0
    
    def test_job_serialization(self):
        """Test job serialization."""
        job = ProductionJob(
            recipe=self.recipe,
            started_by="worker_001",
            batch_size=2
        )
        job.completed_steps = [0]

        data = job.to_dict()
        restored = ProductionJob.from_dict(data)

        assert restored.recipe.name == job.recipe.name
        assert restored.started_by == job.started_by
        assert restored.batch_size == job.batch_size
        assert restored.completed_steps == job.completed_steps

    def test_get_step_remaining_duration_no_current_step(self):
        """Test get_step_remaining_duration when no current step exists."""
        # Create job with no steps
        empty_recipe = Recipe(name="empty", steps=[])
        job = ProductionJob(recipe=empty_recipe)

        # Should return 0 when no current step
        assert job.get_step_remaining_duration(10) == 0

    def test_consume_step_inputs_no_current_step(self):
        """Test consume_step_inputs when no current step exists."""
        # Create job with no steps
        empty_recipe = Recipe(name="empty", steps=[])
        job = ProductionJob(recipe=empty_recipe)

        # Should return False when no current step
        assert not (job.consume_step_inputs())

    def test_consume_step_inputs_with_batch_size(self):
        """Test consume_step_inputs multiplies by batch_size."""
        steps = [
            ProductionStep(
                name="step1",
                step_type=StepType.PROCESSING,
                duration=5,
                inputs={"wood": 2},
                outputs={"planks": 1}
            )
        ]
        recipe = Recipe(name="test", steps=steps)
        job = ProductionJob(recipe=recipe, batch_size=3)

        # Consume inputs for current step
        job.consume_step_inputs()

        # Should be 2 wood * 3 batch_size = 6
        assert job.inputs_consumed["wood"] == 6

    def test_get_step_outputs_no_current_step(self):
        """Test get_step_outputs when no current step exists."""
        # Create job with no steps
        empty_recipe = Recipe(name="empty", steps=[])
        job = ProductionJob(recipe=empty_recipe)

        # Should return empty dict when no current step
        assert job.get_step_outputs() == {}

    def test_get_progress_empty_recipe(self):
        """Test get_progress with empty recipe."""
        empty_recipe = Recipe(name="empty", steps=[])
        job = ProductionJob(recipe=empty_recipe)

        # Should return 1.0 for empty recipe
        assert job.get_progress() == 1.0

    def test_start_current_step(self):
        """Test start_current_step sets step_start_step."""
        job = ProductionJob(recipe=self.recipe)

        # Initially should be None
        assert job.step_start_step is None

        # Start the step at simulation step 100
        job.start_current_step(100)

        # Should now be set
        assert job.step_start_step == 100

    def test_get_step_remaining_duration_before_started(self):
        """Test get_step_remaining_duration before step is started."""
        job = ProductionJob(recipe=self.recipe)

        # step_start_step is None, should return full duration
        assert job.get_step_remaining_duration(10) == 5

    def test_get_step_remaining_duration_in_progress(self):
        """Test get_step_remaining_duration while step is in progress."""
        job = ProductionJob(recipe=self.recipe)
        job.start_current_step(100)

        # At step 102, elapsed is 2, remaining should be 5 - 2 = 3
        assert job.get_step_remaining_duration(102) == 3

        # At step 105, elapsed is 5, remaining should be 0
        assert job.get_step_remaining_duration(105) == 0

    def test_get_total_outputs(self):
        """Test get_total_outputs returns copy of outputs_produced."""
        steps = [
            ProductionStep(
                name="step1",
                step_type=StepType.PROCESSING,
                duration=5,
                outputs={"product": 2}
            )
        ]
        recipe = Recipe(name="test", steps=steps)
        job = ProductionJob(recipe=recipe, batch_size=3)

        # Complete the step
        job.complete_step(10)

        # Should have 2 * 3 = 6 products
        outputs = job.get_total_outputs()
        assert outputs["product"] == 6

        # Modifying returned dict shouldn't affect job
        outputs["product"] = 999
        assert job.outputs_produced["product"] == 6

    def test_get_next_step(self):
        """Test get_next_step returns current uncompleted step."""
        job = ProductionJob(recipe=self.recipe)

        # Initially should be first step
        next_step = job.get_next_step()
        assert next_step is not None
        if next_step:
            assert next_step.name == "step1"

        # Complete first step
        job.complete_step(10)

        # Should now be second step
        next_step = job.get_next_step()
        assert next_step is not None
        if next_step:
            assert next_step.name == "step2"

        # Complete second step
        job.complete_step(20)

        # No more steps
        assert job.get_next_step() is None


@pytest.mark.unit
class TestStartingRecipes:
    """Test get_starting_recipes function."""

    def test_get_starting_recipes(self):
        """Test that get_starting_recipes returns valid recipes."""
        from libertas.economy.production import get_starting_recipes

        recipes = get_starting_recipes()

        # Should return a list
        assert isinstance(recipes, list)

        # Should have some recipes
        assert len(recipes) > 0

        # All items should be Recipe instances
        from libertas.economy import Recipe
        for recipe in recipes:
            assert isinstance(recipe, Recipe)
            assert recipe.name is not None
            assert len(recipe.steps) > 0


@pytest.mark.unit
class TestRecipeRegistry:
    """Test RecipeRegistry class."""

    def setup_method(self):
        self.registry = ResourceRegistry()
        self.recipe = Recipe(name="test", steps=[])
    
    def test_register_and_get(self):
        """Test registering and retrieving recipes."""
        self.registry.register(self.recipe)
        
        retrieved = self.registry.get("test")
        if retrieved:
            assert retrieved.name == "test"
    
    def test_invent_recipe(self):
        """Test inventing new recipes."""
        steps = [ProductionStep(name="step1", duration=5,
            step_type=StepType.QUALITY_CHECK)]
        
        recipe = self.registry.invent(
            name="new_recipe",
            steps=steps,
            inventor_id="worker_001",
            step=50,
            description="A new way to make things"
        )
        
        new_recipe = self.registry.get("new_recipe")
        if new_recipe:
            assert new_recipe.name == "new_recipe"
        assert len(self.registry.invention_history) == 1
        assert self.registry.invention_history[0]["inventor"] == "worker_001"
    
    def test_list_recipes(self):
        """Test listing all recipes."""
        self.registry.register(Recipe(name="recipe1", steps=[]))
        self.registry.register(Recipe(name="recipe2", steps=[]))
        
        recipes = self.registry.list_recipes()
        assert len(recipes) == 2
    
    def test_get_by_category(self):
        """Test filtering recipes by category."""
        self.registry.register(Recipe(name="craft1", steps=[], category="crafting"))
        self.registry.register(Recipe(name="craft2", steps=[], category="crafting"))
        self.registry.register(Recipe(name="process1", steps=[], category="processing"))
        
        crafting = self.registry.get_by_category("crafting")
        assert len(crafting) == 2

@pytest.mark.unit
class TestRecipeEdgeCases:
    """Test edge cases for recipe system."""

    def test_empty_recipe(self):
        """Test recipe with no steps."""
        recipe = Recipe(name="empty", steps=[])

        assert recipe.total_duration == 0
        assert recipe.total_inputs == {}
        assert recipe.total_outputs == {}

        can_start, _ = recipe.can_start({})
        assert can_start

    def test_get_step_inputs_invalid_index(self):
        """Test get_step_inputs with invalid step index."""
        recipe = Recipe(
            name="test_recipe",
            steps=[
                ProductionStep(
                    name="step1",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    inputs={"wood": 2},
                    outputs={"planks": 1}
                )
            ]
        )

        # Invalid index should return empty dict
        assert recipe.get_step_inputs(-1) == {}
        assert recipe.get_step_inputs(999) == {}

    def test_get_step_outputs_invalid_index(self):
        """Test get_step_outputs with invalid step index."""
        recipe = Recipe(
            name="test_recipe",
            steps=[
                ProductionStep(
                    name="step1",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    inputs={"wood": 2},
                    outputs={"planks": 1}
                )
            ]
        )

        # Invalid index should return empty dict
        assert recipe.get_step_outputs(-1) == {}
        assert recipe.get_step_outputs(999) == {}

    def test_get_required_tools_for_step_invalid_index(self):
        """Test get_required_tools_for_step with invalid step index."""
        recipe = Recipe(
            name="test_recipe",
            steps=[
                ProductionStep(
                    name="step1",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    required_tool="hammer"
                )
            ]
        )

        # Invalid index should return None
        assert recipe.get_required_tools_for_step(-1 is None)
        assert recipe.get_required_tools_for_step(999 is None)
    
    def test_job_with_zero_batch(self):
        """Test job with zero batch size."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe, batch_size=0)
        
        job.assign_worker("w1")
        outputs = job.complete_step(10)
        assert outputs == {}
    
    def test_job_with_negative_batch(self):
        """Test job with negative batch size."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe, batch_size=-1)
        
        job.assign_worker("w1")
        outputs = job.complete_step(10)
        # Should handle gracefully (outputs multiplied by negative)
        assert outputs == {}
    
    def test_complete_already_completed_step(self):
        """Test completing already completed step."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe)
        
        job.completed_steps = [0]
        job.current_step_index = 1
        
        outputs = job.complete_step(10)
        assert outputs == {}
    
    def test_get_next_step_from_completed_job(self):
        """Test getting next step from completed job."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe)
        
        job.completed_steps = [0]
        job.current_step_index = 1
        
        assert job.get_next_step() is None


