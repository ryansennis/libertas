# tests/test_recipe.py
import unittest
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.economy import (
    ProductionStep,
    StepType,
    Recipe,
    ProductionJob, 
    RecipeRegistry
)


@pytest.mark.unit
class TestProductionStep(unittest.TestCase):
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
        
        self.assertEqual(step.name, "cut_wood")
        self.assertEqual(step.step_type, StepType.PROCESSING)
        self.assertEqual(step.duration, 10)
        self.assertEqual(step.inputs, {"wood": 2})
        self.assertEqual(step.outputs, {"planks": 1})
    
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
        
        self.assertEqual(step.required_tool, "hammer")
        self.assertEqual(step.required_skill, "crafting")
        self.assertEqual(step.required_skill_level, 2.0)
    
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
        self.assertFalse(can_perform)
        self.assertIn("Need expertise level 3.0", reason)
        
        # Worker with sufficient skill
        can_perform, reason = step.can_perform({"expertise": 4.0}, False)
        self.assertTrue(can_perform)
        
        # Step requiring tool
        step.required_tool = "hammer"
        can_perform, reason = step.can_perform({"expertise": 4.0}, False)
        self.assertFalse(can_perform)
        self.assertIn("Need tool: hammer", reason)
        
        # Worker with tool
        can_perform, reason = step.can_perform({"expertise": 4.0}, True)
        self.assertTrue(can_perform)
    
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
        
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.step_type, original.step_type)
        self.assertEqual(restored.duration, original.duration)
        self.assertEqual(restored.quality_threshold, original.quality_threshold)


@pytest.mark.unit
class TestRecipe(unittest.TestCase):
    """Test Recipe class."""
    
    def setUp(self):
        self.steps = [
            ProductionStep(name="step1", duration=5, inputs={"a": 1}, outputs={"b": 1},
            step_type=StepType.QUALITY_CHECK),
            ProductionStep(name="step2", duration=3, inputs={"b": 1}, outputs={"c": 1},
            step_type=StepType.QUALITY_CHECK)
        ]
        self.recipe = Recipe(name="test_recipe", steps=self.steps)
    
    def test_recipe_creation(self):
        """Test basic recipe creation."""
        self.assertEqual(self.recipe.name, "test_recipe")
        self.assertEqual(len(self.recipe.steps), 2)
        self.assertEqual(self.recipe.total_duration, 8)
    
    def test_aggregated_inputs_outputs(self):
        """Test input/output aggregation."""
        self.assertEqual(self.recipe.total_inputs, {"a": 1, "b": 1})
        self.assertEqual(self.recipe.total_outputs, {"b": 1, "c": 1})
    
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
        
        self.assertIn("hammer", recipe.requires_tools)
        self.assertIn("crafting", recipe.requires_skills)
    
    def test_get_step(self):
        """Test retrieving steps by index."""
        step = self.recipe.get_step(0)
        if step:
            self.assertEqual(step.name, "step1")
        
        self.assertIsNone(self.recipe.get_step(10))
    
    def test_can_start(self):
        """Test checking if recipe can be started with available inputs."""
        # Sufficient inputs
        can_start, reason = self.recipe.can_start({"a": 5, "b": 5})
        self.assertTrue(can_start)
        
        # Insufficient inputs
        can_start, reason = self.recipe.can_start({"a": 0})
        self.assertFalse(can_start)
        self.assertIn("Need 1 a", reason)
    
    def test_requires_parallel_coordination(self):
        """Test detection of parallel steps."""
        # No parallel steps
        self.assertFalse(self.recipe.requires_parallel_coordination())
        
        # Add parallel step
        parallel_step = ProductionStep(
            name="parallel", duration=5,
            can_parallel=True, max_parallel=3,
            step_type=StepType.QUALITY_CHECK
        )
        recipe = Recipe(name="parallel_recipe", steps=[parallel_step])
        self.assertTrue(recipe.requires_parallel_coordination())
    
    def test_recipe_serialization(self):
        """Test recipe serialization."""
        data = self.recipe.to_dict()
        restored = Recipe.from_dict(data)
        
        self.assertEqual(restored.name, self.recipe.name)
        self.assertEqual(len(restored.steps), len(self.recipe.steps))
        self.assertEqual(restored.total_duration, self.recipe.total_duration)


@pytest.mark.unit
class TestProductionJob(unittest.TestCase):
    """Test ProductionJob class."""
    
    def setUp(self):
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
        
        self.assertEqual(job.recipe.name, "test")
        self.assertEqual(job.started_by, "worker_001")
        self.assertEqual(job.batch_size, 2)
        self.assertTrue(job.is_active)
        self.assertEqual(job.current_step_index, 0)
    
    def test_assign_worker(self):
        """Test assigning worker to job."""
        job = ProductionJob(recipe=self.recipe)
        job.assign_worker("worker_001", 0)
        
        self.assertEqual(job.assigned_worker_id, "worker_001")
        self.assertEqual(job.current_step_index, 0)
    
    def test_step_completion(self):
        """Test completing production steps."""
        job = ProductionJob(recipe=self.recipe, batch_size=1)
        job.assign_worker("worker_001", 0)
        
        # Complete first step
        self.assertEqual(job.complete_step(10), {"output": 1})
        self.assertEqual(len(job.completed_steps), 1)
        self.assertEqual(job.current_step_index, 1)
        
        # Not complete yet
        self.assertFalse(job.is_complete())
        
        # Complete second step
        job.assign_worker("worker_001", 1)
        self.assertEqual(job.complete_step(15), {"output": 1})
        self.assertTrue(job.is_complete())
        self.assertFalse(job.is_active)
    
    def test_get_progress(self):
        """Test progress calculation."""
        job = ProductionJob(recipe=self.recipe)
        
        self.assertEqual(job.get_progress(), 0.0)
        
        job.completed_steps = [0]
        self.assertEqual(job.get_progress(), 0.5)
        
        job.completed_steps = [0, 1]
        self.assertEqual(job.get_progress(), 1.0)
    
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
        
        self.assertEqual(restored.recipe.name, job.recipe.name)
        self.assertEqual(restored.started_by, job.started_by)
        self.assertEqual(restored.batch_size, job.batch_size)
        self.assertEqual(restored.completed_steps, job.completed_steps)


@pytest.mark.unit
class TestRecipeRegistry(unittest.TestCase):
    """Test RecipeRegistry class."""
    
    def setUp(self):
        self.registry = RecipeRegistry()
        self.recipe = Recipe(name="test", steps=[])
    
    def test_register_and_get(self):
        """Test registering and retrieving recipes."""
        self.registry.register(self.recipe)
        
        retrieved = self.registry.get("test")
        if retrieved:
            self.assertEqual(retrieved.name, "test")
    
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
            self.assertEqual(new_recipe.name, "new_recipe")
        self.assertEqual(len(self.registry.invention_history), 1)
        self.assertEqual(self.registry.invention_history[0]["inventor"], "worker_001")
    
    def test_list_recipes(self):
        """Test listing all recipes."""
        self.registry.register(Recipe(name="recipe1", steps=[]))
        self.registry.register(Recipe(name="recipe2", steps=[]))
        
        recipes = self.registry.list_recipes()
        self.assertEqual(len(recipes), 2)
    
    def test_get_by_category(self):
        """Test filtering recipes by category."""
        self.registry.register(Recipe(name="craft1", steps=[], category="crafting"))
        self.registry.register(Recipe(name="craft2", steps=[], category="crafting"))
        self.registry.register(Recipe(name="process1", steps=[], category="processing"))
        
        crafting = self.registry.get_by_category("crafting")
        self.assertEqual(len(crafting), 2)

@pytest.mark.unit
class TestRecipeEdgeCases(unittest.TestCase):
    """Test edge cases for recipe system."""
    
    def test_empty_recipe(self):
        """Test recipe with no steps."""
        recipe = Recipe(name="empty", steps=[])
        
        self.assertEqual(recipe.total_duration, 0)
        self.assertEqual(recipe.total_inputs, {})
        self.assertEqual(recipe.total_outputs, {})
        
        can_start, _ = recipe.can_start({})
        self.assertTrue(can_start)
    
    def test_job_with_zero_batch(self):
        """Test job with zero batch size."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe, batch_size=0)
        
        job.assign_worker("w1", 0)
        outputs = job.complete_step(10)
        self.assertEqual(outputs, {})
    
    def test_job_with_negative_batch(self):
        """Test job with negative batch size."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe, batch_size=-1)
        
        job.assign_worker("w1", 0)
        outputs = job.complete_step(10)
        # Should handle gracefully (outputs multiplied by negative)
        self.assertEqual(outputs, {})
    
    def test_complete_already_completed_step(self):
        """Test completing already completed step."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe)
        
        job.completed_steps = [0]
        job.current_step_index = 1
        
        outputs = job.complete_step(10)
        self.assertEqual(outputs, {})
    
    def test_get_next_step_from_completed_job(self):
        """Test getting next step from completed job."""
        recipe = Recipe(name="test", steps=[ProductionStep(name="s1", duration=5,
            step_type=StepType.QUALITY_CHECK)])
        job = ProductionJob(recipe=recipe)
        
        job.completed_steps = [0]
        job.current_step_index = 1
        
        self.assertIsNone(job.get_next_step())


if __name__ == '__main__':
    unittest.main()