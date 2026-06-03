# tests/test_worker.py
import unittest
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.organization import Worker, WorkerConfig, Pod, PodConfig, Federation
from libertas.economy import Recipe, ProductionStep, ProductionJob, Resource, StepType


@pytest.mark.unit
class TestWorkerConfig(unittest.TestCase):
    """Test WorkerConfig class."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            system_prompt="You are a helpful worker",
            initial_skills={"crafting": 2.0},
            initial_tools=["hammer"],
            initial_currency=500.0
        )
        
        self.assertEqual(config.name, "worker_001")
        self.assertEqual(config.llm_model, "ollama/tinyllama")
        self.assertEqual(config.initial_skills, {"crafting": 2.0})
        self.assertEqual(config.initial_tools, ["hammer"])
        self.assertEqual(config.initial_currency, 500.0)
    
    def test_config_serialization(self):
        """Test config serialization to/from JSON."""
        config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model="ollama/tinyllama"
        )
        
        json_str = config.to_json()
        self.assertIsInstance(json_str, str)
        
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()
            loaded = WorkerConfig.from_json(json_str)
            self.assertEqual(loaded.name, config.name)


@pytest.mark.unit
class TestWorker(unittest.TestCase):
    """Test Worker class with real objects."""
    
    def setUp(self):
        self.federation = Federation(pods=[])
        
        tool = Resource(
            name="hammer",
            base_value=15.0,
            is_tool=True,
            durability=100,
            required_skill="crafting",
            enables_recipes=["forge", "assemble"]
        )
        
        self.federation.register_new_resource(tool)
        
        self.worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            initial_skills={"crafting": 1.0, "forging": 0.5},
            initial_tools=["hammer"],
            initial_currency=100.0
        )
        
        self.worker = Worker(self.federation, self.worker_config, coordinate=(0, 0))
        self.worker.tools[tool.name] = [tool]
    
    def test_worker_creation(self):
        """Test basic worker creation."""
        self.assertEqual(self.worker.name, "worker_001")
        self.assertEqual(self.worker.coordinate, (0, 0))
        self.assertEqual(self.worker.currency, 100.0)
        self.assertEqual(self.worker.skills, {"crafting": 1.0, "forging": 0.5})
    
    def test_hash_and_eq(self):
        """Test worker hashing and equality."""
        worker2_config = WorkerConfig(name="worker_002", reasoning=Mock, llm_model="ollama/tinyllama")
        worker2 = Worker(self.federation, worker2_config, (1, 0))
        
        self.assertEqual(hash(self.worker), hash(self.worker))
        self.assertNotEqual(hash(self.worker), hash(worker2))
        self.assertEqual(self.worker, self.worker)
        self.assertNotEqual(self.worker, worker2)
    
    def test_tool_management(self):
        """Test tool inventory management."""
        self.assertTrue(self.worker.has_tool("hammer"))
        
        self.assertTrue(self.worker.equip_tool("hammer"))
        self.assertEqual(self.worker.equipped_tool, "hammer")
        
        self.worker.unequip_tool()
        self.assertIsNone(self.worker.equipped_tool)
    
    def test_use_equipped_tool(self):
        """Test using equipped tool degrades durability."""
        self.worker.equip_tool("hammer")
        
        tool = self.worker.tools["hammer"][0]
        self.assertEqual(tool.durability, 100)
        
        self.worker.use_equipped_tool()
        self.assertEqual(tool.durability, 99)
    
    def test_skill_improvement(self):
        """Test skill improvement through practice."""
        self.assertEqual(self.worker.get_skill_level("crafting"), 1.0)
        
        self.worker.improve_skill("crafting", 0.2)
        self.assertEqual(self.worker.get_skill_level("crafting"), 1.2)
        
        for _ in range(100):
            self.worker.improve_skill("crafting", 0.5)
        self.assertEqual(self.worker.get_skill_level("crafting"), 10.0)
    
    def test_currency_management(self):
        """Test currency addition and subtraction."""
        self.worker.add_currency(50.0)
        self.assertEqual(self.worker.currency, 150.0)
        
        self.assertTrue(self.worker.subtract_currency(30.0))
        self.assertEqual(self.worker.currency, 120.0)
        
        self.assertFalse(self.worker.subtract_currency(200.0))
        self.assertEqual(self.worker.currency, 120.0)
    
    def test_transaction_history(self):
        """Test transaction history tracking."""
        self.worker.add_currency(100.0)
        self.worker.subtract_currency(25.0)
        
        self.assertEqual(len(self.worker.transaction_history), 2)
        self.assertEqual(self.worker.transaction_history[0]["type"], "credit")
        self.assertEqual(self.worker.transaction_history[1]["type"], "debit")
    
    def test_assign_to_job(self):
        """Test assigning worker to production job."""
        step = ProductionStep(name="s1", duration=5, step_type=StepType.ASSEMBLY)
        recipe = Recipe(name="test", steps=[step])
        job = ProductionJob(recipe=recipe)
        
        self.assertTrue(self.worker.assign_to_job(job, 0, 100))
        self.assertEqual(self.worker.current_job, job)
        self.assertEqual(self.worker.assigned_step_index, 0)
        
        job2 = ProductionJob(recipe=recipe)
        self.assertFalse(self.worker.assign_to_job(job2, 0, 101))
    
    def test_is_available(self):
        """Test worker availability."""
        self.assertTrue(self.worker.is_available())
        
        step = ProductionStep(name="s1", duration=5, step_type=StepType.ASSEMBLY)
        recipe = Recipe(name="test", steps=[step])
        job = ProductionJob(recipe=recipe)
        self.worker.assign_to_job(job, 0, 100)
        
        self.assertFalse(self.worker.is_available())
    
    def test_work_on_step_completion(self):
        """Test working on and completing a production step."""
        step = ProductionStep(
            name="craft",
            duration=10,
            outputs={"widget": 1},
            required_skill="crafting",
            step_type=StepType.ASSEMBLY,
            required_tool="hammer"
        )
        recipe = Recipe(name="test", steps=[step])
        job = ProductionJob(recipe=recipe, batch_size=1)
        
        self.worker.equip_tool("hammer")
        self.worker.assign_to_job(job, 0, 100)
        job.start_current_step(100)
        
        outputs = self.worker.work_on_current_step(105)
        self.assertEqual(outputs, {})
        
        outputs = self.worker.work_on_current_step(110)
        self.assertEqual(outputs, {"widget": 1})
        self.assertTrue(job.is_complete())
        self.assertIsNone(self.worker.current_job)
        self.assertGreater(self.worker.get_skill_level("crafting"), 1.0)
    
    def test_cancel_job(self):
        """Test canceling current production job."""
        step = ProductionStep(name="s1", duration=5, step_type=StepType.ASSEMBLY)
        recipe = Recipe(name="test", steps=[step])
        job = ProductionJob(recipe=recipe)
        
        self.worker.assign_to_job(job, 0, 100)
        self.assertIsNotNone(self.worker.current_job)
        self.assertTrue(job.is_active)

        self.worker.cancel_current_job()
        self.assertIsNone(self.worker.current_job)
        self.assertFalse(job.is_active)
        self.assertEqual(job.error_message, "Cancelled by worker")
    
    def test_get_status(self):
        """Test getting worker status."""
        status = self.worker.get_status()
        
        self.assertEqual(status["name"], "worker_001")
        self.assertEqual(status["skills"]["crafting"], 1.0)
        self.assertEqual(status["currency"], 100.0)
        self.assertEqual(status["completed_jobs"], 0)
    
    def test_worker_serialization(self):
        """Test worker state serialization."""
        self.worker.add_currency(50.0)
        self.worker.improve_skill("crafting", 0.5)
        
        data = self.worker.to_dict()
        
        self.assertEqual(data["name"], "worker_001")
        self.assertEqual(data["skills"]["crafting"], 1.5)
        self.assertEqual(data["currency"], 150.0)
        self.assertIn("hammer", data["tools"])
    
    def test_get_federation(self):
        """Test getting federation reference."""
        federation = self.worker.federation
        self.assertEqual(federation, self.federation)


@pytest.mark.unit
class TestWorkerEdgeCases(unittest.TestCase):
    """Test edge cases for Worker class."""
    
    def setUp(self):
        # Create real federation and pod
        self.federation = Federation(pods=[])
        tool = Resource(
            name="hammer",
            base_value=15.0,
            is_tool=True,
            durability=100,
            required_skill="crafting",
            enables_recipes=["forge", "assemble"]
        )
        self.federation.register_new_resource(tool)
        
        pod_config = PodConfig(name="test_pod", workers=[])
        self.pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        
        worker_config = WorkerConfig(
            name="test_worker",
            reasoning=Mock,
            llm_model="ollama/tinyllama"
        )
        
        self.worker = Worker(self.federation, worker_config, coordinate=(0, 0))
        self.pod.add_worker(self.worker)
    
    def test_worker_no_tools(self):
        """Test worker with no initial tools."""
        config = WorkerConfig(
            name="worker_no_tools",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            initial_tools=None
        )
        worker = Worker(self.federation, config, (0, 0))
        
        self.assertFalse(worker.has_tool("hammer"))
        self.assertFalse(worker.equip_tool("hammer"))
    
    def test_worker_no_skills(self):
        """Test worker with no initial skills."""
        config = WorkerConfig(
            name="worker_no_skills",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            initial_skills=None
        )
        worker = Worker(self.federation, config, (0, 0))
        
        self.assertEqual(worker.get_skill_level("any_skill"), 0.0)
    
    def test_use_tool_when_not_equipped(self):
        """Test using tool when none equipped."""
        config = WorkerConfig(
            name="worker",
            reasoning=Mock,
            llm_model="ollama/tinyllama"
        )
        worker = Worker(self.federation, config, (0, 0))
        
        self.assertFalse(worker.use_equipped_tool())
    
    def test_equip_nonexistent_tool(self):
        """Test equipping non-existent tool."""
        self.assertFalse(self.worker.equip_tool("nonexistent"))
    
    def test_work_without_job(self):
        """Test working when no job assigned."""
        outputs = self.worker.work_on_current_step(100)
        self.assertEqual(outputs, {})
    
    def test_complete_step_without_job(self):
        """Test completing step when no job assigned."""
        outputs = self.worker.complete_current_step(100)
        self.assertEqual(outputs, {})
    
    def test_coordinate_update(self):
        """Test updating worker coordinates."""
        self.worker.coordinate = (10, 20)
        self.assertEqual(self.worker.coordinate, (10, 20))


@pytest.mark.unit
class TestWorkerWithTools(unittest.TestCase):
    """Test worker with tool requirements for production."""
    
    def setUp(self):
        self.federation = Federation(pods=[])
        tool = Resource(
            name="hammer",
            base_value=15.0,
            is_tool=True,
            durability=100,
            required_skill="crafting",
            enables_recipes=["forge", "assemble"]
        )
        self.federation.register_new_resource(tool)
        
        self.worker_config = WorkerConfig(
            name="worker",
            reasoning=Mock,
            llm_model="ollama/tinyllama",
            initial_tools=["hammer"],
            initial_skills={"crafting": 2.0}
        )
        
        self.worker = Worker(self.federation, self.worker_config, (0, 0))
    
    def test_step_requires_tool(self):
        """Test completing step that requires a tool."""
        step = ProductionStep(
            name="forge",
            duration=5,
            outputs={"metal": 1},
            required_tool="hammer",
            required_skill="crafting",
            step_type=StepType.ASSEMBLY
        )
        recipe = Recipe(name="forge_metal", steps=[step])
        job = ProductionJob(recipe=recipe)
        
        self.worker.assign_to_job(job, 0, 100)
        job.start_current_step(100)
        
        outputs = self.worker.work_on_current_step(105)
        self.assertEqual(outputs, {"metal": 1})
        self.assertEqual(self.worker.tools["hammer"][0].durability, 99)
    
    def test_step_missing_required_tool(self):
        """Test step that requires a tool the worker doesn't have."""
        step = ProductionStep(
            name="advanced_forge",
            duration=5,
            outputs={"steel": 1},
            required_tool="anvil",
            step_type=StepType.ASSEMBLY
        )
        recipe = Recipe(name="make_steel", steps=[step])
        job = ProductionJob(recipe=recipe)
        
        self.worker.assign_to_job(job, 0, 100)
        job.start_current_step(100)
        
        outputs = self.worker.work_on_current_step(105)
        self.assertEqual(outputs, {})
        self.assertIsNotNone(self.worker.current_job)


if __name__ == '__main__':
    unittest.main()