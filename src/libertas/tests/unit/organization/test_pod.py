# tests/test_pod.py
import unittest
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.organization.pod import Pod, PodConfig
from libertas.organization.worker import Worker, WorkerConfig
from libertas.organization.federation import Federation
from libertas.economy import Recipe, ProductionStep, StepType, Resource

LLM_MODEL = "ollama/qwen3"

@pytest.mark.unit
class TestPodConfig(unittest.TestCase):
    """Test PodConfig class."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL),
            WorkerConfig(name="w2", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        
        config = PodConfig(
            name="pod_001",
            workers=worker_configs,
            initial_inventory={"wood": 100.0},
            initial_tools=["hammer"]
        )
        
        self.assertEqual(config.name, "pod_001")
        self.assertEqual(len(config.workers), 2)
        self.assertEqual(config.initial_inventory, {"wood": 100.0})
        self.assertEqual(config.initial_tools, ["hammer"])
    
    def test_config_serialization(self):
        """Test config serialization to/from JSON."""
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        config = PodConfig(name="pod_001", workers=worker_configs)

        json_str = config.to_json()
        self.assertIsInstance(json_str, str)

        loaded = PodConfig.from_json(json_str)

    def test_config_to_json_with_filepath(self):
        """Test config serialization to file."""
        import tempfile
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        config = PodConfig(name="pod_001", workers=worker_configs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            result = config.to_json(filepath=filepath)
            self.assertIsNone(result)  # Returns None when writing to file

            # Read back
            loaded = PodConfig.from_json(None, filepath=filepath)
            self.assertEqual(loaded.name, config.name)
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_config_from_json_dict(self):
        """Test config from dict."""
        config_dict = {
            "name": "pod_001",
            "workers": [],
            "initial_inventory": {"wood": 50.0}
        }

        loaded = PodConfig.from_json(config_dict)
        self.assertEqual(loaded.name, "pod_001")
        self.assertEqual(loaded.initial_inventory, {"wood": 50.0})

    def test_config_from_json_invalid_type(self):
        """Test config from invalid data type."""
        with self.assertRaises(ValueError) as cm:
            PodConfig.from_json(123)

        self.assertIn("Unsupported data type", str(cm.exception))

    def test_config_from_json_file(self):
        """Test config from_json_file convenience method."""
        import tempfile
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        config = PodConfig(name="pod_001", workers=worker_configs)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            config.to_json(filepath=filepath)
            loaded = PodConfig.from_json_file(filepath)
            self.assertEqual(loaded.name, config.name)
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)
        self.assertEqual(loaded.name, config.name)
        self.assertEqual(len(loaded.workers), len(config.workers))


@pytest.mark.unit
class TestPodInitialization(unittest.TestCase):
    """Test Pod initialization with tools."""

    def test_pod_with_initial_tools(self):
        """Test pod initialization with initial_tools."""
        federation = Federation(pods=[], seed=42)

        # Register tools
        hammer = Resource("hammer", base_value=10.0, is_tool=True, durability=100)
        saw = Resource("saw", base_value=15.0, is_tool=True, durability=80)
        federation.register_new_resource(hammer)
        federation.register_new_resource(saw)

        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(
            name="tool_pod",
            workers=worker_configs,
            initial_tools=["hammer", "saw"]
        )

        pod = Pod(federation, pod_config, coordinate=(0, 0))

        # Should have tools in inventory
        self.assertGreater(len(pod.inventory.instances.get("hammer", [])), 0)
        self.assertGreater(len(pod.inventory.instances.get("saw", [])), 0)

    def test_pod_with_unregistered_initial_tool(self):
        """Test pod initialization with unregistered tool."""
        federation = Federation(pods=[], seed=42)

        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(
            name="tool_pod",
            workers=worker_configs,
            initial_tools=["nonexistent_tool"]
        )

        pod = Pod(federation, pod_config, coordinate=(0, 0))

        # Should not have the nonexistent tool
        self.assertEqual(len(pod.inventory.instances.get("nonexistent_tool", [])), 0)

    def test_pod_equality_different_type(self):
        """Test pod __eq__ with non-Pod object."""
        federation = Federation(pods=[], seed=42)
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="test_pod", workers=worker_configs)
        pod = Pod(federation, pod_config, coordinate=(0, 0))

        self.assertNotEqual(pod, "not a pod")
        self.assertNotEqual(pod, 123)
        self.assertNotEqual(pod, None)


@pytest.mark.unit
class TestPodProperties(unittest.TestCase):
    """Test Pod property accessors."""

    def setUp(self):
        self.federation = Federation(pods=[], seed=42)
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="test_pod", workers=worker_configs)
        self.pod = Pod(self.federation, pod_config, coordinate=(5, 10))

    def test_position_getter(self):
        """Test pod position property getter."""
        position = self.pod.position
        # Position should be accessible
        self.assertTrue(hasattr(self.pod, 'position'))

    def test_position_setter(self):
        """Test pod position property setter."""
        self.pod.position = (20, 30)
        # Setting position should not raise error
        self.assertTrue(hasattr(self.pod, 'position'))

    def test_start_production_recipe_not_found(self):
        """Test start_production with non-existent recipe."""
        success, message = self.pod.start_production("nonexistent_recipe")

        self.assertFalse(success)
        self.assertIn("not found", message)


@pytest.mark.unit
class TestPod(unittest.TestCase):
    """Test Pod class with real Federation."""

    def setUp(self):
        # Create real federation
        self.federation = Federation(pods=[])
        
        wood = Resource("wood", "system", base_value=1.0)
        # Register wood resource
        self.federation.register_new_resource(wood)
        
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL),
            WorkerConfig(name="w2", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        
        self.pod_config = PodConfig(
            name="pod_001",
            workers=worker_configs,
            initial_inventory={"wood": 100.0}
        )
        
        self.pod = Pod(self.federation, self.pod_config, coordinate=(0, 0))
    
    def test_pod_creation(self):
        """Test basic pod creation."""
        self.assertEqual(self.pod.name, "pod_001")
        self.assertEqual(self.pod.coordinate, (0, 0))
        self.assertEqual(self.pod.num_workers(), 2)
    
    def test_inventory_initialization(self):
        """Test inventory initialization from config."""
        self.assertEqual(self.pod.inventory.get_quantity("wood"), 100.0)
    
    def test_hash_and_eq(self):
        """Test pod hashing and equality."""
        pod1 = self.pod
        pod2_config = PodConfig(name="pod_002", workers=[])
        pod2 = Pod(self.federation, pod2_config, coordinate=(1, 0))
        
        self.assertEqual(hash(pod1), hash(pod1))
        self.assertNotEqual(hash(pod1), hash(pod2))
        self.assertEqual(pod1, pod1)
        self.assertNotEqual(pod1, pod2)
    
    def test_start_production(self):
        """Test starting production jobs."""
        # Create a simple recipe
        step = ProductionStep(name="make_wood", duration=5, step_type=StepType.PROCESSING)
        recipe = Recipe(name="wood_processing", steps=[step])
        self.federation.recipe_registry.register(recipe)
        
        success, result = self.pod.start_production("wood_processing", batch_size=2)
        
        self.assertTrue(success)
        self.assertIsNotNone(result)
        self.assertEqual(len(self.pod.production_queue), 1)
    
    def test_start_production_insufficient_inputs(self):
        """Test starting production with insufficient inputs."""
        step = ProductionStep(
            name="consume_wood",
            duration=5,
            inputs={"wood": 1000.0},
            step_type=StepType.PROCESSING
        )
        recipe = Recipe(name="wood_intensive", steps=[step])
        self.federation.recipe_registry.register(recipe)
        
        success, result = self.pod.start_production("wood_intensive")
        
        self.assertFalse(success)
        self.assertIn("Need 1000.0 wood, have 100.0", result)
    
    def test_get_inventory_summary(self):
        """Test getting inventory summary."""
        summary = self.pod.get_inventory_summary()
        self.assertEqual(summary.get("wood"), 100.0)
    
    def test_transfer_to_pod(self):
        """Test transferring resources between pods."""
        target_config = PodConfig(name="pod_002", workers=[])
        target_pod = Pod(self.federation, target_config, coordinate=(1, 0))
        
        success = self.pod.transfer_to_pod("wood", 30.0, target_pod)
        
        self.assertTrue(success)
        self.assertEqual(self.pod.inventory.get_quantity("wood"), 70.0)
        self.assertEqual(target_pod.inventory.get_quantity("wood"), 30.0)
    
    def test_transfer_insufficient(self):
        """Test transfer with insufficient inventory."""
        target_config = PodConfig(name="pod_002", workers=[])
        target_pod = Pod(self.federation, target_config, coordinate=(1, 0))
        
        success = self.pod.transfer_to_pod("wood", 200.0, target_pod)
        
        self.assertFalse(success)
        self.assertEqual(self.pod.inventory.get_quantity("wood"), 100.0)
    
    def test_worker_management(self):
        """Test adding and removing workers."""
        # Register hammer resource for worker
        tool = Resource("hammer", "system", base_value=10.0, is_tool=True)
        self.federation.register_new_resource(tool)
        
        worker = Worker(
            self.federation,
            WorkerConfig(name="w3", reasoning=Mock, llm_model=LLM_MODEL),
            coordinate=(0, 0)
        )
        
        initial_count = self.pod.num_workers()
        self.pod.add_worker(worker)
        self.assertEqual(self.pod.num_workers(), initial_count + 1)
        
        self.pod.remove_worker(worker)
        self.assertEqual(self.pod.num_workers(), initial_count)
    
    def test_get_worker_by_name(self):
        """Test retrieving workers by ID."""
        first_worker = list(self.pod)[0]
        worker = self.pod.get_worker_by_name(first_worker.name)
        self.assertIsNotNone(worker)
        if worker is not None:
            self.assertEqual(worker.name, "w1")
        
        self.assertIsNone(self.pod.get_worker_by_name("nonexistent"))
    
    def test_worker_network_fully_connected(self):
        """Test that worker network is fully connected."""
        self.assertTrue(self.pod.is_fully_connected())
        
        # Register hammer resource for worker
        tool = Resource("hammer", "system", base_value=10.0, is_tool=True)
        self.federation.register_new_resource(tool)
        
        # Add a new worker
        worker = Worker(
            self.federation,
            WorkerConfig(name="w3", reasoning=Mock, llm_model=LLM_MODEL),
            coordinate=(0, 0)
        )
        self.pod.add_worker(worker)
        
        # Should still be fully connected
        self.assertTrue(self.pod.is_fully_connected())
    
    def test_get_worker_neighbors(self):
        """Test getting worker neighbors."""
        worker = self.pod.get_worker_by_name("w1")
        neighbors = []
        if worker is not None:
            neighbors = self.pod.get_worker_neighbors(worker)
        
        # In a complete graph with 2 workers, each has 1 neighbor
        self.assertEqual(len(neighbors), 1)
    
    def test_set_worker_layout(self):
        """Test changing worker layout."""
        # Should not raise errors
        self.pod.set_worker_layout("circular")
        self.pod.set_worker_layout("spring")
        
        with self.assertRaises(ValueError):
            self.pod.set_worker_layout("invalid_layout")
    
    def test_pod_serialization(self):
        """Test pod serialization to JSON."""
        json_str = self.pod.to_json()
        self.assertIsInstance(json_str, str)
        if json_str is not None:
            self.assertIn("pod_001", json_str)


@pytest.mark.unit
class TestPodEdgeCases(unittest.TestCase):
    """Test edge cases for Pod class."""
    
    def setUp(self):
        self.federation = Federation(pods=[])
    
    def test_empty_pod(self):
        """Test pod with no workers."""
        config = PodConfig(name="empty_pod", workers=[])
        pod = Pod(self.federation, config, coordinate=(0, 0))
        
        self.assertEqual(pod.num_workers(), 0)
        self.assertTrue(pod.is_fully_connected())
        self.assertEqual(len(pod.get_worker_network().nodes), 0)
    
    def test_pod_single_worker(self):
        """Test pod with single worker."""
        worker_config = WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        config = PodConfig(name="single_pod", workers=[worker_config])
        pod = Pod(self.federation, config, coordinate=(0, 0))
        
        self.assertEqual(pod.num_workers(), 1)
        self.assertTrue(pod.is_fully_connected())
    
    def test_pod_coordinate_update(self):
        """Test updating pod coordinates."""
        config = PodConfig(name="coord_pod", workers=[])
        pod = Pod(self.federation, config, coordinate=(0, 0))
        
        pod.coordinate = (5, 10)
        self.assertEqual(pod.coordinate, (5, 10))
    
    def test_transfer_nonexistent_resource(self):
        """Test transferring non-existent resource."""
        config1 = PodConfig(name="pod1", workers=[])
        config2 = PodConfig(name="pod2", workers=[])
        pod1 = Pod(self.federation, config1, coordinate=(0, 0))
        pod2 = Pod(self.federation, config2, coordinate=(1, 0))
        
        success = pod1.transfer_to_pod("nonexistent", 10.0, pod2)
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()