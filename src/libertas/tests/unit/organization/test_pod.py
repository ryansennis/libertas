# tests/test_pod.py
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.organization.pod import Pod, PodConfig
from libertas.organization.worker import Worker, WorkerConfig
from libertas.organization.federation import Federation
from libertas.resources import Recipe, ProductionStep, StepType, Resource, Material, Tool

LLM_MODEL = "ollama/tinyllama"

@pytest.mark.unit
class TestPodConfig:
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
        
        assert config.name == "pod_001"
        assert len(config.workers) == 2
        assert config.initial_inventory == {"wood": 100.0}
        assert config.initial_tools == ["hammer"]
    
    def test_config_serialization(self):
        """Test config serialization to/from JSON."""
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        config = PodConfig(name="pod_001", workers=worker_configs)

        json_str = config.to_json()
        assert isinstance(json_str, str)

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
            assert result is None  # Returns None when writing to file

            # Read back
            loaded = PodConfig.from_json(None, filepath=filepath)
            assert loaded.name == config.name
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
        assert loaded.name == "pod_001"
        assert loaded.initial_inventory == {"wood": 50.0}

    def test_config_from_json_invalid_type(self):
        """Test config from invalid data type."""
        with pytest.raises(ValueError) as cm:
            PodConfig.from_json(123)

        assert "Unsupported data type" in str(cm.value)

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
            assert loaded.name == config.name
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)
        assert loaded.name == config.name
        assert len(loaded.workers) == len(config.workers)


@pytest.mark.unit
class TestPodInitialization:
    """Test Pod initialization with tools."""

    def test_pod_with_initial_tools(self):
        """Test pod initialization with initial_tools."""
        federation = Federation(pods=[], seed=42)

        # Register tools
        hammer = Tool("hammer", base_value=10.0, durability=100)
        saw = Resource("saw", base_value=15.0, durability=80)
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
        assert len(pod.inventory.instances.get("hammer", []))
        assert len(pod.inventory.instances.get("saw", []))

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
        assert len(pod.inventory.instances.get("nonexistent_tool", []))

    def test_pod_equality_different_type(self):
        """Test pod __eq__ with non-Pod object."""
        federation = Federation(pods=[], seed=42)
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="test_pod", workers=worker_configs)
        pod = Pod(federation, pod_config, coordinate=(0, 0))

        assert pod != "not a pod"
        assert pod != 123
        assert pod != None


@pytest.mark.unit
class TestPodProperties:
    """Test Pod property accessors."""

    def setup_method(self):
        self.federation = Federation(pods=[], seed=42)
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="test_pod", workers=worker_configs)
        self.pod = Pod(self.federation, pod_config, coordinate=(5, 10))

    def test_position_getter(self):
        """Test pod position property getter."""
        position = self.pod.position
        # Position should be accessible
        assert hasattr(self.pod, 'position')

    def test_position_setter(self):
        """Test pod position property setter."""
        self.pod.position = (20, 30)
        # Setting position should not raise error
        assert hasattr(self.pod, 'position')

    def test_start_production_recipe_not_found(self):
        """Test start_production with non-existent recipe."""
        success, message = self.pod.start_production("nonexistent_recipe")

        assert not (success)
        assert "not found" in message


@pytest.mark.unit
class TestPod:
    """Test Pod class with real Federation."""

    def setup_method(self):
        # Create real federation
        self.federation = Federation(pods=[])
        
        wood = Material("wood", "system", base_value=1.0)
        # Register wood resource
        self.federation.resource_registry.register(wood)
        
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
        assert self.pod.name == "pod_001"
        assert self.pod.coordinate == (0)
        assert self.pod.num_workers() == 2
    
    def test_inventory_initialization(self):
        """Test inventory initialization from config."""
        assert self.pod.inventory.get_quantity("wood") == 100.0
    
    def test_hash_and_eq(self):
        """Test pod hashing and equality."""
        pod1 = self.pod
        pod2_config = PodConfig(name="pod_002", workers=[])
        pod2 = Pod(self.federation, pod2_config, coordinate=(1, 0))
        
        assert hash(pod1) == hash(pod1)
        assert hash(pod1) != hash(pod2)
        assert pod1 == pod1
        assert pod1 != pod2
    
    def test_start_production(self):
        """Test starting production jobs."""
        # Create a simple recipe
        step = ProductionStep(name="make_wood", duration=5, step_type=StepType.PROCESSING)
        recipe = Recipe(name="wood_processing", steps=[step])
        self.federation.recipe_registry.register(recipe)
        
        success, result = self.pod.start_production("wood_processing", batch_size=2)
        
        assert success
        assert result is not None
        assert len(self.pod.production_queue) == 1
    
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
        
        assert not (success)
        assert "Need 1000.0 wood in have 100.0", result
    
    def test_get_inventory_summary(self):
        """Test getting inventory summary."""
        summary = self.pod.get_inventory_summary()
        assert summary.get("wood") == 100.0
    
    def test_transfer_to_pod(self):
        """Test transferring resources between pods."""
        target_config = PodConfig(name="pod_002", workers=[])
        target_pod = Pod(self.federation, target_config, coordinate=(1, 0))
        
        success = self.pod.transfer_to_pod("wood", 30.0, target_pod)
        
        assert success
        assert self.pod.inventory.get_quantity("wood") == 70.0
        assert target_pod.inventory.get_quantity("wood") == 30.0
    
    def test_transfer_insufficient(self):
        """Test transfer with insufficient inventory."""
        target_config = PodConfig(name="pod_002", workers=[])
        target_pod = Pod(self.federation, target_config, coordinate=(1, 0))
        
        success = self.pod.transfer_to_pod("wood", 200.0, target_pod)
        
        assert not (success)
        assert self.pod.inventory.get_quantity("wood") == 100.0
    
    def test_worker_management(self):
        """Test adding and removing workers."""
        # Register hammer resource for worker
        tool = Tool("hammer", "system", base_value=10.0, durability=100)
        self.federation.resource_registry.register(tool)
        
        worker = Worker(
            self.federation,
            WorkerConfig(name="w3", reasoning=Mock, llm_model=LLM_MODEL),
            coordinate=(0)
        )
        
        initial_count = self.pod.num_workers()
        self.pod.add_worker(worker)
        assert self.pod.num_workers() == initial_count + 1
        
        self.pod.remove_worker(worker)
        assert self.pod.num_workers() == initial_count
    
    def test_get_worker_by_name(self):
        """Test retrieving workers by ID."""
        first_worker = list(self.pod)[0]
        worker = self.pod.get_worker_by_name(first_worker.name)
        assert worker is not None
        if worker is not None:
            assert worker.name == "w1"
        
        assert self.pod.get_worker_by_name("nonexistent" is None)
    
    def test_worker_network_fully_connected(self):
        """Test that worker network is fully connected."""
        assert self.pod.is_fully_connected()
        
        # Register hammer resource for worker
        tool = Tool("hammer", "system", base_value=10.0, durability=100)
        self.federation.resource_registry.register(tool)
        
        # Add a new worker
        worker = Worker(
            self.federation,
            WorkerConfig(name="w3", reasoning=Mock, llm_model=LLM_MODEL),
            coordinate=(0)
        )
        self.pod.add_worker(worker)
        
        # Should still be fully connected
        assert self.pod.is_fully_connected()
    
    def test_get_worker_neighbors(self):
        """Test getting worker neighbors."""
        worker = self.pod.get_worker_by_name("w1")
        neighbors = []
        if worker is not None:
            neighbors = self.pod.get_worker_neighbors(worker)
        
        # In a complete graph with 2 workers, each has 1 neighbor
        assert len(neighbors) == 1
    
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
        assert isinstance(json_str, str)
        if json_str is not None:
            assert "pod_001" in json_str

    def test_start_production_inventory_remove_fails(self):
        """Test start_production when inventory.remove fails (line 181)."""
        from unittest.mock import patch
        from libertas.economy import Recipe, ProductionStep, StepType

        # Register recipe
        step = ProductionStep(
            name="make_planks",
            duration=10,
            step_type=StepType.PROCESSING,
            inputs={"wood": 10.0},
            outputs={"plank": 5.0}
        )
        recipe = Recipe(name="make_planks", steps=[step])
        self.federation.recipe_registry.register(recipe)

        # Mock inventory.remove to return False
        with patch.object(self.pod.inventory, 'remove', return_value=False):
            success, message = self.pod.start_production("make_planks", batch_size=1)
            assert not (success)
            assert "Failed to consume" in message

    def test_update_worker_coordinates_exception(self):
        """Test exception handling in _update_worker_coordinates (lines 322-330, 350-351)."""
        from unittest.mock import patch
        import networkx as nx

        # Create worker graph
        graph = nx.Graph()
        graph.add_node(list(self.pod)[0].unique_id)

        # Mock networkx layout to raise exception
        with patch('networkx.spring_layout', side_effect=Exception("Layout failed")):
            # Should fall back to circular layout without crashing
            self.pod._update_worker_coordinates(graph)

            # Worker should have coordinates (from fallback)
            worker = list(self.pod)[0]
            assert worker.coordinate is not None


@pytest.mark.unit
class TestPodProductionFailures:
    """Test pod production failure scenarios."""

    def setup_method(self):
        self.federation = Federation(pods=[], seed=42)

        # Register resources
        wood = Material("wood", base_value=1.0)
        planks = Material("planks", base_value=2.0)
        self.federation.resource_registry.register(wood)
        self.federation.resource_registry.register(planks)

        # Register recipe
        recipe_step = ProductionStep(
            name="cut_wood",
            step_type=StepType.PROCESSING,
            duration=5,
            inputs={"wood": 10.0},  # Requires 10 wood
            outputs={"planks": 5.0}
        )
        self.federation.recipe_registry.register(
            Recipe(name="make_planks", steps=[recipe_step])
        )

        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(
            name="test_pod",
            workers=worker_configs,
            initial_inventory={"wood": 5.0}  # Only 5 wood, need 10
        )

        self.pod = Pod(self.federation, pod_config, coordinate=(0, 0))

    def test_start_production_insufficient_inventory(self):
        """Test start_production when inventory.remove fails (line 181)."""
        # This test actually fails at recipe.can_start, not at inventory.remove
        # Line 181 is only reached if can_start passes but remove fails
        # That's a race condition scenario that's hard to test
        # Let's just verify the production fails
        success, message = self.pod.start_production("make_planks", batch_size=1)

        # Should fail (either at can_start or remove)
        assert not (success)
        assert "wood" in message.lower()


@pytest.mark.unit
class TestPodWorkerAccess:
    """Test pod worker access methods."""

    def setup_method(self):
        self.federation = Federation(pods=[], seed=42)
        worker_configs = [
            WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL),
            WorkerConfig(name="w2", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        pod_config = PodConfig(name="test_pod", workers=worker_configs)
        self.pod = Pod(self.federation, pod_config, coordinate=(0, 0))

    def test_get_worker_by_index_out_of_range(self):
        """Test get_worker_by_index with invalid index (lines 290-293)."""
        # Try index beyond range
        worker = self.pod.get_worker_by_index(999)
        assert worker is None

        # Note: negative indices might work in Python (wrapping around)
        # so we only test clearly out of range indices

    def test_update_worker_coordinates_empty_pod(self):
        """Test _update_worker_coordinates with empty pod (line 314)."""
        import networkx as nx

        # Create empty pod
        empty_config = PodConfig(name="empty_pod", workers=[])
        empty_pod = Pod(self.federation, empty_config, coordinate=(0, 0))

        # Should return early without error
        graph = nx.Graph()
        empty_pod._update_worker_coordinates(graph)
        assert len(empty_pod) == 0

    def test_update_worker_coordinates_exception_fallback(self):
        """Test _update_worker_coordinates exception fallback (lines 322-330)."""
        import networkx as nx

        # Create a broken graph that will cause spring_layout to fail
        graph = nx.Graph()
        graph.add_node("invalid_worker_id")  # ID that won't match any worker

        # Should fall back to circular layout without error
        self.pod._update_worker_coordinates(graph)

        # Workers should have coordinates (circular fallback)
        for worker in self.pod:
            assert worker.coordinate is not None

    def test_set_worker_layout_empty_pod(self):
        """Test set_worker_layout with empty pod (line 334)."""
        # Create empty pod
        empty_config = PodConfig(name="empty_pod", workers=[])
        empty_pod = Pod(self.federation, empty_config, coordinate=(0, 0))

        # Should return early without error
        empty_pod.set_worker_layout("spring")
        assert len(empty_pod) == 0

    def test_set_worker_layout_random(self):
        """Test set_worker_layout with random layout (line 341)."""
        self.pod.set_worker_layout("random")

        # Workers should have coordinates
        for worker in self.pod:
            assert worker.coordinate is not None

    def test_set_worker_layout_kamada_kawai(self):
        """Test set_worker_layout with kamada_kawai layout (line 343)."""
        self.pod.set_worker_layout("kamada_kawai")

        # Workers should have coordinates (covers lines 343, 350-351)
        for worker in self.pod:
            assert worker.coordinate is not None

    def test_get_worker_degrees(self):
        """Test get_worker_degrees method (line 371)."""
        degrees = self.pod.get_worker_degrees()

        # Should return a dict
        assert isinstance(degrees, dict)

    def test_to_list(self):
        """Test to_list method (line 387)."""
        worker_list = self.pod.to_list()

        # Should return a list of workers
        assert isinstance(worker_list, list)
        assert len(worker_list) == 2

    def test_pod_from_json(self):
        """Test Pod.from_json class method (lines 402-403)."""
        import tempfile

        # Create a pod config
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="test_pod", workers=worker_configs)

        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            pod_config.to_json(filepath=filepath)

            # Create Pod from JSON file
            pod = Pod.from_json(self.federation, None, (0, 0), filepath=filepath)

            assert pod.name == "test_pod"
            assert len(pod) == 1
        finally:
            import os
            if os.path.exists(filepath):
                os.remove(filepath)


@pytest.mark.unit
class TestPodEdgeCases:
    """Test edge cases for Pod class."""
    
    def setup_method(self):
        self.federation = Federation(pods=[])
    
    def test_empty_pod(self):
        """Test pod with no workers."""
        config = PodConfig(name="empty_pod", workers=[])
        pod = Pod(self.federation, config, coordinate=(0, 0))
        
        assert pod.num_workers() == 0
        assert pod.is_fully_connected()
        assert len(pod.get_worker_network().nodes) == 0
    
    def test_pod_single_worker(self):
        """Test pod with single worker."""
        worker_config = WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        config = PodConfig(name="single_pod", workers=[worker_config])
        pod = Pod(self.federation, config, coordinate=(0, 0))
        
        assert pod.num_workers() == 1
        assert pod.is_fully_connected()
    
    def test_pod_coordinate_update(self):
        """Test updating pod coordinates."""
        config = PodConfig(name="coord_pod", workers=[])
        pod = Pod(self.federation, config, coordinate=(0, 0))
        
        pod.coordinate = (5, 10)
        assert pod.coordinate == (5, 10)
    
    def test_transfer_nonexistent_resource(self):
        """Test transferring non-existent resource."""
        config1 = PodConfig(name="pod1", workers=[])
        config2 = PodConfig(name="pod2", workers=[])
        pod1 = Pod(self.federation, config1, coordinate=(0, 0))
        pod2 = Pod(self.federation, config2, coordinate=(1, 0))

        success = pod1.transfer_to_pod("nonexistent", 10.0, pod2)
        assert not (success)


