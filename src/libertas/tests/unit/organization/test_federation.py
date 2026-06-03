# tests/test_federation.py
import unittest
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.organization.federation import Federation
from libertas.organization.pod import PodConfig
from libertas.organization.worker import WorkerConfig
from libertas.economy import Recipe, ProductionStep, Resource, StepType

LLM_MODEL = "ollama/functiongemma"

@pytest.mark.unit
class TestFederation(unittest.TestCase):
    """Test Federation class with real objects."""
    
    def setUp(self):
        # Create worker configs
        self.worker_configs = [
            WorkerConfig(name=f"w{i}", reasoning=Mock, llm_model=LLM_MODEL)
            for i in range(3)
        ]
        
        # Create pod configs - use 'name' not 'unique_id'
        self.pod_configs = [
            PodConfig(
                name=f"pod_{i}",
                workers=self.worker_configs,
                initial_inventory={"wood": 100.0}
            )
            for i in range(2)
        ]
        
        self.federation = Federation(pods=self.pod_configs, seed=42)
        
        wood = Resource("wood", base_value=1.0)

        test_resource = Resource("test_resource", base_value=5.0)

        # Register resources for testing
        self.federation.register_new_resource(wood)
        self.federation.register_new_resource(test_resource)
    
    def test_federation_creation(self):
        """Test basic federation creation."""
        self.assertEqual(len(self.federation), 2)
        self.assertEqual(self.federation.steps, 0)
        
        # Check that pods exist (IDs will be numeric from Mesa)
        pod_ids = [pod.name for pod in self.federation]
        self.assertEqual(len(pod_ids), 2)
    
    def test_get_item_by_index(self):
        """Test accessing pod by index."""
        pod = self.federation[0]
        self.assertIsNotNone(pod)
        
        with self.assertRaises(IndexError):
            _ = self.federation[10]
    
    def test_get_item_by_id(self):
        """Test accessing pod by name (string)."""
        first_pod = self.federation[0]
        # Use pod.name for string lookup
        pod = self.federation[first_pod.name]
        self.assertEqual(pod.name, first_pod.name)
        
        with self.assertRaises(KeyError):
            _ = self.federation["nonexistent"]
    
    def test_iteration(self):
        """Test iterating over pods."""
        pods = list(self.federation)
        self.assertEqual(len(pods), 2)
    
    def test_contains(self):
        """Test membership test."""
        pod = self.federation[0]
        self.assertIn(pod, self.federation)
    
    def test_add_pod(self):
        """Test adding a new pod."""
        from libertas.organization.pod import Pod
        
        new_worker_config = WorkerConfig(name="w_new", reasoning=Mock, llm_model=LLM_MODEL)
        new_pod_config = PodConfig(
            name="pod_new",
            workers=[new_worker_config],
            initial_inventory={"wood": 50.0}
        )
        new_pod = Pod(self.federation, new_pod_config, coordinate=(2, 0))
        
        initial_count = len(self.federation)
        self.federation.add(new_pod)
        
        self.assertEqual(len(self.federation), initial_count + 1)
        self.assertIn(new_pod, self.federation)
    
    def test_discard_pod(self):
        """Test removing a pod."""
        pod = self.federation[0]
        initial_count = len(self.federation)
        
        self.federation.discard(pod)
        
        self.assertEqual(len(self.federation), initial_count - 1)
        self.assertNotIn(pod, self.federation)
    
    def test_remove_pod(self):
        """Test removing a pod (raises KeyError if not found)."""
        pod = self.federation[0]
        self.federation.remove(pod)
        self.assertNotIn(pod, self.federation)
        
        with self.assertRaises(KeyError):
            self.federation.remove(pod)
    
    def test_get_pod_by_id(self):
        """Test getting pod by name."""
        first_pod = self.federation[0]
        pod = self.federation.get_pod_by_name(first_pod.name)
        if pod is not None:
            self.assertEqual(pod.name, first_pod.name)
        
        self.assertIsNone(self.federation.get_pod_by_name("nonexistent"))
    
    def test_get_pod_by_index(self):
        """Test getting pod by index."""
        pod = self.federation.get_pod_by_index(0)
        self.assertIsNotNone(pod)
        
        self.assertIsNone(self.federation.get_pod_by_index(10))
    
    def test_get_neighbors(self):
        """Test getting neighboring pods."""
        pod = self.federation[0]
        neighbors = self.federation.get_neighbors(pod)
        
        # Complete graph with 2 pods: each has 1 neighbor
        self.assertEqual(len(neighbors), 1)
    
    def test_get_pod_network(self):
        """Test getting pod network graph."""
        graph = self.federation.get_pod_network()
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)
    
    def test_get_resource(self):
        """Test getting resource by name."""
        resource = self.federation.get_resource("wood")
        self.assertIsNotNone(resource)
        if resource is not None:
            self.assertEqual(resource.name, "wood")
        
        self.assertIsNone(self.federation.get_resource("nonexistent"))
    
    def test_get_recipe(self):
        """Test getting recipe by name."""
        # Register a test recipe
        step = ProductionStep(name="test_step", duration=5, step_type=StepType.PROCESSING)
        recipe = Recipe(name="test_recipe", steps=[step])
        self.federation.recipe_registry.register(recipe)
        
        result = self.federation.get_recipe("test_recipe")
        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(result.name, "test_recipe")
        
    def test_register_new_resource(self):
        """Test inventing new resource."""
        resource = Resource(
            name="mithril",
            invented_by="worker_001",
            base_value=100.0
        )
        self.federation.register_new_resource(resource)
        
        self.assertEqual(resource.name, "mithril")
        self.assertEqual(resource.invented_by, "worker_001")
        self.assertIn("mithril", self.federation.list_resources())
    
    def test_register_new_recipe(self):
        """Test inventing new recipe."""
        steps = [ProductionStep(name="step1", duration=5, step_type=StepType.PROCESSING)]
        
        recipe = self.federation.register_new_recipe(
            name="mithril_forging",
            steps=steps,
            inventor_id="worker_001"
        )
        
        self.assertEqual(recipe.name, "mithril_forging")
        self.assertEqual(recipe.invented_by, "worker_001")
        self.assertIn("mithril_forging", self.federation.list_recipes())
    
    def test_list_resources(self):
        """Test listing all resources."""
        resources = self.federation.list_resources()
        self.assertIn("wood", resources)
        self.assertIn("test_resource", resources)
    
    def test_list_recipes(self):
        """Test listing all recipes."""
        # Register a test recipe
        step = ProductionStep(name="test_step", duration=5, step_type=StepType.PROCESSING)
        recipe = Recipe(name="test_recipe", steps=[step])
        self.federation.recipe_registry.register(recipe)
        
        recipes = self.federation.list_recipes()
        self.assertIn("test_recipe", recipes)
    
    def test_step(self):
        """Test stepping the federation."""
        initial_step = self.federation.steps
        
        self.federation.step()
        
        self.assertEqual(self.federation.steps, initial_step + 1)
    
    def test_to_list(self):
        """Test converting to list."""
        pod_list = self.federation.to_list()
        self.assertEqual(len(pod_list), 2)
        self.assertIsInstance(pod_list, list)
    
    def test_set_pod_layout(self):
        """Test changing pod layout."""
        # Should not raise errors
        self.federation.set_pod_layout("circular")
        self.federation.set_pod_layout("spring")
        
        with self.assertRaises(ValueError):
            self.federation.set_pod_layout("invalid")
    
    def test_get_economic_summary(self):
        """Test getting economic summary."""
        summary = self.federation.get_economic_summary()
        
        self.assertEqual(summary["step"], 0)
        self.assertEqual(summary["num_pods"], 2)
        self.assertEqual(summary["num_workers"], 6)  # 2 pods * 3 workers
        self.assertIn("known_resources", summary)
        self.assertIn("known_recipes", summary)


@pytest.mark.unit
class TestFederationWithCustomRegistries(unittest.TestCase):
    """Test Federation with custom resource and recipe registries."""
    
    def setUp(self):
        from libertas.economy import ResourceRegistry, RecipeRegistry
        
        self.custom_resource_registry = ResourceRegistry()
        self.custom_recipe_registry = RecipeRegistry()
        
        # Add custom resources
        self.custom_resource_registry.register(Resource(name="custom_ore", base_value=5.0))
        
        # Add custom recipe
        step = ProductionStep(name="custom_step", duration=5, step_type=StepType.PROCESSING)
        self.custom_recipe_registry.register(Recipe(name="custom_smelt", steps=[step]))
        
        self.pod_configs = [
            PodConfig(name="pod_0", workers=[]),
            PodConfig(name="pod_1", workers=[])
        ]
        
        self.federation = Federation(
            pods=self.pod_configs,
            resource_registry=self.custom_resource_registry,
            recipe_registry=self.custom_recipe_registry
        )
    
    def test_custom_registries_used(self):
        """Test that custom registries are used instead of defaults."""
        resources = self.federation.list_resources()
        self.assertIn("custom_ore", resources)
        
        recipes = self.federation.list_recipes()
        self.assertIn("custom_smelt", recipes)


@pytest.mark.unit
class TestFederationEdgeCases(unittest.TestCase):
    """Test edge cases for Federation class."""
    
    def setUp(self):
        self.empty_federation = Federation(pods=[])
        
        # Create a federation with one pod for single pod tests
        worker_config = WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)
        pod_config = PodConfig(name="single_pod", workers=[worker_config])
        self.single_pod_federation = Federation(pods=[pod_config], seed=42)
    
    def test_empty_federation(self):
        """Test federation with no pods."""
        self.assertEqual(len(self.empty_federation), 0)
        self.assertEqual(self.empty_federation.get_pod_network().number_of_nodes(), 0)
    
    def test_single_pod_federation(self):
        """Test federation with single pod."""
        self.assertEqual(len(self.single_pod_federation), 1)
        
        # Single pod should have no neighbors
        pod = self.single_pod_federation[0]
        neighbors = self.single_pod_federation.get_neighbors(pod)
        self.assertEqual(len(neighbors), 0)
    
    def test_step_with_no_pods(self):
        """Test stepping federation with no pods."""
        initial_step = self.empty_federation.steps
        
        self.empty_federation.step()
        
        self.assertEqual(self.empty_federation.steps, initial_step + 1)
    
    def test_get_neighbors_nonexistent_pod(self):
        """Test getting neighbors for non-existent pod."""
        from libertas.organization.pod import PodConfig, Pod
        
        # Create a pod not in federation
        fake_config = PodConfig(name="fake_pod", workers=[])
        fake_pod = Pod(self.empty_federation, fake_config, (0, 0))
        
        neighbors = self.single_pod_federation.get_neighbors(fake_pod)
        self.assertEqual(neighbors, [])
    
    def test_remove_nonexistent_pod(self):
        """Test removing pod not in federation."""
        from libertas.organization.pod import PodConfig, Pod
        
        fake_config = PodConfig(name="fake_pod", workers=[])
        fake_pod = Pod(self.empty_federation, fake_config, (0, 0))
        
        with self.assertRaises(KeyError):
            self.single_pod_federation.remove(fake_pod)


if __name__ == '__main__':
    unittest.main()