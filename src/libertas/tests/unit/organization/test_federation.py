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
from libertas.resources import Recipe, ProductionStep, Resource, StepType

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

    def test_discard_pod_rebuilds_graph(self):
        """Test discard() rebuilds graph with complete edges (lines 214-216)."""
        worker_configs = [WorkerConfig(name=f"w{i}", reasoning=Mock, llm_model=LLM_MODEL) for i in range(3)]
        pod_configs = [
            PodConfig(name="pod_0", workers=worker_configs),
            PodConfig(name="pod_1", workers=worker_configs),
            PodConfig(name="pod_2", workers=worker_configs)
        ]
        fed = Federation(pods=pod_configs, seed=42)

        # Remove one pod
        pod_to_remove = fed[0]
        fed.discard(pod_to_remove)

        # Should have complete graph with remaining 2 pods
        graph = fed.get_pod_network()
        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.number_of_edges(), 1)  # Complete graph with 2 nodes has 1 edge

    def test_economic_summary_counts_tools(self):
        """Test get_economic_summary counts tools from pods (line 292)."""
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs, initial_inventory={"hammer": 3.0})
        fed = Federation(pods=[pod_config], seed=42)

        # Register hammer as tool
        hammer = Resource("hammer", base_value=10.0, is_tool=True, durability=100)
        fed.register_new_resource(hammer)

        # Add tools to pod inventory
        pod = fed[0]
        pod.inventory.add(hammer, 3)

        summary = fed.get_economic_summary()

        # Should count tools
        self.assertIn("total_tools", summary)
        self.assertIn("hammer", summary["total_tools"])

    def test_step_with_market_transactions(self):
        """Test step() processes market transactions (lines 317-323)."""
        from unittest.mock import patch

        worker_configs = [
            WorkerConfig(name="buyer", reasoning=Mock, llm_model=LLM_MODEL),
            WorkerConfig(name="seller", reasoning=Mock, llm_model=LLM_MODEL)
        ]
        pod_config = PodConfig(name="pod_0", workers=worker_configs, initial_inventory={"wood": 100.0})
        fed = Federation(pods=[pod_config], seed=42, initialize_market=True)

        # Get workers by name
        buyer = fed._find_worker_by_name("buyer")
        seller = fed._find_worker_by_name("seller")

        # Give buyer currency
        buyer.add_currency(1000.0)
        initial_buyer_currency = buyer.currency
        initial_seller_currency = seller.currency

        # Mock market to return transactions
        mock_transactions = [{
            'buyer_worker': "buyer",
            'seller_worker': "seller",
            'total_value': 100.0
        }]

        with patch.object(fed.market, 'process_market', return_value=mock_transactions):
            fed.step()

        # Currency should be transferred (lines 317-323)
        self.assertEqual(buyer.currency, initial_buyer_currency - 100.0)
        self.assertEqual(seller.currency, initial_seller_currency + 100.0)


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

    def test_register_new_resource_no_registry(self):
        """Test register_new_resource when resource_registry is None."""
        # Manually set resource_registry to None after construction
        fed = Federation(pods=[], seed=42)
        fed.resource_registry = None

        # Try to register resource - should return False
        resource = Resource("test", base_value=10.0)
        result = fed.register_new_resource(resource)
        self.assertFalse(result)

    def test_update_pod_coordinates_empty_federation(self):
        """Test _update_pod_coordinates with empty federation."""
        import networkx as nx
        graph = nx.Graph()

        # Should not raise error
        self.empty_federation._update_pod_coordinates(graph)

    def test_set_pod_layout_circular_with_single_pod(self):
        """Test set_pod_layout with circular layout and single pod."""
        # Set circular layout
        self.single_pod_federation.set_pod_layout("circular")

        # Should not raise error
        self.assertEqual(len(self.single_pod_federation), 1)

    def test_set_pod_layout_grid_with_no_pods(self):
        """Test set_pod_layout with grid layout and no pods."""
        # Should not raise error
        self.empty_federation.set_pod_layout("grid")
        self.assertEqual(len(self.empty_federation), 0)

    def test_set_pod_layout_spring(self):
        """Test set_pod_layout with spring layout."""
        # Set spring layout
        self.single_pod_federation.set_pod_layout("spring")

        # Should not raise error
        self.assertEqual(len(self.single_pod_federation), 1)

    def test_set_pod_layout_random(self):
        """Test set_pod_layout with random layout."""
        # Set random layout
        self.single_pod_federation.set_pod_layout("random")

        # Should not raise error
        self.assertEqual(len(self.single_pod_federation), 1)

    def test_set_pod_layout_kamada_kawai(self):
        """Test set_pod_layout with kamada_kawai layout."""
        # Create federation with 2+ pods (kamada_kawai needs connected graph)
        worker_configs = [WorkerConfig(name=f"w{i}", reasoning=Mock, llm_model=LLM_MODEL) for i in range(2)]
        pod_configs = [
            PodConfig(name="pod_0", workers=worker_configs),
            PodConfig(name="pod_1", workers=worker_configs)
        ]
        fed = Federation(pods=pod_configs, seed=42)

        # Set kamada_kawai layout
        fed.set_pod_layout("kamada_kawai")

        # Should not raise error
        self.assertEqual(len(fed), 2)

    def test_get_economic_summary_with_tools(self):
        """Test get_economic_summary includes tool inventory."""
        # Add tools to pod inventory
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs, initial_inventory={"hammer": 3.0})
        fed = Federation(pods=[pod_config], seed=42)

        # Register hammer as a tool
        hammer = Resource("hammer", base_value=10.0, is_tool=True, durability=100)
        fed.register_new_resource(hammer)

        summary = fed.get_economic_summary()

        # Should include tools section
        self.assertIn("total_tools", summary)

    def test_set_pod_layout_invalid(self):
        """Test set_pod_layout with invalid layout type."""
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs)
        fed = Federation(pods=[pod_config], seed=42)

        # Should raise ValueError for unknown layout
        with self.assertRaises(ValueError) as cm:
            fed.set_pod_layout("invalid_layout")

        self.assertIn("Unknown layout type", str(cm.exception))

    def test_getitem_invalid_type(self):
        """Test __getitem__ with invalid key type."""
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs)
        fed = Federation(pods=[pod_config], seed=42)

        # Should raise TypeError for invalid key type
        with self.assertRaises(TypeError) as cm:
            _ = fed[1.5]  # float key

        self.assertIn("Invalid key type", str(cm.exception))

    def test_get_neighbors_no_graph(self):
        """Test get_neighbors when _pod_graph is None."""
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs)
        fed = Federation(pods=[pod_config], seed=42)

        # Manually set _pod_graph to None
        fed._pod_graph = None

        pod = list(fed)[0]
        neighbors = fed.get_neighbors(pod)

        # Should return empty list
        self.assertEqual(neighbors, [])

    def test_discard_with_graph_rebuild(self):
        """Test discard removes pod and rebuilds graph."""
        worker_configs = [WorkerConfig(name=f"w{i}", reasoning=Mock, llm_model=LLM_MODEL) for i in range(2)]
        pod_configs = [
            PodConfig(name="pod_0", workers=worker_configs),
            PodConfig(name="pod_1", workers=worker_configs)
        ]
        fed = Federation(pods=pod_configs, seed=42)

        # Get first pod
        pod = list(fed)[0]

        # Discard it
        fed.discard(pod)

        # Should have one pod left
        self.assertEqual(len(fed), 1)
        self.assertNotIn(pod, fed)

    def test_discard_nonexistent_pod(self):
        """Test discard with pod not in federation (should not raise)."""
        from libertas.organization.pod import Pod

        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs)
        fed = Federation(pods=[pod_config], seed=42)

        # Create fake pod
        fake_config = PodConfig(name="fake_pod", workers=[])
        fake_pod = Pod(fed, fake_config, (0, 0))

        # Discard should not raise error
        fed.discard(fake_pod)
        self.assertEqual(len(fed), 1)

    def test_get_economic_summary_with_pod_tools(self):
        """Test get_economic_summary includes tools from pods."""
        worker_configs = [WorkerConfig(name="w1", reasoning=Mock, llm_model=LLM_MODEL)]
        pod_config = PodConfig(name="pod_0", workers=worker_configs, initial_inventory={"wood": 50.0})
        fed = Federation(pods=[pod_config], seed=42)

        # Register tool
        hammer = Resource("hammer", base_value=10.0, is_tool=True, durability=100)
        fed.register_new_resource(hammer)

        summary = fed.get_economic_summary()

        # Should have total_tools key
        self.assertIn("total_tools", summary)


if __name__ == '__main__':
    unittest.main()