# tests/unit/organization/test_federation_additional.py
"""Additional unit tests for Federation class to improve coverage."""

import pytest
import unittest
from unittest.mock import Mock

from libertas.organization import Federation, Pod, PodConfig, WorkerConfig
from libertas.resources import Resource, Recipe, ProductionStep, StepType


LLM_MODEL = "ollama/qwen3"


@pytest.mark.unit
class TestFederationHelperMethods(unittest.TestCase):
    """Test Federation private helper methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])

        # Register resources
        self.federation.register_new_resource(
            Resource("wood", "system", base_value=10.0)
        )

        # Create pods with workers
        worker1_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        worker2_config = WorkerConfig(
            name="worker2",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )

        pod1_config = PodConfig(
            name="pod1",
            workers=[worker1_config],
            initial_inventory={"wood": 50.0}
        )
        pod2_config = PodConfig(
            name="pod2",
            workers=[worker2_config],
            initial_inventory={"wood": 30.0}
        )

        self.pod1 = Pod(self.federation, pod1_config, coordinate=(0, 0))
        self.pod2 = Pod(self.federation, pod2_config, coordinate=(1, 0))

        self.federation.add(self.pod1)
        self.federation.add(self.pod2)

    def test_get_pod_inventory(self):
        """Test _get_pod_inventory helper method."""
        quantity = self.federation._get_pod_inventory("pod1", "wood")
        self.assertEqual(quantity, 50.0)

    def test_get_pod_inventory_nonexistent_pod(self):
        """Test _get_pod_inventory with nonexistent pod."""
        quantity = self.federation._get_pod_inventory("nonexistent", "wood")
        self.assertEqual(quantity, 0.0)

    def test_get_pod_inventory_nonexistent_resource(self):
        """Test _get_pod_inventory with nonexistent resource."""
        quantity = self.federation._get_pod_inventory("pod1", "gold")
        self.assertEqual(quantity, 0.0)

    def test_update_pod_inventory_add(self):
        """Test _update_pod_inventory adding resources."""
        initial = self.pod1.inventory.get_quantity("wood")
        self.federation._update_pod_inventory("pod1", "wood", 20.0)
        final = self.pod1.inventory.get_quantity("wood")
        self.assertEqual(final, initial + 20.0)

    def test_update_pod_inventory_remove(self):
        """Test _update_pod_inventory removing resources."""
        initial = self.pod1.inventory.get_quantity("wood")
        self.federation._update_pod_inventory("pod1", "wood", -10.0)
        final = self.pod1.inventory.get_quantity("wood")
        self.assertEqual(final, initial - 10.0)

    def test_update_pod_inventory_nonexistent_pod(self):
        """Test _update_pod_inventory with nonexistent pod."""
        # Should not raise error
        self.federation._update_pod_inventory("nonexistent", "wood", 10.0)

    def test_update_pod_inventory_nonexistent_resource(self):
        """Test _update_pod_inventory with unregistered resource."""
        # Should not raise error (resource not registered)
        self.federation._update_pod_inventory("pod1", "gold", 10.0)

    def test_find_worker_by_name_exists(self):
        """Test _find_worker_by_name with existing worker."""
        worker = self.federation._find_worker_by_name("worker1")
        self.assertIsNotNone(worker)
        self.assertEqual(worker.name, "worker1")

    def test_find_worker_by_name_nonexistent(self):
        """Test _find_worker_by_name with nonexistent worker."""
        worker = self.federation._find_worker_by_name("nonexistent")
        self.assertIsNone(worker)


@pytest.mark.unit
class TestFederationMarketIntegration(unittest.TestCase):
    """Test Federation market integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create federation with market
        self.federation = Federation(pods=[], initialize_market=True)
        self.federation.steps = 0

    def test_step_without_market(self):
        """Test federation step when market is not initialized."""
        fed = Federation(pods=[], initialize_market=False)

        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        pod_config = PodConfig(name="pod1", workers=[worker_config])
        pod = Pod(fed, pod_config, coordinate=(0, 0))
        fed.add(pod)

        # Should not raise error
        fed.step()


@pytest.mark.unit
class TestFederationPodManagement(unittest.TestCase):
    """Test Federation pod management methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])

    def test_remove_pod_that_exists(self):
        """Test removing a pod that exists."""
        pod_config = PodConfig(name="pod1", workers=[])
        pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.federation.add(pod)

        self.assertIn(pod, self.federation)

        self.federation.remove(pod)

        self.assertNotIn(pod, self.federation)

    def test_remove_pod_that_doesnt_exist(self):
        """Test removing a pod that doesn't exist raises KeyError."""
        pod_config = PodConfig(name="pod1", workers=[])
        pod = Pod(self.federation, pod_config, coordinate=(0, 0))

        with self.assertRaises(KeyError):
            self.federation.remove(pod)

    def test_set_pod_layout_empty_pods(self):
        """Test set_pod_layout with no pods."""
        # Should not raise error
        self.federation.set_pod_layout("circular")

    def test_set_pod_layout_creates_grid(self):
        """Test that set_pod_layout creates grid properly."""
        pod1_config = PodConfig(name="pod1", workers=[])
        pod2_config = PodConfig(name="pod2", workers=[])

        pod1 = Pod(self.federation, pod1_config, coordinate=(0, 0))
        pod2 = Pod(self.federation, pod2_config, coordinate=(1, 0))

        self.federation.add(pod1)
        self.federation.add(pod2)

        self.federation.set_pod_layout("circular")

        # Grid should be created
        self.assertIsNotNone(self.federation._grid)
        self.assertIsNotNone(self.federation._pod_graph)


@pytest.mark.unit
class TestFederationGetters(unittest.TestCase):
    """Test Federation getter methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[])

        pod1_config = PodConfig(name="pod1", workers=[])
        pod2_config = PodConfig(name="pod2", workers=[])

        self.pod1 = Pod(self.federation, pod1_config, coordinate=(0, 0))
        self.pod2 = Pod(self.federation, pod2_config, coordinate=(1, 0))

        self.federation.add(self.pod1)
        self.federation.add(self.pod2)

    def test_get_neighbors_with_no_graph(self):
        """Test get_neighbors when no graph exists."""
        # Create federation without setting layout
        fed = Federation(pods=[])
        pod_config = PodConfig(name="pod1", workers=[])
        pod = Pod(fed, pod_config, coordinate=(0, 0))
        fed.add(pod)

        neighbors = fed.get_neighbors(pod)
        self.assertEqual(neighbors, [])

    def test_get_neighbors_nonexistent_pod(self):
        """Test get_neighbors with pod not in graph."""
        self.federation.set_pod_layout("circular")

        # Create pod not in federation
        pod_config = PodConfig(name="pod3", workers=[])
        pod3 = Pod(self.federation, pod_config, coordinate=(2, 0))

        neighbors = self.federation.get_neighbors(pod3)
        self.assertEqual(neighbors, [])

    def test_get_pod_by_name_exists(self):
        """Test get_pod_by_name with existing pod."""
        pod = self.federation.get_pod_by_name("pod1")
        self.assertIsNotNone(pod)
        self.assertEqual(pod.name, "pod1")

    def test_get_pod_by_name_nonexistent(self):
        """Test get_pod_by_name with nonexistent pod."""
        pod = self.federation.get_pod_by_name("nonexistent")
        self.assertIsNone(pod)

    def test_get_pod_by_index_valid(self):
        """Test get_pod_by_index with valid index."""
        pod = self.federation.get_pod_by_index(0)
        self.assertIsNotNone(pod)
        self.assertEqual(pod, self.pod1)

    def test_get_pod_by_index_negative(self):
        """Test get_pod_by_index with negative index."""
        pod = self.federation.get_pod_by_index(-1)
        self.assertIsNone(pod)

    def test_get_pod_by_index_out_of_bounds(self):
        """Test get_pod_by_index with out of bounds index."""
        pod = self.federation.get_pod_by_index(100)
        self.assertIsNone(pod)


@pytest.mark.unit
class TestFederationEconomicSummary(unittest.TestCase):
    """Test Federation economic summary methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.federation = Federation(pods=[], initialize_market=True)
        self.federation.steps = 10

        # Register resources
        self.federation.register_new_resource(
            Resource("wood", "system", base_value=10.0)
        )

        # Create pod with worker and inventory
        worker_config = WorkerConfig(
            name="worker1",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_currency=100.0
        )

        pod_config = PodConfig(
            name="pod1",
            workers=[worker_config],
            initial_inventory={"wood": 50.0}
        )

        pod = Pod(self.federation, pod_config, coordinate=(0, 0))
        self.federation.add(pod)

    def test_get_economic_summary_structure(self):
        """Test that get_economic_summary returns expected structure."""
        summary = self.federation.get_economic_summary()

        self.assertIn("num_pods", summary)
        self.assertIn("num_workers", summary)
        self.assertIn("step", summary)
        self.assertIn("total_inventory", summary)

    def test_get_economic_summary_values(self):
        """Test that get_economic_summary returns correct values."""
        summary = self.federation.get_economic_summary()

        self.assertEqual(summary["num_pods"], 1)
        self.assertEqual(summary["num_workers"], 1)
        self.assertEqual(summary["step"], 10)
        self.assertIn("wood", summary["total_inventory"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
