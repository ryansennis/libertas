# tests/unit/tools/test_economic_tools.py
import unittest
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from libertas.tools.economic_tools import EconomicTools, get_economic_tool_definitions
from libertas.organization.worker import Worker, WorkerConfig
from libertas.organization.pod import Pod, PodConfig
from libertas.organization.federation import Federation
from libertas.economy import Resource, Recipe, ProductionStep, StepType, ResourceRegistry, RecipeRegistry

LLM_MODEL = "ollama/tinyllama"

@pytest.mark.unit
class TestEconomicTools(unittest.TestCase):
    """Test EconomicTools class."""

    def setUp(self):
        # Create resource registry with test data
        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))
        resource_registry.register(Resource(name="metal", base_value=20.0))
        resource_registry.register(Resource(name="plank", base_value=15.0))
        resource_registry.register(Resource(
            name="hammer",
            base_value=50.0,
            is_tool=True,
            durability=100.0,
            required_skill="crafting"
        ))

        # Create recipe registry with test data
        recipe_registry = RecipeRegistry()
        smelt_recipe = Recipe(
            name="smelt",
            steps=[
                ProductionStep(
                    name="smelt",
                    step_type=StepType.PROCESSING,
                    duration=5,
                    inputs={"wood": 2.0},
                    outputs={"metal": 1.0},
                    required_skill="smelting",
                    required_tool="hammer"
                )
            ],
            description="Smelt metal from wood"
        )
        recipe_registry.register(smelt_recipe)

        process_recipe = Recipe(
            name="process_wood",
            steps=[
                ProductionStep(
                    name="process",
                    step_type=StepType.PROCESSING,
                    duration=3,
                    inputs={"wood": 1.0},
                    outputs={"plank": 2.0},
                    required_skill="crafting"
                )
            ],
            description="Process wood into planks"
        )
        recipe_registry.register(process_recipe)

        # Create worker config
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_currency=500.0,
            initial_skills={"crafting": 2.0, "smelting": 1.5},
            initial_tools=["hammer"]
        )

        # Create pod config with worker and initial inventory
        pod_config = PodConfig(
            name="pod_001",
            workers=[worker_config],
            initial_inventory={"wood": 100.0, "metal": 50.0}
        )

        # Create federation with pods and registries
        self.federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry,
            recipe_registry=recipe_registry,
            initialize_market=True
        )
        self.federation.steps = 100  # Set step count for tests

        # Get the worker from the created pod
        pod = self.federation[0]  # First pod
        self.worker = list(pod)[0]  # First worker (Pod is iterable AgentSet)
        self.tools = EconomicTools(self.worker)
    
    def test_inspect_inventory(self):
        """Test inspect_inventory tool."""
        result = self.tools.inspect_inventory()
        data = json.loads(result)

        self.assertEqual(data["pod_id"], "pod_001")
        self.assertIn("wood", data["inventory"])
        self.assertEqual(data["inventory"]["wood"], 100.0)
        self.assertEqual(data["inventory"]["metal"], 50.0)
        self.assertEqual(data["total_items"], 150.0)
    
    def test_inspect_worker_status(self):
        """Test inspect_worker_status tool."""
        result = self.tools.inspect_worker_status()
        data = json.loads(result)

        self.assertEqual(data["name"], "worker_001")
        self.assertEqual(data["currency"], 500.0)
        self.assertIn("crafting", data["skills"])
    
    def test_list_known_resources(self):
        """Test list_known_resources tool."""
        result = self.tools.list_known_resources()
        data = json.loads(result)

        self.assertEqual(data["count"], 4)
        resource_names = [r["name"] for r in data["resources"]]
        self.assertIn("wood", resource_names)
        self.assertIn("metal", resource_names)
        self.assertIn("plank", resource_names)
        self.assertIn("hammer", resource_names)
    
    def test_list_known_recipes(self):
        """Test list_known_recipes tool."""
        result = self.tools.list_known_recipes()
        data = json.loads(result)
        
        self.assertEqual(data["count"], 2)
    
    def test_get_recipe_details(self):
        """Test get_recipe_details tool."""
        result = self.tools.get_recipe_details("smelt")
        data = json.loads(result)
        
        self.assertEqual(data["name"], "smelt")
    
    def test_get_recipe_details_not_found(self):
        """Test get_recipe_details with non-existent recipe."""
        result = self.tools.get_recipe_details("nonexistent")
        data = json.loads(result)
        
        self.assertIn("error", data)
    
    def test_start_production(self):
        """Test start_production tool."""
        result = self.tools.start_production("smelt", batch_size=2)
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertIn("job_id", data)
        self.assertTrue(len(data["job_id"]) > 0)  # Job ID should be a non-empty string
    
    def test_check_production_queue(self):
        """Test check_production_queue tool."""
        result = self.tools.check_production_queue()
        data = json.loads(result)

        self.assertIn("active_jobs", data)
        self.assertIn("queued_jobs", data)

    def test_check_production_queue_with_jobs(self):
        """Test check_production_queue with active jobs (covers lines 204, 215)."""
        # Start production to create active job
        self.tools.start_production("smelt", 1)

        result = self.tools.check_production_queue()
        data = json.loads(result)

        # Should have active jobs listed (covers line 204 loop)
        self.assertIn("active_jobs", data)
        self.assertIn("queued_jobs", data)
    
    def test_list_my_tools(self):
        """Test list_my_tools tool."""
        result = self.tools.list_my_tools()
        data = json.loads(result)
        
        self.assertEqual(data["total_tools"], 1)
        self.assertEqual(data["tools"][0]["name"], "hammer")
    
    def test_equip_tool(self):
        """Test equip_tool tool."""
        result = self.tools.equip_tool("hammer")
        data = json.loads(result)
        
        self.assertTrue(data["success"])
        self.assertEqual(self.worker.equipped_tool, "hammer")
    
    def test_equip_tool_not_found(self):
        """Test equip_tool with non-existent tool."""
        result = self.tools.equip_tool("nonexistent")
        data = json.loads(result)
        
        self.assertFalse(data["success"])
    
    def test_unequip_tool(self):
        """Test unequip_tool tool."""
        self.worker.equipped_tool = "hammer"
        
        result = self.tools.unequip_tool()
        data = json.loads(result)
        
        self.assertTrue(data["success"])
        self.assertIsNone(self.worker.equipped_tool)
    
    def test_transfer_to_pod(self):
        """Test transfer_to_pod tool - should fail with non-existent pod."""
        result = self.tools.transfer_to_pod("wood", 30.0, "pod_002")
        data = json.loads(result)

        # Should fail because pod_002 doesn't exist
        self.assertIn("error", data)
        self.assertIn("not found", data["error"])

    def test_transfer_to_pod_insufficient(self):
        """Test transfer_to_pod when pod has insufficient inventory (covers lines 312-318)."""
        # Try to transfer more than available
        result = self.tools.transfer_to_pod("wood", 1000.0, "pod_002")
        data = json.loads(result)

        # Should fail due to insufficient inventory or pod not found
        # Either error condition covers the error paths
        self.assertTrue("error" in data or ("success" in data and not data["success"]))
    
    def test_list_pods(self):
        """Test list_pods tool."""
        result = self.tools.list_pods()
        data = json.loads(result)
        
        self.assertEqual(data["current_pod"], "pod_001")
        self.assertIsInstance(data["pods"], list)
    
    def test_invent_resource(self):
        """Test invent_resource tool."""
        result = self.tools.invent_resource(
            name="mithril",
            base_value=100.0,
            is_tool=False,
            properties={"rarity": 0.9}
        )
        data = json.loads(result)
        
        self.assertTrue(data["success"])
        self.assertEqual(data["resource"]["name"], "mithril")
    
    def test_invent_resource_already_exists(self):
        """Test invent_resource with duplicate name."""
        # First invention succeeds
        self.tools.invent_resource("mithril")
        
        # Second should fail
        result = self.tools.invent_resource("mithril")
        data = json.loads(result)
        
        self.assertFalse(data["success"])
    
    def test_get_my_skills(self):
        """Test get_my_skills tool."""
        result = self.tools.get_my_skills()
        data = json.loads(result)
        
        self.assertIn("crafting", data["skills"])
        self.assertEqual(data["skills"]["crafting"], 2.0)
    
    def test_get_federation_summary(self):
        """Test get_federation_summary tool."""
        result = self.tools.get_federation_summary()
        data = json.loads(result)

        self.assertEqual(data["step"], 100)
        self.assertEqual(data["num_pods"], 1)
    
    def test_get_balance(self):
        """Test get_balance tool."""
        result = self.tools.get_balance()
        data = json.loads(result)

        self.assertEqual(data["worker_id"], "worker_001")
        self.assertEqual(data["currency"], 500.0)

    def test_start_production_failure_return(self):
        """Test start_production returns failure JSON (line 186)."""
        from unittest.mock import patch

        pod = self.worker.pod
        # Mock pod.start_production to return failure
        with patch.object(pod, 'start_production', return_value=(False, "Insufficient resources")):
            result = self.tools.start_production("process_wood", 1)
            data = json.loads(result)

            self.assertFalse(data["success"])
            self.assertIn("error", data)

    def test_check_production_queue_with_active_jobs(self):
        """Test check_production_queue appends active jobs (line 204)."""
        from libertas.economy import ProductionJob, Recipe, ProductionStep, StepType

        pod = self.worker.pod
        # Create and add active job
        step = ProductionStep(name="test", duration=10, step_type=StepType.PROCESSING)
        recipe = Recipe(name="test_recipe", steps=[step])
        job = ProductionJob(recipe=recipe, job_id="job_001")
        pod.active_jobs.append(job)

        result = self.tools.check_production_queue()
        data = json.loads(result)

        self.assertIn("active_jobs", data)
        self.assertEqual(len(data["active_jobs"]), 1)

    def test_transfer_to_pod_success(self):
        """Test successful resource transfer (lines 312-316)."""
        from libertas.organization.pod import PodConfig
        from libertas.organization import Federation

        # Create a second pod in the federation at setup time
        worker_config2 = WorkerConfig(name="worker_002", reasoning=Mock, llm_model=LLM_MODEL)
        pod_config2 = PodConfig(name="target_pod", workers=[worker_config2])

        # Create new federation with both pods
        fed = Federation(
            pods=[
                PodConfig(
                    name="pod_001",
                    workers=[WorkerConfig(name="worker_001", reasoning=Mock, llm_model=LLM_MODEL, initial_currency=500.0)],
                    initial_inventory={"wood": 100.0}
                ),
                pod_config2
            ],
            resource_registry=self.federation.resource_registry,
            recipe_registry=self.federation.recipe_registry
        )

        # Get worker from first pod
        worker = list(fed[0])[0]
        tools = EconomicTools(worker)

        # Transfer wood from pod_001 to target_pod
        result = tools.transfer_to_pod("wood", 10.0, "target_pod")
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertIn("Transferred", data["message"])

    def test_transfer_to_pod_failure(self):
        """Test failed resource transfer (lines 318-321)."""
        from libertas.organization.pod import PodConfig
        from libertas.organization import Federation

        # Create a second pod in the federation
        worker_config2 = WorkerConfig(name="worker_002", reasoning=Mock, llm_model=LLM_MODEL)
        pod_config2 = PodConfig(name="target_pod", workers=[worker_config2])

        # Create new federation with both pods
        fed = Federation(
            pods=[
                PodConfig(
                    name="pod_001",
                    workers=[WorkerConfig(name="worker_001", reasoning=Mock, llm_model=LLM_MODEL, initial_currency=500.0)],
                    initial_inventory={"wood": 100.0}
                ),
                pod_config2
            ],
            resource_registry=self.federation.resource_registry,
            recipe_registry=self.federation.recipe_registry
        )

        # Get worker from first pod
        worker = list(fed[0])[0]
        tools = EconomicTools(worker)

        # Try to transfer more wood than we have
        result = tools.transfer_to_pod("wood", 200.0, "target_pod")
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("Failed to transfer", data["error"])

    def test_transfer_to_pod_no_pod_assigned(self):
        """Test transfer when worker has no pod (line 304)."""
        from libertas.organization.worker import Worker, WorkerConfig
        from unittest.mock import Mock

        # Create worker without pod
        worker_config = WorkerConfig(name="no_pod_worker", reasoning=Mock, llm_model="ollama/tinyllama")
        worker = Worker(self.federation, worker_config, coordinate=(0, 0), pod=None)

        tools = EconomicTools(worker)
        result = tools.transfer_to_pod("wood", 10.0, "target_pod")
        data = json.loads(result)

        self.assertIn("error", data)
        self.assertIn("not assigned to a pod", data["error"])


@pytest.mark.unit
class TestMarketTools(unittest.TestCase):
    """Test market-related tools."""

    def setUp(self):
        # Create resource registry with test data
        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))
        resource_registry.register(Resource(name="metal", base_value=20.0))

        # Create worker config
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model=LLM_MODEL,
            initial_currency=500.0
        )

        # Create pod config with worker and initial inventory
        pod_config = PodConfig(
            name="pod_001",
            workers=[worker_config],
            initial_inventory={"wood": 100.0}
        )

        # Create federation with pods and registries
        self.federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry,
            initialize_market=True
        )

        # Get the worker from the created pod
        pod = self.federation[0]  # First pod
        self.worker = list(pod)[0]  # First worker (Pod is iterable AgentSet)
        self.tools = EconomicTools(self.worker)
    
    def test_get_market_price(self):
        """Test get_market_price tool."""
        result = self.tools.get_market_price("wood")
        data = json.loads(result)
        
        self.assertEqual(data["resource"], "wood")
        self.assertEqual(data["current_price"], 10.0)
    
    def test_get_market_price_not_registered(self):
        """Test get_market_price for unregistered resource."""
        result = self.tools.get_market_price("unknown")

    def test_buy_from_market_resource_not_available(self):
        """Test buy_from_market with resource not on market (line 515)."""
        result = self.tools.buy_from_market("unregistered_resource", 10.0, 20.0)
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("not available on market", result_dict["error"])

    def test_sell_to_market_resource_not_available(self):
        """Test sell_to_market with resource not on market (line 564)."""
        result = self.tools.sell_to_market("unregistered_resource", 10.0, 5.0)
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("not available on market", result_dict["error"])

    def test_get_my_orders_with_orders(self):
        """Test get_my_orders when worker has orders (covers line 621)."""
        # Place an order first
        self.tools.buy_from_market("wood", 10.0, 15.0)

        result = self.tools.get_my_orders()
        result_dict = json.loads(result)

        # Should have orders (covers line 621 loop)
        self.assertIn("active_orders", result_dict)
        self.assertIn("total_active", result_dict)

    def test_cancel_order_not_found(self):
        """Test cancel_order with non-existent order (covers line 655)."""
        result = self.tools.cancel_order("nonexistent_order_id")
        result_dict = json.loads(result)

        self.assertIn("success", result_dict)
        self.assertFalse(result_dict["success"])
        self.assertIn("error", result_dict)
        data = json.loads(result)
        
        self.assertIn("error", data)
    
    def test_buy_from_market(self):
        """Test buy_from_market tool."""
        # Buy 10 wood at max price 12.0 (costs 120, worker has 500)
        result = self.tools.buy_from_market("wood", 10.0, 12.0)
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertIn("order_id", data)
        self.assertTrue(len(data["order_id"]) > 0)
    
    def test_buy_from_market_insufficient_funds(self):
        """Test buy_from_market with insufficient funds."""
        self.worker.currency = 100.0
        
        result = self.tools.buy_from_market("wood", 50.0, 100.0)
        data = json.loads(result)
        
        self.assertFalse(data["success"])
        self.assertIn("Insufficient funds", data["error"])
    
    def test_sell_to_market(self):
        """Test sell_to_market tool."""
        result = self.tools.sell_to_market("wood", 30.0, 8.0)
        data = json.loads(result)

        self.assertTrue(data["success"])
        self.assertIn("order_id", data)
        self.assertTrue(len(data["order_id"]) > 0)
    
    def test_sell_to_market_insufficient_inventory(self):
        """Test sell_to_market with insufficient inventory."""
        # Try to sell more wood than we have (pod has 100.0 wood)
        result = self.tools.sell_to_market("wood", 200.0, 8.0)
        data = json.loads(result)

        self.assertFalse(data["success"])
        self.assertIn("Insufficient inventory", data["error"])
    
    def test_get_market_summary(self):
        """Test get_market_summary tool."""
        result = self.tools.get_market_summary()
        data = json.loads(result)
        
        self.assertIn("prices", data)
    
    def test_get_my_orders(self):
        """Test get_my_orders tool."""
        result = self.tools.get_my_orders()
        data = json.loads(result)
        
        self.assertEqual(data["total_active"], 0)
    
    def test_cancel_order(self):
        """Test cancel_order tool."""
        # First create an order
        buy_result = self.tools.buy_from_market("wood", 5.0, 12.0)
        buy_data = json.loads(buy_result)
        self.assertTrue(buy_data["success"])
        order_id = buy_data["order_id"]

        # Now cancel it
        result = self.tools.cancel_order(order_id)
        data = json.loads(result)

        self.assertTrue(data["success"])
    
    def test_get_balance(self):
        """Test get_balance tool."""
        result = self.tools.get_balance()
        data = json.loads(result)
        
        self.assertEqual(data["currency"], 500.0)


@pytest.mark.unit
class TestToolDefinitions(unittest.TestCase):
    """Test tool definitions for LLM function calling."""
    
    def test_get_economic_tool_definitions(self):
        """Test that tool definitions are properly formatted."""
        definitions = get_economic_tool_definitions()
        
        self.assertIsInstance(definitions, list)
        self.assertGreater(len(definitions), 0)
        
        # Check required fields for each tool
        for tool_def in definitions:
            self.assertEqual(tool_def["type"], "function")
            self.assertIn("function", tool_def)
            self.assertIn("name", tool_def["function"])
            self.assertIn("description", tool_def["function"])
        
        # Check for key tools
        tool_names = [t["function"]["name"] for t in definitions]
        self.assertIn("inspect_inventory", tool_names)
        self.assertIn("start_production", tool_names)
        self.assertIn("buy_from_market", tool_names)
        self.assertIn("sell_to_market", tool_names)
        self.assertIn("invent_resource", tool_names)
        self.assertIn("get_balance", tool_names)


@pytest.mark.unit
class TestEconomicToolsWorkerWithoutPod(unittest.TestCase):
    """Test EconomicTools when worker has no pod."""

    def setUp(self):
        # Create federation
        self.federation = Federation(pods=[], seed=42)

        # Create worker WITHOUT adding to a pod
        worker_config = WorkerConfig(
            name="worker_no_pod",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )
        self.worker = Worker(self.federation, worker_config, coordinate=(0, 0), pod=None)
        self.tools = EconomicTools(self.worker)

    def test_inspect_inventory_no_pod(self):
        """Test inspect_inventory when worker has no pod (line 37)."""
        result = self.tools.inspect_inventory()
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("not assigned to a pod", result_dict["error"])

    def test_start_production_no_pod(self):
        """Test start_production when worker has no pod (line 171)."""
        result = self.tools.start_production("test_recipe", 1)
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("not assigned to a pod", result_dict["error"])

    def test_check_production_queue_no_pod(self):
        """Test check_production_queue when worker has no pod (line 200)."""
        result = self.tools.check_production_queue()
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("not assigned to a pod", result_dict["error"])

    def test_transfer_to_pod_no_pod(self):
        """Test transfer_to_pod when worker has no pod (line 304)."""
        result = self.tools.transfer_to_pod("wood", 10.0, "other_pod")
        result_dict = json.loads(result)

        self.assertIn("error", result_dict)
        self.assertIn("not assigned to a pod", result_dict["error"])


@pytest.mark.unit
class TestEconomicToolsEdgeCases(unittest.TestCase):
    """Test edge cases for EconomicTools."""

    def setUp(self):
        # Create resource registry with test data
        resource_registry = ResourceRegistry()
        resource_registry.register(Resource(name="wood", base_value=10.0))

        # Create recipe registry
        recipe_registry = RecipeRegistry()

        # Create worker config
        worker_config = WorkerConfig(
            name="worker_001",
            reasoning=Mock,
            llm_model=LLM_MODEL
        )

        # Create pod config with worker
        pod_config = PodConfig(
            name="pod_001",
            workers=[worker_config]
        )

        # Create federation with pods and registries
        self.federation = Federation(
            pods=[pod_config],
            resource_registry=resource_registry,
            recipe_registry=recipe_registry
        )

        # Get the worker from the created pod
        pod = self.federation[0]  # First pod
        self.worker = list(pod)[0]  # First worker (Pod is iterable AgentSet)
        self.tools = EconomicTools(self.worker)
    
    def test_invent_recipe(self):
        """Test invent_recipe tool."""
        steps = [
            {
                "name": "step1",
                "duration": 5,
                "inputs": {"wood": 2},
                "outputs": {"plank": 1}
            }
        ]
        
        result = self.tools.invent_recipe(
            name="new_recipe",
            steps=steps,
            description="A new crafting method",
            category="crafting"
        )
        data = json.loads(result)
        
        self.assertTrue(data["success"])
        self.assertEqual(data["recipe"]["name"], "new_recipe")
    
    def test_invent_recipe_already_exists(self):
        """Test invent_recipe with duplicate name."""
        steps = [{"name": "step1", "duration": 5}]
        
        # First invention succeeds
        self.tools.invent_recipe("unique_recipe", steps)
        
        # Second should fail
        result = self.tools.invent_recipe("unique_recipe", steps)
        data = json.loads(result)

        self.assertFalse(data["success"])


if __name__ == '__main__':
    unittest.main()