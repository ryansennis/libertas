"""Tests for OrganizationTools class."""

import json
import pytest
from unittest.mock import Mock, MagicMock

from libertas.tools.organization_tools import OrganizationTools, get_organization_tool_definitions
from libertas.organization.worker import WorkerConfig
from libertas.organization.pod import PodConfig
from libertas.organization.federation import Federation
from libertas.cognitive import PersonalityTraits, Background
from libertas.resources import Material, Tool
from mesa_llm.reasoning.cot import CoTReasoning

LLM_MODEL = "ollama/tinyllama"


@pytest.mark.unit
class TestOrganizationTools:
    """Test OrganizationTools class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a federation with two workers in a pod
        worker_configs = [
            WorkerConfig(
                name="Alice",
                reasoning=CoTReasoning,
                llm_model=LLM_MODEL,
                initial_currency=1000.0,
                initial_skills={"farming": 5.0, "crafting": 3.0},
                personality=PersonalityTraits(openness=0.8, conscientiousness=0.7),
                background=Background(education_level=4, years_experience=5)
            ),
            WorkerConfig(
                name="Bob",
                reasoning=CoTReasoning,
                llm_model=LLM_MODEL,
                initial_currency=500.0,
                initial_skills={"mining": 4.0},
                personality=PersonalityTraits(openness=0.5, conscientiousness=0.9),
                background=Background(education_level=3, years_experience=3)
            )
        ]

        pod_config = PodConfig(
            name="TestPod",
            workers=worker_configs,
            initial_inventory={"wood": 100.0, "stone": 50.0}
        )

        self.federation = Federation(pods=[pod_config], seed=42)

        # Register resources
        self.federation.resource_registry.register(Material("wood", base_value=10.0))
        self.federation.resource_registry.register(Material("stone", base_value=15.0))
        self.federation.resource_registry.register(Tool("hammer", base_value=50.0))

        self.pod = self.federation[0]
        self.alice = [w for w in self.pod if w.name == "Alice"][0]
        self.bob = [w for w in self.pod if w.name == "Bob"][0]

        self.tools = OrganizationTools(self.alice)

    def test_initialization(self):
        """Test OrganizationTools initializes correctly."""
        assert self.tools.worker == self.alice

    def test_list_pod_members(self):
        """Test listing pod members."""
        result = self.tools.list_pod_members()
        data = json.loads(result)

        assert data["pod_name"] == "TestPod"
        assert data["total_members"] == 2
        assert data["other_members"] == 1
        assert len(data["members"]) == 1

        # Should show Bob but not Alice (self)
        member = data["members"][0]
        assert member["name"] == "Bob"
        assert "currency" in member
        assert "skills" in member

    def test_list_pod_members_no_pod(self):
        """Test listing members when worker has no pod."""
        self.alice._pod = None
        result = self.tools.list_pod_members()
        data = json.loads(result)

        assert "error" in data
        assert "not assigned to a pod" in data["error"]

    def test_get_worker_info(self):
        """Test getting detailed worker information."""
        result = self.tools.get_worker_info("Bob")
        data = json.loads(result)

        assert data["name"] == "Bob"
        assert data["currency"] == 500.0
        assert "mining" in data["skills"]
        assert data["skills"]["mining"] == 4.0
        assert "mood" in data
        assert "personality" in data
        assert "happiness" in data["mood"]
        assert "openness" in data["personality"]

    def test_get_worker_info_not_found(self):
        """Test getting info for non-existent worker."""
        result = self.tools.get_worker_info("Charlie")
        data = json.loads(result)

        assert "error" in data
        assert "not found" in data["error"]

    def test_get_worker_info_no_pod(self):
        """Test getting worker info when not in pod."""
        self.alice._pod = None
        result = self.tools.get_worker_info("Bob")
        data = json.loads(result)

        assert "error" in data

    def test_check_pod_resources(self):
        """Test checking pod resources."""
        result = self.tools.check_pod_resources()
        data = json.loads(result)

        assert data["pod_name"] == "TestPod"
        assert "inventory" in data
        assert "tools" in data
        assert data["total_workers"] == 2
        assert data["active_jobs"] == 0
        assert data["queued_jobs"] == 0

    def test_check_pod_resources_no_pod(self):
        """Test checking resources when not in pod."""
        self.alice._pod = None
        result = self.tools.check_pod_resources()
        data = json.loads(result)

        assert "error" in data

    def test_view_production_queue(self):
        """Test viewing production queue."""
        result = self.tools.view_production_queue()
        data = json.loads(result)

        assert data["pod_name"] == "TestPod"
        assert isinstance(data["active_jobs"], list)
        assert isinstance(data["queued_jobs"], list)
        assert data["total_queued"] == 0

    def test_view_production_queue_no_pod(self):
        """Test viewing queue when not in pod."""
        self.alice._pod = None
        result = self.tools.view_production_queue()
        data = json.loads(result)

        assert "error" in data

    def test_request_tool_from_pod(self):
        """Test requesting a tool from pod."""
        # Add a hammer to pod inventory
        hammer = self.federation.get_resource("hammer")
        if hammer:
            self.pod.inventory.add(hammer, 1)

        result = self.tools.request_tool_from_pod("hammer")
        data = json.loads(result)

        assert data["success"]
        assert "available" in data["message"]

    def test_request_tool_not_available(self):
        """Test requesting unavailable tool."""
        result = self.tools.request_tool_from_pod("chainsaw")
        data = json.loads(result)

        assert not (data["success"])
        assert "error" in data
        assert "not available" in data["error"]

    def test_request_tool_no_pod(self):
        """Test requesting tool when not in pod."""
        self.alice._pod = None
        result = self.tools.request_tool_from_pod("hammer")
        data = json.loads(result)

        assert "error" in data

    def test_view_federation_pods(self):
        """Test viewing all federation pods."""
        result = self.tools.view_federation_pods()
        data = json.loads(result)

        assert data["total_pods"] == 1
        assert len(data["pods"]) == 1
        assert data["your_pod"] == "TestPod"

        pod_info = data["pods"][0]
        assert pod_info["name"] == "TestPod"
        assert pod_info["workers"] == 2
        assert pod_info["active_jobs"] == 0


@pytest.mark.unit
class TestOrganizationToolDefinitions:
    """Test organization tool definitions."""

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        defs = get_organization_tool_definitions()

        assert isinstance(defs, list)
        assert len(defs) > 0

        # Check structure
        for tool_def in defs:
            assert "type" in tool_def
            assert tool_def["type"] == "function"
            assert "function" in tool_def
            assert "name" in tool_def["function"]
            assert "description" in tool_def["function"]
            assert "parameters" in tool_def["function"]

    def test_tool_names(self):
        """Test that all expected tools are defined."""
        defs = get_organization_tool_definitions()
        tool_names = [d["function"]["name"] for d in defs]

        expected_tools = [
            "list_pod_members",
            "get_worker_info",
            "check_pod_resources",
            "view_production_queue",
            "request_tool_from_pod",
            "view_federation_pods"
        ]

        for expected in expected_tools:
            assert expected in tool_names

    def test_list_pod_members_definition(self):
        """Test list_pod_members tool definition."""
        defs = get_organization_tool_definitions()
        tool = next(d for d in defs if d["function"]["name"] == "list_pod_members")

        assert "List all workers" in tool["function"]["description"]
        assert tool["function"]["parameters"]["type"] == "object"

    def test_get_worker_info_definition(self):
        """Test get_worker_info tool definition has required parameters."""
        defs = get_organization_tool_definitions()
        tool = next(d for d in defs if d["function"]["name"] == "get_worker_info")

        params = tool["function"]["parameters"]
        assert "worker_name" in params["properties"]
        assert "worker_name" in params["required"]


