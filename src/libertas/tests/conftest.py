# src/libertas/tests/conftest.py
"""Shared pytest fixtures for libertas tests."""

import pytest
from mesa_llm.reasoning.cot import CoTReasoning

from libertas.economy import Resource, Recipe, ProductionStep, StepType, ResourceRegistry, RecipeRegistry
from libertas.organization import WorkerConfig, PodConfig, Federation


@pytest.fixture
def resource_registry():
    """Create a resource registry with common test resources."""
    registry = ResourceRegistry()

    # Basic resources
    registry.register(Resource(name="wood", base_value=10.0))
    registry.register(Resource(name="metal", base_value=20.0))
    registry.register(Resource(name="plank", base_value=15.0))

    # Tools
    registry.register(Resource(
        name="hammer",
        base_value=50.0,
        is_tool=True,
        durability=100.0,
        required_skill="crafting"
    ))

    return registry


@pytest.fixture
def recipe_registry():
    """Create a recipe registry with common test recipes."""
    registry = RecipeRegistry()

    # Smelt recipe
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
    registry.register(smelt_recipe)

    # Process wood recipe
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
    registry.register(process_recipe)

    return registry


@pytest.fixture
def basic_worker_config():
    """Create a basic worker configuration for testing (non-autonomous)."""
    from unittest.mock import Mock
    return WorkerConfig(
        name="worker_001",
        reasoning=Mock,  # Mock reasoning for non-autonomous tests
        llm_model="ollama/tinyllama",  # Use Ollama to avoid API key requirements
        api_base="http://localhost:9999",  # Non-existent endpoint (will be caught by try-except)
        initial_currency=500.0,
        initial_skills={"crafting": 2.0, "smelting": 1.5},
        initial_tools=["hammer"]
    )


@pytest.fixture
def basic_pod_config(basic_worker_config):
    """Create a basic pod configuration for testing."""
    return PodConfig(
        name="pod_001",
        workers=[basic_worker_config],
        initial_inventory={"wood": 100.0, "metal": 50.0}
    )


@pytest.fixture
def basic_federation(basic_pod_config, resource_registry, recipe_registry):
    """Create a basic federation for testing (non-autonomous)."""
    fed = Federation(
        pods=[basic_pod_config],
        resource_registry=resource_registry,
        recipe_registry=recipe_registry,
        initialize_market=True,
        enable_cognitive_loop=False  # Disable for basic tests
    )
    fed.steps = 100  # Set initial step count for tests

    # Disable memory display to prevent Rich console issues in tests
    for pod in fed:
        for worker in pod:
            worker.memory.display = False

    return fed


@pytest.fixture
def basic_worker(basic_federation):
    """Get a worker from the basic federation."""
    pod = basic_federation[0]
    return list(pod)[0]


@pytest.fixture
def multi_pod_federation(resource_registry, recipe_registry):
    """Create a federation with multiple pods for integration testing."""
    worker1_config = WorkerConfig(
        name="worker_001",
        reasoning=CoTReasoning,
        llm_model="ollama/tinyllama",
        initial_currency=500.0,
        initial_skills={"crafting": 2.0}
    )

    worker2_config = WorkerConfig(
        name="worker_002",
        reasoning=CoTReasoning,
        llm_model="ollama/tinyllama",
        initial_currency=500.0,
        initial_skills={"smelting": 3.0}
    )

    pod1_config = PodConfig(
        name="pod_001",
        workers=[worker1_config],
        initial_inventory={"wood": 100.0}
    )

    pod2_config = PodConfig(
        name="pod_002",
        workers=[worker2_config],
        initial_inventory={"metal": 50.0}
    )

    return Federation(
        pods=[pod1_config, pod2_config],
        resource_registry=resource_registry,
        recipe_registry=recipe_registry,
        initialize_market=True
    )
