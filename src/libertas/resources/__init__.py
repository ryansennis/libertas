"""Resource module - Materials, Tools, Equipment, Consumables, Recipes, and Inventories.

This module provides the foundational resource system for Libertas.
Resources and their production recipes are tightly coupled - every resource must have a recipe.
"""

from .resource import Resource, ResourceTag
from .material import Material
from .tool import Tool
from .equipment import Equipment
from .consumable import Consumable, ConsumableType
from .recipes import Recipe, ProductionStep, StepType
from .inventory import (
    BaseInventory,
    WorkerInventory,
    PodInventory,
    FederationInventory
)
from .registry import ResourceRegistry

__all__ = [
    # Resource base class and tags
    'Resource',
    'ResourceTag',
    # Resource types
    'Material',
    'Tool',
    'Equipment',
    'Consumable',
    'ConsumableType',
    # Recipes
    'Recipe',
    'ProductionStep',
    'StepType',
    # Inventories
    'BaseInventory',
    'WorkerInventory',
    'PodInventory',
    'FederationInventory',
    # Registry
    'ResourceRegistry',
]
