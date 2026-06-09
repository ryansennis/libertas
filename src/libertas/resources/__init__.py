"""Resource module - Materials, Tools, Equipment, Consumables, Recipes, and Inventories.

This module provides the foundational resource system for Libertas.
Resources and their production recipes are tightly coupled - every resource must have a recipe.
"""

from .consumable import Consumable, ConsumableType
from .equipment import Equipment
from .inventory import (
    BaseInventory,
    WorkerInventory,
    PodInventory,
    FederationInventory
)
from .material import Material
from .recipes import Recipe, ProductionStep, StepType
from .registry import ResourceRegistry
from .resource import Resource, ResourceTag
from .tool import Tool

__all__ = [
    'BaseInventory',
    'Consumable',
    'ConsumableType',
    'Equipment',
    'FederationInventory',
    'Material',
    'PodInventory',
    'ProductionStep',
    'Recipe',
    'Resource',
    'ResourceRegistry',
    'ResourceTag',
    'StepType',
    'Tool',
    'WorkerInventory'
]
