"""Resource module - Materials, Tools, Equipment, Consumables, and Inventories.

This module provides the foundational resource system for Libertas.
Resources are no longer stored in economy/ - they're used across production, economy, organization.
"""

from .resource import ResourceInfo, ResourceTag
from .material import Material
from .tool import Tool
from .equipment import Equipment
from .consumable import Consumable
from .inventory import (
    BaseInventory,
    WorkerInventory,
    PodInventory,
    FederationInventory,
    ResourceType
)
from .registry import ResourceRegistry

__all__ = [
    # Resource info
    'ResourceInfo',
    'ResourceTag',
    # Resource types
    'Material',
    'Tool',
    'Equipment',
    'Consumable',
    # Inventories
    'BaseInventory',
    'WorkerInventory',
    'PodInventory',
    'FederationInventory',
    'ResourceType',
    # Registry
    'ResourceRegistry',
]
