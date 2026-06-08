# Migration Guide: Old Resources → New Resources System

This guide helps you migrate code from the old `Resource` class to the new type-based system.

## Quick Reference

### Old System (Deprecated)
```python
from libertas.economy import Resource, Inventory

# Creating resources with boolean flags
wood = Resource(name="wood", base_value=10.0, is_tool=False)
hammer = Resource(name="hammer", base_value=50.0, is_tool=True, durability=100)

# Old inventory with quantities/instances
inventory = Inventory()
inventory.add(wood, 10.0)
inventory.quantities["wood"]  # Direct dict access
inventory.instances["hammer"]  # Direct dict access
```

### New System (Recommended)
```python
from libertas.resources import (
    ResourceInfo, Material, Tool, Equipment, Consumable,
    WorkerInventory, PodInventory, FederationInventory
)

# Creating resources with composition
wood = Material(info=ResourceInfo(name="wood", base_value=10.0), quantity=10.0)
hammer = Tool(info=ResourceInfo(name="hammer", base_value=50.0), durability=100)

# New inventories with type-specific storage
pod_inventory = PodInventory()
pod_inventory.add(wood, 10.0)
pod_inventory.materials["wood"]  # Type-safe access
pod_inventory.tools  # Dict of Tool instances
```

---

## Type Mapping

### Resource Types

| Old Approach | New Type | Use Case |
|-------------|----------|----------|
| `Resource(is_tool=False)` | `Material` | Fungible/bulk items (wood, stone, wheat) |
| `Resource(is_tool=True)` | `Tool` | Worker-held tools (hammer, saw) |
| N/A | `Equipment` | Pod-level machinery (lathe, CNC, forge) |
| N/A | `Consumable` | Single-use items (food, water, entertainment) |

### Inventory Types

| Old Approach | New Type | Use Case |
|-------------|----------|----------|
| `Inventory()` | `PodInventory` | Pod storage (materials + tools + equipment) |
| `Dict[str, List[Resource]]` | `WorkerInventory` | Worker personal tools |
| N/A | `FederationInventory` | Federation shared resource pool |

---

## Migration Steps

### Step 1: Update Resource Creation

**Before:**
```python
from libertas.economy import Resource

wood = Resource(name="wood", base_value=10.0)
hammer = Resource(name="hammer", base_value=50.0, is_tool=True, durability=100)
```

**After:**
```python
from libertas.resources import ResourceInfo, Material, Tool

wood = Material(
    info=ResourceInfo(name="wood", base_value=10.0),
    quantity=1.0
)

hammer = Tool(
    info=ResourceInfo(name="hammer", base_value=50.0),
    durability=100
)
```

### Step 2: Update Inventory Access

**Before:**
```python
# Direct dict access
quantity = inventory.quantities["wood"]
tools = inventory.instances["hammer"]

# Iteration
for name, qty in inventory.quantities.items():
    print(f"{name}: {qty}")
```

**After:**
```python
# Method access
quantity = inventory.get_material_quantity("wood")
tool_count = inventory.count_tools("hammer")

# Iteration
for material in inventory.materials.values():
    print(f"{material.info.name}: {material.quantity}")
```

### Step 3: Update Registry Usage

**Before:**
```python
registry = ResourceRegistry()
registry.register(Resource(name="wood", base_value=10.0))
resource = registry.get("wood")
```

**After:**
```python
from libertas.resources import ResourceRegistry, Material, ResourceInfo

registry = ResourceRegistry()
wood = Material(info=ResourceInfo(name="wood", base_value=10.0))
registry.register_material(wood)

# Get by type
material = registry.get_material("wood")
tool = registry.get_tool("hammer")
```

---

## New Features

### 1. Equipment (Pod-Level Machinery)

```python
from libertas.resources import ResourceInfo, Equipment

lathe = Equipment(
    info=ResourceInfo(name="lathe", base_value=500.0),
    durability=1000,
    capacity=2,  # 2 workers can use simultaneously
    maintenance_cost=50.0,
    maintenance_interval=100
)

# Equipment needs periodic maintenance
if lathe.needs_maintenance(current_step):
    cost = lathe.perform_maintenance(current_step)
```

### 2. Consumables (Worker Needs)

```python
from libertas.resources import ResourceInfo, Consumable

bread = Consumable(
    info=ResourceInfo(name="bread", base_value=2.0),
    need_type="hunger",
    satisfaction_value=0.4,
    quantity=10.0
)

# Workers consume to satisfy needs
satisfaction = bread.consume()  # Returns 0.4, quantity now 9.0
```

### 3. Worker Needs System

```python
from libertas.cognitive import WorkerNeeds

worker.needs = WorkerNeeds()

# Needs degrade over time
worker.needs.degrade_needs()

# Affect mood
worker.needs.affect_mood(worker.mood)

# Workers purchase to satisfy needs (via LLM tools)
# check_my_needs(), view_available_goods(), purchase_and_consume()
```

---

## Backward Compatibility

The old system is **maintained for backward compatibility** during the transition:

- Old `Resource` class still works
- Old `Inventory` with `quantities`/`instances` still works  
- Old tests continue to pass
- Both systems can coexist

However, **new code should use the new system** for:
- Type safety
- Better semantics
- Access to new features (Equipment, Consumables, Worker Needs)

---

## Common Pitfalls

### 1. Forgetting ResourceInfo wrapper

❌ **Wrong:**
```python
Material(name="wood", base_value=10.0)  # No 'name' field!
```

✅ **Correct:**
```python
Material(info=ResourceInfo(name="wood", base_value=10.0))
```

### 2. Accessing old dict fields directly

❌ **Wrong:**
```python
inventory.quantities["wood"]  # Direct access to old system
```

✅ **Correct:**
```python
inventory.get_material_quantity("wood")  # Use accessor method
```

### 3. Mixing old and new registries

❌ **Wrong:**
```python
registry.register(Material(...))  # Old method, new type
```

✅ **Correct:**
```python
registry.register_material(Material(...))  # Type-specific method
```

---

## Testing Your Migration

1. **Run existing tests:**
   ```bash
   python scripts/test.py --all
   ```

2. **Check for deprecation warnings** (future enhancement)

3. **Verify type safety:**
   ```python
   # New system provides type hints
   material: Material = registry.get_material("wood")
   tool: Tool = worker_inventory.get_tool("hammer")
   ```

---

## Timeline

- **Phase 1-2:** New system created, parallel tracking added ✅
- **Phase 3-5:** Resource creation updated, needs system added ✅
- **Phase 6:** Deprecation markers added ✅ (current)
- **Phase 7:** Documentation updated
- **Future:** Old system removal (major version bump)

---

## Getting Help

- See `REFACTOR_STATUS.md` for implementation details
- Check `src/libertas/tests/unit/resources/` for usage examples
- File issues at: https://github.com/ryansennis/libertas/issues
