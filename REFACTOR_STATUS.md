# Resources & Inventory Refactor - Status Report

**Branch:** `refactor/resources-inventory-hierarchy`  
**Date:** 2026-06-08  
**Status:** In Progress (Phase 3 of 7)

---

## Test Results ✅

### All Tests Passing
- **Unit Tests:** 653 passed ✅
- **Integration Tests:** 38 passed ✅
- **Total:** 691 tests passed ✅
- **Test Coverage:** 88.24% (required: 70%) ✅
- **Warnings:** 548 (mostly deprecation warnings from Mesa framework)

### Test Command
```bash
source .venv/bin/activate
python scripts/test.py --all --coverage
```

---

## Completed Phases

### ✅ Phase 1: Create New Module Structure
**Commit:** `44e9b72`

Created `src/libertas/resources/` module with:
- `ResourceInfo` - shared metadata component (composition, not inheritance)
- `Material` - fungible resources (wood, stone, wheat) 
- `Tool` - worker-held tools with durability (hammer, saw)
- `Equipment` - pod-level machinery (lathe, CNC, forge)
- `Consumable` - worker needs items (food, water, entertainment)
- `WorkerInventory` - capacity-limited tool storage
- `PodInventory` - shared materials/tools/equipment/consumables
- `FederationInventory` - federation-level resource pool
- `ResourceRegistry` - type-specific registration

**Files:** 8 new files, 926 lines  
**Design:** Composition over inheritance, collection-based like `Federation[Pod]`

---

### ✅ Phase 2: Add Parallel Inventory Systems
**Commit:** `1dab754`

Updated existing classes to track both old and new systems:
- **Inventory class** - added `materials`, `tools`, `equipment`, `consumables` dicts
  - `add()` writes to BOTH old (quantities/instances) and new systems
  - `remove()` removes from BOTH systems
  - `get_quantity()` reads from new system with fallback
  - Serialization handles both formats
- **Worker class** - added `inventory: WorkerInventory` field
- **Federation class** - added `shared_inventory: FederationInventory` field

**Result:** Both systems work in parallel, no breaking changes

---

### 🔄 Phase 3: Update Resource Creation Sites (PARTIAL)
**Commit:** `f28f964`

Progress so far:
- ✅ Updated test fixtures in `conftest.py` to create `Material` and `Tool` instances
- ✅ Enhanced old `ResourceRegistry` with new methods:
  - `register_material()`, `register_tool()`, `register_equipment()`, `register_consumable()`
  - `get_material()`, `get_tool()` for new types
- ✅ Dual registration: both old `Resource` and new types for backward compatibility

**Remaining work:**
- Update production recipes to specify Material/Tool/Equipment outputs
- Update all direct `Resource(...)` instantiations across codebase
- Update federation resource initialization

---

## Remaining Phases

### ⏳ Phase 4: Migrate Inventory Access
- Replace direct `.quantities` access → `.materials` or `.get_material_quantity()`
- Replace direct `.instances` access → `.tools` or `.count_tools()`
- Update pod.py inventory methods (get_inventory_summary, get_tools_summary)
- Update worker.py observations
- Update all test assertions

### ⏳ Phase 5: Add Worker Needs System
- Create `src/libertas/cognitive/needs.py` with `WorkerNeeds` class
- Create `src/libertas/tools/needs_tools.py` with LLM-callable tools
- Add hunger, thirst, rest, recreation, housing tracking
- Integrate with mood system
- Add purchasing decisions via LLM

### ⏳ Phase 6: Remove Old Systems
- Delete `quantities` and `instances` from Inventory
- Delete old `Resource` class (keep ResourceInfo)
- Remove parallel-write logic
- Clean up imports

### ⏳ Phase 7: Documentation & Roadmap
- Update README with new resource system
- Add roadmap entry for Phase 4 completion
- Update docstrings
- Document economic circulation loop

---

## Architecture Overview

### Resource Types (Composition-Based)
```python
# NOT inheritance - composition!
Material(info=ResourceInfo(...), quantity=1.0)
Tool(info=ResourceInfo(...), durability=100)
Equipment(info=ResourceInfo(...), maintenance_cost=50.0)
Consumable(info=ResourceInfo(...), need_type="hunger")
```

### Inventory Hierarchy (Collection-Based)
```python
WorkerInventory    # holds tools only, capacity-limited
PodInventory       # holds all types, shared by pod
FederationInventory # holds all types, federation pool
```

### Parallel Systems (Migration Strategy)
```python
# OLD SYSTEM (Phase 6 removal)
inventory.quantities = {"wood": 100.0}
inventory.instances = {"hammer": [tool1, tool2]}

# NEW SYSTEM (active now)
inventory.materials = {"wood": Material(...)}
inventory.tools = {"uuid1": Tool(...)}
```

---

## Key Metrics

- **Total Commits:** 3 (Phase 1, 2, 3 partial)
- **Files Changed:** 13
- **Lines Added:** ~1,164
- **Tests:** All passing (691/691)
- **Coverage:** 88.24%
- **Breaking Changes:** 0 (parallel systems prevent breaks)

---

## Next Steps

1. Complete Phase 3 - update remaining resource creation sites
2. Begin Phase 4 - migrate inventory access patterns
3. Continue with Phases 5-7

**Estimated remaining work:** ~4-6 hours for Phases 4-7

---

## Notes

- Using composition (ResourceInfo) instead of inheritance prevents type hierarchy complexity
- Parallel systems allow incremental migration without breaking existing code
- New inventory classes follow Mesa's collection pattern (like `Federation[Pod]`)
- All tests maintained throughout refactor - no regression
