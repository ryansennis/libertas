# libertas
Agent-based modeling of organizational structures.

## Quick Start

### Running Tests

```bash
# Run all tests
python scripts/test.py

# Run unit tests only (fast)
python scripts/test.py --quick

# Run with coverage
python scripts/test.py --coverage

# Filter by name
python scripts/test.py --filter worker
```

See [scripts/README.md](scripts/README.md) for detailed testing documentation.

### Test Categories

- **248 Unit Tests**: Fast, isolated component tests
- **6 Integration Tests**: Component interaction tests
- **5 E2E Tests**: Complete simulation scenarios

**Total: 259 tests with 100% passing**
**Coverage: 91.54%** 🎯
