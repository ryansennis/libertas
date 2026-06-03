# libertas

[![Test Suite](https://github.com/ryansennis/libertas/actions/workflows/test.yml/badge.svg)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-91.54%25-brightgreen)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Tests](https://img.shields.io/badge/tests-259%20passing-brightgreen)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Agent-based modeling of organizational structures for economic democracy research.

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
