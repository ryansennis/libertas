# Contributing to Libertas

Thank you for your interest in contributing to Libertas! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.11 or 3.12
- Git
- Ollama (for integration and e2e tests)

### Initial Setup

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/libertas.git
cd libertas
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install development dependencies**:
```bash
pip install -e ".[dev]"
```

4. **Set up Ollama** (optional, for full test suite):
```bash
python scripts/setup_ollama.py
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes
- Write clean, documented code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests Locally

**Quick check (before committing)**:
```bash
python scripts/test.py --quick
```

**Full test suite**:
```bash
python scripts/test.py --all --coverage
```

**Specific test categories**:
```bash
python scripts/test.py --unit        # Unit tests only
python scripts/test.py --integration # Integration tests
python scripts/test.py --e2e        # End-to-end tests
```

### 4. Check Code Quality

**Format code**:
```bash
black src/
isort src/
```

**Lint**:
```bash
ruff check src/
```

**Type checking** (optional):
```bash
mypy src/ --ignore-missing-imports
```

### 5. Commit Changes
```bash
git add .
git commit -m "Brief description of changes

Detailed explanation if needed.

Co-Authored-By: Your Name <your.email@example.com>"
```

### 6. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### PR Description
Include:
- **Purpose**: What problem does this solve?
- **Changes**: What did you change?
- **Testing**: How did you test this?
- **Screenshots**: If UI-related (not applicable yet)

### PR Checklist
- [ ] Tests pass locally (`python scripts/test.py --all`)
- [ ] Code is formatted (`black src/`)
- [ ] New tests added for new features
- [ ] Documentation updated if needed
- [ ] Coverage maintained or improved (≥90%)
- [ ] No breaking changes (or clearly documented)

### CI Checks
Your PR will automatically run:
1. **Quick Test & Lint** (~2 min): Fast feedback
2. **Full Test Suite** (~15 min): Comprehensive testing
3. **Coverage Report**: Posted as comment

All checks must pass before merging.

## Writing Tests

### Test Organization
```
tests/
├── unit/          # Fast, isolated tests
├── integration/   # Component interaction tests
└── e2e/          # Full simulation tests
```

### Test Guidelines

**Unit Tests**:
- Fast (<1s per test)
- Isolated (mock external dependencies)
- Test single component
- Use `@pytest.mark.unit` decorator

```python
import pytest
from libertas.economy import Inventory, Resource

@pytest.mark.unit
def test_inventory_add():
    """Test adding resources to inventory."""
    inv = Inventory()
    wood = Resource("wood", "system", base_value=10.0)
    
    result = inv.add(wood, 50.0)
    
    assert result is True
    assert inv.get_quantity("wood") == 50.0
```

**Integration Tests**:
- Test component interactions
- May use real Ollama
- Use `@pytest.mark.integration` decorator

**E2E Tests**:
- Full simulation scenarios
- Test complete workflows
- Use `@pytest.mark.e2e` decorator

### Test Fixtures

Use shared fixtures from `conftest.py`:
```python
def test_worker_in_federation(basic_federation):
    """Test using shared fixture."""
    pod = basic_federation[0]
    worker = list(pod)[0]
    assert worker.pod == pod
```

### Coverage Requirements
- Maintain ≥90% overall coverage
- New code should have ≥95% coverage
- Critical paths must be 100% covered

## Code Style

### Python Style
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for public functions

### Example:
```python
from typing import Dict, Optional

def calculate_total_value(
    inventory: Dict[str, float],
    prices: Dict[str, float]
) -> float:
    """Calculate total inventory value.
    
    Args:
        inventory: Resource name to quantity mapping
        prices: Resource name to price mapping
        
    Returns:
        Total value of inventory
    """
    return sum(
        quantity * prices.get(resource, 0)
        for resource, quantity in inventory.items()
    )
```

### Documentation
- Use clear, concise docstrings
- Include examples for complex functionality
- Update README.md for user-facing changes
- Update WORKFLOWS.md for CI/CD changes

## Project Structure

```
libertas/
├── src/libertas/
│   ├── economy/         # Economic system
│   ├── organization/    # Workers, Pods, Federation
│   └── tools/          # LLM tools for agents
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── .github/
    └── workflows/      # CI/CD pipelines
```

## Common Tasks

### Adding a New Feature

1. **Write tests first** (TDD):
```bash
# Create test file
touch tests/unit/feature/test_new_feature.py

# Write failing tests
python scripts/test.py --filter test_new_feature
```

2. **Implement feature**:
```python
# src/libertas/feature/new_feature.py
```

3. **Verify tests pass**:
```bash
python scripts/test.py --filter test_new_feature
python scripts/test.py --all  # Full suite
```

4. **Check coverage**:
```bash
python scripts/test.py --filter test_new_feature --coverage
```

### Fixing a Bug

1. **Write test that reproduces bug**
2. **Verify test fails**
3. **Fix bug**
4. **Verify test passes**
5. **Run full test suite**

### Adding Dependencies

1. **Add to `pyproject.toml`**:
```toml
[project]
dependencies = [
    "existing-dep>=1.0",
    "new-dep>=2.0",
]
```

2. **Install and test**:
```bash
pip install -e ".[dev]"
python scripts/test.py --all
```

3. **Document usage** in appropriate files

## Release Process

Releases are automated but require version tags:

1. **Update version** in `pyproject.toml`
2. **Create tag**:
```bash
git tag -a v0.1.0 -m "Release 0.1.0: Description"
git push origin v0.1.0
```

3. **CI automatically**:
   - Runs full test suite
   - Creates GitHub release
   - Publishes to PyPI (if configured)
   - Creates announcement

See [WORKFLOWS.md](WORKFLOWS.md) for details.

## Getting Help

- **Documentation**: Check README.md and WORKFLOWS.md
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Tests**: Look at existing tests for examples

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project guidelines

## Recognition

Contributors are recognized:
- In commit messages (`Co-Authored-By`)
- In release notes
- In CONTRIBUTORS.md (coming soon)

Thank you for contributing to Libertas! 🎉
