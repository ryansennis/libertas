# Scripts

Utility scripts for the libertas project.

## Test Runner (`test.py`)

Comprehensive test runner with filtering, coverage, and category selection.

### Quick Start

```bash
# Run all tests
python scripts/test.py

# Run unit tests only (fastest)
python scripts/test.py --quick

# Run with coverage report
python scripts/test.py --coverage

# Filter tests by name
python scripts/test.py --filter worker
```

### Usage

```
python scripts/test.py [options]

Options:
  --all              Run all tests (default)
  --quick            Run unit tests only (fastest)
  --unit             Run unit tests
  --integration      Run integration tests
  --e2e              Run end-to-end tests
  --coverage         Generate coverage report
  --filter PATTERN   Filter tests by name pattern
  --list             List available tests without running
  -x, --failfast     Stop on first failure
  -s, --show-output  Show test output (don't capture)
  -v, --verbose      Verbose output
  -h, --help         Show help message
```

### Examples

```bash
# Run only unit tests (fast feedback loop)
python scripts/test.py --quick

# Run integration tests only
python scripts/test.py --integration

# Run all tests with coverage report
python scripts/test.py --coverage

# Run only tests matching "worker"
python scripts/test.py --filter worker

# Run unit tests with coverage
python scripts/test.py --unit --coverage

# List all available tests without running
python scripts/test.py --list

# List tests matching "market"
python scripts/test.py --filter market --list

# Run tests and stop on first failure
python scripts/test.py -x

# Run tests with verbose output
python scripts/test.py -v

# Combine options
python scripts/test.py --unit --filter production -x -v
```

### Test Categories

- **Unit Tests** (248 tests): Fast, isolated tests for individual components
- **Integration Tests** (6 tests): Test interactions between components
- **E2E Tests** (5 tests): Full simulation scenarios

**Total: 259 tests | Coverage: 91.54%** 🎯

### Coverage Reports

When running with `--coverage`, reports are generated in two formats:
- **Terminal**: Summary in console output
- **HTML**: Detailed report at `htmlcov/index.html`

```bash
# Generate coverage report
python scripts/test.py --coverage

# Open HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Ollama Setup (`setup_ollama.py`)

Script for setting up and managing the Ollama server for LLM testing.

```bash
python scripts/setup_ollama.py
```

---

## Development Workflow

### Fast Feedback Loop

During active development, use quick tests for fast feedback:

```bash
# Run only unit tests (completes in ~0.5s)
python scripts/test.py --quick
```

### Before Committing

Run full test suite with coverage:

```bash
# Verify all tests pass with good coverage
python scripts/test.py --coverage
```

### Debugging Specific Tests

Filter to specific test areas:

```bash
# Debug worker-related tests
python scripts/test.py --filter worker -s -x

# Debug market tests with verbose output
python scripts/test.py --filter market -v
```

### CI/CD

For continuous integration, run all tests:

```bash
python scripts/test.py --all
```
