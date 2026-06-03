# libertas

[![Test Suite](https://github.com/ryansennis/libertas/actions/workflows/test.yml/badge.svg)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-93.07%25-brightgreen)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Tests](https://img.shields.io/badge/tests-310%20passing-brightgreen)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Agent-based modeling framework for simulating constitutional democracies and economic systems with LLM-powered autonomous agents.

## Overview

Libertas enables large-scale experiments to identify optimal organizational structures for economic democracy. Autonomous agents read natural language constitutions, understand their rights and duties, propose motions, vote on decisions, and manage cooperative enterprises.

### Key Features

- **Constitutional Democracy**: Natural language constitutions that LLM agents can read and interpret
- **Democratic Governance**: Multiple voting systems (simple majority, supermajority, unanimous)
- **Economic Simulation**: Production, markets, resource management, and skill development
- **Autonomous Agents**: LLM-powered workers with personalities, backgrounds, and decision-making
- **Research-Oriented**: Designed for running batch experiments to derive statistical insights

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ryansennis/libertas.git
cd libertas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all dependencies (dev + test)
pip install -e ".[all]"
```

### Basic Usage

```python
from libertas.organization import Federation, PodConfig, WorkerConfig
from libertas.governance import MotionType, VoteType
from mesa_llm.reasoning.cot import CoTReasoning

# Create a federation with a worker cooperative
worker_config = WorkerConfig(
    name="Alice",
    reasoning=CoTReasoning,
    llm_model="ollama/mistral",
    initial_skills={"crafting": 3.0}
)

pod_config = PodConfig(
    name="TechCoop",
    workers=[worker_config]
)

federation = Federation(pods=[pod_config])

# Read the constitution
constitution_text = federation.constitution.get_full_text()
print(constitution_text)

# Propose and vote on a motion
pod = list(federation)[0]
workers = list(pod)

motion = federation.governance.propose_motion(
    proposer=workers[0],
    motion_type=MotionType.PRODUCTION_PRIORITY,
    title="Focus on tool production",
    description="Prioritize tool manufacturing this quarter",
    scope="pod",
    eligible_voters={w.unique_id for w in workers},
    voting_duration=50
)

# Workers vote
federation.governance.cast_vote(motion.motion_id, workers[0].unique_id, "for")

# Run simulation
for step in range(100):
    federation.steps = step
    federation.step()
```

### Running Tests

```bash
# Unit tests (fast, ~4s)
python scripts/test.py --quick

# Unit tests with coverage
python scripts/test.py --unit --coverage

# Governance tests only
python scripts/test.py --filter governance

# Integration tests (requires Ollama)
python scripts/test.py --integration
```

See [scripts/README.md](scripts/README.md) for detailed testing documentation.

## Architecture

### Governance Module

**Constitutional System** (`libertas.governance.constitution`):
- `Constitution`: Living documents with natural language articles
- `Article`: Individual rules with permissions and requirements
- `AmendmentProposal`: Democratic amendment process
- Permission checking based on agent attributes (currency, skills, etc.)
- Default constitutions for federations and pods

**Voting System** (`libertas.governance.voting`):
- `Motion`: Proposals with configurable vote types
- `GovernanceEngine`: Central coordinator for all voting processes
- Support for simple majority, supermajority, unanimous voting
- Motion lifecycle: propose → vote → tally → execute
- Scope-based queries (federation, pod, custom)

### Economic System

**Production** (`libertas.economy.production`):
- Time-dependent production jobs with multiple steps
- Worker skill requirements and tool dependencies
- Batch processing and parallel production
- Skill improvement through practice

**Markets** (`libertas.economy.market`):
- Order book for buy/sell orders
- Price discovery through supply/demand
- Resource trading between pods
- Market statistics and price history

**Resources** (`libertas.economy.resource`):
- Bulk resources (wood, metal, food)
- Tools with durability and skill requirements
- Resource invention system
- Registry for global resource definitions

### Organizational Structure

**Federation** (`libertas.organization.federation`):
- Top-level coordinator for multiple pods
- Constitutional governance at federation level
- Market facilitation between pods
- Resource and recipe registries

**Pod** (`libertas.organization.pod`):
- Worker cooperatives with shared inventory
- Individual pod constitutions (direct democracy by default)
- Production queue management
- Worker networks for collaboration

**Worker** (`libertas.organization.worker`):
- LLM-powered autonomous agents (via mesa_llm)
- Skills, tools, and personal currency
- Production work and market trading
- Future: personality, mood, political views

## Research Applications

Libertas enables experiments to answer questions like:

- **Governance**: Do direct democracies or hierarchies produce better economic outcomes?
- **Participation**: What constitutional features maximize voting participation?
- **Innovation**: Do worker-owned firms invent more resources and processes?
- **Stability**: Which structures are most resilient to economic shocks?
- **Well-being**: What ownership models maximize worker happiness?
- **Efficiency**: Do cooperatives or corporations reach higher productivity?
- **Inequality**: How do different structures affect wealth distribution?
- **Evolution**: Which constitutions adapt best through amendments?

### Running Experiments

```python
from libertas.organization import Federation, PodConfig, WorkerConfig

# Design experiment with different governance structures
configs = [
    {"governance": "democracy", "pods": 5},
    {"governance": "hierarchy", "pods": 5}
]

results = []
for config in configs:
    # Create federation with specified structure
    federation = create_federation(config)
    
    # Run simulation
    for step in range(1000):
        federation.steps = step
        federation.step()
    
    # Collect metrics
    results.append({
        "governance": config["governance"],
        "avg_happiness": get_avg_happiness(federation),
        "total_production": get_total_production(federation),
        "voting_rate": get_voting_rate(federation)
    })

# Analyze results
compare_governance_models(results)
```

## Project Structure

```
libertas/
├── src/libertas/
│   ├── economy/              # Economic system
│   │   ├── inventory.py      # Resource storage
│   │   ├── market.py         # Trading system
│   │   ├── production.py     # Jobs and recipes
│   │   └── resource.py       # Resources and tools
│   ├── governance/           # Democratic systems
│   │   ├── constitution.py   # Constitutional framework
│   │   └── voting.py         # Voting and motions
│   ├── organization/         # Organizational structure
│   │   ├── federation.py     # Top-level coordinator
│   │   ├── pod.py            # Worker cooperatives
│   │   └── worker.py         # LLM-powered agents
│   └── tools/                # LLM tools for agents
│       └── economic_tools.py # Production, trading, invention
├── scripts/                  # Utility scripts
│   ├── test.py              # Unified test runner
│   └── setup_ollama.py      # LLM setup
└── tests/                   # Test suite
    ├── unit/                # Fast component tests
    ├── integration/         # Interaction tests
    └── e2e/                 # Full simulation tests
```

## Development

### Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development setup and guidelines.

### Roadmap

**Phase 1: Foundation** (In Progress)
- ✅ Constitutional democracy system
- ✅ Voting and governance engine
- ✅ Integration with Federation/Pod
- ⏳ Agent personality and mood
- ⏳ LLM tools for governance

**Phase 2: Cognitive Architecture**
- Agent observe-reason-act loop
- Memory systems (episodic, semantic)
- Permission checking before actions
- Autonomous decision-making

**Phase 3: Advanced Economics**
- Time-dependent tasks
- Banking and loans
- Contracts and negotiations
- Resource invention
- Supply chains

**Phase 4: Experimentation**
- Batch experiment framework
- Metrics collection
- Statistical analysis
- Visualization dashboards

## License

MIT License - see [LICENSE](LICENSE) for details.
