# libertas

[![Test Suite](https://github.com/ryansennis/libertas/actions/workflows/test.yml/badge.svg)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ryansennis/libertas/main/.github/badges/coverage.json)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ryansennis/libertas/main/.github/badges/tests.json)](https://github.com/ryansennis/libertas/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Agent-based modeling framework for simulating constitutional democracies and economic systems with LLM-powered autonomous agents.

## Overview

Libertas enables large-scale experiments to identify optimal organizational structures for economic democracy. Autonomous agents read natural language constitutions, understand their rights and duties, propose motions, vote on decisions, and manage cooperative enterprises.

### Key Features

- **Constitutional Democracy**: Natural language constitutions that LLM agents can read and interpret
- **Democratic Governance**: Multiple voting systems (simple majority, supermajority, unanimous)
- **Economic Simulation**: Production, markets, resource management, and skill development
- **Autonomous Agents**: LLM-powered workers with cognitive architecture
  - Observe-reason-act loop for autonomous decision-making
  - Personality traits (OCEAN + political dimensions)
  - Dynamic mood system and episodic memory
  - Constitutional permission checking
- **Research-Oriented**: Designed for running batch experiments to derive statistical insights

## System Requirements

### Recommended Hardware

Libertas is designed for **large-scale agent-based simulations** with LLM-powered autonomous agents. To get the most out of the framework:

**Minimum** (for development/testing):
- CPU: 4+ cores
- RAM: 8GB
- Storage: 10GB (for model weights)
- Python 3.12+

**Recommended** (for serious research):
- CPU: 16+ cores (for parallel agent processing)
- RAM: 32GB+ (large simulations with many agents)
- GPU: NVIDIA GPU with 8GB+ VRAM (for local LLM inference)
- Storage: 50GB+ SSD (model weights, experiment data)
- OS: Linux or macOS (better for long-running simulations)

**For Large-Scale Experiments** (100+ agents, batch experiments):
- CPU: 32+ cores / Threadripper / Server CPU
- RAM: 64GB-128GB
- GPU: NVIDIA A100, H100, or multiple consumer GPUs
- Storage: NVMe SSD with 500GB+
- Consider: Cloud compute (AWS, GCP, Azure) with spot instances

### LLM Backend Options

1. **Local (Ollama)** - Best for serious research
   - Full control over models and inference
   - No API costs
   - Requires powerful GPU (RTX 3090, 4090, or better)
   - Install: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Cloud APIs** (OpenAI, Anthropic, etc.)
   - No hardware requirements
   - Pay per token (can get expensive with many agents)
   - Potential rate limits
   - Good for prototyping

3. **Remote Ollama Server**
   - Run Ollama on a powerful server
   - Access from lightweight machines
   - Good middle ground

**Note**: Integration and E2E tests run actual LLM inference, so they can take several minutes even on powerful hardware.

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
from libertas.cognitive import PersonalityTraits, Background
from libertas.governance import MotionType, VoteType
from mesa_llm.reasoning.cot import CoTReasoning

# Create autonomous workers with different personalities
alice_config = WorkerConfig(
    name="Alice",
    reasoning=CoTReasoning,
    llm_model="ollama/mistral",
    initial_currency=1000.0,
    personality=PersonalityTraits(
        openness=0.8,
        conscientiousness=0.7,
        extraversion=0.6,
        agreeableness=0.7,
        neuroticism=0.3,
        economic_left_right=-0.7,  # Collectivist
        authority_libertarian=0.5   # Libertarian
    ),
    background=Background(
        education_level=4,
        years_experience=5
    )
)

pod_config = PodConfig(
    name="TechCoop",
    workers=[alice_config],
    initial_inventory={"wood": 1000.0}
)

federation = Federation(pods=[pod_config])

# Workers autonomously observe, reason, decide, and act each step
for step in range(100):
    federation.step()  # Workers automatically participate in governance and economy

# Check worker state
pod = federation[0]
alice = list(pod)[0]
print(f"Alice's mood: happiness={alice.mood.happiness:.2f}, stress={alice.mood.stress:.2f}")
print(f"Alice's memory: {len(alice.episodic_memory)} episodes")
print(f"Alice's currency: ${alice.currency:.2f}")
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
- Cognitive loop: observe → reason → decide → act
- Personality traits (OCEAN model + political dimensions)
- Dynamic mood system (happiness, stress, motivation)
- Episodic memory (rolling 100-entry window)
- Skills, tools, and personal currency
- Production work and market trading
- Autonomous voting and governance participation

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
│   ├── cognitive/            # Agent cognitive architecture
│   │   └── __init__.py       # PersonalityTraits, Background, MoodState
│   └── tools/                # LLM tools for agents
│       ├── economic_tools.py # Production, trading, invention
│       └── governance_tools.py # Voting, motions, constitution
├── scripts/                  # Utility scripts
│   ├── test.py              # Unified test runner
│   └── setup_ollama.py      # LLM setup
└── tests/                   # Test suite
    ├── unit/                # Fast component tests
    ├── integration/         # Interaction tests
    └── e2e/                 # Full simulation tests
```

## Performance Tips

### Optimizing for Large Simulations

1. **Use Local LLMs**: Ollama with a good GPU is 10-100x faster than cloud APIs
2. **Batch Experiments**: Run multiple simulations in parallel on multi-core systems
3. **Profile First**: Use small simulations to tune before scaling up
4. **Model Selection**: Smaller models (7B-13B) can be sufficient for many experiments
5. **Checkpoint Often**: Save simulation state periodically for long-running experiments
6. **Monitor Resources**: Watch CPU, RAM, and GPU usage to find bottlenecks

### Expected Performance

With a high-end consumer setup (RTX 4090, 32GB RAM, 16-core CPU):
- **Unit tests**: ~4 seconds
- **Integration tests**: ~30 seconds (with LLM calls)
- **Small simulation** (5 agents, 100 steps): ~5-10 minutes
- **Medium simulation** (20 agents, 1000 steps): ~1-2 hours
- **Large simulation** (100 agents, 1000 steps): Several hours to days
- **Batch experiments** (10 runs × medium): Can parallelize across cores

### Scaling Strategies

For research requiring massive scale:
1. **Cloud Computing**: Use AWS/GCP with GPU instances (p3, g5, a2 instances)
2. **Cluster Computing**: Distribute across multiple machines
3. **Model Caching**: Cache LLM responses for repeated queries
4. **Reduced Reasoning**: Limit LLM calls to critical decision points
5. **Hierarchical Simulation**: Use simpler agents for routine tasks

## Development

### Contributing

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development setup and guidelines.

### Roadmap

**Phase 1: Foundation** ✅ **COMPLETE**
- ✅ Constitutional democracy system
- ✅ Voting and governance engine
- ✅ Integration with Federation/Pod
- ✅ Agent personality and mood
- ✅ LLM tools for governance (EconomicTools, GovernanceTools)
- ✅ Cognitive attributes (PersonalityTraits, Background, MoodState)

**Phase 2: Cognitive Architecture** ✅ **COMPLETE**
- ✅ Agent observe-reason-act loop
- ✅ Memory systems (episodic memory)
- ✅ Permission checking before actions
- ✅ Autonomous decision-making
- ✅ LLM-based reasoning with personality influence
- ✅ Dynamic mood updates
- ✅ Semantic memory
- ✅ Explicit goal tracking

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
