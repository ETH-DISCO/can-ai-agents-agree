# Byzantine Consensus Game (BCG)

A multi-agent simulation framework for studying Byzantine fault tolerance using LLM-based agents. This is the code accompanying the ICLR 2026 Workshop paper.

## Overview

This implementation simulates a Byzantine Consensus Game where:
- **Honest agents** try to reach consensus on a common integer value
- **Byzantine agents** attempt to disrupt consensus (identity hidden during simulation)
- All agents use **vLLM** with structured JSON output for decision-making
- Communication happens through the **A2A-sim protocol** (Agent-to-Agent)
- Consensus achieved through **supermajority voting** (≥66% agreement)

## Project Structure

```
byzantine_consensus_game/
├── main.py                    # Entry point - run single simulations
├── config.py                  # All configuration (models, game params, LLM settings)
├── byzantine_consensus.py     # Game logic and state management
├── bcg_agents.py              # LLM-based agent implementations (honest + Byzantine)
├── vllm_agent.py              # vLLM interface with JSON schema enforcement
├── a2a_sim.py                 # A2A-sim communication protocol
├── agent_network.py           # Network topology management
├── communication_protocol.py  # Protocol abstractions
├── protocol_factory.py        # Protocol instantiation
├── requirements.txt           # Python dependencies
└── results/analysis/          # Aggregated experiment results
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 24GB+ VRAM for 14B models)
- vLLM 0.6.0+

## Installation

```bash
pip install -r requirements.txt
```

## Models

The framework supports the following models (configured in `config.py`):

| Model | HuggingFace ID | Parameters |
|-------|----------------|------------|
| Qwen3-8B | `Qwen/Qwen3-8B` | 8B |
| Qwen3-14B | `Qwen/Qwen3-14B` | 14B (default) |
| Qwen3-32B | `Qwen/Qwen3-32B` | 32B |
| Mistral-22B | `mistralai/Mistral-Small-Instruct-2409` | 22B |

## Usage

### Reproducing Paper Experiments

**Q1 - Cooperative consensus (no Byzantine agents):**
```bash
# 8 honest agents, may_exist prompt (agents told Byzantine may exist)
python main.py --honest 8 --byzantine 0 --byzantine-awareness may_exist

# 4 honest agents, none_exist prompt (agents told no Byzantine exist)
python main.py --honest 4 --byzantine 0 --byzantine-awareness none_exist
```

**Q2 - Byzantine resilience:**
```bash
# 8 honest + 2 Byzantine agents
python main.py --honest 8 --byzantine 2 --byzantine-awareness may_exist
```

### All CLI Options

```bash
python main.py \
  --honest 8 \              # Number of honest agents (default: 8)
  --byzantine 0 \           # Number of Byzantine agents (default: 0)
  --rounds 50 \             # Max rounds before deadline (default: 50)
  --threshold 66.0 \        # Consensus threshold % (default: 66.0)
  --value-range 0-50 \      # Initial value range (default: 0-50)
  --byzantine-awareness may_exist \  # may_exist or none_exist (default: may_exist)
  --verbose                 # Print detailed output
```

## Configuration

All parameters are centralized in `config.py`:

```python
# Game parameters (matching cluster experiments)
BCG_CONFIG = {
    "num_honest": 8,
    "num_byzantine": 0,
    "value_range": (0, 50),
    "consensus_threshold": 66.0,
    "max_rounds": 50,
}

# LLM generation settings (single source of truth)
LLM_CONFIG = {
    "temperature_decide": 0.5,   # Proposal decisions
    "temperature_vote": 0.3,     # Termination votes
    "max_tokens_decide": 300,
    "max_tokens_vote": 200,
    "max_json_retries": 3,
}
```

## How It Works

1. **Initialization**: Agents receive random integer values from the value range [0, 50]
2. **Each Round**:
   - **Decision Phase**: LLM generates JSON with new proposed value and reasoning
   - **Broadcast Phase**: Agents broadcast values via A2A protocol
   - **Receive Phase**: Agents receive all proposals and update state
   - **Voting Phase**: LLM decides whether to vote stop (consensus) or continue
3. **Termination**: Game ends when ≥66% vote to stop, or max rounds (50) reached

## Agent Types

- **Honest Agents**: Use LLM to analyze proposals, track suspicion, and converge toward consensus
- **Byzantine Agents**: Use LLM to strategically disrupt honest consensus while appearing cooperative

## License

MIT
