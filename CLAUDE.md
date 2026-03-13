# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Branch: `expanded_walk`

This branch extends the random walks experiment to analyze how steering interventions affect activation trajectory dynamics. Key additions:
- New `trajectory` module for capturing activation trajectories with intervention support
- New `intervention_geometry.py` experiment comparing trajectory dynamics across conditions
- 29 new tests for trajectory analysis (100 total tests now)

**PR**: https://github.com/AlliedToasters/llama-introspection/pull/3

## Project Overview

This project reproduces and extends experiments from Anthropic's introspection paper on LLaMA-3.1-Instruct models. It investigates whether language models can detect when "thoughts" (activation patterns) are injected into their neural networks during generation.

Paper reference: https://transformer-circuits.pub/2025/introspection/index.html

Key extension: Adds a "natural activation scaling" control condition (Neel Nanda's suggestion) to test whether detection is due to injected vectors being anomalously large vs. simply amplifying natural activation patterns.

## Commands

### Setup
```bash
# Install package in development mode
pip install -e ".[dev]"

# Copy .env.template to .env and add API keys:
# - NNSIGHT_API_KEY (for remote model execution via NDIF)
# - ANTHROPIC_API_KEY (for Claude-as-judge evaluation)
```

### Run Tests
```bash
pytest                    # Run all tests (100 tests)
pytest tests/ -v          # Verbose output
pytest tests/test_trajectory.py      # Trajectory module tests
pytest tests/test_experiment_e2e.py  # End-to-end tests only
```

### Generate Prompt Configurations
```bash
python experiments/generate_prompts.py  # Creates prompts.pt
```

### Pre-compute Steering Vectors
```bash
python experiments/batch_compute.py              # Local models (1B, 8B)
python experiments/batch_compute.py --use-remote # Include remote models (70B, 405B)
```

### Run Experiments

Single experiment with specific conditions:
```bash
python experiments/introspection.py --model 1B --condition concept --concept ocean --n-trials 5
python experiments/introspection.py --model 8B --condition random --n-trials 10
python experiments/introspection.py --model 70B --condition scale --use-remote
```

Full battery test:
```bash
python experiments/battery.py --model 8B \
  --n-control-trials 18 \
  --n-random-trials 18 \
  --n-concept-trials 1 \
  --n-scale-trials 18
```

### Analysis
```bash
python experiments/plot_vector_norms.py --results-dir results/
python experiments/compare_walks.py --model 1B --n-gens 5
```

### Trajectory / Intervention Geometry (expanded_walk branch)
```bash
# Compare trajectory dynamics across intervention types
python experiments/intervention_geometry.py --model 1B --n-trials 3

# With specific concepts and strengths
python experiments/intervention_geometry.py --model 8B \
  --concepts ocean fire music \
  --strengths 0.5 1.0 2.0 \
  --scale-factors 2.0 4.0

# Test intervention propagation (diagnostic)
python experiments/test_intervention_propagation.py
```

## Architecture

### Package Structure
```
llama-introspection/
├── src/llama_introspection/      # Core library
│   ├── __init__.py
│   ├── models.py                 # Model shortcuts, config
│   ├── steering.py               # Steering vector computation
│   ├── evaluation.py             # Claude-as-judge grading
│   ├── geometry.py               # Vector norm/distance metrics
│   ├── trajectory.py             # Activation trajectory analysis (NEW)
│   └── utils.py                  # KL divergence, token analysis
├── experiments/                   # Experiment scripts
│   ├── introspection.py          # Main experiment runner
│   ├── battery.py                # Batch test runner
│   ├── batch_compute.py          # Pre-compute vectors
│   ├── generate_prompts.py       # Generate prompt configs
│   ├── compare_walks.py          # Random walk analysis
│   ├── intervention_geometry.py  # Trajectory dynamics experiment (NEW)
│   ├── test_intervention_propagation.py  # Diagnostic script (NEW)
│   ├── plot_vector_norms.py      # Result visualization
│   └── activation_geometry_ablation.py
├── tests/                         # Test suite
│   ├── conftest.py               # Fixtures (tiny model)
│   ├── mocks.py                  # Mock clients for testing
│   ├── test_evaluation.py        # Grading tests
│   ├── test_experiment_e2e.py    # End-to-end tests
│   ├── test_geometry.py
│   ├── test_models.py
│   ├── test_steering.py
│   ├── test_trajectory.py        # Trajectory module tests (NEW - 29 tests)
│   └── test_utils.py
├── config.py                      # Legacy wrapper (re-exports from package)
├── steering_vectors.py            # Legacy wrapper (re-exports from package)
├── util.py                        # Legacy wrapper (re-exports from package)
└── prompts.pt                     # Generated prompt configurations
```

### Core Modules

**src/llama_introspection/models.py**
- `MODEL_SHORTCUTS`: Maps short names (1B, 8B, 70B, 405B) to full HuggingFace IDs
- `REMOTE_MODELS`: Models requiring NDIF remote execution
- `DEFAULT_STRENGTHS`, `DEFAULT_SCALE_FACTORS`: Experiment parameters
- `resolve_model_id()`, `is_remote_model()`: Helper functions

**src/llama_introspection/steering.py**
- `SteeringVectorResult`: Container for computed vectors with metadata
- `compute_bespoke_vector()`: Contrastive pair steering vectors
- `compute_generic_vector()`: Mean-subtracted concept vectors
- `compute_random_vector()`: Random baseline vectors
- `compute_injection_position()`: Find token position for intervention
- `get_layer_accessor()`, `get_num_layers()`: Model layer utilities

**src/llama_introspection/evaluation.py**
- `GradingClient`: Protocol for grading clients (real or mock)
- `AnthropicGradingClient`: Wrapper for real Anthropic API
- `grade_response()`: Grade model responses using Claude-as-judge
- `build_grader_prompt()`: Construct grading prompts

**src/llama_introspection/geometry.py**
- `compute_l2_norm()`, `compute_l2_distance()`, `compute_cosine_similarity()`
- `compute_intervention_geometry()`: Pre/post intervention metrics
- `compute_trajectory_stats()`: Activation norm statistics

**src/llama_introspection/trajectory.py** (NEW)
- `TrajectoryData`: Dataclass for trajectory metrics and intervention metadata
  - Properties: `norm_growth_ratio`, `mean_update_alignment`
  - Fields: `state_norms`, `update_norms`, `update_alignments`, `update_state_alignments`
  - Intervention fields: `pre_norm`, `post_norm`, `perturbation_norm`, `perturbation_cosine`
- `compute_trajectory_metrics(states)`: Pure function computing metrics from layer tensors
- `simulate_random_walks()`: Baseline random walk simulation for comparison
- `capture_trajectory()`: Capture full trajectory with optional intervention
  - Supports: `baseline`, `concept`, `random`, `scale` intervention types
  - Verified: interventions propagate to downstream layers in trace mode

**src/llama_introspection/utils.py**
- `compute_kl_divergence()`: KL divergence with nucleus filtering
- `compute_kl_curves()`: KL curves across alpha sweep
- `get_token_transitions()`: Find where top token changes

### Key Data Structures

**SteeringVectorResult**: Container for computed vectors with metadata and cache paths.

**Experiment results**: Nested dict `results[layer][strength] = [trial_results]` where each trial includes generated text, geometry metrics (pre/post norms, L2 distance), and Claude grades (affirmative, correct_id, early_detection, coherent).

## Testing

### Test Infrastructure
- **Tiny model**: `llamafactory/tiny-random-Llama-3` (4.11M params) for fast testing
- **Mock clients**: `MockAnthropicClient` mimics real API for testing without credits
- **100 tests total**: Unit tests + end-to-end experiment tests + trajectory tests

### Mock Clients (tests/mocks.py)
```python
from tests.mocks import MockAnthropicClient

# Drop-in replacement for Anthropic client
client = MockAnthropicClient()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=300,
    messages=[{"role": "user", "content": "..."}]
)
```

Available mocks:
- `MockGradingClient`: Heuristic-based grading
- `MockAnthropicClient`: Full API interface mock
- `MockAnthropicClientFixedResponse`: Returns fixed responses
- `MockGradingClientSequence`: Returns responses from sequence

## Critical Constraints

### File I/O: Use torch.save/load Only

**MUST use `torch.save()` and `torch.load()` for ALL file I/O operations**, not native `open()`. Python's `open()` breaks nnsight's tracing mechanism. This affects prompts.pt, cached vectors, and results serialization.

### Model Execution

- **Local models**: 1B, 8B (run with available GPU)
- **Remote models**: 70B, 405B (require `--use-remote` flag, execute on NDIF servers via nnsight)

### Dependencies

Core: nnsight, anthropic, python-dotenv, pandas, matplotlib, numpy, torch
Dev: pytest, pytest-cov

## File Reference

- `prompts.pt` - Generated prompt configurations (17 configs including bespoke and generic vectors)
- `results/` - Cached steering vectors and experiment outputs
- `assets/` - Result plots and raw CSV data from trials

### Legacy Wrappers (for backwards compatibility)
These files re-export from the `llama_introspection` package:
- `config.py` → `llama_introspection.models`
- `steering_vectors.py` → `llama_introspection.steering`
- `util.py` → `llama_introspection.utils`

New code should import directly from `llama_introspection.*`.
