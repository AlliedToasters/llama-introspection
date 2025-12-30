# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project reproduces and extends experiments from Anthropic's introspection paper on LLaMA-3.1-Instruct models. It investigates whether language models can detect when "thoughts" (activation patterns) are injected into their neural networks during generation.

Paper reference: https://transformer-circuits.pub/2025/introspection/index.html

Key extension: Adds a "natural activation scaling" control condition (Neel Nanda's suggestion) to test whether detection is due to injected vectors being anomalously large vs. simply amplifying natural activation patterns.

## Commands

### Setup
```bash
# Copy .env.template to .env and add API keys:
# - NNSIGHT_API_KEY (for remote model execution via NDIF)
# - ANTHROPIC_API_KEY (for Claude-as-judge evaluation)
```

### Generate Prompt Configurations
```bash
python generate_prompts.py  # Creates prompts.pt
```

### Pre-compute Steering Vectors
```bash
python batch_compute.py              # Local models (1B, 8B)
python batch_compute.py --use-remote # Include remote models (70B, 405B)
```

### Run Experiments

Single experiment with specific conditions:
```bash
python introspection.py --model 1B --condition concept --concept ocean --n-trials 5
python introspection.py --model 8B --condition random --n-trials 10
python introspection.py --model 70B --condition scale --use-remote
```

Full battery test:
```bash
python battery.py --model 8B \
  --n-control-trials 18 \
  --n-random-trials 18 \
  --n-concept-trials 1 \
  --n-scale-trials 18
```

### Analysis
```bash
python plot_vector_norms.py --results-dir results/
python compare_walks.py --model 1B --n-gens 5
```

## Architecture

### Core Modules

- **steering_vectors.py** - Central library for computing steering vectors. Handles bespoke (contrastive pairs), generic (mean-subtracted), and random control vectors. Includes caching system and injection position calculation.

- **introspection.py** - Main experiment runner with four conditions: concept injection, random vector injection, activation scaling, and baseline (no intervention). Supports layer sweeping, early stopping on incoherence, and Claude-as-judge evaluation.

- **battery.py** - Batch test runner that combines all intervention conditions. Outputs results to pandas DataFrame for analysis.

- **util.py** - Model loading, KL divergence computation, plotting utilities, and conditional text generation with hooks.

- **config.py** - Model shortcuts (1B, 8B, 70B, 405B), remote model list, default strengths/scale factors.

### Key Data Structures

**SteeringVectorResult**: Container for computed vectors with metadata and cache paths.

**Experiment results**: Nested dict `results[layer][strength] = [trial_results]` where each trial includes generated text, geometry metrics (pre/post norms, L2 distance), and Claude grades (affirmative, correct_id, early_detection, coherent).

## Critical Constraints

### File I/O: Use torch.save/load Only

**MUST use `torch.save()` and `torch.load()` for ALL file I/O operations**, not native `open()`. Python's `open()` breaks nnsight's tracing mechanism. This affects prompts.pt, cached vectors, and results serialization.

### Model Execution

- **Local models**: 1B, 8B (run with available GPU)
- **Remote models**: 70B, 405B (require `--use-remote` flag, execute on NDIF servers via nnsight)

### Dependencies

Core: nnsight, anthropic, python-dotenv, pandas, matplotlib, numpy, torch

## Development Workflow

### Test-Driven Development
- Use pytest for all tests, located in `tests/` directory
- Write tests before and after each module extraction
- Run tests with `pytest` or `pytest tests/` from project root

### Package Structure (Refactor in Progress)
- Core library code goes in `src/llama_introspection/`
- Experiment scripts go in `experiments/` and import from the package
- Extract modules incrementally—never break existing functionality during refactor

### Refactoring Process
1. Write tests for existing functionality
2. Extract module to `src/llama_introspection/`
3. Verify tests still pass
4. Update imports in dependent code
5. Remove old module only after all imports updated

## File Reference

- `prompts.pt` - Generated prompt configurations (17 configs including bespoke and generic vectors)
- `results/` - Cached steering vectors and experiment outputs
- `assets/` - Result plots and raw CSV data from trials
- `run_experiment.py` - Legacy experiment runner (superseded by introspection.py)
- `_old.py` - Deprecated code, safe to ignore

### Module Structure (Target)
```
llama-introspection/
├── src/
│   └── llama_introspection/
│       ├── __init__.py
│       ├── steering.py        # from steering_vectors.py
│       ├── experiment.py      # core experiment logic from introspection.py
│       ├── evaluation.py      # Claude-as-judge grading
│       ├── geometry.py        # vector norm/distance computations
│       ├── models.py          # model loading, config
│       └── utils.py           # shared utilities
├── experiments/
│   ├── introspection.py       # main introspection experiment
│   ├── battery.py             # batch runner
│   └── analysis/
│       ├── plot_vector_norms.py
│       └── compare_walks.py
├── tests/
│   └── ...
└── pyproject.toml