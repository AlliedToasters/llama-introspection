# Injected Thoughts Detection Experiment

Reproduces experiments from Anthropic's introspection paper on Llama-405B-Instruct using NDIF/nnsight.

**Paper:** https://transformer-circuits.pub/2025/introspection/index.html

## Setup

1. Install dependencies:
```bash
pip install nnsight anthropic python-dotenv
```

2. Create `.env` file with your API keys:
```bash
cp .env.template .env
# Edit .env with your keys
```

3. Get API keys:
   - **NNSIGHT_API_KEY**: Register at https://login.ndif.us
   - **ANTHROPIC_API_KEY**: From https://console.anthropic.com

## Usage

see `commands.md` for now

## Output

Results are saved to `results/` directory:
- `sweep_<hash>.json` - Single trial per strength
- `multi_<hash>.json` - Multiple trials
- `graded_*.json` - LLM-judged results

## Experiment Overview

1. **Compute steering vector**: Record activations on "HI! HOW ARE YOU?" vs "Hi! How are you?" pair (contrastive) or "generic"
2. **Inject during generation**: Add scaled vector to layer ~2/3 through model during all generation steps  
3. **Evaluate responses**: Use Claude as judge to classify:
   - Did model affirm detecting an injected thought?
   - Did it correctly identify the concept (caps/shouting/loudness)?
   - Did it detect BEFORE mentioning the concept?
   - Was the response coherent?

## Anthropic Introspectio Paper Question

The paper tested random vectors as controls. Question: what if you simply *amplify* the model's natural activations (scale them up) instead of adding a semantic steering vector? Would that also trigger detection, suggesting the model is just detecting "loudness" in activation space rather than true introspection?

To test this, modify the script to:
1. Compute `amplified = original_activation * scale_factor` instead of `original + steering_vec * strength`
2. Compare detection rates between semantic injection vs simple amplification