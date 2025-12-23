# Injected Thoughts Detection Experiment

Reproduces experiments from Anthropic's introspection paper on Llama-3.1-Instruct models using NDIF/nnsight.

**Paper:** https://transformer-circuits.pub/2025/introspection/index.html

## Setup

1. Install dependencies (recommend using a venv):
```bash
pip install nnsight anthropic python-dotenv pandas matplotlib
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

After setting up your venv and installing the things, run 
```
python generate_prompts.py
```
Then, pre-compute the steering vectors:
```
python batch_compute.py --use-remote
```
You can omit the `--use-remote` flag if you can run the big models on your local system.

After that, to run a battery of interventions on (for example) 1 
```
python battery.py --model 8B --n-control-trials 18 --n-random-trials 18 --n-concept-trials 1 --n-scale-trials 18
```

## Output

Results are saved to `results/` directory.

## Experiment Overview

1. **Compute steering vector**: Record activations on "HI! HOW ARE YOU?" vs "Hi! How are you?" pair (contrastive) or "generic"
2. **Inject during generation**: Add scaled vector to layer ~2/3 through model during all generation steps  
3. **Evaluate responses**: Use Claude as judge to classify:
   - Did model affirm detecting an injected thought?
   - Did it correctly identify the concept (caps/shouting/loudness)?
   - Did it detect BEFORE mentioning the concept?
   - Was the response coherent?

## Anthropic Introspection Paper Question

This is a response to “Emergent Introspective Awareness in Large Language Models” by Jack Lindsey (Anthropic) and “Can LLMs Introspect? A Live Paper Review,” a video review of the paper by Neel Nanda.

Nanda says “I predict [detection] only happens because the injected thing is anomalously big,” and asks “what if you took what the model was thinking about naturally and dialed it up a ton?” as an alternative to steering with random vectors as a control.

This experiment first reproduces a subset of results from Lindsey, and adds a "natural activation scaling" control intervention to the tests.
