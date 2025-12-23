MODEL_SHORTCUTS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Llama-3.1-70B-Instruct",
    "405B": "meta-llama/Llama-3.1-405B-Instruct",
}

# REMOTE_MODELS = {"70B", "405B", "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-405B-Instruct"}
REMOTE_MODELS = {"405B", "meta-llama/Llama-3.1-405B-Instruct"}

DEFAULT_STRENGTHS = [0.5, 1.0, 2.0, 4.0, 8.0]

# Scale factors for activation scaling condition
DEFAULT_SCALE_FACTORS = [0.5, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]

MAX_NEW_TOKENS = 100