#!/usr/bin/env python3
"""
Compare "directed" semantic walks (real LLM trajectories) vs random walks.

Hypothesis: LLM trajectories are directed (updates roughly aligned), leading to
faster norm growth than a random walk with the same step sizes.

- Random walk in d dimensions: ||h||² grows as sum of ||u_i||² (sqrt growth)
- Directed walk: ||h|| grows as sum of ||u_i|| (linear growth)

Usage:
    python compare_walks.py --model 1B --n-gens 5
    python compare_walks.py --model 1B --prompt "Hello, how are you?"
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json
import hashlib

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from nnsight import LanguageModel, CONFIG
    api_key = os.getenv("NNSIGHT_API_KEY")
    if api_key:
        CONFIG.API.APIKEY = api_key
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False
    print("Warning: nnsight not available")

# Import from shared trajectory module
from llama_introspection.trajectory import (
    simulate_random_walks,
    get_layer_accessor,
)

# =============================================================================
# Configuration
# =============================================================================


MODEL_SHORTCUTS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "8B": "meta-llama/Llama-3.1-8B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    "70B": "meta-llama/Llama-3.1-70B-Instruct",
    "70b": "meta-llama/Llama-3.1-70B-Instruct",
    "70B-base": "meta-llama/Llama-3.1-70B",
    "70b-base": "meta-llama/Llama-3.1-70B",
    "405B": "meta-llama/Llama-3.1-405B-Instruct",
    "405b": "meta-llama/Llama-3.1-405B-Instruct",
    "405B-base": "meta-llama/Llama-3.1-405B",
    "405b-base": "meta-llama/Llama-3.1-405B"
}

# Diverse prompts for trajectory analysis
DIVERSE_PROMPTS = [
    "Explain the concept of entropy in thermodynamics.",
    "Write a short poem about the ocean at sunset.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis in plants.",
    "Tell me about the history of the Roman Empire.",
    "How does a neural network learn from data?",
    "What makes a good leader?",
    "Explain quantum entanglement in simple terms.",
    "Write a recipe for chocolate chip cookies.",
    "What are the ethical implications of artificial intelligence?",
    "Describe the water cycle.",
    "How do airplanes stay in the air?",
    "What is the significance of the Mona Lisa?",
    "Explain how vaccines work.",
    "What causes the seasons to change?",
    "How do black holes form?",
    "What is the difference between mitosis and meiosis?",
    "Explain the concept of supply and demand.",
    "How does memory work in the human brain?",
    "What are the main causes of climate change?",
    "Describe the structure of DNA.",
    "How do computers store information?",
    "What is the theory of relativity?",
    "Explain the process of fermentation.",
    "What are the benefits of meditation?",
    "How do electric cars work?",
    "What is the significance of the Renaissance?",
    "Explain how the immune system fights infection.",
    "What are prime numbers and why are they important?",
    "How do earthquakes occur?",
]


def get_prompts(n_prompts: int) -> list[str]:
    """Get n prompts from the diverse prompts list."""
    if n_prompts > len(DIVERSE_PROMPTS):
        # Cycle through if we need more
        prompts = []
        while len(prompts) < n_prompts:
            prompts.extend(DIVERSE_PROMPTS)
        return prompts[:n_prompts]
    return DIVERSE_PROMPTS[:n_prompts]


@dataclass
class WalkData:
    """Data from a single forward pass / generation."""
    # State vector norms at each layer: [n_layers + 1] (includes embedding)
    state_norms: list[float] = field(default_factory=list)
    
    # Update vector norms: [n_layers] (one per layer)
    update_norms: list[float] = field(default_factory=list)
    
    # Cosine similarity between consecutive updates (measures "directedness")
    update_alignments: list[float] = field(default_factory=list)
    
    # Cosine similarity between update and current state (measures "overwriting")
    # Negative = update opposes current state direction
    update_state_alignments: list[float] = field(default_factory=list)
    
    # Metadata
    n_layers: int = 0
    hidden_dim: int = 0
    prompt: str = ""
    response: str = ""  # Generated response text
    
def get_cache_key(model_name: str, prompt: str, max_tokens: int) -> str:
    """Generate a cache key for a trajectory run."""
    import hashlib
    content = f"{model_name}|{prompt}|{max_tokens}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_cached_trajectory(cache_dir: Path, cache_key: str) -> Optional[WalkData]:
    """Load a cached trajectory if it exists."""
    cache_path = cache_dir / f"trajectory_{cache_key}.json"
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path) as f:
            data = json.load(f)
        
        return WalkData(
            state_norms=data["state_norms"],
            update_norms=data["update_norms"],
            update_alignments=data["update_alignments"],
            update_state_alignments=data["update_state_alignments"],
            n_layers=data["n_layers"],
            hidden_dim=data["hidden_dim"],
            prompt=data["prompt"],
            response=data.get("response", ""),
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: corrupted cache file {cache_path}: {e}")
        return None


def save_cached_trajectory(cache_dir: Path, cache_key: str, walk: WalkData):
    """Save a trajectory to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"trajectory_{cache_key}.json"
    
    data = {
        "state_norms": walk.state_norms,
        "update_norms": walk.update_norms,
        "update_alignments": walk.update_alignments,
        "update_state_alignments": walk.update_state_alignments,
        "n_layers": walk.n_layers,
        "hidden_dim": walk.hidden_dim,
        "prompt": walk.prompt,
        "response": walk.response,
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)

def capture_trajectory(
    model,
    prompt: str,
    max_new_tokens: int = 1,
    use_remote: bool = False,
) -> WalkData:
    """
    Capture the activation trajectory through all layers.
    
    Returns state norms and update norms for analysis.
    """
    # Get layers reference before tracing
    if hasattr(model, '_model') and hasattr(model._model, 'h'):
        # GPT-2 style
        layers = model._model.h
    else:
        # LLaMA style
        layers = model.model.layers
    
    num_layers = len(layers)
    
    # Use the pattern from run_experiment.py that works with remote
    with model.trace(remote=use_remote) as tracer:
        layer_outputs = list().save()  # Create and save list FIRST
        
        with tracer.invoke(prompt):
            for l in range(num_layers):
                # Append to the pre-saved list
                layer_outputs.append(layers[l].output[:, -1, :])
    
    # Process captured activations
    walk_data = WalkData(
        n_layers=num_layers,
        prompt=prompt[:100],
    )
    
    walk_data.response = "(trace mode - no generation)"
    
    # Convert to tensors and compute norms
    states = [lo.detach().cpu().squeeze(0) for lo in layer_outputs]  # [hidden_dim] each
    walk_data.hidden_dim = states[0].shape[-1]
    
    # State norms
    walk_data.state_norms = [s.norm().item() for s in states]
    
    # Update norms: ||h_{l+1} - h_l||
    for l in range(num_layers - 1):
        update = states[l + 1] - states[l]
        walk_data.update_norms.append(update.norm().item())
    
    # Update-state alignments: cosine similarity between update and current state
    for l in range(num_layers - 1):
        update = states[l + 1] - states[l]
        state = states[l]
        cos_sim = torch.nn.functional.cosine_similarity(
            update.unsqueeze(0), state.unsqueeze(0)
        ).item()
        walk_data.update_state_alignments.append(cos_sim)
    
    # Update alignments: cosine similarity between consecutive updates
    for l in range(num_layers - 2):
        u1 = states[l + 1] - states[l]
        u2 = states[l + 2] - states[l + 1]
        cos_sim = torch.nn.functional.cosine_similarity(
            u1.unsqueeze(0), u2.unsqueeze(0)
        ).item()
        walk_data.update_alignments.append(cos_sim)
    
    return walk_data


def plot_comparison(
    real_walks: list[WalkData],
    random_trajectories: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Directed vs Random Walk",
):
    """
    Plot comparison of real (directed) walks vs random walks.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: State norms ---
    ax1 = axes[0, 0]
    
    # Real walks
    real_norms = np.array([w.state_norms for w in real_walks])
    n_layers = real_norms.shape[1]
    layers = np.arange(n_layers)
    
    real_mean = real_norms.mean(axis=0)
    real_std = real_norms.std(axis=0)
    real_min = real_norms.min(axis=0)
    real_max = real_norms.max(axis=0)
    
    ax1.plot(layers, real_mean, 'b-', lw=2, label='LLM trajectory (mean)')
    ax1.fill_between(layers, real_mean - real_std, real_mean + real_std, 
                     alpha=0.3, color='blue', label='LLM ± std')
    ax1.plot(layers, real_min, 'b--', lw=1, alpha=0.5)
    ax1.plot(layers, real_max, 'b--', lw=1, alpha=0.5)
    
    # Random walks
    random_mean = random_trajectories.mean(axis=0)
    random_std = random_trajectories.std(axis=0)
    random_min = random_trajectories.min(axis=0)
    random_max = random_trajectories.max(axis=0)
    
    # Align x-axis (random walks have one more point due to initial state)
    random_layers = np.arange(len(random_mean))
    
    ax1.plot(random_layers, random_mean, 'r-', lw=2, label='Random walk (mean)')
    ax1.fill_between(random_layers, random_mean - random_std, random_mean + random_std,
                     alpha=0.3, color='red', label='Random ± std')
    ax1.plot(random_layers, random_min, 'r--', lw=1, alpha=0.5)
    ax1.plot(random_layers, random_max, 'r--', lw=1, alpha=0.5)
    
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Activation Norm', fontsize=12)
    ax1.set_title('State Vector Norms', fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Update norms by layer ---
    ax2 = axes[0, 1]
    
    update_norms = np.array([w.update_norms for w in real_walks])
    if update_norms.size > 0:
        update_mean = update_norms.mean(axis=0)
        update_std = update_norms.std(axis=0)
        update_min = update_norms.min(axis=0)
        update_max = update_norms.max(axis=0)
        update_layers = np.arange(len(update_mean))
        
        ax2.plot(update_layers, update_mean, 'purple', lw=2, marker='o', markersize=4, label='Mean')
        ax2.fill_between(update_layers, update_mean - update_std, update_mean + update_std,
                        alpha=0.3, color='purple', label='± std')
        ax2.plot(update_layers, update_min, 'purple', lw=1, alpha=0.5, linestyle='--')
        ax2.plot(update_layers, update_max, 'purple', lw=1, alpha=0.5, linestyle='--')
        
        # Show overall mean as reference
        overall_mean = update_mean.mean()
        ax2.axhline(y=overall_mean, color='gray', linestyle=':', lw=1.5, 
                   label=f'Overall mean: {overall_mean:.1f}')
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Update Norm ||h_{l+1} - h_l||', fontsize=12)
    ax2.set_title('Update Vector Norms by Layer', fontsize=13)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Update-State alignment (overwriting) ---
    ax3 = axes[1, 0]
    
    update_state_aligns = np.array([w.update_state_alignments for w in real_walks])
    if update_state_aligns.size > 0:
        usa_mean = update_state_aligns.mean(axis=0)
        usa_std = update_state_aligns.std(axis=0)
        usa_layers = np.arange(len(usa_mean))
        
        ax3.plot(usa_layers, usa_mean, 'darkorange', lw=2, marker='o', markersize=4, label='Mean')
        ax3.fill_between(usa_layers, usa_mean - usa_std, usa_mean + usa_std,
                        alpha=0.3, color='orange')
        
        # Color regions
        ax3.axhline(y=0, color='black', linestyle='-', lw=1)
        ax3.axhspan(-1, 0, alpha=0.1, color='red', label='Overwriting (opposing)')
        ax3.axhspan(0, 1, alpha=0.1, color='green', label='Reinforcing (aligned)')
    
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('cos(update, state)', fontsize=12)
    ax3.set_title('Update-State Alignment\n(negative = overwriting)', fontsize=13)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.0, 1.0)
    
    # --- Plot 4: Consecutive update alignments ---
    ax4 = axes[1, 1]
    
    # Real update alignments (cosine similarity between consecutive updates)
    alignments = np.array([w.update_alignments for w in real_walks])
    if alignments.size > 0:
        align_mean = alignments.mean(axis=0)
        align_std = alignments.std(axis=0)
        align_layers = np.arange(len(align_mean))
        
        ax4.plot(align_layers, align_mean, 'g-', lw=2, marker='o', markersize=4, label='LLM updates')
        ax4.fill_between(align_layers, align_mean - align_std, align_mean + align_std,
                        alpha=0.3, color='green')
    
    # Expected alignment for random vectors in high-d: ~0
    ax4.axhline(y=0, color='red', linestyle='--', lw=1.5, label='Random expectation')
    
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Cosine Similarity', fontsize=12)
    ax4.set_title('Consecutive Update Alignment\n(u_l vs u_{l+1})', fontsize=13)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.5, 1.0)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    return fig


def print_summary(real_walks: list[WalkData], random_trajectories: np.ndarray):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY: Directed vs Random Walk")
    print("=" * 60)
    
    real_norms = np.array([w.state_norms for w in real_walks])
    update_norms = np.array([w.update_norms for w in real_walks])
    
    # Basic stats
    initial_norm = real_norms[:, 0].mean()
    final_norm = real_norms[:, -1].mean()
    mean_update_norm = update_norms.mean()
    sum_update_norms = update_norms.sum(axis=1).mean()
    
    print(f"\nState norms:")
    print(f"  Initial (layer 0 output): {initial_norm:.2f}")
    print(f"  Final (layer {real_norms.shape[1]-1} output): {final_norm:.2f}")
    
    print(f"\nUpdate norms:")
    print(f"  Mean per layer: {mean_update_norm:.2f}")
    print(f"  Sum across layers: {sum_update_norms:.2f}")
    print(f"  Ratio (sum / initial): {sum_update_norms / initial_norm:.2f}x")
    
    # Growth ratios
    real_growth = real_norms[:, -1] / real_norms[:, 0]
    random_growth = random_trajectories[:, -1] / random_trajectories[:, 0]
    
    print(f"\nNorm growth (final / initial):")
    print(f"  LLM trajectory: {real_growth.mean():.2f}x (± {real_growth.std():.2f})")
    print(f"  Random walk:    {random_growth.mean():.2f}x (± {random_growth.std():.2f})")
    
    # Theoretical comparison with actual update norms
    n_layers = real_norms.shape[1] - 1  # number of updates
    sqrt_sum_sq = np.sqrt((update_norms ** 2).sum(axis=1)).mean()
    
    print(f"\nTheoretical predictions (using actual update norms):")
    print(f"  Random walk: sqrt(sum(||u||²)) = {sqrt_sum_sq:.2f}")
    print(f"  Random growth: {sqrt_sum_sq / initial_norm:.2f}x")
    print(f"  Directed walk: sum(||u||) = {sum_update_norms:.2f}")
    print(f"  Directed growth: {sum_update_norms / initial_norm:.2f}x")
    
    # Consecutive update alignments
    alignments = np.array([w.update_alignments for w in real_walks])
    if alignments.size > 0:
        print(f"\nConsecutive update alignment (u_l vs u_{{l+1}}):")
        print(f"  Mean: {alignments.mean():.3f} (expected 0.0 for random)")
        print(f"  Std:  {alignments.std():.3f}")
    
    # Update-state alignments (overwriting)
    update_state = np.array([w.update_state_alignments for w in real_walks])
    if update_state.size > 0:
        n_update_layers = update_state.shape[1]
        print(f"\nUpdate-state alignment (overwriting metric):")
        print(f"  Mean: {update_state.mean():.3f}")
        if n_update_layers >= 5:
            print(f"  Early layers (0-4): {update_state[:, :5].mean():.3f}")
        mid_start = n_update_layers // 3
        mid_end = 2 * n_update_layers // 3
        if mid_end > mid_start:
            print(f"  Middle layers ({mid_start}-{mid_end}): {update_state[:, mid_start:mid_end].mean():.3f}")
            print(f"  Late layers ({mid_end}+): {update_state[:, mid_end:].mean():.3f}")
        print(f"  (negative = update opposes current state direction)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LLM trajectories vs random walks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="1B",
                        help="Model to use (1B, 8B, 70B, 405B or full name)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (default: introspection prompt)")
    parser.add_argument("--n-gens", type=int, default=5,
                        help="Number of generations to run")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens per generation")
    parser.add_argument("--n-random-walks", type=int, default=100,
                        help="Number of random walks to simulate")
    parser.add_argument("--output-dir", type=str, default="walk_comparison",
                        help="Output directory for plots")
    parser.add_argument("--use-remote", action="store_true",
                        help="Use NDIF remote inference")
    parser.add_argument("--no-cache", action="store_true",
                    help="Disable caching (re-run all trajectories)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve model name
    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("DIRECTED VS RANDOM WALK COMPARISON")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Generations: {args.n_gens}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Random walks: {args.n_random_walks}")
    
    if not NNSIGHT_AVAILABLE:
        print("\nERROR: nnsight not available")
        return 1
    
    # Load model
    print(f"\nLoading model...")
    model = LanguageModel(model_name, device_map="auto")
    
    # Detect if this is a base model (no chat template)
    is_base_model = not hasattr(model.tokenizer, 'chat_template') or model.tokenizer.chat_template is None
    if is_base_model:
        print(f"Detected base model (no chat template) - using raw prompts")
    
    # Generate or prepare prompts
    raw_prompts_for_json = []  # Human-readable prompts for JSON
    if args.prompt:
        # Single custom prompt - use for all generations
        prompts = [args.prompt] * args.n_gens
        raw_prompts_for_json = [args.prompt] * args.n_gens
        print(f"Using custom prompt for all {args.n_gens} generations")
    else:
        # Use diverse prompts from static list
        print(f"\nUsing {args.n_gens} diverse prompts...")
        raw_prompts = get_prompts(args.n_gens)
        raw_prompts_for_json = raw_prompts
        
        if is_base_model:
            # Base models: use raw prompts directly (maybe with a simple prefix)
            prompts = raw_prompts
        else:
            # Instruct models: format prompts as chat messages
            prompts = []
            for raw_prompt in raw_prompts:
                messages = [{"role": "user", "content": raw_prompt}]
                formatted = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(formatted)
        
        # Show first few prompts
        print("Sample prompts:")
        for i, p in enumerate(raw_prompts[:3]):
            preview = p[:60] + "..." if len(p) > 60 else p
            print(f"  {i+1}. {preview}")
        if len(raw_prompts) > 3:
            print(f"  ... and {len(raw_prompts) - 3} more")
    
    # Warm up (only needed for local execution)
    if not args.use_remote:
        print("\nWarming up...")
        with model.trace("test", remote=False):
            pass
    else:
        print("\nSkipping warmup (remote execution)")
    
    # Run generations and capture trajectories
    print(f"\nCapturing {args.n_gens} trajectories...")
    real_walks = []
    
    cache_dir = output_dir / "cache" / model_name.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_count = 0
    computed_count = 0

    for i, prompt in enumerate(prompts):
        cache_key = get_cache_key(model_name, prompt, args.max_tokens)
        
        # Try cache first (unless --no-cache)
        walk_data = None if args.no_cache else load_cached_trajectory(cache_dir, cache_key)
        
        if walk_data:
            print(f"  Generation {i + 1}/{args.n_gens}... [CACHED]")
            cached_count += 1
        else:
            print(f"  Generation {i + 1}/{args.n_gens}...", end=" ", flush=True)
            walk_data = capture_trajectory(
                model, prompt, max_new_tokens=args.max_tokens, use_remote=args.use_remote
            )
            save_cached_trajectory(cache_dir, cache_key, walk_data)
            computed_count += 1
            print(f"layers={walk_data.n_layers}, final_norm={walk_data.state_norms[-1]:.1f}")
        
        real_walks.append(walk_data)

    print(f"\nTrajectories: {cached_count} cached, {computed_count} computed")
    
    # Use average update norms across all real walks
    avg_update_norms = np.array([w.update_norms for w in real_walks]).mean(axis=0).tolist()
    initial_norm = np.mean([w.state_norms[0] for w in real_walks])
    hidden_dim = real_walks[0].hidden_dim
    
    random_trajectories = simulate_random_walks(
        update_norms=avg_update_norms,
        initial_norm=initial_norm,
        hidden_dim=hidden_dim,
        n_walks=args.n_random_walks,
    )
    
    # Print summary
    print_summary(real_walks, random_trajectories)
    
    # Plot
    model_short = model_name.split('/')[-1]
    output_path = output_dir / f"walk_comparison_{model_short}.png"
    
    plot_comparison(
        real_walks=real_walks,
        random_trajectories=random_trajectories,
        output_path=output_path,
        title=f"Directed vs Random Walk ({model_short})",
    )
    
    # Save data with prompts for reproducibility
    data_path = output_dir / f"walk_data_{model_short}.json"
    save_data = {
        "model": model_name,
        "n_gens": args.n_gens,
        "max_tokens": args.max_tokens,
        "prompts": raw_prompts_for_json,  # Human-readable prompts
        "real_walks": [
            {
                "prompt": raw_prompts_for_json[i] if i < len(raw_prompts_for_json) else "",
                "response": w.response,
                "state_norms": w.state_norms,
                "update_norms": w.update_norms,
                "update_alignments": w.update_alignments,
                "update_state_alignments": w.update_state_alignments,
            }
            for i, w in enumerate(real_walks)
        ],
        "random_trajectories_mean": random_trajectories.mean(axis=0).tolist(),
        "random_trajectories_std": random_trajectories.std(axis=0).tolist(),
        "avg_update_norms": avg_update_norms,
        "initial_norm": initial_norm,
        "hidden_dim": hidden_dim,
    }
    with open(data_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved data: {data_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())