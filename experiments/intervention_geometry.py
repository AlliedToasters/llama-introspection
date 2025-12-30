#!/usr/bin/env python3
"""
Intervention Geometry Experiment

Analyzes how different intervention types affect activation trajectory dynamics:
1. BASELINE - No intervention (directed walk baseline)
2. CONCEPT - Inject steering vector for specific concept
3. RANDOM - Inject random vector (matched norm)
4. SCALE - Multiply activations by scale factor

Captures full trajectory metrics:
- State norms at each layer
- Update norms (step sizes)
- Update alignments (directedness measure)
- Intervention geometry (pre/post norms, perturbation)

This extends compare_walks.py by adding intervention support.

Usage:
    python intervention_geometry.py --model 1B --n-trials 5
    python intervention_geometry.py --model 8B --concepts ocean fire --n-trials 3
    python intervention_geometry.py --model 70B --use-remote --n-trials 10
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_introspection.trajectory import (
    TrajectoryData,
    capture_trajectory,
    simulate_random_walks,
)
from llama_introspection.models import MODEL_SHORTCUTS, REMOTE_MODELS
from llama_introspection.steering import (
    compute_generic_vector,
    compute_injection_position,
    get_layer_accessor,
    compute_mean_steering_norm,
)

# =============================================================================
# Configuration
# =============================================================================

# Prompts for trajectory capture
TRAJECTORY_PROMPTS = [
    "Do you detect an injected thought? If so, what is the injected thought about?",
    "Explain the concept of entropy in thermodynamics.",
    "What makes a good leader?",
    "Describe the process of photosynthesis in plants.",
    "How does a neural network learn from data?",
]

# Default concepts for steering vectors
DEFAULT_CONCEPTS = ["ocean", "fire", "music"]

# Baseline words for generic vector computation
BASELINE_WORDS = [
    "object", "thing", "item", "entity", "element",
    "matter", "subject", "unit", "piece", "part",
]

GENERIC_PROMPT_TEMPLATE = "The word \"{word}\" is on my mind."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intervention geometry experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="1B",
                        help="Model to use (1B, 8B, 70B, 405B or full name)")
    parser.add_argument("--concepts", type=str, nargs="+", default=DEFAULT_CONCEPTS,
                        help="Concepts for steering vectors")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="Number of trials per condition")
    parser.add_argument("--intervention-layer", type=int, default=None,
                        help="Layer for intervention (default: 2/3 depth)")
    parser.add_argument("--strengths", type=float, nargs="+", default=[1.0],
                        help="Intervention strengths to test")
    parser.add_argument("--scale-factors", type=float, nargs="+", default=[2.0],
                        help="Scale factors to test")
    parser.add_argument("--output-dir", type=str, default="results/geometry")
    parser.add_argument("--use-remote", action="store_true",
                        help="Force remote execution")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def get_prompts(n_prompts: int) -> list[str]:
    """Get prompts for trajectory capture."""
    if n_prompts <= len(TRAJECTORY_PROMPTS):
        return TRAJECTORY_PROMPTS[:n_prompts]
    # Cycle if more needed
    prompts = []
    while len(prompts) < n_prompts:
        prompts.extend(TRAJECTORY_PROMPTS)
    return prompts[:n_prompts]


def format_prompt_for_model(model, prompt: str) -> str:
    """Format prompt for chat model if applicable."""
    if hasattr(model.tokenizer, 'chat_template') and model.tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def run_experiment(
    model,
    model_name: str,
    args,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the full intervention geometry experiment."""

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get layer info
    layers = get_layer_accessor(model)(model)
    num_layers = len(layers)

    # Default intervention layer: 2/3 depth
    int_layer = args.intervention_layer if args.intervention_layer else (num_layers * 2 // 3)
    print(f"Intervention layer: {int_layer}/{num_layers}")

    # Get hidden dimension
    with model.trace(remote=args.use_remote) as tracer:
        hidden_dim_proxy = list().save()
        with tracer.invoke("test"):
            hidden_dim_proxy.append(layers[0].output.shape[-1])
    hidden_dim = int(hidden_dim_proxy[0])
    print(f"Hidden dim: {hidden_dim}")

    # Pre-compute steering vectors for concepts
    print(f"\nComputing steering vectors for {len(args.concepts)} concepts...")
    steering_vectors = {}
    for concept in args.concepts:
        result = compute_generic_vector(
            model=model,
            model_slug=model_name,
            concept_word=concept,
            baseline_words=BASELINE_WORDS,
            prompt_template=GENERIC_PROMPT_TEMPLATE,
            cache_dir=str(output_dir),
            use_remote=args.use_remote,
        )
        steering_vectors[concept] = result.vectors[int_layer]
        print(f"  {concept}: norm={steering_vectors[concept].norm().item():.2f}")

    # Compute mean steering norm for random vector matching
    mean_steering_norm = np.mean([v.norm().item() for v in steering_vectors.values()])
    print(f"Mean steering norm: {mean_steering_norm:.2f}")

    # Get prompts
    prompts = get_prompts(args.n_trials)
    formatted_prompts = [format_prompt_for_model(model, p) for p in prompts]

    results = []

    # 1. Baseline trajectories
    print(f"\n=== Baseline Trajectories ===")
    for i, (raw_prompt, prompt) in enumerate(zip(prompts, formatted_prompts)):
        print(f"  Trial {i+1}/{len(prompts)}...")
        traj = capture_trajectory(
            model=model,
            prompt=prompt,
            intervention_type="baseline",
            use_remote=args.use_remote,
        )
        results.append({
            "condition": "baseline",
            "trial": i,
            "prompt": raw_prompt[:50],
            "strength": 0.0,
            "concept": None,
            **trajectory_to_dict(traj),
        })

    # 2. Concept injection trajectories
    for concept in args.concepts:
        sv = steering_vectors[concept]
        for strength in args.strengths:
            print(f"\n=== Concept: {concept}, Strength: {strength} ===")
            for i, (raw_prompt, prompt) in enumerate(zip(prompts, formatted_prompts)):
                print(f"  Trial {i+1}/{len(prompts)}...")
                traj = capture_trajectory(
                    model=model,
                    prompt=prompt,
                    intervention_type="concept",
                    intervention_layer=int_layer,
                    steering_vector=sv,
                    intervention_strength=strength,
                    concept=concept,
                    use_remote=args.use_remote,
                )
                results.append({
                    "condition": "concept",
                    "trial": i,
                    "prompt": raw_prompt[:50],
                    "strength": strength,
                    "concept": concept,
                    **trajectory_to_dict(traj),
                })

    # 3. Random vector trajectories
    for strength in args.strengths:
        print(f"\n=== Random Vector, Strength: {strength} ===")
        for i, (raw_prompt, prompt) in enumerate(zip(prompts, formatted_prompts)):
            print(f"  Trial {i+1}/{len(prompts)}...")
            # Generate random vector matched to mean steering norm
            random_vec = torch.randn(hidden_dim)
            random_vec = random_vec / random_vec.norm() * mean_steering_norm

            traj = capture_trajectory(
                model=model,
                prompt=prompt,
                intervention_type="random",
                intervention_layer=int_layer,
                steering_vector=random_vec,
                intervention_strength=strength,
                use_remote=args.use_remote,
            )
            results.append({
                "condition": "random",
                "trial": i,
                "prompt": raw_prompt[:50],
                "strength": strength,
                "concept": None,
                **trajectory_to_dict(traj),
            })

    # 4. Scale trajectories
    for scale_factor in args.scale_factors:
        print(f"\n=== Scale Factor: {scale_factor} ===")
        for i, (raw_prompt, prompt) in enumerate(zip(prompts, formatted_prompts)):
            print(f"  Trial {i+1}/{len(prompts)}...")
            traj = capture_trajectory(
                model=model,
                prompt=prompt,
                intervention_type="scale",
                intervention_layer=int_layer,
                intervention_strength=scale_factor,
                use_remote=args.use_remote,
            )
            results.append({
                "condition": "scale",
                "trial": i,
                "prompt": raw_prompt[:50],
                "strength": scale_factor,
                "concept": None,
                **trajectory_to_dict(traj),
            })

    df = pd.DataFrame(results)
    return df


def trajectory_to_dict(traj: TrajectoryData) -> dict:
    """Convert TrajectoryData to flat dict for DataFrame."""
    return {
        "n_layers": traj.n_layers,
        "hidden_dim": traj.hidden_dim,
        "intervention_type": traj.intervention_type,
        "intervention_layer": traj.intervention_layer,
        "intervention_strength": traj.intervention_strength,
        "pre_norm": traj.pre_norm,
        "post_norm": traj.post_norm,
        "perturbation_norm": traj.perturbation_norm,
        "perturbation_cosine": traj.perturbation_cosine,
        "steering_vector_norm": traj.steering_vector_norm,
        "norm_growth_ratio": traj.norm_growth_ratio,
        "mean_update_alignment": traj.mean_update_alignment,
        "initial_norm": traj.state_norms[0] if traj.state_norms else 0,
        "final_norm": traj.state_norms[-1] if traj.state_norms else 0,
        "state_norms_json": json.dumps(traj.state_norms),
        "update_norms_json": json.dumps(traj.update_norms),
        "update_alignments_json": json.dumps(traj.update_alignments),
    }


def plot_results(df: pd.DataFrame, output_dir: Path, model_name: str):
    """Generate visualization of trajectory differences."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get one representative trajectory per condition for state norms plot
    conditions = df["condition"].unique()
    colors = {"baseline": "blue", "concept": "green", "random": "orange", "scale": "red"}

    # 1. State norms by layer (compare conditions)
    ax1 = axes[0, 0]
    for condition in conditions:
        cond_df = df[df["condition"] == condition]
        # Parse state norms from JSON
        all_norms = [json.loads(row["state_norms_json"]) for _, row in cond_df.iterrows()]
        norms_array = np.array(all_norms)
        mean_norms = norms_array.mean(axis=0)
        std_norms = norms_array.std(axis=0)
        layers = np.arange(len(mean_norms))

        color = colors.get(condition, "gray")
        ax1.plot(layers, mean_norms, color=color, lw=2, label=condition)
        ax1.fill_between(layers, mean_norms - std_norms, mean_norms + std_norms,
                        color=color, alpha=0.2)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("State Norm")
    ax1.set_title("State Norms by Condition")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Norm growth ratio by condition
    ax2 = axes[0, 1]
    growth_data = df.groupby("condition")["norm_growth_ratio"].agg(["mean", "std"])
    x = np.arange(len(growth_data))
    ax2.bar(x, growth_data["mean"], yerr=growth_data["std"],
            color=[colors.get(c, "gray") for c in growth_data.index],
            capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(growth_data.index)
    ax2.set_ylabel("Norm Growth Ratio (final/initial)")
    ax2.set_title("Norm Growth by Condition")
    ax2.grid(alpha=0.3, axis="y")

    # 3. Mean update alignment by condition
    ax3 = axes[1, 0]
    align_data = df.groupby("condition")["mean_update_alignment"].agg(["mean", "std"])
    ax3.bar(x, align_data["mean"], yerr=align_data["std"],
            color=[colors.get(c, "gray") for c in align_data.index],
            capsize=5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(align_data.index)
    ax3.set_ylabel("Mean Update Alignment")
    ax3.set_title("Update Alignment (Directedness)")
    ax3.axhline(0, color="black", linestyle="--", lw=1)
    ax3.grid(alpha=0.3, axis="y")

    # 4. Intervention geometry (perturbation norm vs cosine)
    ax4 = axes[1, 1]
    intervention_df = df[df["condition"] != "baseline"]
    if len(intervention_df) > 0:
        for condition in intervention_df["condition"].unique():
            cond_df = intervention_df[intervention_df["condition"] == condition]
            ax4.scatter(
                cond_df["perturbation_norm"],
                cond_df["perturbation_cosine"],
                c=colors.get(condition, "gray"),
                label=condition,
                alpha=0.7,
                s=50,
            )
        ax4.set_xlabel("Perturbation Norm")
        ax4.set_ylabel("Perturbation-State Cosine")
        ax4.set_title("Intervention Geometry")
        ax4.legend()
        ax4.grid(alpha=0.3)

    plt.suptitle(f"Intervention Geometry Analysis: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plot_path = output_dir / f"intervention_geometry_{model_name.replace('/', '_')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved plot: {plot_path}")
    plt.show()


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("INTERVENTION GEOMETRY SUMMARY")
    print("=" * 60)

    for condition in df["condition"].unique():
        cond_df = df[df["condition"] == condition]
        print(f"\n{condition.upper()} (n={len(cond_df)})")
        print(f"  Norm growth: {cond_df['norm_growth_ratio'].mean():.3f} +/- {cond_df['norm_growth_ratio'].std():.3f}")
        print(f"  Update alignment: {cond_df['mean_update_alignment'].mean():.3f} +/- {cond_df['mean_update_alignment'].std():.3f}")

        if condition != "baseline":
            print(f"  Perturbation norm: {cond_df['perturbation_norm'].mean():.2f}")
            print(f"  Perturbation cosine: {cond_df['perturbation_cosine'].mean():.3f}")


def main():
    args = parse_args()

    # Load environment
    load_dotenv()

    try:
        from nnsight import LanguageModel, CONFIG
        api_key = os.getenv("NNSIGHT_API_KEY")
        if api_key:
            CONFIG.API.APIKEY = api_key
    except ImportError:
        print("ERROR: nnsight not available")
        return 1

    # Resolve model name
    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    use_remote = args.use_remote or args.model in REMOTE_MODELS or model_name in REMOTE_MODELS

    print("=" * 60)
    print("INTERVENTION GEOMETRY EXPERIMENT")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Remote: {use_remote}")
    print(f"Trials: {args.n_trials}")
    print(f"Concepts: {args.concepts}")
    print(f"Strengths: {args.strengths}")
    print(f"Scale factors: {args.scale_factors}")

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model...")
    if use_remote:
        model = LanguageModel(model_name)
    else:
        model = LanguageModel(model_name, device_map="auto")

    # Run experiment
    df = run_experiment(model, model_name, args, output_dir)

    # Save results
    csv_path = output_dir / f"intervention_geometry_{model_name.replace('/', '_')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    # Print summary
    print_summary(df)

    # Plot
    plot_results(df, output_dir, model_name)

    return 0


if __name__ == "__main__":
    exit(main())
