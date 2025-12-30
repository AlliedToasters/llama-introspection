#!/usr/bin/env python3
"""
Plot steering vector norms as a function of layer number.

Usage:
    # Plot from cached vectors
    python plot_vector_norms.py --results-dir results/
    
    # Compute and plot for specific prompts
    python plot_vector_norms.py --prompt-labels "all-caps/shouting/loudness" "dog" "justice" --model meta-llama/Llama-3.2-1B-Instruct
    
    # Plot all available cached vectors
    python plot_vector_norms.py --results-dir results/ --all-cached

Outputs are saved to plots/ directory with unique filenames encoding:
    - Model name
    - Vector types (bespoke/generic)
    - Concept labels
    - Timestamp
"""

import argparse
import hashlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Only import heavy dependencies if needed
def get_steering_imports():
    from steering_vectors import (
        compute_bespoke_vector,
        compute_generic_vector,
        list_cached_vectors,
        load_cached_vectors,
        get_num_layers,
    )
    return compute_bespoke_vector, compute_generic_vector, list_cached_vectors, load_cached_vectors, get_num_layers


def sanitize_filename(s: str) -> str:
    """Sanitize a string for use in filenames."""
    # Replace problematic characters
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        s = s.replace(char, '_')
    return s[:50]  # Limit length


def generate_plot_filename(
    model_slug: str,
    vectors_dict: Dict[str, Dict],
    plot_type: str = "norms",
    extension: str = "png",
) -> str:
    """Generate a unique filename encoding model and vector details."""
    
    # Extract model short name
    model_name = model_slug.split('/')[-1] if '/' in model_slug else model_slug
    model_name = sanitize_filename(model_name)
    
    # Collect vector info
    vec_types = set()
    concepts = []
    for label, data in vectors_dict.items():
        metadata = data.get("metadata", {})
        vec_type = metadata.get("type", "unknown")
        vec_types.add(vec_type)
        
        if vec_type == "generic":
            concepts.append(metadata.get("concept_word", ""))
        elif vec_type == "bespoke":
            # Use first word of positive prompt as identifier
            pos = metadata.get("positive_prompt", label)
            concepts.append(pos.split()[0] if pos else label[:10])
        else:
            concepts.append(label[:10])
    
    # Build filename components
    types_str = "+".join(sorted(vec_types))
    
    # Create concept hash if too many concepts
    if len(concepts) > 3:
        concepts_str = f"{len(concepts)}concepts_{hashlib.md5('_'.join(concepts).encode()).hexdigest()[:6]}"
    else:
        concepts_str = "_".join(sanitize_filename(c) for c in concepts[:3])
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{plot_type}_{model_name}_{types_str}_{concepts_str}_{timestamp}.{extension}"
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description="Plot steering vector norms by layer")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing cached steering vectors")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to save plots")
    parser.add_argument("--prompt-labels", type=str, nargs="+",
                        help="Specific prompt labels to compute/plot")
    parser.add_argument("--prompts-file", type=str, default="prompts.pt",
                        help="Path to prompts config file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model to use for computing vectors")
    parser.add_argument("--all-cached", action="store_true",
                        help="Plot all cached vectors in results directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output filename (default: auto-generated)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 6],
                        help="Figure size (width, height)")
    parser.add_argument("--log-scale", action="store_true",
                        help="Use log scale for y-axis")
    return parser.parse_args()


def load_cached_vectors_from_dir(results_dir: Path) -> Dict[str, Dict]:
    """Load all cached steering vectors from a directory."""
    vectors = {}
    
    for path in results_dir.glob("steering_*.pt"):
        try:
            data = torch.load(path, weights_only=True)
            metadata = data.get("metadata", {})
            vecs = data.get("vectors", [])
            
            # Create a label from metadata
            vec_type = metadata.get("type", "unknown")
            if vec_type == "bespoke":
                pos = metadata.get("positive_prompt", "")[:20]
                label = f"bespoke: {pos}..."
            elif vec_type == "generic":
                word = metadata.get("concept_word", "unknown")
                label = f"generic: {word}"
            elif vec_type == "random":
                seed = metadata.get("seed", 0)
                label = f"random (seed={seed})"
            else:
                label = path.stem
            
            vectors[label] = {
                "vectors": vecs,
                "metadata": metadata,
                "path": path,
            }
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    return vectors


def compute_vectors_for_prompts(
    prompt_labels: List[str],
    prompts_config: Dict,
    model,
    model_slug: str,
    results_dir: Path,
    use_remote: bool = False,
) -> Dict[str, Dict]:
    """Compute steering vectors for specified prompt labels."""
    compute_bespoke_vector, compute_generic_vector, _, _, _ = get_steering_imports()
    
    PROMPTS = prompts_config["prompts"]
    BASELINE_WORDS = prompts_config["baseline_words"]
    GENERIC_PROMPT_TEMPLATE = prompts_config["generic_prompt_template"]
    
    vectors = {}
    
    for label in prompt_labels:
        if label not in PROMPTS:
            print(f"Warning: Unknown prompt label '{label}', skipping")
            continue
        
        prompt_config = PROMPTS[label]
        prompt_type = prompt_config["type"]
        
        print(f"Computing vectors for '{label}' ({prompt_type})...")
        
        if prompt_type == "bespoke":
            result = compute_bespoke_vector(
                model=model,
                model_slug=model_slug,
                positive_prompt=prompt_config["steering_positive"],
                negative_prompt=prompt_config["steering_negative"],
                cache_dir=results_dir,
                use_remote=use_remote,
            )
        elif prompt_type == "generic":
            result = compute_generic_vector(
                model=model,
                model_slug=model_slug,
                concept_word=prompt_config["concept_word"],
                baseline_words=BASELINE_WORDS,
                prompt_template=GENERIC_PROMPT_TEMPLATE,
                cache_dir=results_dir,
                use_remote=use_remote,
            )
        else:
            print(f"Warning: Unknown prompt type '{prompt_type}' for '{label}', skipping")
            continue
        
        vectors[label] = {
            "vectors": result.vectors,
            "metadata": result.metadata,
        }
    
    return vectors


def plot_vector_norms(
    vectors_dict: Dict[str, Dict],
    plots_dir: Path,
    model_slug: str = "unknown",
    output_filename: Optional[str] = None,
    figsize: tuple = (12, 6),
    log_scale: bool = False,
    title: Optional[str] = None,
) -> Path:
    """Plot steering vector norms as a function of layer.
    
    Returns:
        Path to the saved PNG file.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    # Generate filename if not provided
    if output_filename is None:
        output_filename = generate_plot_filename(model_slug, vectors_dict, "norms", "png")
    
    output_path = plots_dir / output_filename
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color cycle for different vectors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    
    for idx, (label, data) in enumerate(vectors_dict.items()):
        vecs = data["vectors"]
        norms = [v.norm().item() for v in vecs]
        layers = list(range(len(norms)))
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(layers, norms, 
                label=label, 
                color=color, 
                marker=marker,
                markersize=4,
                linewidth=1.5,
                alpha=0.8)
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Steering Vector Norms by Layer", fontsize=14)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(vecs) - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    print(f"{'Label':<30} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print("-" * 60)
    
    for label, data in vectors_dict.items():
        vecs = data["vectors"]
        norms = [v.norm().item() for v in vecs]
        print(f"{label[:30]:<30} {min(norms):>8.2f} {max(norms):>8.2f} {np.mean(norms):>8.2f} {np.std(norms):>8.2f}")
    
    return output_path


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    vectors_dict = {}
    model_slug = args.model  # Track model for filename generation
    
    if args.all_cached:
        # Load all cached vectors
        print(f"Loading cached vectors from {results_dir}...")
        vectors_dict = load_cached_vectors_from_dir(results_dir)
        
        if not vectors_dict:
            print("No cached vectors found. Use --prompt-labels to compute some.")
            return 1
        
        # Try to extract model from cached metadata
        for data in vectors_dict.values():
            cached_model = data.get("metadata", {}).get("model_slug")
            if cached_model:
                model_slug = cached_model
                break
    
    elif args.prompt_labels:
        # Compute vectors for specified prompts
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"Prompts file not found: {prompts_path}")
            print("Run 'python generate_prompts.py' first to create it.")
            return 1
        
        prompts_config = torch.load(prompts_path, weights_only=False)
        
        # Load model
        import os
        from dotenv import load_dotenv
        from nnsight import CONFIG, LanguageModel
        
        load_dotenv()
        CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY")
        
        print(f"Loading model: {args.model}")
        model = LanguageModel(args.model)
        
        vectors_dict = compute_vectors_for_prompts(
            prompt_labels=args.prompt_labels,
            prompts_config=prompts_config,
            model=model,
            model_slug=args.model,
            results_dir=results_dir,
        )
    
    else:
        # Default: try to load cached, or compute a few examples
        print(f"Looking for cached vectors in {results_dir}...")
        vectors_dict = load_cached_vectors_from_dir(results_dir)
        
        if not vectors_dict:
            print("No cached vectors found.")
            print("Use --prompt-labels to specify which vectors to compute, or")
            print("Use --all-cached after running the main experiment.")
            return 1
        
        # Try to extract model from cached metadata
        for data in vectors_dict.values():
            cached_model = data.get("metadata", {}).get("model_slug")
            if cached_model:
                model_slug = cached_model
                break
    
    if not vectors_dict:
        print("No vectors to plot!")
        return 1
    
    print(f"\nPlotting {len(vectors_dict)} vector sets...")
    
    # Generate title
    model_short = model_slug.split('/')[-1] if '/' in model_slug else model_slug
    title = f"Steering Vector Norms ({model_short})"
    
    output_path = plot_vector_norms(
        vectors_dict=vectors_dict,
        plots_dir=plots_dir,
        model_slug=model_slug,
        output_filename=args.output,  # None = auto-generate
        figsize=tuple(args.figsize),
        log_scale=args.log_scale,
        title=title,
    )
    
    print(f"\nPlot saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
