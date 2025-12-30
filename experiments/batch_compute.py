#!/usr/bin/env python3
"""
Batch compute steering vectors and generate plots for all prompts across multiple models.

This script is designed to run overnight to exhaustively compute:
- All bespoke and generic steering vectors
- For multiple models
- Generate comparison plots

Usage:
    python batch_compute.py
    python batch_compute.py --models meta-llama/Llama-3.2-1B-Instruct
    python batch_compute.py --skip-existing
    python batch_compute.py --plots-only  # Skip vector computation, just make plots
"""

import os
import sys
import argparse
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from dotenv import load_dotenv


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    # "meta-llama/Llama-3.1-405B-Instruct",
]

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch compute steering vectors and plots")
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS,
                        help="Models to compute vectors for")
    parser.add_argument("--prompts-file", type=str, default="prompts.pt",
                        help="Path to prompts config file")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to store results")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to store plots")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip vectors that are already cached")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip vector computation, only generate plots")
    parser.add_argument("--prompt-labels", type=str, nargs="+", default=None,
                        help="Specific prompt labels to process (default: all)")
    parser.add_argument("--use-remote", action="store_true",
                        help="Use NDIF remote execution")
    return parser.parse_args()


def setup_environment():
    """Load environment variables and configure nnsight."""
    load_dotenv()
    
    from nnsight import CONFIG
    CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY")
    
    if not CONFIG.API.APIKEY:
        print("Warning: NNSIGHT_API_KEY not found. Remote execution will fail.")


def load_prompts_config(prompts_file: str) -> Dict:
    """Load prompts configuration."""
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        print(f"Prompts file not found: {prompts_path}")
        print("Run 'python generate_prompts.py' first to create it.")
        sys.exit(1)
    
    return torch.load(prompts_path, weights_only=False)


def compute_vectors_for_model(
    model_slug: str,
    prompts_config: Dict,
    results_dir: Path,
    prompt_labels: Optional[List[str]] = None,
    skip_existing: bool = False,
    use_remote: bool = False,
) -> Dict[str, Dict]:
    """Compute all steering vectors for a single model."""
    from nnsight import LanguageModel
    from steering_vectors import (
        compute_bespoke_vector,
        compute_generic_vector,
        compute_baseline_means,
        get_cache_path,
    )
    
    PROMPTS = prompts_config["prompts"]
    BASELINE_WORDS = prompts_config["baseline_words"]
    GENERIC_PROMPT_TEMPLATE = prompts_config["generic_prompt_template"]
    
    # Filter prompts if specified
    if prompt_labels:
        labels_to_process = [l for l in prompt_labels if l in PROMPTS]
    else:
        labels_to_process = list(PROMPTS.keys())
    
    print(f"\n{'='*70}")
    print(f"MODEL: {model_slug}")
    print(f"{'='*70}")
    print(f"Processing {len(labels_to_process)} prompts...")
    
    # Load model
    print(f"Loading model...")
    start_time = time.time()
    model = LanguageModel(model_slug)
    print(f"Model loaded in {time.time() - start_time:.1f}s")
    
    # Pre-compute baseline means for generic vectors (expensive, do once)
    generic_labels = [l for l in labels_to_process if PROMPTS[l]["type"] == "generic"]
    baseline_means = None
    
    if generic_labels:
        print(f"\nPre-computing baseline means for {len(generic_labels)} generic vectors...")
        baseline_means, _ = compute_baseline_means(
            model=model,
            model_slug=model_slug,
            baseline_words=BASELINE_WORDS,
            prompt_template=GENERIC_PROMPT_TEMPLATE,
            cache_dir=results_dir,
            use_remote=use_remote,
        )
        print(f"Baseline means ready (will be reused for all generic vectors)")
    
    vectors = {}
    errors = []
    
    for i, label in enumerate(labels_to_process):
        prompt_config = PROMPTS[label]
        prompt_type = prompt_config["type"]
        
        print(f"\n[{i+1}/{len(labels_to_process)}] {label} ({prompt_type})")
        
        # Check if already cached
        if skip_existing:
            if prompt_type == "bespoke":
                cache_path = get_cache_path(
                    results_dir, model_slug, "bespoke",
                    positive=prompt_config["steering_positive"],
                    negative=prompt_config["steering_negative"],
                )
            else:
                cache_path = get_cache_path(
                    results_dir, model_slug, "generic",
                    concept_word=prompt_config["concept_word"],
                    n_baseline=len(BASELINE_WORDS),
                )
            
            if cache_path.exists():
                print(f"  Skipping (cached at {cache_path.name})")
                # Load cached for return value
                cached = torch.load(cache_path, weights_only=True)
                vectors[label] = {
                    "vectors": cached["vectors"],
                    "metadata": cached["metadata"],
                }
                continue
        
        try:
            start_time = time.time()
            
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
                    baseline_means=baseline_means,
                )
            else:
                print(f"  Unknown prompt type: {prompt_type}")
                continue
            
            elapsed = time.time() - start_time
            norms = result.norms()
            print(f"  Computed in {elapsed:.1f}s | norm range: [{min(norms):.1f}, {max(norms):.1f}]")
            
            vectors[label] = {
                "vectors": result.vectors,
                "metadata": result.metadata,
            }
            
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            errors.append((label, str(e)))
    
    # Clean up model to free memory before next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for label, err in errors:
            print(f"  - {label}: {err}")
    
    return vectors


def generate_plots_for_model(
    model_slug: str,
    vectors: Dict[str, Dict],
    plots_dir: Path,
):
    """Generate plots for a single model's vectors."""
    from plot_vector_norms import plot_vector_norms, generate_plot_filename
    
    if not vectors:
        print(f"No vectors to plot for {model_slug}")
        return
    
    model_short = model_slug.split('/')[-1]
    
    # Plot all vectors together
    print(f"\nGenerating combined plot for {model_short}...")
    plot_vector_norms(
        vectors_dict=vectors,
        plots_dir=plots_dir,
        model_slug=model_slug,
        title=f"All Steering Vector Norms ({model_short})",
    )
    
    # Separate plots by type
    bespoke_vectors = {k: v for k, v in vectors.items() 
                       if v.get("metadata", {}).get("type") == "bespoke"}
    generic_vectors = {k: v for k, v in vectors.items() 
                       if v.get("metadata", {}).get("type") == "generic"}
    
    if bespoke_vectors:
        print(f"Generating bespoke-only plot...")
        plot_vector_norms(
            vectors_dict=bespoke_vectors,
            plots_dir=plots_dir,
            model_slug=model_slug,
            output_filename=f"norms_{model_short}_bespoke_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            title=f"Bespoke Steering Vectors ({model_short})",
        )
    
    if generic_vectors:
        print(f"Generating generic-only plot...")
        plot_vector_norms(
            vectors_dict=generic_vectors,
            plots_dir=plots_dir,
            model_slug=model_slug,
            output_filename=f"norms_{model_short}_generic_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            title=f"Generic Steering Vectors ({model_short})",
        )


def generate_comparison_plots(
    all_vectors: Dict[str, Dict[str, Dict]],
    prompts_config: Dict,
    plots_dir: Path,
):
    """Generate comparison plots across models."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    PROMPTS = prompts_config["prompts"]
    models = list(all_vectors.keys())
    
    if len(models) < 2:
        print("Need at least 2 models for comparison plots")
        return
    
    # Find common prompts
    common_labels = set(all_vectors[models[0]].keys())
    for model in models[1:]:
        common_labels &= set(all_vectors[model].keys())
    
    if not common_labels:
        print("No common prompts across models")
        return
    
    print(f"\nGenerating comparison plots for {len(common_labels)} common prompts...")
    
    # Create comparison plot: side-by-side norms for each prompt
    for label in sorted(common_labels):
        fig, ax = plt.subplots(figsize=(12, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for idx, model_slug in enumerate(models):
            vecs = all_vectors[model_slug][label]["vectors"]
            norms = [v.norm().item() for v in vecs]
            layers = list(range(len(norms)))
            
            model_short = model_slug.split('/')[-1]
            ax.plot(layers, norms, label=model_short, color=colors[idx], 
                    marker='o', markersize=3, linewidth=1.5)
        
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("L2 Norm", fontsize=12)
        ax.set_title(f"Steering Vector Comparison: {label}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_label = label.replace('/', '_').replace(' ', '_')[:30]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"comparison_{safe_label}_{timestamp}.png"
        
        output_path = plots_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        # plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename}")
    
    # Create summary comparison: mean norm across layers for each prompt/model
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels_sorted = sorted(common_labels)
    x = np.arange(len(labels_sorted))
    width = 0.8 / len(models)
    
    for idx, model_slug in enumerate(models):
        mean_norms = []
        for label in labels_sorted:
            vecs = all_vectors[model_slug][label]["vectors"]
            norms = [v.norm().item() for v in vecs]
            mean_norms.append(np.mean(norms))
        
        model_short = model_slug.split('/')[-1]
        offset = (idx - len(models)/2 + 0.5) * width
        ax.bar(x + offset, mean_norms, width, label=model_short)
    
    ax.set_xlabel("Prompt", fontsize=12)
    ax.set_ylabel("Mean L2 Norm", fontsize=12)
    ax.set_title("Mean Steering Vector Norms by Prompt and Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([l[:20] + "..." if len(l) > 20 else l for l in labels_sorted], 
                        rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = plots_dir / f"comparison_summary_{timestamp}.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    # plt.savefig(summary_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  Saved summary: {summary_path.name}")


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)
    
    # Load prompts config
    prompts_config = load_prompts_config(args.prompts_file)
    
    print(f"Batch Steering Vector Computation")
    print(f"=" * 70)
    print(f"Models: {args.models}")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Results dir: {results_dir}")
    print(f"Plots dir: {plots_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Plots only: {args.plots_only}")
    
    if args.prompt_labels:
        print(f"Prompt labels: {args.prompt_labels}")
    else:
        print(f"Prompt labels: all ({len(prompts_config['prompts'])} prompts)")
    
    start_time = time.time()
    
    # Collect all vectors
    all_vectors = {}
    
    if args.plots_only:
        # Load cached vectors
        print("\nLoading cached vectors...")
        from plot_vector_norms import load_cached_vectors_from_dir
        
        for model_slug in args.models:
            # Filter cached vectors by model
            cached = load_cached_vectors_from_dir(results_dir)
            model_vectors = {
                k: v for k, v in cached.items()
                if v.get("metadata", {}).get("model_slug") == model_slug
            }
            if model_vectors:
                all_vectors[model_slug] = model_vectors
                print(f"  {model_slug}: {len(model_vectors)} vectors")
    else:
        # Compute vectors
        setup_environment()
        
        for model_slug in args.models:
            vectors = compute_vectors_for_model(
                model_slug=model_slug,
                prompts_config=prompts_config,
                results_dir=results_dir,
                prompt_labels=args.prompt_labels,
                skip_existing=args.skip_existing,
                use_remote=args.use_remote,
            )
            all_vectors[model_slug] = vectors
    
    # Generate plots for each model
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    for model_slug, vectors in all_vectors.items():
        generate_plots_for_model(model_slug, vectors, plots_dir)
    
    # Generate comparison plots if multiple models
    if len(all_vectors) > 1:
        generate_comparison_plots(all_vectors, prompts_config, plots_dir)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
