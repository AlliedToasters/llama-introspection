#!/usr/bin/env python3
"""
Steering Vector Computation and Caching Utilities

Handles all steering vector computation for the injected thoughts experiment.
Supports multiple vector types for ablation studies.

IMPORTANT: Uses torch.save/load for ALL file I/O because Python's native open()
function breaks nnsight's tracing mechanism.

Supported vector types:
- BESPOKE: vec = activations(positive_prompt) - activations(negative_prompt)
- GENERIC: vec = activations("Tell me about {word}") - mean_baseline
- RANDOM: random vectors matching activation dimensions (control)
- AMPLIFICATION: scaling factor for original activations (control)
"""

import hashlib
import torch
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union, Tuple
from dataclasses import dataclass


@dataclass
class SteeringVectorResult:
    """Container for steering vector computation results."""
    vectors: List[torch.Tensor]  # One per layer
    metadata: Dict
    cache_path: Optional[Path] = None
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]
    
    def __len__(self) -> int:
        return len(self.vectors)
    
    @property
    def num_layers(self) -> int:
        return len(self.vectors)
    
    @property
    def hidden_dim(self) -> int:
        return self.vectors[0].shape[-1] if self.vectors else 0
    
    def norms(self) -> List[float]:
        """Return L2 norms for each layer's vector."""
        return [v.norm().item() for v in self.vectors]
    
    def summary(self) -> str:
        """Return a summary string."""
        norms = self.norms()
        return (
            f"SteeringVectors: {self.num_layers} layers, dim={self.hidden_dim}\n"
            f"  Type: {self.metadata.get('type', 'unknown')}\n"
            f"  Model: {self.metadata.get('model_slug', 'unknown')}\n"
            f"  Norms: min={min(norms):.2f}, max={max(norms):.2f}, mean={sum(norms)/len(norms):.2f}"
        )


def get_cache_path(
    cache_dir: Path,
    model_slug: str,
    vector_type: str,
    **kwargs
) -> Path:
    """Generate a deterministic cache path based on configuration."""
    # Build a unique identifier string
    id_parts = [model_slug, vector_type]
    
    # Add type-specific identifiers
    if vector_type == "bespoke":
        id_parts.extend([kwargs.get("positive", ""), kwargs.get("negative", "")])
    elif vector_type == "generic":
        id_parts.extend([kwargs.get("concept_word", ""), str(kwargs.get("n_baseline", 0))])
    elif vector_type == "random":
        id_parts.append(str(kwargs.get("seed", 0)))
    elif vector_type == "amplification":
        id_parts.append("amplification")
    
    id_string = "|".join(id_parts)
    hash_str = hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    return cache_dir / f"steering_{vector_type}_{hash_str}.pt"


def get_layer_accessor(model) -> Callable:
    """Return appropriate layer accessor function based on model architecture."""
    model_name = type(model._model).__name__.lower() if hasattr(model, '_model') else ""
    
    if "gpt2" in model_name:
        return lambda m: m._model.h
    else:
        # Default: LLaMA-style architecture
        return lambda m: m.model.layers


def get_num_layers(model) -> int:
    """Get the number of layers in the model."""
    accessor = get_layer_accessor(model)
    return len(accessor(model))


def compute_bespoke_vector(
    model,
    model_slug: str,
    positive_prompt: str,
    negative_prompt: str,
    cache_dir: Optional[Path] = None,
    use_remote: bool = False,
    force_recompute: bool = False,
) -> SteeringVectorResult:
    """
    Compute bespoke (contrastive) steering vector.
    
    vec[layer] = activations(positive_prompt)[-1] - activations(negative_prompt)[-1]
    
    Args:
        model: nnsight LanguageModel instance
        model_slug: Model identifier string (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        positive_prompt: Prompt representing the concept to steer toward
        negative_prompt: Prompt representing the concept to steer away from
        cache_dir: Directory for caching vectors (uses torch.save)
        use_remote: Whether to use NDIF remote execution
        force_recompute: If True, ignore cache and recompute
    
    Returns:
        SteeringVectorResult with vectors for each layer
    """
    num_layers = get_num_layers(model)
    layers_accessor = get_layer_accessor(model)
    
    # Check cache
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_path = get_cache_path(
            cache_dir, model_slug, "bespoke",
            positive=positive_prompt, negative=negative_prompt
        )
        
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached bespoke vectors from {cache_path}")
            cache_data = torch.load(cache_path, weights_only=True)
            cached_vectors = cache_data["vectors"]
            cached_metadata = cache_data["metadata"]
            
            # Verify layer count matches
            if len(cached_vectors) != num_layers:
                print(f"  WARNING: Cached vectors have {len(cached_vectors)} layers, model has {num_layers}")
                print(f"  Recomputing...")
            else:
                return SteeringVectorResult(
                    vectors=cached_vectors,
                    metadata=cached_metadata,
                    cache_path=cache_path,
                )
    
    print(f"Computing bespoke steering vectors for {num_layers} layers...")
    print(f"  Model: {model_slug}")
    print(f"  Positive: {positive_prompt[:50]}...")
    print(f"  Negative: {negative_prompt[:50]}...")
    
    with model.trace(remote=use_remote) as tracer:
        pos_activations = list().save()
        neg_activations = list().save()
        
        with tracer.invoke(positive_prompt):
            for layer_idx in range(num_layers):
                pos_activations.append(layers_accessor(model)[layer_idx].output[:, -1, :])
        
        with tracer.invoke(negative_prompt):
            for layer_idx in range(num_layers):
                neg_activations.append(layers_accessor(model)[layer_idx].output[:, -1, :])
    
    vectors = [
        (pos_activations[i] - neg_activations[i]).detach().clone().cpu()
        for i in range(num_layers)
    ]
    
    metadata = {
        "type": "bespoke",
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "model_slug": model_slug,
        "num_layers": num_layers,
    }
    
    # Cache results
    if cache_path is not None:
        torch.save({"vectors": vectors, "metadata": metadata}, cache_path)
        print(f"  Cached to {cache_path}")
    
    return SteeringVectorResult(vectors=vectors, metadata=metadata, cache_path=cache_path)


def compute_baseline_means(
    model,
    model_slug: str,
    baseline_words: List[str],
    prompt_template: str = "Tell me about {word}.",
    cache_dir: Optional[Path] = None,
    use_remote: bool = False,
    force_recompute: bool = False,
    batch_size: int = 10,
) -> Tuple[List[torch.Tensor], Path]:
    """
    Compute mean baseline activations across a set of words.
    
    This is the expensive part of generic vector computation and can be
    cached and reused across multiple concept words for the same model.
    
    Args:
        model: nnsight LanguageModel instance
        model_slug: Model identifier string
        baseline_words: List of baseline words
        prompt_template: Template with {word} placeholder
        cache_dir: Directory for caching
        use_remote: Whether to use NDIF remote execution
        force_recompute: If True, ignore cache and recompute
        batch_size: Number of words to process per batch
    
    Returns:
        Tuple of (baseline_means list, cache_path)
    """
    num_layers = get_num_layers(model)
    layers_accessor = get_layer_accessor(model)
    
    # Check cache
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        
        # Cache key includes model, number of baseline words, and template
        cache_id = f"{model_slug}|baseline|{len(baseline_words)}|{prompt_template}"
        cache_hash = hashlib.md5(cache_id.encode()).hexdigest()[:12]
        cache_path = cache_dir / f"baseline_means_{cache_hash}.pt"
        
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached baseline means from {cache_path}")
            cache_data = torch.load(cache_path, weights_only=True)
            cached_means = cache_data["baseline_means"]
            
            # Verify layer count matches
            if len(cached_means) != num_layers:
                print(f"  WARNING: Cached baseline has {len(cached_means)} layers, model has {num_layers}")
                print(f"  Recomputing...")
            else:
                return cached_means, cache_path
    
    print(f"Computing baseline activations across {len(baseline_words)} words...")
    print(f"  Model: {model_slug}")
    print(f"  Layers: {num_layers}")
    
    # Compute baseline mean activations
    baseline_sums = [None] * num_layers
    n_baseline = 0
    
    for batch_start in range(0, len(baseline_words), batch_size):
        batch_words = baseline_words[batch_start:batch_start + batch_size]
        
        with model.trace(remote=use_remote) as tracer:
            batch_activations = {i: list().save() for i in range(num_layers)}
            
            for word in batch_words:
                prompt = prompt_template.format(word=word.lower())
                with tracer.invoke(prompt):
                    for layer_idx in range(num_layers):
                        batch_activations[layer_idx].append(
                            layers_accessor(model)[layer_idx].output[:, -1, :]
                        )
            batch_activations.save()
        
        # Accumulate sums
        for layer_idx in range(num_layers):
            for act in batch_activations[layer_idx]:
                act_cpu = act.detach().clone().cpu()
                if baseline_sums[layer_idx] is None:
                    baseline_sums[layer_idx] = act_cpu
                else:
                    baseline_sums[layer_idx] = baseline_sums[layer_idx] + act_cpu
                n_baseline += 1
        
        processed = min(batch_start + batch_size, len(baseline_words))
        print(f"    Processed {processed}/{len(baseline_words)} baseline words")
    
    # Compute means
    n_per_layer = n_baseline // num_layers
    baseline_means = [s / n_per_layer for s in baseline_sums]
    
    # Cache results
    if cache_path is not None:
        metadata = {
            "model_slug": model_slug,
            "n_baseline_words": len(baseline_words),
            "prompt_template": prompt_template,
            "num_layers": num_layers,
        }
        torch.save({"baseline_means": baseline_means, "metadata": metadata}, cache_path)
        print(f"  Cached baseline means to {cache_path}")
    
    return baseline_means, cache_path


def compute_generic_vector(
    model,
    model_slug: str,
    concept_word: str,
    baseline_words: List[str],
    prompt_template: str = "Tell me about {word}.",
    cache_dir: Optional[Path] = None,
    use_remote: bool = False,
    force_recompute: bool = False,
    batch_size: int = 10,
    baseline_means: Optional[List[torch.Tensor]] = None,
) -> SteeringVectorResult:
    """
    Compute generic (mean-subtracted) steering vector per paper protocol.
    
    vec[layer] = activations(concept_prompt)[-1] - mean(activations(baseline_prompts))[-1]
    
    Args:
        model: nnsight LanguageModel instance
        model_slug: Model identifier string (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        concept_word: The concept word to create a vector for
        baseline_words: List of baseline words for mean subtraction
        prompt_template: Template with {word} placeholder
        cache_dir: Directory for caching vectors
        use_remote: Whether to use NDIF remote execution
        force_recompute: If True, ignore cache and recompute
        batch_size: Number of baseline words to process per batch
        baseline_means: Pre-computed baseline means (skips expensive baseline computation)
    
    Returns:
        SteeringVectorResult with vectors for each layer
    """
    num_layers = get_num_layers(model)
    layers_accessor = get_layer_accessor(model)
    
    # Check cache
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_path = get_cache_path(
            cache_dir, model_slug, "generic",
            concept_word=concept_word, n_baseline=len(baseline_words)
        )
        
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached generic vectors from {cache_path}")
            cache_data = torch.load(cache_path, weights_only=True)
            cached_vectors = cache_data["vectors"]
            cached_metadata = cache_data["metadata"]
            
            # Verify layer count matches
            if len(cached_vectors) != num_layers:
                print(f"  WARNING: Cached vectors have {len(cached_vectors)} layers, model has {num_layers}")
                print(f"  Recomputing...")
            else:
                return SteeringVectorResult(
                    vectors=cached_vectors,
                    metadata=cached_metadata,
                    cache_path=cache_path,
                )
    
    print(f"Computing generic steering vector for '{concept_word}'...")
    print(f"  Model: {model_slug}")
    print(f"  Layers: {num_layers}")
    
    # Get or compute baseline means
    if baseline_means is not None:
        print(f"  Using pre-computed baseline means")
        if len(baseline_means) != num_layers:
            raise ValueError(f"Baseline means have {len(baseline_means)} layers, model has {num_layers}")
    else:
        # Compute baseline (this is the expensive part)
        baseline_means, _ = compute_baseline_means(
            model=model,
            model_slug=model_slug,
            baseline_words=baseline_words,
            prompt_template=prompt_template,
            cache_dir=cache_dir,
            use_remote=use_remote,
            batch_size=batch_size,
        )
    
    # Compute concept activations (cheap - just one forward pass)
    print(f"  Computing concept activations for '{concept_word}'...")
    concept_prompt = prompt_template.format(word=concept_word.lower())
    
    with model.trace(remote=use_remote) as tracer:
        concept_activations = list().save()
        
        with tracer.invoke(concept_prompt):
            for layer_idx in range(num_layers):
                concept_activations.append(
                    layers_accessor(model)[layer_idx].output[:, -1, :]
                )
    
    # Compute steering vectors
    vectors = [
        (concept_activations[i].detach().clone().cpu() - baseline_means[i])
        for i in range(num_layers)
    ]
    
    metadata = {
        "type": "generic",
        "concept_word": concept_word,
        "n_baseline_words": len(baseline_words),
        "prompt_template": prompt_template,
        "model_slug": model_slug,
        "num_layers": num_layers,
    }
    
    # Cache results
    if cache_path is not None:
        torch.save({"vectors": vectors, "metadata": metadata}, cache_path)
        print(f"  Cached to {cache_path}")
    
    return SteeringVectorResult(vectors=vectors, metadata=metadata, cache_path=cache_path)


def compute_random_vector(
    model,
    model_slug: str,
    seed: int = 42,
    normalize_to: Optional[float] = None,
    match_norms_from: Optional[SteeringVectorResult] = None,
    cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
) -> SteeringVectorResult:
    """
    Compute random steering vectors (control condition).
    
    Args:
        model: nnsight LanguageModel instance (for dimension info)
        model_slug: Model identifier string
        seed: Random seed for reproducibility
        normalize_to: If set, normalize all vectors to this L2 norm
        match_norms_from: If set, match norms layer-by-layer from another result
        cache_dir: Directory for caching vectors
        force_recompute: If True, ignore cache and recompute
    
    Returns:
        SteeringVectorResult with random vectors for each layer
    """
    num_layers = get_num_layers(model)
    layers_accessor = get_layer_accessor(model)
    
    # Get hidden dimension from model
    # Run a dummy forward to get dimensions
    with model.trace(remote=False) as tracer:
        with tracer.invoke("test"):
            hidden_dim = layers_accessor(model)[0].output.shape[-1]
            hidden_dim_saved = hidden_dim.save()
    
    hidden_dim = int(hidden_dim_saved)
    
    # Check cache
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_path = get_cache_path(cache_dir, model_slug, "random", seed=seed)
        
        if cache_path.exists() and not force_recompute:
            print(f"Loading cached random vectors from {cache_path}")
            cache_data = torch.load(cache_path, weights_only=True)
            result = SteeringVectorResult(
                vectors=cache_data["vectors"],
                metadata=cache_data["metadata"],
                cache_path=cache_path,
            )
            # Still apply normalization if requested
            if normalize_to is not None:
                result = normalize_vectors(result, target_norm=normalize_to)
            if match_norms_from is not None:
                result = match_vector_norms(result, match_norms_from)
            return result
    
    print(f"Computing random steering vectors (seed={seed})...")
    print(f"  Model: {model_slug}")
    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")
    
    torch.manual_seed(seed)
    vectors = [
        torch.randn(1, hidden_dim)
        for _ in range(num_layers)
    ]
    
    metadata = {
        "type": "random",
        "seed": seed,
        "model_slug": model_slug,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
    }
    
    result = SteeringVectorResult(vectors=vectors, metadata=metadata, cache_path=cache_path)
    
    # Apply normalization
    if normalize_to is not None:
        result = normalize_vectors(result, target_norm=normalize_to)
        result.metadata["normalized_to"] = normalize_to
    
    if match_norms_from is not None:
        result = match_vector_norms(result, match_norms_from)
        result.metadata["matched_norms_from"] = match_norms_from.metadata.get("type", "unknown")
    
    # Cache results (before normalization for reusability)
    if cache_path is not None:
        torch.save({"vectors": vectors, "metadata": metadata}, cache_path)
        print(f"  Cached to {cache_path}")
    
    return result


def normalize_vectors(
    result: SteeringVectorResult,
    target_norm: float,
) -> SteeringVectorResult:
    """Normalize all vectors to a target L2 norm."""
    normalized = []
    for v in result.vectors:
        norm = v.norm()
        if norm > 0:
            normalized.append(v * (target_norm / norm))
        else:
            normalized.append(v)
    
    new_metadata = result.metadata.copy()
    new_metadata["normalized_to"] = target_norm
    
    return SteeringVectorResult(
        vectors=normalized,
        metadata=new_metadata,
        cache_path=result.cache_path,
    )


def match_vector_norms(
    result: SteeringVectorResult,
    reference: SteeringVectorResult,
) -> SteeringVectorResult:
    """Match vector norms layer-by-layer from a reference result."""
    assert len(result.vectors) == len(reference.vectors), \
        f"Layer count mismatch: {len(result.vectors)} vs {len(reference.vectors)}"
    
    matched = []
    for v, ref_v in zip(result.vectors, reference.vectors):
        ref_norm = ref_v.norm()
        v_norm = v.norm()
        if v_norm > 0:
            matched.append(v * (ref_norm / v_norm))
        else:
            matched.append(v)
    
    new_metadata = result.metadata.copy()
    new_metadata["matched_norms_from"] = reference.metadata.get("type", "unknown")
    
    return SteeringVectorResult(
        vectors=matched,
        metadata=new_metadata,
        cache_path=result.cache_path,
    )


def compute_activation_statistics(
    model,
    prompts: List[str],
    use_remote: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Compute activation statistics across prompts (for amplification baseline).
    
    Returns mean and std of activation norms per layer.
    """
    num_layers = get_num_layers(model)
    layers_accessor = get_layer_accessor(model)
    
    all_norms = [[] for _ in range(num_layers)]
    
    for prompt in prompts:
        with model.trace(remote=use_remote) as tracer:
            with tracer.invoke(prompt):
                for layer_idx in range(num_layers):
                    act = layers_accessor(model)[layer_idx].output[:, -1, :]
                    norm = act.norm().save()
                    all_norms[layer_idx].append(norm)
    
    stats = {
        "mean_norms": torch.tensor([
            sum(norms) / len(norms) for norms in all_norms
        ]),
        "std_norms": torch.tensor([
            torch.std(torch.tensor([n.item() for n in norms])).item()
            for norms in all_norms
        ]),
    }
    
    return stats


def load_cached_vectors(cache_path: Path) -> Optional[SteeringVectorResult]:
    """Load cached steering vectors from a path."""
    if not cache_path.exists():
        return None
    
    cache_data = torch.load(cache_path, weights_only=True)
    return SteeringVectorResult(
        vectors=cache_data["vectors"],
        metadata=cache_data["metadata"],
        cache_path=cache_path,
    )


def list_cached_vectors(cache_dir: Path) -> List[Dict]:
    """List all cached steering vectors in a directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    
    results = []
    for path in cache_dir.glob("steering_*.pt"):
        try:
            data = torch.load(path, weights_only=True)
            metadata = data.get("metadata", {})
            results.append({
                "path": path,
                "type": metadata.get("type", "unknown"),
                "model": metadata.get("model_slug", "unknown"),
                "info": metadata,
            })
        except Exception as e:
            results.append({
                "path": path,
                "error": str(e),
            })
    
    return results


# =============================================================================
# Injection utilities
# =============================================================================

def compute_injection_position(
    tokenizer,
    messages: List[Dict],
    marker: str = "\n\nTrial 1",
) -> Tuple[str, int, int]:
    """
    Compute the token position where injection should start.
    
    Per the paper: "beginning at the newline token prior to 'Trial 1'"
    
    Args:
        tokenizer: HuggingFace tokenizer
        messages: Chat messages to format
        marker: String marker to find injection point
    
    Returns:
        Tuple of (formatted_prompt, injection_start_pos, total_tokens)
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    marker_char_pos = prompt.find(marker)
    if marker_char_pos == -1:
        raise ValueError(f"Could not find '{marker}' in prompt")
    
    # Include the first newline character
    prefix_with_newline = prompt[:marker_char_pos + 1]
    injection_start_pos = len(tokenizer.encode(prefix_with_newline))
    total_tokens = len(tokenizer.encode(prompt))
    
    return prompt, injection_start_pos, total_tokens


# =============================================================================
# Convenience functions for ablation studies
# =============================================================================

def create_ablation_vectors(
    model,
    model_slug: str,
    reference_vectors: SteeringVectorResult,
    ablation_types: List[str] = ["random_matched", "scaled_0.5", "scaled_2.0"],
    seed: int = 42,
    cache_dir: Optional[Path] = None,
) -> Dict[str, SteeringVectorResult]:
    """
    Create a set of ablation vectors based on a reference.
    
    Args:
        model: nnsight LanguageModel instance
        model_slug: Model identifier string
        reference_vectors: The original steering vectors to create ablations for
        ablation_types: List of ablation types to create
        seed: Random seed for random vectors
        cache_dir: Directory for caching
    
    Returns:
        Dict mapping ablation type to SteeringVectorResult
    """
    results = {"reference": reference_vectors}
    
    for ablation_type in ablation_types:
        if ablation_type == "random_matched":
            # Random vectors with matched layer-wise norms
            results[ablation_type] = compute_random_vector(
                model, model_slug, seed=seed, match_norms_from=reference_vectors, cache_dir=cache_dir
            )
        
        elif ablation_type.startswith("random_norm_"):
            # Random vectors with fixed norm
            norm = float(ablation_type.split("_")[-1])
            results[ablation_type] = compute_random_vector(
                model, model_slug, seed=seed, normalize_to=norm, cache_dir=cache_dir
            )
        
        elif ablation_type.startswith("scaled_"):
            # Scaled version of reference
            scale = float(ablation_type.split("_")[-1])
            scaled_vectors = [v * scale for v in reference_vectors.vectors]
            results[ablation_type] = SteeringVectorResult(
                vectors=scaled_vectors,
                metadata={
                    **reference_vectors.metadata,
                    "ablation": "scaled",
                    "scale_factor": scale,
                },
            )
        
        elif ablation_type == "negated":
            # Negated reference vectors
            negated_vectors = [-v for v in reference_vectors.vectors]
            results[ablation_type] = SteeringVectorResult(
                vectors=negated_vectors,
                metadata={
                    **reference_vectors.metadata,
                    "ablation": "negated",
                },
            )
        
        elif ablation_type == "orthogonalized":
            # Orthogonalize each layer to previous layer (decorrelate)
            ortho_vectors = [reference_vectors.vectors[0]]
            for i in range(1, len(reference_vectors.vectors)):
                v = reference_vectors.vectors[i]
                prev = ortho_vectors[-1]
                # Project out component along previous vector
                proj = (v @ prev.T) / (prev @ prev.T + 1e-8) * prev
                ortho_vectors.append(v - proj)
            
            results[ablation_type] = SteeringVectorResult(
                vectors=ortho_vectors,
                metadata={
                    **reference_vectors.metadata,
                    "ablation": "orthogonalized",
                },
            )
    
    return results
