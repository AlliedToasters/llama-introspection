"""
Utility functions for LLM steering experiments.

Provides:
- Model and SAE loading
- Steering hook creation
- Batched logit computation
- KL divergence computation
- Token transition analysis
- Plotting utilities
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_and_sae(model_name="google/gemma-2-2b", sae_release="gemma-scope-2b-pt-res-canonical", sae_id="layer_20/width_16k/canonical"):
    """
    Load the model with TransformerLens and the SAE.
    
    Returns:
        model, sae, device
    """
    from transformer_lens import HookedTransformer
    from sae_lens import SAE
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    print(f"Loading SAE: {sae_release} / {sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device
    )
    
    return model, sae, device


def get_steering_vector(sae, feature_index):
    """
    Extract the steering vector for a given feature from the SAE decoder.
    
    Args:
        sae: The loaded SAE
        feature_index: Index of the feature to extract
    
    Returns:
        steering_vector: [d_model] tensor, normalized
        hook_point: Name of the hook point for this SAE
    """
    steering_vector = sae.W_dec[feature_index].detach().clone()
    steering_vector = steering_vector / steering_vector.norm()
    hook_point = sae.cfg.metadata['hook_name']
    
    return steering_vector, hook_point


def create_steering_hook(vector, alpha, max_activation):
    """
    Create a hook function that adds scaled steering vector to activations.
    
    Args:
        vector: The steering direction (from SAE decoder)
        alpha: Steering strength multiplier
        max_activation: Scale factor (typically max observed activation)
    
    Returns:
        A hook function
    """
    scale = alpha * max_activation
    
    def hook_fn(activations, hook):
        # activations shape: [batch, seq_len, d_model]
        # vector shape: [d_model]
        # We add the vector to ALL token positions
        return activations + scale * vector
    
    return hook_fn


def get_next_token_logits(model, prompt, steering_vector, hook_point, alpha=0.0, max_activation=1.0):
    """Get logits for the next token prediction."""
    tokens = model.to_tokens(prompt)
    
    if alpha != 0:
        hook_fn = create_steering_hook(steering_vector, alpha, max_activation)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_point, hook_fn)])
    else:
        with torch.no_grad():
            logits = model(tokens)
    
    return logits[0, -1, :]  # Last position, shape: [vocab_size]


def get_all_logits_batched(model, prompt, alpha_values, steering_vector, hook_point, max_activation=1.0):
    """
    Single forward pass with all alphas batched.
    
    Args:
        model: The HookedTransformer model
        prompt: Input prompt string
        alpha_values: Array/list of alpha values to sweep
        steering_vector: The steering direction [d_model]
        hook_point: Name of the hook point
        max_activation: Scale factor for steering
    
    Returns:
        Tensor of shape [num_alphas, vocab_size]
    """
    tokens = model.to_tokens(prompt)
    batch_size = len(alpha_values)
    tokens_batched = tokens.repeat(batch_size, 1)
    
    alphas = torch.tensor(alpha_values, device=model.cfg.device, dtype=torch.float32)
    
    def batched_steering_hook(activation, hook):
        # activation: [batch, seq, d_model]
        # Add to ALL positions (matching loop behavior)
        return activation + alphas[:, None, None] * max_activation * steering_vector
    
    model.reset_hooks()
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_point, batched_steering_hook)]):
            logits_batched = model(tokens_batched)
    
    return logits_batched[:, -1, :]  # [batch, vocab]


def compute_kl_divergence(logits_base, logits_steered, p_top=1.0):
    """
    Compute KL(steered || base) for next-token distributions.
    
    Args:
        logits_base: Logits from unsteered model
        logits_steered: Logits from steered model
        p_top: Nucleus threshold - include tokens that make up top p_top 
               cumulative probability in either distribution.
               Set to 1.0 for raw KL over full vocabulary.
    
    Returns:
        KL divergence value (float)
    """
    # Convert to probabilities
    p_base = torch.softmax(logits_base, dim=-1)
    p_steered = torch.softmax(logits_steered, dim=-1)
    
    if p_top < 1.0:
        # Nucleus selection for base distribution
        sorted_base, indices_base = torch.sort(p_base, descending=True)
        cumsum_base = torch.cumsum(sorted_base, dim=-1)
        nucleus_mask_base = cumsum_base <= p_top
        nucleus_mask_base[0] = True  # Always include at least the top token
        nucleus_indices_base = indices_base[nucleus_mask_base]
        
        # Nucleus selection for steered distribution
        sorted_steered, indices_steered = torch.sort(p_steered, descending=True)
        cumsum_steered = torch.cumsum(sorted_steered, dim=-1)
        nucleus_mask_steered = cumsum_steered <= p_top
        nucleus_mask_steered[0] = True
        nucleus_indices_steered = indices_steered[nucleus_mask_steered]
        
        # Union of both nucleus sets
        union_indices = torch.unique(torch.cat([nucleus_indices_base, nucleus_indices_steered]))
        
        # Select only the union tokens
        p_base = p_base[union_indices]
        p_steered = p_steered[union_indices]
        
        # Renormalize to valid probability distributions
        p_base = p_base / p_base.sum()
        p_steered = p_steered / p_steered.sum()
    
    # KL divergence
    eps = 1e-10  # Numerical stability
    kl = (p_steered * (torch.log(p_steered + eps) - torch.log(p_base + eps))).sum()
    
    return kl.item()


def compute_kl_curves(all_logits, alpha_values):
    """
    Compute KL divergence curves from pre-computed logits.
    
    Args:
        all_logits: [num_alphas, vocab_size] tensor of logits
        alpha_values: array of alpha values corresponding to each row
    
    Returns:
        dict with 'full' and 'nucleus' KL arrays
    """
    alpha_values = np.array(alpha_values)
    
    # Find baseline (α closest to 0)
    idx_zero = np.argmin(np.abs(alpha_values))
    logits_base = all_logits[idx_zero]
    
    kl_full = []
    kl_nucleus = []
    
    for i in range(len(alpha_values)):
        kl_full.append(compute_kl_divergence(logits_base, all_logits[i], p_top=1.0))
        kl_nucleus.append(compute_kl_divergence(logits_base, all_logits[i], p_top=0.4))
    
    return {
        'full': np.array(kl_full),
        'nucleus': np.array(kl_nucleus)
    }


def get_token_transitions(all_logits, alpha_values, tokenizer):
    """
    Find where the top predicted token changes along the alpha sweep.
    
    Args:
        all_logits: [num_alphas, vocab_size] tensor
        alpha_values: array of alpha values
        tokenizer: model tokenizer for decoding
    
    Returns:
        List of dicts with 'idx', 'alpha', 'token_id', 'token_str' at each transition
    """
    top_tokens = all_logits.argmax(dim=-1)  # [num_alphas]
    
    transitions = []
    prev_token = None
    
    for i, (alpha, token_id) in enumerate(zip(alpha_values, top_tokens)):
        token_id = token_id.item()
        if token_id != prev_token:
            token_str = tokenizer.decode([token_id]).strip()
            if not token_str:
                token_str = repr(tokenizer.decode([token_id]))
            transitions.append({
                'idx': i,
                'alpha': alpha,
                'token_id': token_id,
                'token_str': token_str
            })
            prev_token = token_id
    
    return transitions


def plot_pareto_curve(all_logits, alpha_values, tokenizer, 
                      title="Steering Strength vs Output Perturbation",
                      show_method_markers=False,
                      figsize=(14, 8),
                      save_path=None):
    """
    Unified plotting function for Pareto frontier visualization.
    
    Args:
        all_logits: [num_alphas, vocab_size] tensor
        alpha_values: array of alpha values
        tokenizer: model tokenizer for decoding tokens
        title: plot title
        show_method_markers: if True, show DAC and Azizi operating points
        figsize: figure size tuple
        save_path: if provided, save figure to this path
    
    Returns:
        fig, ax for further customization
    """
    alpha_values = np.array(alpha_values)
    
    # Compute KL curves
    kl_curves = compute_kl_curves(all_logits, alpha_values)
    kl_full = kl_curves['full']
    kl_nucleus = kl_curves['nucleus']
    
    # Find token transitions
    transitions = get_token_transitions(all_logits, alpha_values, tokenizer)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Main KL curve
    ax.plot(alpha_values, kl_full, 'b-o', markersize=3, linewidth=1.5,
            label='Full-vocab KL(α)', zorder=2)
    
    # Nucleus KL curve (dashed)
    ax.plot(alpha_values, kl_nucleus, 'b--', alpha=0.4, linewidth=1,
            label='Nucleus KL (p_top=0.4)')
    
    # Annotate token transitions
    y_max = max(kl_full.max(), 1.0)
    annotated_positions = []
    
    for t in transitions:
        idx = t['idx']
        alpha = t['alpha']
        kl_val = kl_full[idx]
        token_str = t['token_str']
        
        # Mark the transition point
        ax.scatter([alpha], [kl_val], color='red', s=80, zorder=5,
                   marker='v', edgecolors='darkred', linewidths=0.5)
        
        # Smart label positioning
        y_offset = 0.08 * y_max if len(annotated_positions) % 2 == 0 else -0.12 * y_max
        
        for prev_alpha, prev_y in annotated_positions:
            if abs(alpha - prev_alpha) < 0.5:
                y_offset = -y_offset
                break
        
        label_y = kl_val + y_offset
        label_y = max(0.02 * y_max, min(label_y, y_max * 0.95))
        
        ax.annotate(
            f'"{token_str}"',
            xy=(alpha, kl_val),
            xytext=(alpha, label_y),
            fontsize=8,
            color='darkred',
            ha='center',
            fontweight='bold',
            arrowprops=dict(arrowstyle='-', color='red', alpha=0.4, linewidth=0.5),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red', alpha=0.7)
        )
        annotated_positions.append((alpha, label_y))
    
    # Optional method markers
    if show_method_markers:
        _add_method_markers(ax, alpha_values, kl_full, kl_nucleus)
    
    # Formatting
    ax.set_xlabel('Steering Strength (α)', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title(title)
    ax.legend(loc='upper center', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax, transitions, kl_curves


def _add_method_markers(ax, alpha_values, kl_full, kl_nucleus):
    """Add DAC and Azizi operating point markers to plot."""
    
    # DAC: uses nucleus KL at α=1 as the adaptive coefficient (capped at 2)
    idx_alpha_1 = np.argmin(np.abs(alpha_values - 1.0))
    kl_nucleus_at_1 = kl_nucleus[idx_alpha_1]
    alpha_dac = min(kl_nucleus_at_1, 2.0)
    idx_dac = np.argmin(np.abs(alpha_values - alpha_dac))
    kl_at_dac = kl_full[idx_dac]
    
    ax.scatter([alpha_dac], [kl_at_dac], color='green', s=200, zorder=6,
               marker='*', edgecolors='darkgreen', linewidths=1)
    ax.annotate(f'DAC\nα={alpha_dac:.2f}', 
                xy=(alpha_dac, kl_at_dac), 
                xytext=(alpha_dac + 0.5, kl_at_dac + 0.3),
                fontsize=9, color='green',
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    # Azizi: given KL budget ε, find max α under budget
    epsilon_targets = [0.1, 0.5, 1.0]
    colors = ['orange', 'purple', 'brown']
    
    for eps, color in zip(epsilon_targets, colors):
        positive_mask = alpha_values > 0
        pos_alphas = alpha_values[positive_mask]
        pos_kls = kl_full[positive_mask]
        
        under_budget = pos_kls <= eps
        if under_budget.any():
            alpha_max = pos_alphas[under_budget][-1]
        else:
            alpha_max = pos_alphas[0]
        
        ax.axhline(y=eps, color=color, linestyle=':', alpha=0.5, linewidth=1)
        ax.scatter([alpha_max], [eps], color=color, s=100, zorder=5,
                   marker='s', edgecolors='black', linewidths=0.5)
        ax.annotate(f'ε={eps}', xy=(alpha_max, eps), 
                    xytext=(alpha_max + 0.3, eps + 0.08),
                    fontsize=8, color=color)


def generate_text(model, prompt, steering_vector, hook_point, alpha=0.0, max_activation=1.0, max_tokens=20):
    """Generate text with optional steering."""
    tokens = model.to_tokens(prompt)
    
    if alpha != 0:
        hook_fn = create_steering_hook(steering_vector, alpha, max_activation)
        hooks = [(hook_point, hook_fn)]
    else:
        hooks = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                logits = model(tokens)
            
            next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == model.tokenizer.eos_token_id:
                break
    
    return model.tokenizer.decode(tokens[0])
