#!/usr/bin/env python3
"""
Test whether interventions in trace mode propagate to downstream layers.

This investigates whether modifying activations at layer L affects
the activations at layers L+1, L+2, etc.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from nnsight import LanguageModel

def test_propagation():
    # Load tiny model
    print("Loading tiny model...")
    model = LanguageModel('llamafactory/tiny-random-Llama-3', device_map='cpu')
    layers = model.model.layers
    num_layers = len(layers)
    print(f'Model has {num_layers} layers')

    prompt = 'Hello, world!'

    # 1. Baseline trace - capture layer outputs
    print('\n=== Baseline Trace ===')
    with model.trace(remote=False) as tracer:
        baseline_outputs = list().save()
        with tracer.invoke(prompt):
            for l in range(num_layers):
                baseline_outputs.append(layers[l].output[:, -1, :].clone())

    baseline_norms = [o.detach().norm().item() for o in baseline_outputs]
    print(f'Layer norms: {[round(n, 2) for n in baseline_norms]}')

    # 2. Intervention trace - add large vector at layer 0
    print('\n=== Intervention Trace (add vector at layer 0) ===')
    hidden_dim = baseline_outputs[0].shape[-1]
    big_vector = torch.randn(hidden_dim) * 100  # Large perturbation

    with model.trace(remote=False) as tracer:
        intervened_outputs = list().save()
        with tracer.invoke(prompt):
            for l in range(num_layers):
                if l == 0:
                    # Apply intervention BEFORE capturing
                    layers[l].output[:, -1, :] += big_vector
                intervened_outputs.append(layers[l].output[:, -1, :].clone())

    intervened_norms = [o.detach().norm().item() for o in intervened_outputs]
    print(f'Layer norms: {[round(n, 2) for n in intervened_norms]}')

    # 3. Compare
    print('\n=== Comparison ===')
    for l in range(num_layers):
        diff = abs(intervened_norms[l] - baseline_norms[l])
        status = "CHANGED" if diff > 1.0 else "same"
        print(f'Layer {l}: baseline={baseline_norms[l]:.2f}, intervened={intervened_norms[l]:.2f}, diff={diff:.2f} [{status}]')

    # Check if downstream layers changed
    if num_layers > 1:
        downstream_diff = abs(intervened_norms[-1] - baseline_norms[-1])
        print(f'\nDownstream propagation test:')
        if downstream_diff > 1.0:
            print(f'  ✓ Intervention at layer 0 PROPAGATED to layer {num_layers-1}')
        else:
            print(f'  ✗ Intervention at layer 0 did NOT propagate to layer {num_layers-1}')
            print(f'    This means trace-mode interventions may not affect downstream computation!')


def test_generation_vs_trace():
    """Compare whether intervention affects actual generation."""
    print("\n" + "="*60)
    print("=== Generation vs Trace Comparison ===")
    print("="*60)

    model = LanguageModel('llamafactory/tiny-random-Llama-3', device_map='cpu')
    layers = model.model.layers
    num_layers = len(layers)

    prompt = 'The capital of France is'

    # Get hidden dim from model config
    hidden_dim = model._model.config.hidden_size
    print(f'Hidden dim: {hidden_dim}')

    # 1. Baseline generation
    print('\n--- Baseline Generation ---')
    with model.generate(prompt, max_new_tokens=5, remote=False):
        baseline_output = model.generator.output.save()

    baseline_text = model.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    print(f'Output: {baseline_text}')

    # 2. Generation with intervention
    print('\n--- Generation with Intervention (layer 0) ---')
    big_vector = torch.randn(hidden_dim) * 1000

    with model.generate(prompt, max_new_tokens=5, remote=False):
        # Intervene on all generated tokens - need to match shape
        layers[0].output[:, :, :] = layers[0].output[:, :, :] + big_vector
        intervened_output = model.generator.output.save()

    intervened_text = model.tokenizer.decode(intervened_output[0], skip_special_tokens=True)
    print(f'Output: {intervened_text}')

    # Compare
    print('\n--- Comparison ---')
    if baseline_text != intervened_text:
        print('✓ Intervention CHANGED the generated text')
    else:
        print('✗ Intervention did NOT change generated text (concerning!)')


if __name__ == '__main__':
    test_propagation()
    test_generation_vs_trace()
