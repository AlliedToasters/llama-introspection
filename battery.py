import argparse
from pathlib import Path
import os
from dotenv import load_dotenv
import hashlib

from nnsight import CONFIG, LanguageModel
from anthropic import Anthropic
import pandas as pd

from introspection import run_experiment, compute_layer_indices, INJECTION_MESSAGES, ALL_CONCEPTS, BASELINE_WORDS, GENERIC_PROMPT_TEMPLATE
from config import MODEL_SHORTCUTS, REMOTE_MODELS
from steering_vectors import (
    compute_generic_vector,
    compute_injection_position,
    get_layer_accessor,
    compute_mean_steering_norm,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Introspection experiment with multiple conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="1B",
                        help="Model to use (1B, 8B, 70B, 405B or full name)")
    parser.add_argument("--use-remote", action="store_true",
                        help="Force remote execution")
    parser.add_argument("--n-control-trials", type=int, default=5, 
                        help="Number of runs without intervention")
    parser.add_argument("--n-random-trials", type=int, default=5, 
                        help="Number of random vector runs per strength")
    parser.add_argument("--n-scale-trials", type=int, default=5, 
                        help="Number of random vector runs per strength")
    parser.add_argument("--n-concept-trials", type=int, default=5, 
                        help="Number of random vector runs per strength")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed for random vector generation")
    return parser.parse_args()

def unpack_rows(results: dict) -> list:
    intervention_layer = list(results.keys())[0]
    _data = results[intervention_layer]
    strength = list(_data.keys())[0]
    _data = _data[strength]
    rows = []
    for item in _data:
        full_text = item['text']
        response = "assistant".join(full_text.split("assistant")[2:])
        geometry = item['geometry']
        pre_norm = geometry['pre_norm']
        post_norm = geometry['post_norm']
        l2_distance = geometry.get('l2_distance')
        if l2_distance is None:
            print(f"Weird geometry: {geometry}")
            print(f"Deriving l2_distance from norms...")
            # this seems to happen only for scale interventions,
            # so we can just subtract post - pre
            l2_distance = post_norm - pre_norm
        grade = item['grade']
        # we use 'get' accessor for claude responses because some fields may be missing
        # thanks, Claude
        refusal = grade.get('refusal')
        affirmative = grade.get('affirmative')
        correct_id = grade.get('correct_id')
        early_detection = grade.get('early_detection')
        coherent = grade.get('coherent')
        grader_reasoning = grade.get('reasoning')
        concept = item.get('concept', 'N/A')
        row = {
            "intervention_layer": intervention_layer,
            "strength": strength,
            "concept": concept,
            "full_text": full_text,
            "response": response,
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "l2_distance": l2_distance,
            "refusal": refusal,
            "affirmative": affirmative,
            "correct_id": correct_id,
            "early_detection": early_detection,
            "coherent": coherent,
            "grader_reasoning": grader_reasoning,
            "trial": item.get('trial', -1),
        }
        rows.append(row)
    return rows
        

if __name__ == "__main__":
    args = parse_args()

    output_cols = [
        "intervention",
        "intervention_layer",
        "strength",
        "concept",
        "full_text",
        "response",
        "pre_norm",
        "post_norm",
        "l2_distance",
        "refusal",
        "affirmative",
        "correct_id",
        "early_detection",
        "coherent",
        "grader_reasoning",
        "trial",
    ]
    rows = []
    
    # Resolve model name
    model_name = MODEL_SHORTCUTS.get(args.model, args.model)
    use_remote = args.use_remote or args.model in REMOTE_MODELS or model_name in REMOTE_MODELS

    # Setup API keys
    load_dotenv()
    CONFIG.API.APIKEY = os.getenv("NNSIGHT_API_KEY")
    if not CONFIG.API.APIKEY:
        raise ValueError("NNSIGHT_API_KEY not found")
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY required for grading")
    
    client = Anthropic(api_key=anthropic_key)

    # Load model
    print(f"\nLoading model...")
    if use_remote:
        model = LanguageModel(model_name)
    else:
        model = LanguageModel(model_name, device_map="auto")

    # Get layers
    layers = get_layer_accessor(model)(model)
    num_layers = len(layers)
    layer_indices = compute_layer_indices(num_layers, 1)
    
    print(f"Layers: {num_layers}, Testing: {layer_indices}")

    # Compute injection position
    injection_prompt, injection_start_pos, total_tokens = compute_injection_position(
        model.tokenizer, INJECTION_MESSAGES
    )
    print(f"Injection starts at token {injection_start_pos}/{total_tokens}")

    # Get hidden dim
    with model.trace(remote=use_remote) as tracer:
        hidden_dim_proxy = list().save()
        with tracer.invoke("test"):
            hidden_dim_proxy.append(layers[0].output.shape[-1])
    hidden_dim = int(hidden_dim_proxy[0])
    print(f"Hidden dim: {hidden_dim}")

    # run experiment serially:

    # Pre-compute all steering vectors
    concepts = ALL_CONCEPTS
    n_concepts = len(concepts)
    print(f"\nPre-computing steering vectors for {n_concepts} concepts...")
    mean_steering_norm = compute_mean_steering_norm(
        model_slug=model_name,
        layer_idx=layer_indices[0],
        cache_dir=args.output_dir,
        concepts=concepts,
    )



    
    # control runs
    # Create experiment hash
    exp_id = f"{model_name}|control"
    exp_hash = hashlib.md5(exp_id.encode()).hexdigest()[:8]
    # Run experiment
    save_path = Path(args.output_dir) / f"exp_battery_{exp_hash}.pt"
    print(f"\n=== Control Trials ({exp_hash}) ===")
    results, stopped_at, config = run_experiment(
        model,
        hidden_dim,
        mean_steering_norm,
        args,
        model_name,
        "concept", # with strength 0.0, this is effectively control
        None, # dummy concept name
        layers,
        injection_prompt,
        injection_start_pos,
        layer_indices,
        strengths=[0.0],
        n_trials=args.n_control_trials,
        incoherence_tolerance=2,
        client=client,
        save_path=save_path,
        use_remote=use_remote,
    )
    _new_rows = unpack_rows(results)
    # add intervention type
    for row in _new_rows:
        row['intervention'] = 'control'
    rows.extend(_new_rows)
    df = pd.DataFrame(rows, columns=output_cols)
    # save intermediate results
    df.to_csv(Path(args.output_dir) / "introspection_battery_results_tmp.csv", index=False)

    # concept vector runs
    print(f"\n=== Concept Vector Trials ===")
    for concept in concepts:
        # Create experiment hash
        exp_id = f"{model_name}|concept|{concept}"
        exp_hash = hashlib.md5(exp_id.encode()).hexdigest()[:8]
        # Run experiment
        save_path = Path(args.output_dir) / f"exp_battery_{exp_hash}.pt"
        print(f"\n--- Concept: {concept} ---")
        print(f"\nComputing/fetching steering vector for '{concept}'...")
        result = compute_generic_vector(
            model=model,
            model_slug=model_name,
            concept_word=concept,
            baseline_words=BASELINE_WORDS,
            prompt_template=GENERIC_PROMPT_TEMPLATE,
            cache_dir=args.output_dir,
            use_remote=use_remote,
        )
        # Use vector from injection layer
        injection_layer = layer_indices[0]
        steering_vector = result.vectors[injection_layer]
        print(f"Steering vector norm: {steering_vector.norm().item():.2f}")
    
        results, stopped_at, config = run_experiment(
            model,
            hidden_dim,
            mean_steering_norm,
            args,
            model_name,
            "concept",
            concept,
            layers,
            injection_prompt,
            injection_start_pos,
            layer_indices,
            strengths=[0.5, 1.0, 2.0, 4.0],
            n_trials=args.n_concept_trials,
            incoherence_tolerance=2,
            client=client,
            save_path=save_path,
            steering_vector=steering_vector,
            use_remote=use_remote,
        )
        _new_rows = unpack_rows(results)
        # add intervention type
        for row in _new_rows:
            row['intervention'] = 'concept'
        rows.extend(_new_rows)
        df = pd.DataFrame(rows, columns=output_cols)
        # save intermediate results
        df.to_csv(Path(args.output_dir) / "introspection_battery_results_tmp.csv", index=False)

    # random vector runs
    # Create experiment hash
    exp_id = f"{model_name}|random"
    exp_hash = hashlib.md5(exp_id.encode()).hexdigest()[:8]
    # Run experiment
    save_path = Path(args.output_dir) / f"exp_battery_{exp_hash}.pt"
    print(f"\n=== Random Vector Trials ({exp_hash}) ===")
    results, stopped_at, config = run_experiment(
        model,
        hidden_dim,
        mean_steering_norm,
        args,
        model_name,
        "random",
        None, # there is no known concept for a random vector
        layers,
        injection_prompt,
        injection_start_pos,
        layer_indices,
        strengths=[0.5, 1.0, 2.0, 4.0],
        n_trials=args.n_random_trials,
        incoherence_tolerance=2,
        client=client,
        save_path=save_path,
        use_remote=use_remote,
    )
    _new_rows = unpack_rows(results)
    # add intervention type
    for row in _new_rows:
        row['intervention'] = 'random'
    rows.extend(_new_rows)
    df = pd.DataFrame(rows, columns=output_cols)
    # save intermediate results
    df.to_csv(Path(args.output_dir) / "introspection_battery_results_tmp.csv", index=False)

    # scale vector runs
    # Create experiment hash
    exp_id = f"{model_name}|scale"
    exp_hash = hashlib.md5(exp_id.encode()).hexdigest()[:8]
    # Run experiment
    save_path = Path(args.output_dir) / f"exp_battery_{exp_hash}.pt"
    print(f"\n=== Scale Vector Trials ({exp_hash}) ===")
    results, stopped_at, config = run_experiment(
        model,
        hidden_dim,
        mean_steering_norm,
        args,
        model_name,
        "scale",
        None, # there is no known concept for a scale vector
        layers,
        injection_prompt,
        injection_start_pos,
        layer_indices,
        strengths=[0.5, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0],
        n_trials=args.n_scale_trials,
        incoherence_tolerance=2,
        client=client,
        save_path=save_path,
        use_remote=use_remote,
    )
    _new_rows = unpack_rows(results)
    # add intervention type
    for row in _new_rows:
        row['intervention'] = 'scale'
    rows.extend(_new_rows)
    df = pd.DataFrame(rows, columns=output_cols)
    # save final results
    final_results_path = Path(args.output_dir) / f"introspection_battery_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(final_results_path, index=False)

    print("\nExperiment complete.")
    print(f"Results saved to {final_results_path}")

    exit(0)