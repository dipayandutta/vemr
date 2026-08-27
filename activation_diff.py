import torch
from .neuron_viewer import capture_neuron_activations

def _pool(activations, mode="mean"):
    """
    mode = "mean": average across every token - a general
    "how did this neuron behave over the whole prompt"
    mode = "last" just the final token - what the neuron is doing
    right where the model is about to predict the next word
    """
    if mode =="mean":
        return activations[0].mean(dim=0)
    if mode == "last":
        return activations[0, -1]
    raise ValueError(f"unknown pool mode: {mode}")

def neuron_diff(model, tokenizer, prompt_a, prompt_b, layer_idx, pool="mean", top_k=10):
    inputs_a = tokenizer(prompt_a, return_tensors="pt")
    inputs_b = tokenizer(prompt_b, return_tensors="pt")

    acts_a = capture_neuron_activations(model, inputs_a, layer_idx)
    acts_b = capture_neuron_activations(model, inputs_b, layer_idx)

    pooled_a = _pool(acts_a, pool)
    pooled_b = _pool(acts_b, pool)

    diff = (pooled_a - pooled_b).abs()
    top_vals, top_idxs = torch.topk(diff, top_k)

    results = []

    for idx, val in zip(top_idxs.tolist(), top_vals.tolist()):
        results.append({
            "neuron": idx,
            "diff": val,
            "a": pooled_a[idx].item(),
            "b": pooled_b[idx].item(),
        })

    return results