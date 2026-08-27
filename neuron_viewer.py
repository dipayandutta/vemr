"""
Part 2 - Neuron Viewer.

"Neurons" = the 3072 intermediate units inside one layer's MLP (the expanded
space between mlp.c_fc and mlp.c_proj), read right after the activation
function fires. Different, much larger space than the 768-wide residual
stream the logit lens reads - 3072 x 12 layers = 36,864 of these in GPT-2
small. Each one is a candidate "specialist" worth poking at.
"""
import torch

from .model_loader import layers


def capture_neuron_activations(model, inputs, layer_idx):
    """Hook the activation function inside one layer's MLP.
    Returns shape (batch, seq, 3072) - one row of 3072 neuron values per token.
    """
    block = layers(model)[layer_idx]
    captured = {}

    def hook(module, inp, output):
        captured["acts"] = output.detach()

    handle = block.mlp.act.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["acts"]


def neuron_trace(activations, neuron_idx):
    """One neuron's activation value at every token position in the prompt."""
    return activations[0, :, neuron_idx].tolist()


def top_neurons_at_token(activations, token_idx, top_k=10):
    """Which neurons fired strongest for one specific token."""
    token_acts = activations[0, token_idx]  # (3072,)
    top_vals, top_idxs = torch.topk(token_acts, top_k)
    return list(zip(top_idxs.tolist(), top_vals.tolist()))


def top_neurons_overall(activations, top_k=10):
    """Which neurons had the highest peak activation anywhere in the prompt -
    a quick way to discover interesting neurons worth drilling into."""
    max_per_neuron, _ = activations[0].max(dim=0)  # (3072,)
    top_vals, top_idxs = torch.topk(max_per_neuron, top_k)
    return list(zip(top_idxs.tolist(), top_vals.tolist()))


def probe_neuron(model, tokenizer, layer_idx, neuron_idx, positive_prompts, negative_prompts):
    """Screen one candidate neuron against a hypothesis in a single call:
    a set of prompts it's hypothesized to fire on (positive), and a set of
    unrelated prompts it should stay quiet on (negative). This is exactly
    the replicate + control pattern used by hand across several separate
    `neurons` calls to confirm neuron 792 (Part 3) and Head 0 (Part 5) -
    automated here so Part 7's discovery-mode candidates (found by
    selectivity alone, with no hypothesis testing yet) can be triaged
    quickly instead of one manual multi-command cycle per candidate.

    Verdict is a simple, honest heuristic, not a claim of confirmation:
    "LOOKS REAL" only means the positive group's weakest peak beat the
    negative group's strongest peak - a clean separation worth digging
    into further, the same bar neuron 1253 FAILED (its positive and
    negative prompts overlapped almost completely). It is not a
    substitute for the judgment calls that actually confirmed 792 and
    Head 0 (checking WHICH token triggers it, whether the shape makes
    semantic sense, etc.) - just a fast first filter.
    """
    def _peak(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
        acts = capture_neuron_activations(model, inputs, layer_idx)
        trace = neuron_trace(acts, neuron_idx)
        peak_val = max(trace)
        peak_tok = tokens[trace.index(peak_val)]
        return peak_val, peak_tok

    positive = [(p,) + _peak(p) for p in positive_prompts]
    negative = [(p,) + _peak(p) for p in negative_prompts]

    pos_vals = [v for _, v, _ in positive]
    neg_vals = [v for _, v, _ in negative]

    if pos_vals and neg_vals and min(pos_vals) > max(neg_vals):
        verdict = "LOOKS REAL - clean separation, worth a closer manual look"
    else:
        verdict = "INCONCLUSIVE - positive/negative activations overlap, likely not selective for this hypothesis"

    return {"positive": positive, "negative": negative, "verdict": verdict}
