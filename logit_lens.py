"""
Part 4 - Token Evolution + Logit Lens.

Ports day2_logit_lens.py's logic into the package (using hooks.py's
ActivationCapture instead of its old hand-rolled hook loop), and adds a new
capability: watching the logit lens evolve across an entire GENERATED
sequence, not just a single fixed prompt's last token. That's the
"token evolution" half - generate a few tokens, and for each one, replay
what every layer believed at the moment it was produced.
"""
import torch

from .model_loader import layers
from .hooks import ActivationCapture


def reverse_lookup(model, hidden_vec, top_k=5):
    """The core trick, unchanged from Day 2: project a hidden state through
    the model's own final layer-norm + unembedding matrix to read it as
    token probabilities."""
    with torch.no_grad():
        normed = model.transformer.ln_f(hidden_vec)
        logits = model.lm_head(normed)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)
    return top_probs, top_ids


def trace_prompt(model, tokenizer, prompt, top_k=5, position=-1):
    """Original Day 2 behaviour: one prompt, one token position (default:
    the last token), logit lens read out at every layer.

    Returns: list of (layer_idx, [(token_str, prob), ...]) per layer.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    pos = position if position >= 0 else seq_len + position

    with ActivationCapture(model) as cap:
        with torch.no_grad():
            model(**inputs)
        captured = cap.captured

    results = []
    for layer_idx in range(len(layers(model))):
        hidden = captured[layer_idx][0, pos]
        top_probs, top_ids = reverse_lookup(model, hidden, top_k)
        guesses = [(tokenizer.decode([tid]), float(p)) for tid, p in zip(top_ids, top_probs)]
        results.append((layer_idx, guesses))
    return results, inputs


def trace_generation(model, tokenizer, prompt, num_new_tokens=5, top_k=5, layers_to_show=None):
    """Token Evolution: generate num_new_tokens one at a time (greedy), and
    for EACH generated token, run the logit lens at every layer at the
    position that just produced it. Lets you watch, step by step, whether
    the model 'knew' the answer early or only converged at the last layer -
    for every token it actually outputs, not just one fixed spot.

    Returns: list of dicts, one per generated step:
        {"token": str, "layers": [(layer_idx, [(tok, prob), ...]), ...]}
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    all_layer_indices = list(range(len(layers(model))))
    layers_to_show = layers_to_show if layers_to_show is not None else all_layer_indices

    steps = []
    for _ in range(num_new_tokens):
        current_inputs = {"input_ids": input_ids}
        with ActivationCapture(model, layer_indices=layers_to_show) as cap:
            with torch.no_grad():
                outputs = model(**current_inputs)
            captured = cap.captured

        last_pos = input_ids.shape[1] - 1
        layer_results = []
        for layer_idx in layers_to_show:
            hidden = captured[layer_idx][0, last_pos]
            top_probs, top_ids = reverse_lookup(model, hidden, top_k)
            guesses = [(tokenizer.decode([tid]), float(p)) for tid, p in zip(top_ids, top_probs)]
            layer_results.append((layer_idx, guesses))

        next_token_id = outputs.logits[0, -1].argmax().item()
        next_token_str = tokenizer.decode([next_token_id])
        steps.append({"token": next_token_str, "layers": layer_results})

        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

    return steps
