"""
Part 7 - Circuit Discovery + Feature Search.

Two techniques, both reusing infrastructure from earlier parts instead of
needing anything new under the hood:

1. Causal activation patching (a simplified version of the "causal tracing"
   technique behind the ROME paper). Everything so far in this project has
   been correlational: "layer 9 tends to favor Paris." Patching answers a
   causal question instead: "if I forcibly transplant layer 9's activation
   from a run that gets the right answer into a run that doesn't, does the
   wrong run's answer change?" If yes, that layer is doing real causal work,
   not just happening to correlate with the answer.

2. Feature search - Part 2's neuron viewer required already knowing which
   neuron to check. This scans a bank of candidate prompts either against
   one specific neuron (confirms/refutes a hypothesis) or across ALL 3072
   neurons in a layer at once, ranked by selectivity - genuine discovery.
"""
import torch

from .model_loader import layers
from .hooks import ActivationCapture, ActivationPatcher
from .logit_lens import reverse_lookup
from .neuron_viewer import capture_neuron_activations


# A small, deliberately diverse prompt bank spanning different topics, used
# as the "corpus" for feature search - diverse on purpose, so a neuron that
# fires on only one of these is a real candidate for being selective.
CANDIDATE_PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "My favorite food is pizza with extra cheese and",
    "She ordered sushi and green tea for",
    "The committee reviewed the quarterly budget report and",
    "My favorite tool is Kubernetes and",
    "The weather today is sunny with a chance of",
    "He plays basketball every weekend with his",
    "The stock market fell sharply after the",
    "Scientists discovered a new species of",
    "The concert was postponed due to heavy",
    "Our flight was delayed because of the",
    "The recipe calls for two cups of",
    "The startup raised ten million dollars in",
    "The museum's new exhibit features ancient",
    "The dog chased the ball across the",
]


# ---------------------------------------------------------------------------
# Circuit discovery: causal activation patching
# ---------------------------------------------------------------------------
def _last_token_activations(model, tokenizer, prompt):
    """Capture every layer's activation at the LAST token position only.
    Using the last position (rather than trying to align two differently-
    tokenized prompts word-for-word) keeps this robust: both prompts end at
    the same grammatical spot ("...the city of"), regardless of how many
    tokens their subjects took earlier."""
    inputs = tokenizer(prompt, return_tensors="pt")
    pos = inputs["input_ids"].shape[1] - 1
    with ActivationCapture(model) as cap:
        with torch.no_grad():
            model(**inputs)
        acts = {li: t[0, pos].clone() for li, t in cap.captured.items()}
    return acts, pos, inputs


def _make_position_patch_fn(position, replacement_vector):
    def fn(hidden_states):
        edited = hidden_states.clone()
        edited[0, position] = replacement_vector
        return edited
    return fn


def _final_layer_guess(model, tokenizer, inputs, position, top_k):
    last_layer = len(layers(model)) - 1
    with ActivationCapture(model, layer_indices=[last_layer]) as cap:
        with torch.no_grad():
            model(**inputs)
        hidden = cap.captured[last_layer][0, position]
    top_probs, top_ids = reverse_lookup(model, hidden, top_k)
    return [(tokenizer.decode([tid]), float(p)) for tid, p in zip(top_ids, top_probs)]


def patch_sweep(model, tokenizer, clean_prompt, corrupted_prompt, top_k=5):
    """For each layer, transplant the CLEAN run's last-token activation into
    the CORRUPTED run at that same layer, and see how much the corrupted
    run's final answer shifts toward the clean answer. Returns
    (corrupted_baseline, clean_answer_top1, per_layer_results)."""
    clean_acts, _, _ = _last_token_activations(model, tokenizer, clean_prompt)

    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt")
    corrupted_pos = corrupted_inputs["input_ids"].shape[1] - 1

    corrupted_baseline = _final_layer_guess(model, tokenizer, corrupted_inputs, corrupted_pos, top_k)
    clean_answer_top1 = reverse_lookup(
        model,
        clean_acts[len(layers(model)) - 1],
        1,
    )
    clean_top1_str = tokenizer.decode([clean_answer_top1[1][0]])

    results = []
    for layer_idx in range(len(layers(model))):
        patch_fn = _make_position_patch_fn(corrupted_pos, clean_acts[layer_idx])
        with ActivationPatcher(model, layer_idx, patch_fn):
            guesses = _final_layer_guess(model, tokenizer, corrupted_inputs, corrupted_pos, top_k)
        results.append((layer_idx, guesses))

    return corrupted_baseline, clean_top1_str, results


# ---------------------------------------------------------------------------
# Feature search
# ---------------------------------------------------------------------------
def search_one_neuron(model, tokenizer, layer_idx, neuron_idx, top_k=10):
    """Rank the candidate prompt bank by how strongly each one activates a
    SPECIFIC neuron - confirms or refutes a hypothesis about that neuron."""
    results = []
    for prompt in CANDIDATE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
        acts = capture_neuron_activations(model, inputs, layer_idx)
        col = acts[0, :, neuron_idx]
        peak_val, peak_pos = col.max(dim=0)
        results.append((prompt, tokens[peak_pos.item()], float(peak_val)))
    results.sort(key=lambda r: r[2], reverse=True)
    return results[:top_k]


def discover_selective_neurons(model, tokenizer, layer_idx, top_k=10):
    """Scan ALL 3072 neurons in a layer across the whole candidate prompt
    bank, ranked by selectivity: peak activation minus average activation.
    A high-selectivity neuron fires hard on one specific thing and stays
    quiet otherwise - exactly the shape neuron 792 turned out to have, and
    the opposite of neuron 1253's generic, everywhere-active shape."""
    per_prompt = []
    for prompt in CANDIDATE_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
        acts = capture_neuron_activations(model, inputs, layer_idx)[0]  # (seq, 3072)
        per_prompt.append((prompt, tokens, acts))

    all_rows = torch.cat([a for _, _, a in per_prompt], dim=0)  # (total_tokens, 3072)
    peak, _ = all_rows.max(dim=0)
    mean = all_rows.mean(dim=0)
    selectivity = peak - mean

    top_vals, top_idxs = torch.topk(selectivity, top_k)
    results = []
    for neuron_idx, sel in zip(top_idxs.tolist(), top_vals.tolist()):
        best_prompt, best_token, best_val = None, None, -1e9
        for prompt, tokens, acts in per_prompt:
            col = acts[:, neuron_idx]
            v, p = col.max(dim=0)
            if v.item() > best_val:
                best_val, best_prompt, best_token = v.item(), prompt, tokens[p.item()]
        results.append((neuron_idx, sel, best_prompt, best_token, best_val))
    return results
