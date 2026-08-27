"""
Part 6 - Activation Editor.

Uses ActivationPatcher (built in Part 1's hooks.py) to change what a layer
outputs DURING a live forward pass. Nothing here is saved to disk or
persists past one call - this is the "try an edit and see what happens"
step, and the direct precursor to VEMR's MODIFY step (Parts 8-9 make edits
like this persist to real weights, with automatic verification and rollback).
"""
import torch

from .model_loader import layers
from .hooks import ActivationPatcher, ActivationCapture
from .logit_lens import reverse_lookup


def make_edit_fn(op, factor=1.0):
    """Build the function ActivationPatcher calls on the target layer's
    output tensor during the forward pass. VEMR treats this as a black box
    edit function E - same idea, just running live instead of persisted."""
    if op == "zero":
        return lambda h: torch.zeros_like(h)
    if op == "scale":
        return lambda h: h * factor
    raise ValueError(f"Unknown op: {op!r}. Supported: zero, scale")


def traced_run(model, tokenizer, prompt, top_k=5, position=-1, edit_layer=None, edit_fn=None):
    """Run the model (optionally with a live edit applied at edit_layer) and
    return the full per-layer logit-lens trace - same shape as Part 4's
    trace_prompt, so an edited and an unedited run can be compared layer by
    layer, watching the edit's effect ripple forward through later layers."""
    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    pos = position if position >= 0 else seq_len + position

    def _run():
        with ActivationCapture(model) as cap:
            with torch.no_grad():
                model(**inputs)
            return cap.captured

    if edit_layer is not None and edit_fn is not None:
        # ActivationPatcher's hook must be registered before ActivationCapture's
        # so the capture on the edited layer records the EDITED value, and
        # every later layer naturally sees the edited value flow forward.
        with ActivationPatcher(model, edit_layer, edit_fn):
            captured = _run()
    else:
        captured = _run()

    results = []
    for layer_idx in range(len(layers(model))):
        hidden = captured[layer_idx][0, pos]
        top_probs, top_ids = reverse_lookup(model, hidden, top_k)
        guesses = [(tokenizer.decode([tid]), float(p)) for tid, p in zip(top_ids, top_probs)]
        results.append((layer_idx, guesses))
    return results


def diff_top1(baseline, edited):
    """Which layers had their #1 guess change because of the edit - a quick
    summary of how far the edit's effect actually propagated."""
    changes = []
    for (li_b, gb), (li_e, ge) in zip(baseline, edited):
        top1_b = gb[0][0] if gb else None
        top1_e = ge[0][0] if ge else None
        if top1_b != top1_e:
            changes.append((li_b, top1_b, top1_e))
    return changes
