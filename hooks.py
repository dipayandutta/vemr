"""
Reusable forward-hook infrastructure.

day2_logit_lens.py and day3_bloom_cache.py each hand-rolled their own
hook-registration loop. This module generalizes that pattern once, as a
context manager, so every later part (neuron viewer, attention explorer,
activation editor, circuit discovery...) can reuse the same tap instead of
re-implementing it.
"""
import torch
from .model_loader import layers


def _unwrap_hidden_states(output):
    """Newer transformers versions return the hidden-state tensor directly
    from a block's forward; older versions wrap it in a tuple. Handle both -
    this is the exact bug we hit and fixed in Day 2."""
    return output[0] if isinstance(output, tuple) else output


class ActivationCapture:
    """Captures every layer's output hidden state during one forward pass.

    Usage:
        with ActivationCapture(model) as cap:
            model(**inputs)
        cap.captured[layer_idx]  # -> tensor (batch, seq, hidden)
    """

    def __init__(self, model, layer_indices=None):
        self.model = model
        self.layer_indices = layer_indices  # None = all layers
        self.captured = {}
        self._handles = []

    def __enter__(self):
        self.captured = {}
        targets = (
            enumerate(layers(self.model))
            if self.layer_indices is None
            else ((i, layers(self.model)[i]) for i in self.layer_indices)
        )
        for idx, block in targets:
            handle = block.register_forward_hook(self._make_hook(idx))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def _make_hook(self, layer_idx):
        def hook(module, inp, output):
            self.captured[layer_idx] = _unwrap_hidden_states(output).detach()
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class ActivationPatcher:
    """Overwrites a layer's output during the forward pass with a supplied
    tensor - the mechanism Part 6 (Activation Editor) will build on top of.

    Usage:
        def edit_fn(hidden_states):
            return hidden_states * 0   # e.g. zero out the layer
        with ActivationPatcher(model, layer_idx=5, edit_fn=edit_fn):
            model(**inputs)   # layer 5's output is replaced live
    """

    def __init__(self, model, layer_idx: int, edit_fn):
        self.model = model
        self.layer_idx = layer_idx
        self.edit_fn = edit_fn
        self._handle = None

    def __enter__(self):
        block = layers(self.model)[self.layer_idx]
        self._handle = block.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inp, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = _unwrap_hidden_states(output)
        edited = self.edit_fn(hidden_states)
        if is_tuple:
            return (edited,) + output[1:]
        return edited


def run_with_capture(model, inputs, layer_indices=None):
    """Convenience one-liner: run a forward pass and return captured activations."""
    with ActivationCapture(model, layer_indices) as cap:
        with torch.no_grad():
            model(**inputs)
        return cap.captured