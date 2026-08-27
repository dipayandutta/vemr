"""
Part 8 - Layer Extraction.

This is the literal "pull the Stone out of the Gauntlet" step: take one
layer's weights out of a running model and save them to a standalone file,
with enough metadata to know exactly where they came from and enough
integrity-checking to know the file hasn't been corrupted or silently
altered before anyone tries to put it back.

Scope on purpose: Part 8 only does EXTRACT -> SAVE -> LOAD -> RESTORE, and
proves that round-trip is lossless (bit-identical). It does NOT do the
edit-verify-commit-or-rollback logic - that's VEMR proper, built in Part 9
on top of this file's `restore_layer_into_model` plus Part 6's edit_fn
plus a probe-set divergence check. Keeping the boundary here matters: a
clean, trustworthy extraction primitive is what VEMR's safety guarantee
in Part 9 actually rests on.
"""
import hashlib
import time

import torch

from .model_loader import layers, layer_state_dict, DEFAULT_MODEL


def _checksum(state_dict) -> str:
    """Deterministic SHA-256 over every tensor's raw bytes, in a fixed key
    order (dict insertion order isn't guaranteed identical across a
    save/load round-trip on every platform, so we sort explicitly). This is
    what proves a restored layer is bit-identical to what was extracted,
    not just 'close enough' or 'the right shape'."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        h.update(state_dict[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def extract_layer(model, layer_idx: int, model_name: str = DEFAULT_MODEL) -> dict:
    """Pull one layer's weights out of a live model into a standalone,
    self-describing payload - the in-memory extraction. Nothing is written
    to disk yet (see save_layer)."""
    state_dict = {k: v.clone() for k, v in layer_state_dict(model, layer_idx).items()}
    shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    checksum = _checksum(state_dict)
    return {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "shapes": shapes,
        "checksum": checksum,
        "state_dict": state_dict,
    }


def save_layer(payload: dict, path: str) -> None:
    """Write an extracted-layer payload to disk. Plain torch.save - the
    payload is just tensors plus JSON-safe metadata, no custom format
    needed."""
    torch.save(payload, path)


def load_layer(path: str) -> dict:
    """Read an extracted-layer payload back from disk."""
    return torch.load(path, weights_only=False)


def verify_layer(payload: dict) -> bool:
    """Recompute the checksum from the payload's actual tensors and compare
    against the checksum stored at extraction time. True means the tensors
    are exactly what was extracted - nothing was corrupted, truncated, or
    silently modified in between."""
    return _checksum(payload["state_dict"]) == payload["checksum"]


def restore_layer_into_model(model, layer_idx: int, payload: dict, strict: bool = True):
    """Copy an extracted payload's weights into a live model's layer.

    This is the raw mechanism only - no verify-then-commit-or-rollback
    logic lives here (that's Part 9 / VEMR). Used two ways:
      1. Round-trip proof: extract layer N, restore it right back into the
         SAME layer N of the SAME model, confirm nothing changed.
      2. The primitive Part 9 will call after applying an edit function to
         the extracted state_dict, before running VEMR's verification step.

    Raises if shapes don't match the target layer, unless strict=False.
    """
    target_block = layers(model)[layer_idx]
    current_shapes = {k: tuple(v.shape) for k, v in target_block.state_dict().items()}
    if strict and current_shapes != payload["shapes"]:
        raise ValueError(
            f"Shape mismatch: payload was extracted from layer {payload['layer_idx']} "
            f"of {payload['model_name']!r} and does not match target layer {layer_idx}'s "
            f"current shapes. Refusing to load (pass strict=False to override)."
        )
    target_block.load_state_dict(payload["state_dict"])
    return model


def ablate_layer(model, layer_idx: int):
    """Physically remove one layer's block from a LIVE model, in place -
    the model actually shrinks from N layers to N-1. Different in kind from
    extract_layer: extract_layer copies weights out while leaving the
    running model completely untouched (photographing the Stone).
    ablate_layer is the real amputation - deleting the block from
    `model.transformer.h` (an nn.ModuleList), the way Tony Stark actually
    pulls the Stone out of Vision's forehead, leaving a visible gap.

    Structurally safe to do: every GPT-2 block maps a (batch, seq, 768)
    residual stream to another (batch, seq, 768) one - homogeneous shapes
    in and out. Deleting a block never creates a dimension mismatch
    anywhere downstream, so the shortened model still runs end to end and
    still produces a full probability distribution over the vocabulary.
    It just won't be the SAME distribution, because a whole computation
    step is now missing - that's the entire point of this function: making
    it possible to directly observe what a "gutted" model actually does.

    Note this only affects the in-memory `model` object for as long as the
    current process is alive - `llmsurgery` reloads a fresh model on every
    separate CLI invocation, so ablating in one command can't be "seen" by
    a later, separate command. Use `llmsurgery ablate`, which does the
    remove-and-compare in one process, to observe the effect.

    Returns the removed block (the caller should have already called
    extract_layer + save_layer BEFORE this, if the weights need to survive).
    """
    block_list = layers(model)
    removed = block_list[layer_idx]
    del block_list[layer_idx]
    return removed
