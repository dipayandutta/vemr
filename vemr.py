"""
Part 9 - VEMR: Verified Extract-Modify-Reinsert, implemented for real.

This is algorithm-vemr.md's Algorithm 1, line for line. Everything before
this part rehearsed one piece of it: Part 6 (activation_editor) tried edits
live but nothing persisted past one forward pass; Part 8 (layer_extractor)
persisted extraction/restoration but with no automatic correctness check.
VEMR is those two put together, plus the piece neither had on its own: a
fixed probe set, a divergence measurement, and an automatic commit-or-
rollback decision - the model is guaranteed to end in exactly one of two
states, never a silently-broken third one.
"""
import copy

import torch

from .model_loader import layers


# A fixed probe set, deliberately small and diverse (not the same prompt
# repeated) - the spec calls for D_probe to be FIXED so results are
# comparable across runs, not re-chosen per edit to make an edit look good.
DEFAULT_PROBE_SET = [
    "The Eiffel Tower is located in the city of",
    "My favorite food is pizza with extra cheese and",
    "The capital of Japan is",
    "Water boils at a temperature of",
    "The president of the United States lives in the",
]


# ---------------------------------------------------------------------------
# Edit functions E(theta, p) - these operate on a layer's WEIGHTS
# (state_dict), permanently, unlike Part 6's activation_editor functions,
# which only ever touched what flows THROUGH a layer during one forward
# pass. VEMR doesn't care what E does internally (edit-agnostic by design) -
# these four are just enough to exercise the protocol, including "swap,"
# which is mergekit-style layer-swapping wrapped in VEMR's safety net
# instead of applied blind.
# ---------------------------------------------------------------------------
def make_weight_edit_fn(op: str, **params):
    if op == "zero":
        return lambda sd: {k: torch.zeros_like(v) for k, v in sd.items()}

    if op == "scale":
        factor = params["factor"]
        return lambda sd: {k: v * factor for k, v in sd.items()}

    if op == "noise":
        std = params.get("std", 0.01)
        return lambda sd: {k: v + torch.randn_like(v) * std for k, v in sd.items()}

    if op == "swap":
        # params["payload"] is another extracted layer's payload, as
        # returned by layer_extractor.load_layer(). Shapes are guaranteed
        # to match for two layers of the same architecture (GPT-2 blocks
        # are homogeneous), so no explicit shape check is needed here -
        # restore_layer_into_model's strict check covers the case where
        # they don't, if this is ever pointed at a different architecture.
        payload = params["payload"]
        return lambda sd: {k: v.clone() for k, v in payload["state_dict"].items()}

    raise ValueError(f"Unknown op: {op!r}. Supported: zero, scale, noise, swap")


def make_neuron_edit_fn(neuron_idx: int, op: str = "zero", factor: float = 1.0):
    """A SURGICAL, single-neuron edit function E(theta, p) - unlike
    make_weight_edit_fn's ops, which touch every weight in a layer's whole
    state_dict (~7.09M params), this touches only the handful of
    parameters belonging to ONE of the layer's 3072 MLP neurons. This is
    what makes a real "find one neuron, edit exactly that one neuron,
    verify" capstone possible - every other edit function in this project
    could only ever operate on a whole layer at once.

    GPT-2's Conv1D layers store weights as (in_features, out_features) -
    the opposite convention from nn.Linear - so a single neuron's
    parameters are a COLUMN of c_fc and a ROW of c_proj, not a row of both:
      mlp.c_fc.weight:   (768, 3072) - neuron n is COLUMN n
      mlp.c_fc.bias:     (3072,)     - neuron n is ENTRY n
      mlp.c_proj.weight: (3072, 768) - neuron n is ROW n
      mlp.c_proj.bias:   (768,)      - shared across ALL neurons, never touched

    op="zero": neuron n's pre-activation becomes exactly 0 for every
    possible input (its c_fc column and bias are zeroed), so GELU(0)=0
    always - the neuron can never fire, regardless of what's in the
    prompt. Its c_proj row is zeroed too (redundant given the input side
    is already zero, but makes the intent unambiguous rather than relying
    on an indirect consequence).
    op="scale": leaves the neuron's trigger condition completely alone
    (it still activates on whatever it activates on) but scales its
    DOWNSTREAM contribution by `factor` - a softer edit that turns the
    neuron's influence up or down without touching its detection function.
    """
    fc_key, fc_bias_key, proj_key = "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight"

    def fn(sd):
        new_sd = {k: v.clone() for k, v in sd.items()}
        if op == "zero":
            new_sd[fc_key][:, neuron_idx] = 0.0
            new_sd[fc_bias_key][neuron_idx] = 0.0
            new_sd[proj_key][neuron_idx, :] = 0.0
        elif op == "scale":
            new_sd[proj_key][neuron_idx, :] *= factor
        else:
            raise ValueError(f"Unknown neuron op: {op!r}. Supported: zero, scale")
        return new_sd

    return fn


# ---------------------------------------------------------------------------
# f_M(x) and the two divergence signals
# ---------------------------------------------------------------------------
def _final_distribution(model, tokenizer, prompt):
    """The model's REAL final output distribution for a probe prompt - full
    softmax over the vocabulary at the last token position, exactly what
    Part 8's `ablate` command computed, reused here as f_M(x)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return torch.softmax(logits, dim=-1)


def _perplexity(model, tokenizer, prompt):
    """Standard next-token perplexity on the probe's own text - a second,
    simpler divergence signal, independent of any one probe's top-1
    answer. exp(cross-entropy loss) is the standard definition."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


# Public alias - benchmark.py (and anything else outside this module) should
# import THIS name rather than reaching for the underscore-prefixed internal
# one directly.
perplexity = _perplexity


def _kl(p, q, eps=1e-12):
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps))).item()


def _top1(dist, tokenizer):
    return tokenizer.decode([torch.argmax(dist).item()])


# ---------------------------------------------------------------------------
# Algorithm 1: VEMR(M, i, E, p, D_probe, tau)
# ---------------------------------------------------------------------------
def vemr(model, tokenizer, layer_idx, edit_fn, probe_set=None, tau=0.05, metric="kl"):
    """The protocol itself, matching algorithm-vemr.md line by line:

      1. EXTRACT   - deep-copy the target layer's weights
      2-3. BASELINE - run the probe set through the UNEDITED model
      4-5. MODIFY + REINSERT (tentative) - apply E, load it in live
      6-7. VERIFY   - run the probe set again, through the EDITED model
      8.   divergence Delta = d(before, after)
      9-13. commit if Delta <= tau, else roll back to the exact deep copy

    tau=0.05 default is not a guess - it's the empirically-calibrated
    value from `sweep_edit_severity` / `suggest_tau` below, run across
    layers 2/5/8/10 with the default probe set: mild edits topped out at
    delta 0.0122, severe edits started at delta 0.0756, and 0.05 sits in
    the middle of that gap with margin both directions (100% mild-commit,
    100% severe-catch on that grid). See findings.md for the full sweep
    and the real precision/recall tradeoff at tighter/looser tau values.

    Returns an audit_record dict: layer_idx, tau, metric, delta, status
    ("COMMITTED"/"ROLLED_BACK"), perplexity before/after, and each probe's
    top-1 answer before/after.

    Implementation requirement from the spec's safety-property note: no
    exception may escape mid-edit without guaranteeing rollback first -
    handled below with try/except around the verify step, which is the
    only part that runs after weights have already been tentatively
    swapped in.
    """
    probe_set = probe_set if probe_set is not None else DEFAULT_PROBE_SET
    block = layers(model)[layer_idx]

    # 1: EXTRACT. A true deep copy - nothing else holds a reference to
    # these tensors, so restoring them later is restoring the ORIGINAL
    # weights, not whatever the live state_dict happens to be by then.
    theta_orig = copy.deepcopy(block.state_dict())

    # 2-3: BASELINE, before anything is touched.
    b_before = {p: _final_distribution(model, tokenizer, p) for p in probe_set}
    p_before = sum(_perplexity(model, tokenizer, p) for p in probe_set) / len(probe_set)

    # 4-5: MODIFY + REINSERT (tentative) - weights are now live-edited,
    # unverified. Everything from here until commit/rollback is the
    # "unsafe window" the try/except below exists to close.
    theta_new = edit_fn(theta_orig)
    block.load_state_dict(theta_new)

    try:
        # 6-7: VERIFY, with the edit in place.
        b_after = {p: _final_distribution(model, tokenizer, p) for p in probe_set}
        p_after = sum(_perplexity(model, tokenizer, p) for p in probe_set) / len(probe_set)

        # 8: Delta
        if metric == "kl":
            delta = sum(_kl(b_before[p], b_after[p]) for p in probe_set) / len(probe_set)
        elif metric == "perplexity":
            delta = abs(p_after - p_before)
        else:
            raise ValueError(f"Unknown metric: {metric!r}. Supported: kl, perplexity")

        # 9-13: commit or rollback
        if delta <= tau:
            status = "COMMITTED"
        else:
            block.load_state_dict(theta_orig)  # ROLLBACK
            status = "ROLLED_BACK"
    except Exception:
        # Something went wrong mid-verification (e.g. a bad metric name,
        # a probe that crashes the forward pass). Weights are currently
        # theta_new, unverified - the safety property demands they never
        # stay that way. Restore, then let the caller see the real error.
        block.load_state_dict(theta_orig)
        raise

    audit_record = {
        "layer_idx": layer_idx,
        "tau": tau,
        "metric": metric,
        "delta": delta,
        "status": status,
        "perplexity_before": p_before,
        "perplexity_after": p_after,
        "probes": [
            {
                "prompt": p,
                "top1_before": _top1(b_before[p], tokenizer),
                "top1_after": _top1(b_after[p], tokenizer),
            }
            for p in probe_set
        ],
    }
    return audit_record


# ---------------------------------------------------------------------------
# path-to-full-novelty.md item 1: "an empirically-swept tolerance threshold,
# measured catch-rate" - replacing a guessed tau (the first real runs showed
# 0.5 was far too loose) with a real, evidence-based one. A fixed grid of
# edits, each hand-labeled by INTENDED severity (mild edits should ideally
# get committed; severe ones should ideally get caught), swept across
# several layers.
# ---------------------------------------------------------------------------
SWEEP_EDITS = [
    ("mild", "scale x0.95", lambda: make_weight_edit_fn("scale", factor=0.95)),
    ("mild", "scale x1.05", lambda: make_weight_edit_fn("scale", factor=1.05)),
    ("mild", "noise std=0.001", lambda: make_weight_edit_fn("noise", std=0.001)),
    ("moderate", "scale x0.8", lambda: make_weight_edit_fn("scale", factor=0.8)),
    ("moderate", "scale x1.2", lambda: make_weight_edit_fn("scale", factor=1.2)),
    ("moderate", "noise std=0.01", lambda: make_weight_edit_fn("noise", std=0.01)),
    ("severe", "zero", lambda: make_weight_edit_fn("zero")),
    ("severe", "scale x5.0", lambda: make_weight_edit_fn("scale", factor=5.0)),
    ("severe", "scale x-1.0", lambda: make_weight_edit_fn("scale", factor=-1.0)),
    ("severe", "noise std=0.5", lambda: make_weight_edit_fn("noise", std=0.5)),
]


def sweep_edit_severity(model, tokenizer, layers_list=None, probe_set=None, metric="kl"):
    """Measure Delta for a grid of (layer, edit) combinations, WITHOUT
    permanently changing the model. Every single measurement is taken via
    a real vemr() call with tau=-1, which forces a ROLLED_BACK outcome
    every time (Delta is always >= 0, so always > -1) - the model is
    guaranteed clean after each measurement, not just at the end of the
    whole sweep. This reuses the exact same commit/rollback machinery
    already proven correct on layer 5, rather than a separate "just
    measure it" code path that could drift out of sync with the real
    protocol.

    Returns a list of dicts: layer_idx, severity, edit_desc, delta.
    """
    layers_list = layers_list if layers_list is not None else [2, 5, 8, 10]
    records = []
    for layer_idx in layers_list:
        for severity, desc, make_fn in SWEEP_EDITS:
            edit_fn = make_fn()
            result = vemr(model, tokenizer, layer_idx, edit_fn, probe_set, tau=-1.0, metric=metric)
            records.append({
                "layer_idx": layer_idx,
                "severity": severity,
                "edit_desc": desc,
                "delta": result["delta"],
                # perplexity_after is exactly "what the model would look
                # like if this edit were committed blindly" - tau=-1 above
                # forces the real vemr() call to ALWAYS roll back (so the
                # live model stays clean across the whole sweep), but the
                # measurement itself already captured what committing
                # would have produced, before the rollback happened.
                "perplexity_before": result["perplexity_before"],
                "perplexity_after": result["perplexity_after"],
            })
    return records


def suggest_tau(records, tau_candidates=None):
    """For each candidate tau, compute what fraction of MILD edits would be
    (correctly) committed, and what fraction of SEVERE edits would be
    (correctly) caught - a small precision/recall-style table for picking
    a real default tau instead of a guess. A good tau maximizes both
    numbers at once; where they trade off against each other is the
    actual empirical answer to 'what should the default be.'"""
    tau_candidates = tau_candidates if tau_candidates is not None else \
        [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5]
    mild = [r["delta"] for r in records if r["severity"] == "mild"]
    severe = [r["delta"] for r in records if r["severity"] == "severe"]

    rows = []
    for tau in tau_candidates:
        mild_commit_rate = sum(1 for d in mild if d <= tau) / len(mild) if mild else 0.0
        severe_catch_rate = sum(1 for d in severe if d > tau) / len(severe) if severe else 0.0
        rows.append({"tau": tau, "mild_commit_rate": mild_commit_rate, "severe_catch_rate": severe_catch_rate})
    return rows


# ---------------------------------------------------------------------------
# path-to-full-novelty.md item 3: "Benchmark VEMR's verification against a
# no-verification baseline on deliberately bad edits - does it actually
# catch what it should?" Reuses the exact same sweep_edit_severity() records
# (real VEMR calls, real deltas, real perplexity) rather than a separate
# simulation - the "naive" policy and "VEMR" policy are two different ways
# of reading the SAME measured data, not two different experiments that
# could drift apart.
# ---------------------------------------------------------------------------
def benchmark_vs_naive(records, tau=0.05):
    """For each record, compute what a NAIVE policy (commit every edit
    blindly, no verification at all) would leave the model at, versus what
    VEMR would actually do at the given tau. Naive = perplexity_after
    always. VEMR = perplexity_after if delta <= tau (would commit) else
    perplexity_before (would roll back to the original).

    Returns: per-severity-tier summary (mean naive perplexity, mean VEMR
    perplexity, and for severe edits specifically, how many VEMR actually
    catches vs. lets slip through at this tau).
    """
    tiers = {}
    for r in records:
        tiers.setdefault(r["severity"], []).append(r)

    summary = []
    for severity in ("mild", "moderate", "severe"):
        rows = tiers.get(severity, [])
        if not rows:
            continue
        naive_pp = [r["perplexity_after"] for r in rows]
        vemr_pp = [
            r["perplexity_after"] if r["delta"] <= tau else r["perplexity_before"]
            for r in rows
        ]
        entry = {
            "severity": severity,
            "n": len(rows),
            "naive_mean_perplexity": sum(naive_pp) / len(naive_pp),
            "naive_max_perplexity": max(naive_pp),
            "vemr_mean_perplexity": sum(vemr_pp) / len(vemr_pp),
            "vemr_max_perplexity": max(vemr_pp),
        }
        if severity == "severe":
            entry["caught"] = sum(1 for r in rows if r["delta"] > tau)
            entry["missed"] = sum(1 for r in rows if r["delta"] <= tau)
        summary.append(entry)
    return summary
