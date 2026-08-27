"""
Performance / cost benchmark - measures latency, throughput, parameter
count, and quality (perplexity) for the SAME model in two states:
unmodified, and after a real edit - so the cost/performance impact of
model surgery can be reported in concrete, measured numbers, the same
standard the rest of this project holds itself to (see vemr.py's
docstring and findings.md).

Two scenarios are benchmarked, because they represent two different
product stories:

  1. WHOLE-LAYER ABLATION (compression story). Physically removing one of
     the model's 12 transformer blocks (layer_extractor.ablate_layer).
     This measurably shrinks the model - fewer parameters, less compute
     per forward pass - and is expected to measurably speed up inference.
     The tradeoff is a quality cost, which is exactly what VEMR's
     verification step (Section 9 / vemr.py) exists to catch if it's too
     large before this kind of edit ever ships.

  2. SURGICAL SINGLE-NEURON EDIT (behavior-repair story). Editing exactly
     one of a layer's 3072 MLP neurons via VEMR (make_neuron_edit_fn).
     This is NOT a compression play: removing 1 neuron out of the model's
     36,864 total MLP units is not expected to move latency or memory in
     any way a wall-clock benchmark can resolve. It is a "patch one
     learned behavior, verified, without retraining" story. This
     benchmark scenario exists partly to make that distinction concrete
     and measured rather than merely asserted.

Every number this script prints is measured live, on whatever machine it
is run on. It does not hard-code or assume any particular hardware, so
absolute latency numbers are only meaningful relative to the SAME run's
own baseline - not as an absolute claim, and not for comparison across
different machines or across separate runs.

The dollar-cost figures are a deliberately simple, fully transparent
model (see _cost_estimate's docstring) - a way to translate a measured
latency delta into a directionally correct dollar figure, NOT a
calibrated cloud-pricing estimate. Every assumption behind that number is
printed alongside it rather than hidden.
"""
import statistics
import time

import torch

from . import layer_extractor
from . import model_loader
from .vemr import DEFAULT_PROBE_SET, make_neuron_edit_fn, perplexity, vemr as run_vemr


# ---------------------------------------------------------------------------
# Low-level measurement helpers
# ---------------------------------------------------------------------------
def _time_forward_passes(model, tokenizer, prompts, n_repeats=20, warmup=3):
    """Average latency (ms) and throughput (tokens/sec) of a single forward
    pass - not a full generation loop, which would also measure sampling
    strategy and Python-loop overhead rather than the model's own cost.
    Averaged across `n_repeats` repetitions of the full prompt set, after
    `warmup` untimed repetitions (a freshly loaded/edited model's first
    call or two can be slower due to lazy initialization; excluding these
    keeps the measurement representative of steady-state cost)."""
    encoded = [tokenizer(p, return_tensors="pt") for p in prompts]

    for _ in range(warmup):
        for inputs in encoded:
            with torch.no_grad():
                model(**inputs)

    latencies_ms = []
    total_tokens = 0
    for _ in range(n_repeats):
        for inputs in encoded:
            n_tok = inputs["input_ids"].shape[1]
            start = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            elapsed = time.perf_counter() - start
            latencies_ms.append(elapsed * 1000)
            total_tokens += n_tok

    total_time_s = sum(latencies_ms) / 1000
    return {
        "mean_latency_ms": statistics.mean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "stdev_latency_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "throughput_tokens_per_sec": (total_tokens / total_time_s) if total_time_s > 0 else float("nan"),
        "n_forward_passes": len(latencies_ms),
    }


def _quality(model, tokenizer, probe_set):
    ppls = [perplexity(model, tokenizer, p) for p in probe_set]
    return {
        "mean_perplexity": statistics.mean(ppls),
        "per_prompt_perplexity": dict(zip(probe_set, ppls)),
    }


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def _cost_estimate(mean_latency_ms, dollars_per_hour):
    """A deliberately simple, transparent cost model: dollar cost per 1,000
    forward passes, assuming the given $/hour compute rate is used at
    100% utilization in a single-stream (unbatched) setting. This is a
    illustrative unit-cost translation of a MEASURED latency number, not a
    calibrated cloud-pricing model - no batching, queuing, network, or
    memory-bandwidth effects are modeled. `dollars_per_hour` should be set
    to whatever real hourly compute rate (CPU or GPU) the user wants to
    reason about; the default of $1.00/hr has no particular significance
    and should be overridden with a real, current on-demand or reserved
    rate for the target hardware before treating the dollar figure as
    meaningful.
    """
    calls_per_hour = 3600.0 / (mean_latency_ms / 1000.0)
    cost_per_call = dollars_per_hour / calls_per_hour
    return {
        "cost_per_1k_forward_passes_usd": cost_per_call * 1000,
        "calls_per_hour_single_stream": calls_per_hour,
        "assumption": (
            f"${dollars_per_hour:.4f}/hour compute, single-stream (no batching), "
            f"100% utilization assumed - illustrative only, not a calibrated cloud price"
        ),
    }


def _pct_change(before, after):
    if before == 0:
        return float("nan")
    return (after - before) / before * 100


# ---------------------------------------------------------------------------
# Scenario 1: whole-layer ablation (compression story)
# ---------------------------------------------------------------------------
def benchmark_layer_ablation(model_name, layer_idx, probe_set=None, n_repeats=20,
                              warmup=3, dollars_per_hour=1.0):
    probe_set = probe_set if probe_set is not None else DEFAULT_PROBE_SET

    model, tokenizer = model_loader.load(model_name)

    before_params = _param_count(model)
    before_perf = _time_forward_passes(model, tokenizer, probe_set, n_repeats=n_repeats, warmup=warmup)
    before_quality = _quality(model, tokenizer, probe_set)
    before_cost = _cost_estimate(before_perf["mean_latency_ms"], dollars_per_hour)

    layer_extractor.ablate_layer(model, layer_idx)

    after_params = _param_count(model)
    after_perf = _time_forward_passes(model, tokenizer, probe_set, n_repeats=n_repeats, warmup=warmup)
    after_quality = _quality(model, tokenizer, probe_set)
    after_cost = _cost_estimate(after_perf["mean_latency_ms"], dollars_per_hour)

    return {
        "scenario": "whole-layer-ablation",
        "model_name": model_name,
        "layer_idx": layer_idx,
        "before": {"params": before_params, "perf": before_perf, "quality": before_quality, "cost": before_cost},
        "after": {"params": after_params, "perf": after_perf, "quality": after_quality, "cost": after_cost},
        "deltas": {
            "params_pct": _pct_change(before_params, after_params),
            "latency_pct": _pct_change(before_perf["mean_latency_ms"], after_perf["mean_latency_ms"]),
            "throughput_pct": _pct_change(before_perf["throughput_tokens_per_sec"], after_perf["throughput_tokens_per_sec"]),
            "mean_perplexity_delta": after_quality["mean_perplexity"] - before_quality["mean_perplexity"],
            "cost_per_1k_pct": _pct_change(before_cost["cost_per_1k_forward_passes_usd"], after_cost["cost_per_1k_forward_passes_usd"]),
        },
    }


# ---------------------------------------------------------------------------
# Scenario 2: surgical single-neuron edit (behavior-repair story)
# ---------------------------------------------------------------------------
def benchmark_neuron_edit(model_name, layer_idx, neuron_idx, probe_set=None, n_repeats=20,
                           warmup=3, tau=0.05, metric="kl", dollars_per_hour=1.0):
    probe_set = probe_set if probe_set is not None else DEFAULT_PROBE_SET

    model, tokenizer = model_loader.load(model_name)

    before_params = _param_count(model)
    before_perf = _time_forward_passes(model, tokenizer, probe_set, n_repeats=n_repeats, warmup=warmup)
    before_quality = _quality(model, tokenizer, probe_set)
    before_cost = _cost_estimate(before_perf["mean_latency_ms"], dollars_per_hour)

    edit_fn = make_neuron_edit_fn(neuron_idx, op="zero")
    record = run_vemr(model, tokenizer, layer_idx, edit_fn, probe_set=probe_set, tau=tau, metric=metric)

    after_params = _param_count(model)
    after_perf = _time_forward_passes(model, tokenizer, probe_set, n_repeats=n_repeats, warmup=warmup)
    after_quality = _quality(model, tokenizer, probe_set)
    after_cost = _cost_estimate(after_perf["mean_latency_ms"], dollars_per_hour)

    return {
        "scenario": "single-neuron-edit",
        "model_name": model_name,
        "layer_idx": layer_idx,
        "neuron_idx": neuron_idx,
        "vemr_status": record["status"],
        "before": {"params": before_params, "perf": before_perf, "quality": before_quality, "cost": before_cost},
        "after": {"params": after_params, "perf": after_perf, "quality": after_quality, "cost": after_cost},
        "deltas": {
            "params_pct": _pct_change(before_params, after_params),
            "latency_pct": _pct_change(before_perf["mean_latency_ms"], after_perf["mean_latency_ms"]),
            "throughput_pct": _pct_change(before_perf["throughput_tokens_per_sec"], after_perf["throughput_tokens_per_sec"]),
            "mean_perplexity_delta": after_quality["mean_perplexity"] - before_quality["mean_perplexity"],
            "cost_per_1k_pct": _pct_change(before_cost["cost_per_1k_forward_passes_usd"], after_cost["cost_per_1k_forward_passes_usd"]),
        },
    }
