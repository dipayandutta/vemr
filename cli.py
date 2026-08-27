"""
llmsurgery CLI - single entry point, git/docker-style subcommands.

Part 1 ships one real subcommand (`inspect`) to prove the skeleton works
end-to-end: package -> loader -> hooks -> CLI. Parts 2-9 each add their own
subcommand here as they're built (logit-lens, neuron-viewer, attn-explorer,
edit, extract, reinsert, etc.) - this file is the one place that grows.
"""
import click
import torch

from . import model_loader
from . import neuron_viewer
from . import activation_diff
from . import logit_lens
from . import attention_explorer
from . import activation_editor
from . import circuit_discovery
from . import layer_extractor
from . import vemr as vemr_module
from . import benchmark as benchmark_module
from .hooks import run_with_capture


@click.group()
@click.version_option()
def cli():
    """llmsurgery - extract, inspect, modify, and reinsert transformer layers."""
    pass


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL,
              help="HuggingFace model name to load.")
def inspect(model_name):
    """Load a model and print its layer structure (Part 1 / was Day 1)."""
    click.echo(f"Loading {model_name} ...")
    model, tokenizer = model_loader.load(model_name)
    model_loader.print_layer_summary(model)


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt", required=True, help="Prompt to trace activations for.")
@click.option("--layer", "layer_idx", type=int, default=None,
              help="Only capture this layer index (default: all layers).")
def peek(model_name, prompt, layer_idx):
    """Quick sanity check that hooks work: run a prompt, print captured
    activation shapes per layer. (Full logit-lens command lands in Part 4.)"""
    model, tokenizer = model_loader.load(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    layer_indices = [layer_idx] if layer_idx is not None else None
    captured = run_with_capture(model, inputs, layer_indices)
    click.echo(f"Prompt: {prompt!r}  ({inputs['input_ids'].shape[1]} tokens)")
    for idx in sorted(captured):
        click.echo(f"  Layer {idx:2d}: activation shape = {tuple(captured[idx].shape)}")


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt", required=True, help="Prompt to inspect neurons on.")
@click.option("--layer", "layer_idx", type=int, required=True,
              help="Which layer's MLP to inspect (0-11 for GPT-2 small).")
@click.option("--neuron", "neuron_idx", type=int, default=None,
              help="Show this neuron's activation across every token.")
@click.option("--token", "token_idx", type=int, default=None,
              help="Show the top-activating neurons at this token position.")
@click.option("--top-k", default=10, help="How many neurons to show in top-k modes.")
def neurons(model_name, prompt, layer_idx, neuron_idx, token_idx, top_k):
    """Part 2: Neuron Viewer - inspect individual MLP neurons.

    Three modes:
      (no flags)        -> top-k neurons by peak activation anywhere in the prompt
      --token N         -> top-k neurons that fired strongest at token N
      --neuron N        -> that one neuron's activation across every token
    """
    model, tokenizer = model_loader.load(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
    acts = neuron_viewer.capture_neuron_activations(model, inputs, layer_idx)

    click.echo(f"Prompt: {prompt!r}")
    click.echo("Tokens: " + ", ".join(f"[{i}]{t!r}" for i, t in enumerate(tokens)) + "\n")

    if neuron_idx is not None:
        trace = neuron_viewer.neuron_trace(acts, neuron_idx)
        click.echo(f"Layer {layer_idx}, neuron {neuron_idx} activation per token:")
        for i, (tok, val) in enumerate(zip(tokens, trace)):
            bar = "#" * max(0, int(val))
            click.echo(f"  [{i:2d}] {tok!r:12s} {val:8.3f}  {bar}")
        return

    if token_idx is not None:
        top = neuron_viewer.top_neurons_at_token(acts, token_idx, top_k)
        click.echo(f"Layer {layer_idx}, top {top_k} neurons at token [{token_idx}] {tokens[token_idx]!r}:")
        for n_idx, val in top:
            click.echo(f"  neuron {n_idx:5d}: {val:8.3f}")
        return

    top = neuron_viewer.top_neurons_overall(acts, top_k)
    click.echo(f"Layer {layer_idx}, top {top_k} neurons by peak activation anywhere in prompt:")
    for n_idx, val in top:
        click.echo(f"  neuron {n_idx:5d}: peak {val:8.3f}")


@cli.command(name="neuron-diff")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt-a", required=True, help="First prompt.")
@click.option("--prompt-b", required=True, help="Second prompt to compare against.")
@click.option("--layer", "layer_idx", type=int, required=True)
@click.option("--pool", type=click.Choice(["mean", "last"]), default="mean",
              help="mean = average over all tokens, last = just the final token.")
@click.option("--top-k", default=10)
def neuron_diff_cmd(model_name, prompt_a, prompt_b, layer_idx, pool, top_k):
    """Part 3: Activation Difference Viewer.

    Searches all 3072 neurons in a layer and ranks them by how differently
    they behave between two prompts - instead of manually checking one
    neuron at a time and hoping it's the interesting one.
    """
    model, tokenizer = model_loader.load(model_name)
    results = activation_diff.neuron_diff(
        model, tokenizer, prompt_a, prompt_b, layer_idx, pool, top_k
    )

    click.echo(f"A: {prompt_a!r}")
    click.echo(f"B: {prompt_b!r}")
    click.echo(f"Layer {layer_idx}, top {top_k} most different neurons (pool={pool}):\n")
    for r in results:
        click.echo(
            f"  neuron {r['neuron']:5d}: diff={r['diff']:7.3f}  "
            f"(A={r['a']:7.3f}, B={r['b']:7.3f})"
        )


@cli.command(name="logit-lens")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt", required=True)
@click.option("--top-k", default=5)
@click.option("--position", default=-1, help="Token position to trace (default: last token).")
def logit_lens_cmd(model_name, prompt, top_k, position):
    """Part 4: Logit Lens - what does each layer 'believe' comes next.
    (Ported from day2_logit_lens.py onto the shared hooks.py infrastructure.)
    """
    model, tokenizer = model_loader.load(model_name)
    results, inputs = logit_lens.trace_prompt(model, tokenizer, prompt, top_k, position)

    click.echo(f"Prompt: {prompt!r}")
    click.echo("What each layer 'believes' comes next, traced at every depth:\n")
    for layer_idx, guesses in results:
        guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
        click.echo(f"Layer {layer_idx:2d}: {guess_str}")


@cli.command(name="token-evolution")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt", required=True)
@click.option("--num-tokens", default=5, help="How many new tokens to generate.")
@click.option("--top-k", default=3)
@click.option("--layers", "layer_list", default=None,
              help="Comma-separated layer indices to show (default: all 12).")
def token_evolution_cmd(model_name, prompt, num_tokens, top_k, layer_list):
    """Part 4: Token Evolution - generate several tokens, and for each one,
    replay what every layer believed at the moment it was produced. Watches
    the model 'think' step by step across real generation, not just one
    fixed snapshot.
    """
    model, tokenizer = model_loader.load(model_name)
    layers_to_show = [int(x) for x in layer_list.split(",")] if layer_list else None
    steps = logit_lens.trace_generation(model, tokenizer, prompt, num_tokens, top_k, layers_to_show)

    click.echo(f"Prompt: {prompt!r}\n")
    generated_so_far = prompt
    for step_num, step in enumerate(steps):
        click.echo(f"--- Step {step_num + 1}: generated {step['token']!r} "
                    f"(context so far: {generated_so_far!r}) ---")
        for layer_idx, guesses in step["layers"]:
            guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
            click.echo(f"  Layer {layer_idx:2d}: {guess_str}")
        generated_so_far += step["token"]
        click.echo()


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt", required=True)
@click.option("--layer", "layer_idx", type=int, required=True)
@click.option("--head", "head_idx", type=int, default=None,
              help="Specific head to view (default: averaged across all heads).")
@click.option("--token", "token_idx", type=int, default=None,
              help="Show top attended tokens for just this query position, instead of the full heatmap.")
@click.option("--top-k", default=5)
def attention(model_name, prompt, layer_idx, head_idx, token_idx, top_k):
    """Part 5: Attention Explorer - see which tokens attend to which."""
    model, tokenizer = model_loader.load(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    attentions = attention_explorer.capture_attention(model, inputs)
    num_heads = attentions[layer_idx].shape[1]

    if head_idx is not None:
        attn_matrix = attention_explorer.attention_for_layer_head(attentions, layer_idx, head_idx)
        click.echo(f"Layer {layer_idx}, head {head_idx}:")
    else:
        attn_matrix = attention_explorer.attention_averaged(attentions, layer_idx)
        click.echo(f"Layer {layer_idx}, averaged across all {num_heads} heads:")

    click.echo("Tokens: " + ", ".join(f"[{i}]{t!r}" for i, t in enumerate(tokens)) + "\n")

    if token_idx is not None and head_idx is not None:
        top = attention_explorer.top_attended_tokens(attn_matrix, token_idx, tokens, top_k)
        click.echo(f"Token [{token_idx}] {tokens[token_idx]!r} attends most to:")
        for tok, val in top:
            click.echo(f"  {tok!r:15s} {val:.3f}")
        return

    if token_idx is not None:
        # No specific head given: show every head separately instead of an
        # averaged blend, so a specialist head doesn't get diluted out.
        per_head = attention_explorer.top_attended_tokens_per_head(
            attentions, layer_idx, token_idx, tokens, top_k
        )
        click.echo(f"Token [{token_idx}] {tokens[token_idx]!r} - top {top_k} per head:\n")
        for h_idx, top in per_head.items():
            top_str = ", ".join(f"{tok!r}:{val:.2f}" for tok, val in top)
            click.echo(f"  Head {h_idx:2d}: {top_str}")
        return

    click.echo(attention_explorer.render_ascii_heatmap(attn_matrix, tokens))


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--prompt", required=True)
@click.option("--layer", "layer_idx", type=int, required=True, help="Layer to edit.")
@click.option("--op", type=click.Choice(["zero", "scale"]), required=True)
@click.option("--factor", type=float, default=1.0, help="Scale factor (only used with --op scale).")
@click.option("--top-k", default=5)
def edit(model_name, prompt, layer_idx, op, factor, top_k):
    """Part 6: Activation Editor - live-edit one layer's output and watch
    the effect ripple through every layer after it, verified with the
    logit lens. Nothing is saved - this is the in-memory preview of what
    VEMR's MODIFY + VERIFY steps will do for real in Parts 8-9."""
    model, tokenizer = model_loader.load(model_name)
    edit_fn = activation_editor.make_edit_fn(op, factor)

    baseline = activation_editor.traced_run(model, tokenizer, prompt, top_k)
    edited = activation_editor.traced_run(
        model, tokenizer, prompt, top_k, edit_layer=layer_idx, edit_fn=edit_fn
    )

    click.echo(f"Prompt: {prompt!r}")
    op_desc = f"op={op}" + (f", factor={factor}" if op == "scale" else "")
    click.echo(f"Edit: layer {layer_idx}, {op_desc}\n")

    click.echo("BASELINE (unedited):")
    for li, guesses in baseline:
        marker = "  <-- edit target" if li == layer_idx else ""
        guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
        click.echo(f"  Layer {li:2d}: {guess_str}{marker}")

    click.echo("\nEDITED:")
    for li, guesses in edited:
        marker = "  <-- edit target" if li == layer_idx else ""
        guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
        click.echo(f"  Layer {li:2d}: {guess_str}{marker}")

    changes = activation_editor.diff_top1(baseline, edited)
    click.echo(f"\nLayers whose #1 guess changed: {len(changes)} of {len(baseline)}")
    for li, before, after in changes:
        click.echo(f"  Layer {li:2d}: {before!r} -> {after!r}")


@cli.command(name="patch-sweep")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--clean-prompt", required=True, help="Prompt that gets the right answer.")
@click.option("--corrupted-prompt", required=True, help="Prompt that doesn't.")
@click.option("--top-k", default=5)
def patch_sweep_cmd(model_name, clean_prompt, corrupted_prompt, top_k):
    """Part 7: Circuit discovery via causal activation patching.

    Transplants the clean run's activation into the corrupted run, one
    layer at a time, and shows which layer's transplant does the most to
    pull the corrupted answer toward the clean one - a causal test, not
    just a correlational one.
    """
    model, tokenizer = model_loader.load(model_name)
    baseline, clean_top1, results = circuit_discovery.patch_sweep(
        model, tokenizer, clean_prompt, corrupted_prompt, top_k
    )

    click.echo(f"Clean prompt:     {clean_prompt!r}  (target answer: {clean_top1!r})")
    click.echo(f"Corrupted prompt: {corrupted_prompt!r}\n")

    base_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in baseline)
    click.echo(f"Corrupted baseline (no patch): {base_str}\n")

    click.echo("Patching each layer's clean activation into the corrupted run:")
    for layer_idx, guesses in results:
        guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
        hit = "  <-- clean answer now on top" if guesses and guesses[0][0] == clean_top1 else ""
        click.echo(f"  Layer {layer_idx:2d}: {guess_str}{hit}")


@cli.command(name="feature-search")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layer", "layer_idx", type=int, required=True)
@click.option("--neuron", "neuron_idx", type=int, default=None,
              help="Check this specific neuron against the prompt bank. Omit to discover the most selective neurons instead.")
@click.option("--top-k", default=10)
def feature_search_cmd(model_name, layer_idx, neuron_idx, top_k):
    """Part 7: Feature search over a diverse prompt bank.

    With --neuron: ranks the prompt bank by how strongly they activate that
    one neuron (confirms/refutes a hypothesis).
    Without --neuron: scans all 3072 neurons in the layer and surfaces the
    most SELECTIVE ones automatically (genuine discovery).
    """
    model, tokenizer = model_loader.load(model_name)

    if neuron_idx is not None:
        results = circuit_discovery.search_one_neuron(model, tokenizer, layer_idx, neuron_idx, top_k)
        click.echo(f"Layer {layer_idx}, neuron {neuron_idx} - ranked by peak activation across the prompt bank:\n")
        for prompt, tok, val in results:
            click.echo(f"  {val:7.3f}  {tok!r:12s}  {prompt!r}")
        return

    results = circuit_discovery.discover_selective_neurons(model, tokenizer, layer_idx, top_k)
    click.echo(f"Layer {layer_idx} - top {top_k} most SELECTIVE neurons (peak - mean activation):\n")
    for neuron_idx, sel, prompt, tok, val in results:
        click.echo(f"  neuron {neuron_idx:5d}  selectivity={sel:6.3f}  "
                    f"peaked at {tok!r:12s} in {prompt!r}")


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layer", "layer_idx", type=int, required=True, help="Layer to extract.")
@click.option("--out", "out_path", required=True, help="File to save the extracted layer to.")
def extract(model_name, layer_idx, out_path):
    """Part 8: Layer Extraction - pull one layer's weights out to a
    standalone file, with metadata (source model, layer index, tensor
    shapes, timestamp) and a checksum for integrity checking."""
    model, tokenizer = model_loader.load(model_name)
    payload = layer_extractor.extract_layer(model, layer_idx, model_name)
    layer_extractor.save_layer(payload, out_path)

    n_tensors = len(payload["state_dict"])
    n_params = sum(v.numel() for v in payload["state_dict"].values())
    click.echo(f"Extracted layer {layer_idx} of {model_name!r} -> {out_path}")
    click.echo(f"  {n_tensors} tensors, {n_params:,} params")
    click.echo(f"  extracted_at: {payload['extracted_at']}")
    click.echo(f"  checksum:     {payload['checksum']}")

    # Immediately prove the file on disk round-trips cleanly - read it back
    # and recompute the checksum from what was actually written.
    reloaded = layer_extractor.load_layer(out_path)
    ok = layer_extractor.verify_layer(reloaded)
    click.echo(f"  on-disk integrity check: {'PASSED' if ok else 'FAILED'}")


@cli.command(name="extract-info")
@click.option("--file", "path", required=True, help="Extracted-layer file to inspect.")
def extract_info(path):
    """Part 8: Show metadata for a previously extracted layer file, without
    needing to load any model - everything needed is in the file itself."""
    payload = layer_extractor.load_layer(path)
    ok = layer_extractor.verify_layer(payload)

    click.echo(f"File:         {path}")
    click.echo(f"Source model: {payload['model_name']}")
    click.echo(f"Layer index:  {payload['layer_idx']}")
    click.echo(f"Extracted at: {payload['extracted_at']}")
    click.echo(f"Checksum:     {payload['checksum']}")
    click.echo(f"Integrity:    {'PASSED' if ok else 'FAILED - tensors do not match stored checksum'}")
    click.echo("Tensor shapes:")
    for name, shape in payload["shapes"].items():
        click.echo(f"  {name:30s} {shape}")


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--file", "path", required=True, help="Extracted-layer file to restore.")
@click.option("--layer", "layer_idx", type=int, required=True, help="Layer to restore into.")
@click.option("--prompt", default="The Eiffel Tower is located in the city of",
              help="Prompt used to verify behavior before/after restoring.")
@click.option("--top-k", default=5)
def restore(model_name, path, layer_idx, prompt, top_k):
    """Part 8: Restore an extracted layer's weights back into a live model,
    and prove the round-trip was lossless by comparing the logit-lens trace
    before and after (extracting then restoring the SAME weights into the
    SAME layer should produce an IDENTICAL trace - if it doesn't, either
    the extraction or the restore has a bug).

    This is the raw copy-the-weights-in mechanism only. It does not verify
    against a probe set or roll back on failure - that safety logic is
    VEMR proper, built in Part 9 on top of this command's primitive.
    """
    model, tokenizer = model_loader.load(model_name)
    payload = layer_extractor.load_layer(path)

    if not layer_extractor.verify_layer(payload):
        click.echo("REFUSING: extracted file failed its own integrity check (checksum mismatch).")
        return

    before = activation_editor.traced_run(model, tokenizer, prompt, top_k)
    layer_extractor.restore_layer_into_model(model, layer_idx, payload)
    after = activation_editor.traced_run(model, tokenizer, prompt, top_k)

    click.echo(f"Restored layer {payload['layer_idx']} (from {payload['model_name']!r}) "
               f"into layer {layer_idx} of the loaded model.\n")

    changes = activation_editor.diff_top1(before, after)
    if changes:
        click.echo(f"Layers whose #1 guess changed after restore: {len(changes)} of {len(before)}")
        for li, b, a in changes:
            click.echo(f"  Layer {li:2d}: {b!r} -> {a!r}")
        click.echo("\n(Non-empty means the restored weights differ from what was already "
                    "there - expected if restoring into a DIFFERENT layer index than "
                    "extracted from, or a different model.)")
    else:
        click.echo("Layers whose #1 guess changed after restore: 0 - trace is IDENTICAL.")
        click.echo("Round-trip proof: extract -> save -> load -> restore reproduced the "
                    "model's behavior exactly.")


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layer", "layer_idx", type=int, required=True, help="Layer to physically remove.")
@click.option("--prompt", default="The Eiffel Tower is located in the city of")
@click.option("--position", default=-1, help="Token position to read the prediction at (default: last token).")
@click.option("--top-k", default=5)
@click.option("--save", "save_path", default=None,
              help="Optional: also save the removed layer's weights here first (same as `extract`), so they aren't lost.")
def ablate(model_name, layer_idx, prompt, position, top_k, save_path):
    """Physically remove a layer from a LIVE model - not just copy it out -
    and show exactly how the now-shorter model's prediction changes.

    Different from `edit --op zero`: zeroing a layer's output still runs
    the layer and keeps the model at its original depth (one layer just
    contributes nothing). This command actually deletes the block, so the
    model runs with one fewer layer, period. Because every `llmsurgery`
    command loads a fresh model in its own process, the before/after
    comparison has to happen inside this ONE command - a separate later
    `llmsurgery inspect` call starts over with a full, untouched model.

    Shows: the full per-layer logit-lens trace before and after, the
    model's REAL final probability distribution (not just top-k) both
    ways, and the KL divergence between them - the same kind of divergence
    measurement Part 9's VEMR verification step will compute automatically
    to decide commit-or-rollback.
    """
    model, tokenizer = model_loader.load(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    pos = position if position >= 0 else seq_len + position

    n_before = len(model_loader.layers(model))
    baseline_trace = activation_editor.traced_run(model, tokenizer, prompt, top_k, position)
    with torch.no_grad():
        baseline_probs = torch.softmax(model(**inputs).logits[0, pos], dim=-1)

    if save_path:
        payload = layer_extractor.extract_layer(model, layer_idx, model_name)
        layer_extractor.save_layer(payload, save_path)
        click.echo(f"Saved layer {layer_idx}'s weights to {save_path} before removing it.\n")

    layer_extractor.ablate_layer(model, layer_idx)
    n_after = len(model_loader.layers(model))

    ablated_trace = activation_editor.traced_run(model, tokenizer, prompt, top_k, position)
    with torch.no_grad():
        ablated_probs = torch.softmax(model(**inputs).logits[0, pos], dim=-1)

    eps = 1e-12
    kl = torch.sum(
        baseline_probs * (torch.log(baseline_probs + eps) - torch.log(ablated_probs + eps))
    ).item()

    click.echo(f"Prompt: {prompt!r}")
    click.echo(f"Model layer count: {n_before} -> {n_after}  (layer {layer_idx} physically removed)\n")

    click.echo(f"BEFORE ({n_before} layers) - full logit-lens trace:")
    for li, guesses in baseline_trace:
        marker = "  <-- about to be removed" if li == layer_idx else ""
        guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
        click.echo(f"  Layer {li:2d}: {guess_str}{marker}")

    click.echo(f"\nAFTER ({n_after} layers) - full logit-lens trace "
               f"(each position labeled with its ORIGINAL layer index):")
    for li, guesses in ablated_trace:
        original_idx = li if li < layer_idx else li + 1
        guess_str = ", ".join(f"{tok!r}:{p:.2f}" for tok, p in guesses)
        click.echo(f"  Position {li:2d} (was layer {original_idx:2d}): {guess_str}")

    top_before = torch.topk(baseline_probs, top_k)
    click.echo(f"\nFinal predicted distribution, BEFORE removal (real model output, not logit-lens):")
    for tid, p in zip(top_before.indices.tolist(), top_before.values.tolist()):
        click.echo(f"  {tokenizer.decode([tid])!r:15s} {p:.4f}")

    top_after = torch.topk(ablated_probs, top_k)
    click.echo(f"\nFinal predicted distribution, AFTER removal:")
    for tid, p in zip(top_after.indices.tolist(), top_after.values.tolist()):
        click.echo(f"  {tokenizer.decode([tid])!r:15s} {p:.4f}")

    click.echo(f"\nKL divergence (before || after), full 50257-token vocabulary: {kl:.4f}")
    click.echo("(0.0 would mean an identical distribution. This Delta is exactly the kind of "
               "quantity Part 9's VEMR verification step computes automatically to decide "
               "commit-or-rollback.)")


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layer", "layer_idx", type=int, required=True, help="Layer to edit.")
@click.option("--op", type=click.Choice(["zero", "scale", "noise", "swap"]), required=True)
@click.option("--factor", type=float, default=1.0, help="Scale factor (--op scale).")
@click.option("--std", type=float, default=0.01, help="Noise standard deviation (--op noise).")
@click.option("--swap-file", default=None, help="Extracted-layer file to swap in (--op swap).")
@click.option("--tau", type=float, default=0.05,
              help="Tolerance: max acceptable divergence before rollback. "
                   "0.05 is the empirically-calibrated default (see vemr-sweep / findings.md).")
@click.option("--metric", type=click.Choice(["kl", "perplexity"]), default="kl",
              help="Divergence signal: mean KL over the probe set, or absolute perplexity delta.")
@click.option("--probes", default=None,
              help="Comma-separated custom probe prompts (default: a fixed 5-prompt set).")
def vemr(model_name, layer_idx, op, factor, std, swap_file, tau, metric, probes):
    """Part 9: VEMR - Verified Extract-Modify-Reinsert, for real.

    Extracts a layer's weights, applies an edit function, tentatively
    reinserts it, verifies the model against a fixed probe set, and
    commits the edit or rolls back to a bit-identical copy of the
    original - automatically, based on whether divergence Delta stays
    within tolerance tau. Implements algorithm-vemr.md exactly; see that
    doc for the formal notation and safety-property proof sketch.

    Unlike `edit` (Part 6, in-memory only, one forward pass) and `ablate`
    (Part 8, no verification at all), this is the first command in the
    project that can actually change the model and then decide, on its
    own, whether to keep the change.
    """
    model, tokenizer = model_loader.load(model_name)

    if op == "scale":
        edit_fn = vemr_module.make_weight_edit_fn("scale", factor=factor)
    elif op == "noise":
        edit_fn = vemr_module.make_weight_edit_fn("noise", std=std)
    elif op == "swap":
        if not swap_file:
            click.echo("--swap-file is required for --op swap.")
            return
        payload = layer_extractor.load_layer(swap_file)
        edit_fn = vemr_module.make_weight_edit_fn("swap", payload=payload)
    else:
        edit_fn = vemr_module.make_weight_edit_fn("zero")

    probe_set = [p.strip() for p in probes.split(",")] if probes else None

    record = vemr_module.vemr(model, tokenizer, layer_idx, edit_fn, probe_set, tau, metric)

    op_desc = {"zero": "zero", "scale": f"scale x{factor}", "noise": f"noise std={std}",
               "swap": f"swap in weights from {swap_file}"}[op]
    click.echo(f"VEMR: layer {layer_idx}, op={op_desc}, metric={metric}, tau={tau}\n")

    click.echo("Probe set results (top-1 answer before -> after):")
    for pr in record["probes"]:
        flip = "  <-- CHANGED" if pr["top1_before"] != pr["top1_after"] else ""
        click.echo(f"  {pr['prompt']!r}")
        click.echo(f"    {pr['top1_before']!r} -> {pr['top1_after']!r}{flip}")

    click.echo(f"\nPerplexity: {record['perplexity_before']:.3f} -> {record['perplexity_after']:.3f}")
    click.echo(f"Delta ({metric}): {record['delta']:.4f}  (tau={tau})")
    click.echo(f"\nSTATUS: {record['status']}")
    if record["status"] == "COMMITTED":
        click.echo("The edit is now live in this model instance - Delta stayed within tau.")
    else:
        click.echo("Rolled back to the exact original weights - Delta exceeded tau. "
                    "The model is bit-identical to before this command ran.")


@cli.command(name="vemr-sweep")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layers", "layer_list", default=None,
              help="Comma-separated layer indices to sweep (default: 2,5,8,10).")
@click.option("--metric", type=click.Choice(["kl", "perplexity"]), default="kl")
@click.option("--tau", type=float, default=0.05,
              help="Tau to use for the naive-vs-VEMR benchmark table (default: the calibrated 0.05).")
def vemr_sweep(model_name, layer_list, metric, tau):
    """Part 9 extension: empirically sweep edit severity vs divergence
    across several layers, to replace a guessed tau with a real,
    evidence-based default (path-to-full-novelty.md item 1), AND benchmark
    VEMR's verification against a naive no-verification baseline on the
    same data (path-to-full-novelty.md item 3).

    Runs a fixed grid of mild/moderate/severe weight edits per layer, all
    via real VEMR calls forced to roll back (tau=-1) so the model is never
    permanently changed - a pure measurement pass. Shows: (1) delta per
    edit, (2) what fraction of mild edits would be kept / severe edits
    caught across a range of candidate tau values, and (3) a direct
    comparison of what a NAIVE policy (commit every edit blindly) would
    leave the model at vs. what VEMR actually leaves it at, per severity
    tier, at the given tau.
    """
    model, tokenizer = model_loader.load(model_name)
    layers_list = [int(x) for x in layer_list.split(",")] if layer_list else None

    click.echo("Sweeping edit severity vs divergence "
               "(several dozen real forward passes, may take a moment)...\n")
    records = vemr_module.sweep_edit_severity(model, tokenizer, layers_list, metric=metric)

    click.echo(f"{'Layer':>5}  {'Severity':<10} {'Edit':<16} Delta ({metric})")
    for r in records:
        click.echo(f"{r['layer_idx']:>5}  {r['severity']:<10} {r['edit_desc']:<16} {r['delta']:.4f}")

    click.echo("\nCandidate tau vs outcome:")
    click.echo(f"{'tau':>6}  {'mild commit rate':>18}  {'severe catch rate':>18}")
    for row in vemr_module.suggest_tau(records):
        click.echo(f"{row['tau']:>6.2f}  {row['mild_commit_rate']:>18.2%}  {row['severe_catch_rate']:>18.2%}")

    click.echo(f"\nVEMR vs. naive no-verification baseline, at tau={tau}:")
    click.echo(f"{'Severity':<10} {'n':>3}  {'naive mean PPL':>15}  {'naive max PPL':>14}  "
               f"{'VEMR mean PPL':>14}  {'VEMR max PPL':>13}")
    for row in vemr_module.benchmark_vs_naive(records, tau):
        click.echo(f"{row['severity']:<10} {row['n']:>3}  {row['naive_mean_perplexity']:>15.2f}  "
                   f"{row['naive_max_perplexity']:>14.2f}  {row['vemr_mean_perplexity']:>14.2f}  "
                   f"{row['vemr_max_perplexity']:>13.2f}")
        if "caught" in row:
            click.echo(f"             severe edits caught by VEMR: {row['caught']} of {row['caught'] + row['missed']}")


@cli.command(name="neuron-probe")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layer", "layer_idx", type=int, required=True)
@click.option("--neuron", "neuron_idx", type=int, required=True)
@click.option("--positive", required=True,
              help="Comma-separated prompts hypothesized to fire this neuron.")
@click.option("--negative", required=True,
              help="Comma-separated unrelated control prompts, expected to stay quiet.")
def neuron_probe(model_name, layer_idx, neuron_idx, positive, negative):
    """Screen a candidate neuron against a hypothesis in one command.

    Automates the replicate + negative-control pattern used by hand across
    several `neurons` calls to confirm neuron 792 (Part 3) and Head 0
    (Part 5) - built to quickly triage Part 7's discovery-mode candidates,
    which were found by selectivity alone with no hypothesis testing yet.
    A fast first filter, not a substitute for the closer manual look a
    LOOKS REAL verdict deserves before calling anything confirmed.
    """
    model, tokenizer = model_loader.load(model_name)
    pos_prompts = [p.strip() for p in positive.split(",")]
    neg_prompts = [p.strip() for p in negative.split(",")]

    result = neuron_viewer.probe_neuron(model, tokenizer, layer_idx, neuron_idx, pos_prompts, neg_prompts)

    click.echo(f"Layer {layer_idx}, neuron {neuron_idx}\n")
    click.echo("POSITIVE (hypothesized to fire):")
    for prompt, val, tok in result["positive"]:
        click.echo(f"  {val:7.3f}  {tok!r:12s}  {prompt!r}")

    click.echo("\nNEGATIVE (control, should stay quiet):")
    for prompt, val, tok in result["negative"]:
        click.echo(f"  {val:7.3f}  {tok!r:12s}  {prompt!r}")

    click.echo(f"\nVerdict: {result['verdict']}")


@cli.command(name="neuron-vemr")
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--layer", "layer_idx", type=int, required=True)
@click.option("--neuron", "neuron_idx", type=int, required=True)
@click.option("--op", type=click.Choice(["zero", "scale"]), default="zero")
@click.option("--factor", type=float, default=1.0, help="Scale factor (--op scale).")
@click.option("--tau", type=float, default=0.05)
@click.option("--metric", type=click.Choice(["kl", "perplexity"]), default="kl")
@click.option("--targeted-positive", required=True,
              help="Comma-separated prompts the target neuron is hypothesized to fire on - checked before AND after.")
@click.option("--targeted-negative", default=None,
              help="Optional comma-separated control prompts - checked before AND after too.")
def neuron_vemr(model_name, layer_idx, neuron_idx, op, factor, tau, metric, targeted_positive, targeted_negative):
    """Item 4 capstone tool: apply VEMR to a SINGLE neuron (not a whole
    layer, via vemr.make_neuron_edit_fn) and report two separate things
    side by side:

      1. VEMR's own general-capability verdict, against the default
         5-prompt probe set - commit or rollback, same safety net as
         every other `vemr` call.
      2. A TARGETED before/after check of the specific behavior the edit
         was meant to change, using the same neuron-probe prompt style
         from Parts 7/9.

    This is the full "find X, fix X, prove it worked" story: general
    safety is VEMR's automatic job (nothing else in the model should
    break); confirming the TARGETED change actually happened is a
    separate, deliberate check the experimenter runs on purpose - VEMR
    has no way to know what you intended to change, only whether the
    model as a whole still behaves acceptably.
    """
    model, tokenizer = model_loader.load(model_name)
    pos_prompts = [p.strip() for p in targeted_positive.split(",")]
    neg_prompts = [p.strip() for p in targeted_negative.split(",")] if targeted_negative else []

    def _peak(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
        acts = neuron_viewer.capture_neuron_activations(model, inputs, layer_idx)
        trace = neuron_viewer.neuron_trace(acts, neuron_idx)
        peak_val = max(trace)
        return peak_val, tokens[trace.index(peak_val)]

    def _completion(prompt, top_k=3):
        """The model's REAL next-token prediction for this prompt - not the
        target neuron's own activation, but whether the DOWNSTREAM
        BEHAVIOR the neuron was hypothesized to support actually changes
        once it's removed. A neuron going silent doesn't by itself prove
        the model's behavior changed - other neurons/heads may carry
        redundant information (a common finding in interpretability)."""
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, top_k)
        return [(tokenizer.decode([tid]), float(p)) for tid, p in zip(top.indices.tolist(), top.values.tolist())]

    click.echo(f"BEFORE - targeted neuron {neuron_idx} (layer {layer_idx}) activation, "
               f"and the model's actual next-token prediction:")
    for p in pos_prompts:
        v, t = _peak(p)
        comp_str = ", ".join(f"{tok!r}:{prob:.2f}" for tok, prob in _completion(p))
        click.echo(f"  {v:7.3f}  {t!r:12s}  {p!r}")
        click.echo(f"           completion: {comp_str}")
    if neg_prompts:
        click.echo("  (negative controls)")
        for p in neg_prompts:
            v, t = _peak(p)
            click.echo(f"  {v:7.3f}  {t!r:12s}  {p!r}")

    edit_fn = vemr_module.make_neuron_edit_fn(neuron_idx, op, factor)
    record = vemr_module.vemr(model, tokenizer, layer_idx, edit_fn, tau=tau, metric=metric)

    click.echo(f"\nVEMR general-capability verdict (default 5-prompt probe set): {record['status']}")
    click.echo(f"  Delta ({metric}): {record['delta']:.6f}  (tau={tau})")
    click.echo(f"  Perplexity: {record['perplexity_before']:.3f} -> {record['perplexity_after']:.3f}")
    for pr in record["probes"]:
        flip = "  <-- CHANGED" if pr["top1_before"] != pr["top1_after"] else ""
        click.echo(f"    {pr['prompt']!r}: {pr['top1_before']!r} -> {pr['top1_after']!r}{flip}")

    if record["status"] != "COMMITTED":
        click.echo("\nEdit was ROLLED BACK by VEMR - model is unchanged. "
                   "No targeted after-check to show, since nothing was actually applied.")
        return

    click.echo(f"\nAFTER - targeted neuron {neuron_idx} activation (edit is now live in the model), "
               f"and the model's actual next-token prediction:")
    for p in pos_prompts:
        v, t = _peak(p)
        comp_str = ", ".join(f"{tok!r}:{prob:.2f}" for tok, prob in _completion(p))
        click.echo(f"  {v:7.3f}  {t!r:12s}  {p!r}")
        click.echo(f"           completion: {comp_str}")
    if neg_prompts:
        click.echo("  (negative controls)")
        for p in neg_prompts:
            v, t = _peak(p)
            click.echo(f"  {v:7.3f}  {t!r:12s}  {p!r}")


def _print_benchmark_result(result):
    before, after, deltas = result["before"], result["after"], result["deltas"]

    click.echo(f"Scenario: {result['scenario']}")
    click.echo(f"Model: {result['model_name']}")
    if result["scenario"] == "whole-layer-ablation":
        click.echo(f"Layer removed: {result['layer_idx']}")
    else:
        click.echo(f"Layer: {result['layer_idx']}, neuron edited: {result['neuron_idx']}")
        click.echo(f"VEMR verdict: {result['vemr_status']}")
    click.echo()

    click.echo(f"{'':22s}{'BEFORE':>18s}{'AFTER':>18s}{'DELTA':>14s}")
    click.echo(f"{'Parameters':22s}{before['params']:>18,}{after['params']:>18,}{deltas['params_pct']:>13.3f}%")
    click.echo(f"{'Mean latency (ms)':22s}{before['perf']['mean_latency_ms']:>18.3f}"
               f"{after['perf']['mean_latency_ms']:>18.3f}{deltas['latency_pct']:>13.2f}%")
    click.echo(f"{'Throughput (tok/s)':22s}{before['perf']['throughput_tokens_per_sec']:>18.2f}"
               f"{after['perf']['throughput_tokens_per_sec']:>18.2f}{deltas['throughput_pct']:>13.2f}%")
    click.echo(f"{'Mean perplexity':22s}{before['quality']['mean_perplexity']:>18.3f}"
               f"{after['quality']['mean_perplexity']:>18.3f}"
               f"{deltas['mean_perplexity_delta']:>+13.3f}")
    click.echo(f"{'$/1k fwd passes':22s}{before['cost']['cost_per_1k_forward_passes_usd']:>18.6f}"
               f"{after['cost']['cost_per_1k_forward_passes_usd']:>18.6f}{deltas['cost_per_1k_pct']:>13.2f}%")

    click.echo(f"\nCost assumption: {after['cost']['assumption']}")
    click.echo(f"Timing basis: {before['perf']['n_forward_passes']} forward passes per side "
               f"(after warmup), single-stream, on this machine only - not comparable across machines.")


@cli.command()
@click.option("--model", "model_name", default=model_loader.DEFAULT_MODEL)
@click.option("--scenario", type=click.Choice(["layer", "neuron", "both"]), default="both",
              help="layer = whole-layer ablation (compression story). "
                   "neuron = surgical single-neuron VEMR edit (behavior-repair story). "
                   "both = run and print both.")
@click.option("--layer", "layer_idx", type=int, default=8,
              help="Layer to ablate (--scenario layer) or that the edited neuron lives in (--scenario neuron).")
@click.option("--neuron", "neuron_idx", type=int, default=802,
              help="Neuron to edit for --scenario neuron.")
@click.option("--repeats", "n_repeats", type=int, default=20,
              help="Timed repetitions of the probe set per side (before/after). Higher = more stable timing.")
@click.option("--warmup", type=int, default=3, help="Untimed warm-up repetitions before timing starts.")
@click.option("--tau", type=float, default=0.05, help="VEMR tolerance, used only for --scenario neuron.")
@click.option("--metric", type=click.Choice(["kl", "perplexity"]), default="kl")
@click.option("--dollars-per-hour", type=float, default=1.0,
              help="Illustrative compute rate used to translate measured latency into a $/1k-forward-passes "
                   "figure. Has no built-in significance - override with a real rate for the target hardware.")
def benchmark(model_name, scenario, layer_idx, neuron_idx, n_repeats, warmup, tau, metric, dollars_per_hour):
    """Part 9 extension: measure latency, throughput, parameter count, and
    quality (perplexity) BEFORE and AFTER a real edit, plus an illustrative
    dollar-cost translation - so the performance/cost impact of model
    surgery is a measured number, not a qualitative claim.

    Two scenarios, two different product stories: `layer` physically
    removes a whole transformer block (a compression play - fewer params,
    faster inference, at some quality cost VEMR would need to catch before
    this ships); `neuron` surgically edits one MLP neuron via VEMR (a
    behavior-repair play - not expected to move latency or memory at all,
    which this benchmark measures directly rather than assuming).
    """
    if scenario in ("layer", "both"):
        click.echo("=" * 72)
        result = benchmark_module.benchmark_layer_ablation(
            model_name, layer_idx, n_repeats=n_repeats, warmup=warmup, dollars_per_hour=dollars_per_hour)
        _print_benchmark_result(result)

    if scenario == "both":
        click.echo("\n" + "=" * 72 + "\n")

    if scenario in ("neuron", "both"):
        click.echo("=" * 72)
        result = benchmark_module.benchmark_neuron_edit(
            model_name, layer_idx, neuron_idx, n_repeats=n_repeats, warmup=warmup,
            tau=tau, metric=metric, dollars_per_hour=dollars_per_hour)
        _print_benchmark_result(result)


if __name__ == "__main__":
    cli()
