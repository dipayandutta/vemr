"""
Part 5 - Attention Explorer.

Shows which tokens attend to which, per layer and per head. Uses the
model's own output_attentions=True instead of a hand-rolled hook - HF
officially supports and maintains this output, which avoids repeating the
tuple-vs-tensor version fragility we hit in Day 2 with hidden states.

GPT-2 small has 12 heads per layer, each independently deciding what to
"look back" at. Different heads often specialize - e.g. one head might
mostly do "attend to the previous token" (tracking local grammar), another
might do long-range "attend back to the subject of the sentence."
"""
import torch


def capture_attention(model, inputs):
    """Run one forward pass with attention weights recorded.
    Returns a tuple of length num_layers, each (batch, num_heads, seq, seq).
    attentions[layer][0, head, query_pos, key_pos] = how much query_pos
    attends to key_pos.
    """
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    if not attentions:
        raise RuntimeError(
            "Model returned no attention weights. "
            "Eager attention is required for attention visualization."
        )
    return attentions


def attention_for_layer_head(attentions, layer_idx, head_idx):
    """(seq, seq) attention matrix for one layer + one head."""
    return attentions[layer_idx][0, head_idx]


def attention_averaged(attentions, layer_idx):
    """(seq, seq) attention matrix averaged across all heads in a layer -
    a quick default view before drilling into individual heads."""
    return attentions[layer_idx][0].mean(dim=0)


def top_attended_tokens(attn_matrix, token_idx, tokens, top_k=5):
    """For a given query token position, which key tokens it attends to most."""
    row = attn_matrix[token_idx]
    top_vals, top_idxs = torch.topk(row, min(top_k, row.shape[0]))
    return [(tokens[i], float(v)) for i, v in zip(top_idxs.tolist(), top_vals.tolist())]


def top_attended_tokens_per_head(attentions, layer_idx, token_idx, tokens, top_k=3):
    """Same as top_attended_tokens, but broken out per head instead of
    averaged - averaging can dilute a clean signal from one specialist head
    by blending it with the other 11 (same lesson as Part 2/3's neurons)."""
    num_heads = attentions[layer_idx].shape[1]
    results = {}
    for head_idx in range(num_heads):
        attn_matrix = attention_for_layer_head(attentions, layer_idx, head_idx)
        results[head_idx] = top_attended_tokens(attn_matrix, token_idx, tokens, top_k)
    return results


def render_ascii_heatmap(attn_matrix, tokens, cell_width=6):
    """Terminal-friendly heatmap: rows = query token (the one doing the
    attending), columns = key token (the one being attended to), cell =
    attention weight shaded from light to dark."""
    blocks = " .:-=+*#%@"
    header = " " * 7 + "".join(f"{t[:cell_width]:>{cell_width}}" for t in tokens)
    lines = [header]
    for i, row_tok in enumerate(tokens):
        row_vals = attn_matrix[i].tolist()
        cells = []
        for v in row_vals:
            idx = min(int(v * (len(blocks) - 1)), len(blocks) - 1)
            cells.append(f"{blocks[idx]:>{cell_width}}")
        lines.append(f"{row_tok[:6]:>6} " + "".join(cells))
    return "\n".join(lines)
