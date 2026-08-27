"""
Model loading and layer inspection.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_MODEL = DEFAULT_MODEL_NAME

def load(model_name: str = DEFAULT_MODEL_NAME) :
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def layers(model):
    """Returns the hidden layers of the model."""
    return model.transformer.h

def layer_param_count(block) -> int:
    return sum(p.numel() for p in block.parameters())

def print_layer_summary(model):
    """Human readable summary of the model."""
    blocks = layers(model)
    print(f"Number of layers: {len(blocks)}")
    for idx, block in enumerate(blocks):
        n_params = layer_param_count(block)
        print(
            f"Layer {idx:2d}: {n_params:,} params "
            f"(attn.c_attn.weight={block.attn.c_attn.weight.shape}, "
            f"mlp.c_fc.weight={block.mlp.c_fc.weight.shape})"
        )

    total = sum(p.numel() for p in model.parameters())
    print(f"Total model params: {total:,}")

def layer_state_dict(model, layer_idx: int) :
    return layers(model)[layer_idx].state_dict()