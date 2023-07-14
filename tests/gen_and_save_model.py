import torch
import json
import os
from cohere.models import base_medium as cohere_medium, base_xlarge as cohere_xlarge


torch.manual_seed(0)

def _dump_model(model, model_dir):
    state_dict = model.state_dict()
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    torch.save(state_dict, model_path)
    print("model saved to ", model_path)


def _dump_config(config, model_dir):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=2)
    print("config saved to ", config_path)


def gen_model(model, config, name):
    model_dir = os.path.join("dist", "models", name)
    _dump_model(model, model_dir)
    _dump_config(config, model_dir)


config_medium = {'activation_function': 'gelu',
    'attn_pdrop': 0.1,
    'bos_token_id': 50256,
    'embd_pdrop': 0.1,
    'eos_token_id': 50256,
    'initializer_range': 0.02,
    'layer_norm_epsilon': 1e-05,
    'model_type': 'gpt2',
    'n_embd': 4096,
    'n_head': 32,
    'n_inner': 16384,
    'n_layer': 28,
    'n_positions': 2048,
    'reorder_and_upcast_attn': False,
    'resid_pdrop': 0.1,
    'scale_attn_by_inverse_layer_idx': False,
    'scale_attn_weights': True,
    'summary_activation': None,
    'summary_first_dropout': 0.1,
    'summary_proj_to_labels': True,
    'summary_type': 'cls_index',
    'summary_use_proj': True,
    'transformers_version': '4.30.2',
    'use_cache': True,
    'vocab_size': 51200
}

config_xlarge = {'activation_function': 'gelu',
    'attn_pdrop': 0.1,
    'bos_token_id': 50256,
    'embd_pdrop': 0.1,
    'eos_token_id': 50256,
    'initializer_range': 0.02,
    'layer_norm_epsilon': 1e-05,
    'model_type': 'gpt2',
    'n_embd': 8192,
    'n_head': 64,
    'n_inner': 32768,
    'n_layer': 64,
    'n_positions': 2048,
    'reorder_and_upcast_attn': False,
    'resid_pdrop': 0.1,
    'scale_attn_by_inverse_layer_idx': False,
    'scale_attn_weights': True,
    'summary_activation': None,
    'summary_first_dropout': 0.1,
    'summary_proj_to_labels': True,
    'summary_type': 'cls_index',
    'summary_use_proj': True,
    'transformers_version': '4.30.2',
    'use_cache': True,
    'vocab_size': 51200
}

model = cohere_medium(dev="cuda", dtype=torch.float16)
gen_model(model, config_medium, "cohere-gpt2-medium")
del model

# uncomment the following to create 52B model
# model = cohere_xlarge(dev="cuda", dtype=torch.float16)
# gen_model(model, config_xlarge, "cohere-gpt2-xlarge")
# del model

