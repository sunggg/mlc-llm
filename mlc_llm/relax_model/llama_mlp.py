from dataclasses import dataclass
import tvm
from tvm import relax, tir
from tvm.relax.testing import nn
from .param_manager import ParamManager
from ..quantization import ParamQuantKind  # , QuantizationScheme
from typing import Any, List  # , Optional, Tuple
import numpy as np


def get_param_quant_kind(name: str, param_info: relax.TensorStructInfo) -> ParamQuantKind:
    if "embed_tokens" in name:
        return ParamQuantKind.embedding_table
    elif "lm_head.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


@dataclass
class LlamaConfig:
    def __init__(
        self,
        dtype="float32",
        max_sequence_length=2048,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        position_embedding_base=10000,
        combine_matmul=True,
        **kwargs,
    ):
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.position_embedding_base = position_embedding_base
        self.combine_matmul = combine_matmul
        self.kwargs = kwargs


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dtype: str, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), dtype=dtype, name="linear_weight")
        if bias:
            self.bias = nn.Parameter((out_features,), dtype=dtype, name="linear_bias")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(input, self.weight, self.bias))


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        dtype = config.dtype

        self.combine_matmul = config.combine_matmul
        if self.combine_matmul:
            self.gate_up_proj = Linear(hidden_size, 2 * intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
        else:
            self.gate_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, dtype=dtype, bias=False)
            self.up_proj = Linear(hidden_size, intermediate_size, dtype=dtype, bias=False)

    def forward(self, x):
        if self.combine_matmul:
            gate_up_results = nn.emit(
                relax.op.split(
                    self.gate_up_proj(x),
                    indices_or_sections=2,
                    axis=-1,
                )
            )
            gate_result = relax.TupleGetItem(gate_up_results, 0)
            up_result = relax.TupleGetItem(gate_up_results, 1)
        else:
            gate_result = self.gate_proj(x)
            up_result = self.up_proj(x)

        return self.down_proj(relax.op.nn.silu(gate_result) * up_result)


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = args.sep_embed

    config = LlamaConfig(
        **hf_config,
        dtype=dtype,
        combine_matmul=not args.quantization.name.startswith("autogptq"),
    )

    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    # from .modules import named_parameters
    for layer_id in range(config.num_hidden_layers):
        func_name = "LlamaMLP_" + str(layer_id)
        param_manager.params_in_func[func_name] = []
        with bb.function(func_name):
            model = LlamaMLP(config)
            model_params = model.parameters()
            # 'model.layers.1.mlp.gate_up_proj.weight', 'model.layers.1.mlp.down_proj.weight'

            # param_manager.register_params(model, func_name, args.quantization, get_param_quant_kind)
            assert config.combine_matmul
            assert len(model_params) == 2
            named_params = {
                f"model.layers.{layer_id}.mlp.gate_up_proj.weight": model_params[0],
                f"model.layers.{layer_id}.mlp.down_proj.weight": model_params[1],
            }
            for name, relax_param in named_params.items():
                quantization_scheme = args.quantization
                _base_model_prefix = quantization_scheme.get_base_model_prefix()
                if _base_model_prefix:
                    name = f"{_base_model_prefix}.{name}"

                quant_kind = get_param_quant_kind(name, relax_param.struct_info)
                param = param_manager._register_param(
                    name,
                    relax_param,
                    getattr(quantization_scheme, quant_kind.name),
                    func_name,
                )
                param_manager.params_in_func[func_name].append(param)

            seqlen = tir.Var("n", "int64")
            x = nn.Placeholder((seqlen, config.hidden_size), dtype="float16", name="x")
            with bb.dataflow():
                out = model(x)
                params = [x] + model_params
                gv = bb.emit_output(out)
            bb.emit_func_output(gv, params)
        mod = bb.get()
        gv = mod.get_global_var(func_name)
        bb.update_func(gv, mod[gv].with_attr("num_input", 3))

    def f_convert_pname_fwd(pname: str) -> List[str]:
        if not config.combine_matmul:
            return [pname]

        qkv_str = "query_key_value_proj"
        gate_up_str = "gate_up_proj"
        if qkv_str in pname:
            return [
                pname.replace(qkv_str, "q_proj"),
                pname.replace(qkv_str, "k_proj"),
                pname.replace(qkv_str, "v_proj"),
            ]
        elif gate_up_str in pname:
            return [
                pname.replace(gate_up_str, "gate_proj"),
                pname.replace(gate_up_str, "up_proj"),
            ]
        else:
            return [pname]

    def f_convert_param_bkwd(torch_pname: str, torch_param):
        if not config.combine_matmul:
            return [(torch_pname, torch_param.astype(dtype))]

        combined_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
        if any([name in torch_pname for name in combined_layers]):
            return None
        return [(torch_pname, torch_param.astype(dtype))]

    def f_compute_relax_param(relax_pname: str, torch_params: List[Any]):
        # Expected to enter this function only for the combined linear matmul weights.
        # Other weights are supposed to be loaded in `f_convert_param_bkwd` since
        # each other relax param has a unique corresponding torch param.
        if not config.combine_matmul:
            # When matmul combination is not turned on, each relax param has a unique
            # corresponding torch param, and this function is not expected to be entered.
            raise NotImplementedError(
                "Matmul combination is not turned on, and the function "
                "is not expected to be entered"
            )

        import numpy as np

        if "query_key_value_proj" in relax_pname:
            assert len(torch_params) == 3
        elif "gate_up_proj" in relax_pname:
            assert len(torch_params) == 2
        else:
            raise ValueError("Unexpected param loading")
        return np.concatenate(torch_params, axis=0).astype(dtype)

    param_manager.set_param_loading_func(
        args.model_path,
        args.use_safetensors,
        f_convert_pname_fwd,
        f_convert_param_bkwd,
        f_compute_relax_param,
    )
    device = tvm.cpu()
    param_list = [None] * len(param_manager.param_names)
    # param = self._register_param(
    #                name,
    #                relax_param,
    #                getattr(quantization_scheme, quant_kind.name),
    #                func_name,
    #            )
    """
    if args.quantization.pre_quantized:
        param_list = args.quantization.load_quantized_params(
            args.model_path,
            args.use_safetensors,
            param_list,
            param_manager.pidx2pname,
            device,
            excluded_params=["cos_cached", "sin_cached"],
        )

    head_dim = config.hidden_size / config.num_attention_heads
    inv_freq = 1.0 / (
        config.position_embedding_base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
    )
    # Hardcode the cached sin/cos for 2048.
    # This will be eliminated further with online rotary embedding calculation.
    t = np.arange(2048, dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    param_list[-2] = tvm.nd.array(np.cos(emb).astype(config.dtype), device)
    param_list[-1] = tvm.nd.array(np.sin(emb).astype(config.dtype), device)
    """

    return mod, param_manager, param_list
