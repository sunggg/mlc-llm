# pylint: disable=missing-docstring,too-few-public-methods,too-many-instance-attributes,invalid-name,too-many-locals,too-many-arguments
import argparse
import math
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from ..utils import load_torch_pname2binname_map
from .commons import create_metadata_func

import tvm
from tvm import relax, testing, te
from tvm.relax.testing import nn
from tvm.script import relax as R
from tvm.relax.op import (
    astype,
    broadcast_to,
    expand_dims,
    matmul,
    maximum,
    minimum,
    permute_dims,
    reshape,
    squeeze,
    split,
)
from tvm.relax.op.nn import gelu, softmax
from .modules import ModuleList, Embedding, LayerNorm, Linear
from ..quantization import ParamQuantKind, QuantizationScheme
from .param_manager import ParamManager


@dataclass
class GPT2Config:
    def __init__(
        self,
        dtype="float32",
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.max_sequence_length = (
            self.max_position_embeddings
        ) = self.n_positions = n_positions
        self.hidden_size = self.n_embd = n_embd
        self.num_hidden_layers = self.n_layer = n_layer
        self.num_attention_heads = self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.kwargs = kwargs


def _prepare_decoder_attention_mask(input_shape, src_len, dtype):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    if isinstance(input_shape[-1], tvm.tir.Var) or input_shape[-1] > 1:
        bsz, tgt_len = input_shape

        def min_max_triu_te():
            return te.compute(
                (tgt_len, tgt_len),
                lambda i, j: tvm.tir.Select(
                    j > i, tvm.tir.min_value(dtype), tvm.tir.max_value(dtype)
                ),
                name="make_diag_mask_te",
            )

        mask = nn.emit_te(min_max_triu_te)
        diag_mask = nn.emit(broadcast_to(mask, (bsz, 1, tgt_len, tgt_len)))
        if src_len == tgt_len:
            return diag_mask

        def extend_te(x, tgt_len, src_len):
            return te.compute(
                (bsz, 1, tgt_len, src_len),
                lambda b, _, i, j: te.if_then_else(
                    j < src_len - tgt_len,
                    tvm.tir.max_value(dtype),
                    x[b, _, i, j - (src_len - tgt_len)],
                ),
                name="concat_te",
            )

        return nn.emit_te(extend_te, diag_mask, tgt_len, src_len)
    else:
        # Get src_len from input parameters
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, tgt_len = input_shape
        mask = relax.op.full(
            (bsz, 1, tgt_len, src_len),
            relax.const(tvm.tir.max_value(dtype).value, dtype),
            dtype,
        )
    return nn.emit(mask)


def apply_position_embedding(t_embd, weight, offset: int = 0):
    def f_position_embedding(tensor, weight, offset):
        def position_compute(*idx):
            b, s, e = idx
            return weight[s + offset, e] + tensor[b, s, e]

        return tvm.te.compute(tensor.shape, position_compute, name="position")

    hidden_states = nn.emit_te(
        f_position_embedding,
        t_embd,
        weight,
        offset,
        primfunc_name_hint="position_embedding",
    )
    return hidden_states


class Conv1D(nn.Module):
    def __init__(self, nf, nx, dtype):
        self.nf = nf
        self.weight = nn.Parameter(
            (nx, nf),
            dtype=dtype,
            name="conv1d_weight",
        )
        self.bias = nn.Parameter(
            (nf,),
            dtype=dtype,
            name="conv1d_bias",
        )
        self.dtype = dtype

    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(
            astype(
                matmul(input, self.weight, out_dtype="float32")
                + astype(self.bias, "float32"),
                self.dtype,
            )
        )


class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        pass

    def forward(self, input):
        return nn.emit(gelu(input))


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config: GPT2Config, dtype: str):
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim, dtype)
        self.c_proj = Conv1D(embed_dim, intermediate_size, dtype)
        self.act = GELUActivation()
        self.dtype = dtype

    def forward(self, hidden_states):
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        hidden_states = self.c_proj(hidden_states)
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        return hidden_states


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, is_cross_attention=False, layer_idx=None):
        max_positions = config.max_position_embeddings
        self.bias = nn.Parameter((1, 1, max_positions, max_positions), dtype="bool")
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim, config.dtype)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim, config.dtype)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim, config.dtype)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim, config.dtype)
        self.dtype = config.dtype

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr, relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Union[Tuple[None, None], Tuple[relax.Expr, relax.Expr]]]:
        # hidden_states: [batch_size, seq_len, hidden_size]
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))
        batch_size, seq_len, _ = hidden_states.struct_info.shape
        kv_seq_len = all_seq_len_shape.struct_info.values[0]

        if self.is_cross_attention:
            assert False, "not supprot cross attention yet"
            query = self.q_attn(hidden_states)
            c_attn_output = self.c_attn(encoder_hidden_states)
            c_attn_output = nn.emit(split(c_attn_output, 2, axis=2))
            key, value = c_attn_output[0], c_attn_output[1]
        else:
            c_attn_output = self.c_attn(hidden_states)
            c_attn_output = nn.emit(split(c_attn_output, 3, axis=2))
            query, key, value = c_attn_output[0], c_attn_output[1], c_attn_output[2]

        def _split_heads(tensor):
            return nn.emit(
                reshape(
                    tensor,
                    (batch_size, seq_len, self.num_heads, self.head_dim),
                )
            )

        # q/k/v states: [batch_size, seq_len, num_attention_heads, head_size]
        q, k, v = (
            _split_heads(query),
            _split_heads(key),
            _split_heads(value),
        )

        if past_key_value is not None:
            f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
            f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
            k_cache, v_cache = past_key_value
            k_cache = nn.emit(
                relax.Call(
                    f_kv_cache_append,
                    args=[
                        k_cache,
                        reshape(
                            k, (batch_size * seq_len, self.num_heads, self.head_dim)
                        ),
                    ],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            v_cache = nn.emit(
                relax.Call(
                    f_kv_cache_append,
                    args=[
                        v_cache,
                        reshape(
                            v, (batch_size * seq_len, self.num_heads, self.head_dim)
                        ),
                    ],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            batch_size, _, num_heads, head_size = k.struct_info.shape
            kv_cache_shape = R.shape([batch_size * kv_seq_len, num_heads, head_size])
            kv_states_shape = R.shape([batch_size, kv_seq_len, num_heads, head_size])
            k = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[k_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, k.struct_info.dtype)],
                )
            )
            v = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[v_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, v.struct_info.dtype)],
                )
            )
            k = nn.emit(reshape(k, kv_states_shape))
            v = nn.emit(reshape(v, kv_states_shape))
            past_key_value = (k_cache, v_cache)
        else:
            past_key_value = (None, None)

        q = nn.emit(permute_dims(q, [0, 2, 1, 3]))
        k = nn.emit(permute_dims(k, [0, 2, 1, 3]))
        v = nn.emit(permute_dims(v, [0, 2, 1, 3]))

        # Calculate QK
        attn_weights = nn.emit(
            matmul(q, permute_dims(k, [0, 1, 3, 2]))
            / relax.const(
                math.sqrt(self.head_dim),
                q.struct_info.dtype,
            )
        )
        # Apply attention mask
        attn_weights = nn.emit(
            maximum(
                attn_weights,
                relax.const(
                    tvm.tir.min_value(attn_weights.struct_info.dtype).value,
                    attn_weights.struct_info.dtype,
                ),
            )
        )
        attn_weights = nn.emit(minimum(attn_weights, attention_mask))
        # Calculate Softmax(QK)
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != q.struct_info.dtype:
            attn_weights = astype(attn_weights, q.struct_info.dtype)
        # Calculate Softmax(QK)V
        attn_output = nn.emit(matmul(attn_weights, v))
        # Apply output projection
        attn_output = self.c_proj(
            reshape(
                permute_dims(attn_output, [0, 2, 1, 3]),
                (batch_size, seq_len, self.embed_dim),
            )
        )
        return attn_output, past_key_value


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx=None):
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon, dtype=config.dtype
        )
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon, dtype=config.dtype
        )
        self.mlp = GPT2MLP(inner_dim, config, dtype=config.dtype)
        self.dtype = config.dtype

    def forward(
        self,
        hidden_states,
        all_seq_len_shape: relax.Expr,
        past_key_value: Tuple[relax.Expr],
        attention_mask: Optional[relax.Expr] = None,
    ):
        attn_input = self.ln_1(hidden_states)
        attn_output, present_key_value = self.attn(
            attn_input, all_seq_len_shape, past_key_value, attention_mask
        )

        # residual connection
        attn_output = nn.emit(attn_output + hidden_states)

        mlp_input = self.ln_2(attn_output)
        mlp_output = self.mlp(mlp_input)

        # residual connection
        hidden_states = nn.emit(astype(mlp_output, self.dtype) + attn_output)

        return hidden_states, present_key_value


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        self.wte = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.n_embd,
            dtype=config.dtype,
        )
        self.wpe = Embedding(
            num_embeddings=config.n_positions,
            embedding_dim=config.n_embd,
            dtype=config.dtype,
        )

        self.h = ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(
            hidden_size=config.n_embd, dtype=config.dtype, eps=config.layer_norm_epsilon
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        batch_size, seq_length = input_ids.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]

        # Token Embeddings
        t_embd = self.wte(input_ids)

        # Position Embeddings
        offset = seq_length_with_past - seq_length
        hidden_states = apply_position_embedding(t_embd, self.wpe.weight, offset=offset)

        attention_mask = _prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            dtype=hidden_states.struct_info.dtype,
        )

        present_kv_cache = []
        for i, block in enumerate(self.h):
            past_key_value = (
                (past_key_values[i * 2], past_key_values[i * 2 + 1])
                if past_key_values is not None
                else None
            )
            hidden_states, (present_k_cache, present_v_cache) = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
            )
            present_kv_cache.append(present_k_cache)
            present_kv_cache.append(present_v_cache)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states, present_kv_cache


class GPT2ForCausalLM(nn.Module):
    def __init__(self, config: GPT2Config):
        self.dtype = config.dtype

        self.transformer = GPT2Model(config)
        self.lm_head = Linear(
            in_features=config.n_embd,
            out_features=config.vocab_size,
            bias=True,
            dtype=config.dtype,
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: relax.Expr,
    ):
        hidden_states, key_value_cache = self.transformer(
            input_ids=input_ids,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def te_slice_last(x: te.Tensor):
            bsz, seq_len, n_embd = x.shape
            return te.compute(
                shape=(bsz, 1, n_embd),
                fcompute=lambda i, _, k: x[i, seq_len - 1, k],
                name="slice_last",
            )

        hidden_states = nn.emit_te(
            te_slice_last,
            hidden_states,
            primfunc_name_hint="slice_last",
        )
        if hidden_states.struct_info.dtype != self.dtype:
            hidden_states = nn.emit(astype(hidden_states, self.dtype))

        logits = self.lm_head(hidden_states)

        if logits.struct_info.dtype != "float32":
            logits = nn.emit(astype(logits, "float32"))

        return logits, key_value_cache


def get_param_quant_kind(
    name: str, param_info: relax.TensorStructInfo
) -> ParamQuantKind:
    if "wte.weight" in name:
        return ParamQuantKind.embedding_table
    elif "lm_head.weight" in name:
        return ParamQuantKind.final_fc_weight
    elif "wpe" not in name and param_info.ndim == 2 and name.endswith(".weight"):
        return ParamQuantKind.linear_weight
    else:
        return ParamQuantKind.others


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPT2Config,
    quant_scheme: QuantizationScheme,
    batch_size: int,
) -> None:
    func_name = "prefill"

    batch_size = tvm.tir.IntImm("int64", batch_size)
    seq_len = tvm.tir.Var("n", "int64")
    all_seq_len = tvm.tir.Var("m", "int64")
    with bb.function(func_name):
        model = GPT2ForCausalLM(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_ids = nn.Placeholder(
            (batch_size, seq_len), dtype="int32", name="input_ids"
        )
        all_seq_len_shape = relax.Var(
            "all_seq_len", relax.ShapeStructInfo((all_seq_len,))
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.n_layer * 2)]
            ),
        )

        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids=input_ids,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ] + model.parameters()

            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: GPT2Config,
    quant_scheme: QuantizationScheme,
    batch_size: int,
) -> None:
    func_name = "decode"

    bsz = tvm.tir.IntImm("int64", batch_size)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.Var("n", "int64")

    with bb.function(func_name):
        model = GPT2ForCausalLM(config)
        param_manager.register_params(
            model, func_name, quant_scheme, get_param_quant_kind
        )

        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        all_seq_len_shape = relax.Var(
            "all_seq_len", relax.ShapeStructInfo((all_seq_len,))
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.n_layer * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids=input_ids,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_kv_cache_func(
    bb: relax.BlockBuilder, config: GPT2Config, batch_size: int
) -> None:
    init_shape = relax.ShapeExpr(
        (
            batch_size * config.max_sequence_length,
            config.n_head,
            config.n_embd // config.n_head,
        )
    )
    with bb.function("create_kv_cache", []):
        with bb.dataflow():
            zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
            caches = []
            f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
            for _ in range(config.n_layer * 2):
                caches.append(
                    bb.emit(
                        relax.Call(
                            f_kv_cache_create,
                            args=[zeros, init_shape, relax.PrimValue(0)],
                            sinfo_args=[relax.ObjectStructInfo()],
                        )
                    )
                )
            gv = bb.emit_output(caches)
        bb.emit_func_output(gv)


def create_softmax_func(bb: relax.BlockBuilder, config: GPT2Config) -> None:
    with bb.function("softmax_with_temperature"):
        logits = nn.Placeholder(
            (1, 1, config.vocab_size), dtype="float32", name="logits"
        )
        temperature = nn.Placeholder((), dtype="float32", name="temperature")
        with bb.dataflow():
            div = bb.emit(relax.op.divide(logits, temperature))
            softmax = bb.emit(relax.op.nn.softmax(div, axis=-1))
            gv = bb.emit_output(softmax)
        bb.emit_func_output(gv, [logits, temperature])


def get_model(args: argparse.Namespace, hf_config):
    model = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    batch_size = args.batch_size

    config = GPT2Config(
        **hf_config,
        dtype=dtype,
    )
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len
    elif config.max_sequence_length is None:
        config.max_sequence_length = 2048

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    create_encoding_func(bb, param_manager, config, args.quantization, batch_size)
    create_decoding_func(bb, param_manager, config, args.quantization, batch_size)
    create_kv_cache_func(bb, config, batch_size)
    create_softmax_func(bb, config)
    create_metadata_func(
        bb,
        model_name=model,
        max_window_size=config.max_sequence_length,
        stop_tokens=[0],
        add_prefix_space=False,
    )

    mod = bb.get()
    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, relax.Function):
            mod[gv] = func.with_attr(
                "tir_var_upper_bound",
                {
                    "n": config.max_sequence_length,
                    "m": config.max_sequence_length,
                },
            )

    mod, pidx2pname = param_manager.quantization_transform(mod)
    pname2binname = load_torch_pname2binname_map(
        args.model_path, set(pidx2pname.values())
    )

    def f_convert_param_bkwd(torch_pname: str, raw_param):
        # raw_param: numpy.ndarray
        if "ln_" in torch_pname:
            return [(torch_pname, raw_param.astype("float32"))]
        elif ".attn.bias" in torch_pname:
            return [(torch_pname, raw_param.astype("bool"))]
        else:
            return [(torch_pname, raw_param.astype(dtype))]

    args.pidx2pname = pidx2pname
    args.pname2binname = pname2binname
    args.f_convert_pname_fwd = lambda pname: pname
    args.f_convert_param_bkwd = f_convert_param_bkwd

    return mod, [None] * len(pidx2pname)

    raise ValueError(f"Unsupported model {model}")
