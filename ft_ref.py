import numpy as np
import torch

def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


torch.classes.load_library("lib/libth_transformer.so")
torch.classes.load_library("lib/libgemm_dq_unit_ops.so")

unpack_packed_int4s = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8
pack_int4s = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
fused_gemm_dq = torch.ops.gemm_dq_unit_ops.fused_gemm_dq
fused_gemm_dq_bias_act = torch.ops.gemm_dq_unit_ops.fused_gemm_dq_bias_act
bench = torch.ops.gemm_dq_unit_ops.benchmark_against_cublas_fp
preprocess_weights_for_mixed_gemm = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

symmetric_quantizer = torch.ops.fastertransformer._symmetric_quantize_last_axis_of_batched_matrix

compute_type = torch.float16
weight_dtype = torch.quint4x2

m = 64
k = 64
n = 128
torch_weights_cpu = torch.from_numpy(np.load("y.npy"))
ref_torch_weights, processed_torch_weights, torch_weight_scales = symmetric_quantizer(torch_weights_cpu, weight_dtype)

# print(torch_weight_scales)

# np.save("y.npy", torch_weights_cpu.numpy())
# np.save("weights_preprocessed.npy", ref_torch_weights.numpy())
# np.save("weights_packed.npy", processed_torch_weights.numpy())
# np.save("scales.npy", torch_weight_scales.numpy())
ref_torch_weights = unpack_packed_int4s(ref_torch_weights)
ref_torch_weights = ref_torch_weights.to("cuda")
processed_torch_weights = processed_torch_weights.to("cuda")
torch_weight_scales = torch_weight_scales.to("cuda")

torch_activations = torch.randn(size=(m, k), dtype=compute_type, device="cuda")

scales_unsqueezed = torch_weight_scales.unsqueeze(0)
casted_weights = ref_torch_weights.to(torch_activations.dtype)
dequantized_weights = torch.multiply(casted_weights, scales_unsqueezed)

ref = torch.matmul(torch_activations, dequantized_weights).cpu().numpy()
out = fused_gemm_dq(torch_activations, processed_torch_weights, torch_weight_scales).cpu().numpy()

print(np.max(np.abs(ref - out)), np.mean(np.abs(ref - out)))

np.save("out.npy", out)
np.save("ref.npy", ref)
