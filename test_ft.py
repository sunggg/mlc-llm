import numpy as np

import tvm
from tvm import relax
from tvm.relax.testing import get_relax_matmul_module
import mlc_llm
from mlc_llm import utils
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import relax as R


dtype = "float16"
x_shape = (64, 64)
y_shape = (64, 64)

# x_shape = (512, 4096)
# y_shape = (4096, 4096)

use_bias = False

if use_bias:
    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        dtype,
        transposed_y=True,
        bias_shape=(1, y_shape[0]),
        residual_bin_op=R.add,
    )
else:
    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        dtype,
        transposed_y=True,
        activation=R.nn.silu,
        residual_bin_op=R.multiply,
    )

x = np.random.randn(*x_shape).astype("float16")
# y = np.load("/home/masa/projects/ml/deploy/FasterTransformer/build-docker/y.npy").transpose()
# y = np.random.randn(*y_shape).astype("float16")
y = np.random.normal(0, 0.002, size=y_shape).astype("float16")

bias = np.random.randn(1, y_shape[0]).astype("float16")

mod = mlc_llm.transform.RowWiseQuantize(dtype="float32")(mod)

mod = partition_for_cutlass(mod)
print(mod)
mod = relax.transform.RunCodegen(
    {"cutlass": {"sm": 80, "find_first_valid": False}},
)(mod)

mod = relax.pipeline.get_pipeline()(mod)
mod = relax.transform.LiftTransformParams()(mod)
mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, ["main"])

ex = relax.build(mod_transform, target="llvm")
vm = relax.vm.VirtualMachine(ex, tvm.cpu(0))

if use_bias:
    packed_weight, scales, bias_preprocessed = vm["main_transform_params"]((tvm.nd.array(y), tvm.nd.array(bias)))
else:
    packed_weight, scales = vm["main_transform_params"]((tvm.nd.array(y),))

out_weight = packed_weight.numpy()
out_scales = scales.numpy()

# print(mod_deploy.without_attr("external_mods").without_attr("const_name_to_constant"))

dev = tvm.device("cuda", 0)
ex = relax.build(mod_deploy, target="cuda")
vm = relax.vm.VirtualMachine(ex, dev)

if use_bias:
    inp = [tvm.nd.array(x, dev), (packed_weight.copyto(dev), scales.copyto(dev), bias_preprocessed.copyto(dev))]
else:
    inp = [tvm.nd.array(x, dev), (packed_weight.copyto(dev), scales.copyto(dev))]

out = vm["main"](*inp).numpy()

# ref = np.load("/home/masa/projects/ml/deploy/FasterTransformer/build-docker/out.npy")

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

if use_bias:
    ref = np.dot(x, y.transpose()) + bias + x
else:
    ref = silu(np.dot(x, y.transpose())) * x

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))
print(out)
print(ref)
