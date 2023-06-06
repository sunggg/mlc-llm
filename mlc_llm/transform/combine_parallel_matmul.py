from tvm.relax.dpl import PatternContext, is_op, rewrite_bindings, wildcard
from tvm.script import relax as R


def combine_parallel_transposed_matmul(f, num_branches):
    with PatternContext() as ctx:
        inp_pat = wildcard()

        weight_patterns = []
        matmul_patterns = []

        for _ in range(num_branches):
            w_pat = wildcard()
            weight_patterns.append(w_pat)
            matmul_patterns.append(
                is_op("relax.matmul")(inp_pat, is_op("relax.permute_dims")(w_pat))
            )

    def rewriter(matchings, _):
        inp = matchings[inp_pat]

        weights = [matchings[w_pat] for w_pat in weight_patterns]
        concat = R.concat(weights, axis=0)
        matmul = R.matmul(inp, R.permute_dims(concat))

        replacements = {}

        sections = []
        ind = 0
        for i, matmul_pat in enumerate(matmul_patterns[:-1]):
            width = weights[i].struct_info.shape[0]
            ind += width
            sections.append(int(ind))

        if len(inp.struct_info.shape) == 3:
            slice_axis = 2
        elif len(inp.struct_info.shape) == 2:
            slice_axis = 1
        else:
            assert False

        chunks = R.split(matmul, sections, slice_axis)

        for i, matmul_pat in enumerate(matmul_patterns):
            bound_var = matchings[matmul_pat]
            replacements[bound_var] = chunks[i]

        return replacements

    return rewrite_bindings(ctx, rewriter, f)
