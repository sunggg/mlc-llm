import torch
from mlc_serve.model.sampler import SamplingMetadata, adjust_logits
from mlc_serve.engine import SamplingParams


def _test_top_p_top_k():
    batch_size = 1
    vocab_size = 32000
    shape = (batch_size, vocab_size)
    dtype = torch.float32
    dev = "cuda"
    logits = torch.rand(shape, dtype=dtype, device=dev)

    sampling_param = SamplingParams(
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        logit_bias={1: -1, 3: 1, 2: 2},
        # logprobs=logprobs,
        # top_logprobs=top_logprobs,
    )

    _copy_stream: torch.cuda.Stream = torch.cuda.Stream()
    with torch.cuda.stream(_copy_stream):
        sampling_metadata = SamplingMetadata.from_sampling_params(
            [sampling_param],
            dtype,
            dev,
            vocab_size,
        )
    torch.cuda.current_stream().wait_stream(_copy_stream)

    new_logits = adjust_logits(logits, sampling_metadata, batch_size, vocab_size)
    print(new_logits)


if __name__ == "__main__":
    _test_top_p_top_k()
